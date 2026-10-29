# SPDX-License-Identifier: Apache-2.0
"""Tests for the strict-lossless KV codec on the LMCache-driven transfer path.

CPU tests cover the configuration and the store tally contract. GPU tests run
a real cache server with ``--lossless-kv-codec`` and check that stored KV
comes back bit-exact from L1 and, after L1 eviction, from a filesystem L2.
"""

# Standard
from pathlib import Path
from typing import Generator
import multiprocessing as mp
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.config import LosslessKVConfig, MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.lossless_kv_transfer import StoreTally, codec_byte_layout

_HOST = "localhost"
_PORT = 5613
_CHUNK = 256
_PAGE = 16
_BLOCKS_PER_CHUNK = _CHUNK // _PAGE
_LAYERS, _HEADS, _HEAD_SIZE = 4, 8, 128
_NUM_PAGES = 2048
# L1 holds every L1 test below the eviction watermark; the eviction test
# fills it several times over to push its early chunks out to L2.
_L1_BYTES = 128 * 1024 * 1024
_TIMEOUT = 600.0


# =============================================================================
# CPU tests
# =============================================================================


def test_config_disabled_by_default() -> None:
    assert MPServerConfig().lossless_kv.enabled is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"store_pool_chunks": 0},
        {"capacity_ratio": 0.0},
        {"capacity_ratio": 1.5},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        LosslessKVConfig(enabled=True, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"supported_transfer_mode": "engine_driven"}, {"engine_type": "blend"}],
)
def test_config_requires_default_lmcache_driven_transfer(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MPServerConfig(lossless_kv=LosslessKVConfig(enabled=True), **kwargs)


def test_cli_flags_parse_into_config() -> None:
    # Standard
    import argparse

    # First Party
    from lmcache.v1.multiprocess.config import (
        add_mp_server_args,
        parse_args_to_mp_server_config,
    )

    parser = add_mp_server_args(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--lossless-kv-codec",
            "--lossless-kv-codebooks",
            "books",
            "--lossless-kv-batch",
            "4",
            "--lossless-kv-store-pool",
            "12",
        ]
    )
    config = parse_args_to_mp_server_config(args).lossless_kv
    assert config == LosslessKVConfig(
        enabled=True, codebook_dir="books", batch_size=4, store_pool_chunks=12
    )


def test_codec_byte_layout_is_flat_bytes() -> None:
    layout = codec_byte_layout(8192)
    assert layout.shapes == [torch.Size((8192,))]
    assert layout.dtypes == [torch.uint8]


class _Sized:
    def __init__(self, size: int) -> None:
        self._size = size

    def get_size(self) -> int:
        return self._size


def _key(index: int) -> ObjectKey:
    return ObjectKey(bytes([index]) * 8, "model", 0)


def test_store_tally_reports_once_after_seal_and_batches() -> None:
    reports: list[tuple[int, int]] = []
    tally = StoreTally(lambda count, nbytes: reports.append((count, nbytes)))
    tally.add_batch()
    tally.add_batch()
    tally.batch_done({_key(0): _Sized(4096)}, True)
    tally.seal(True)
    assert reports == []
    tally.batch_done({_key(1): _Sized(8192), _key(2): _Sized(4096)}, True)
    assert reports == [(3, 16384)]


def test_store_tally_reports_on_seal_without_batches() -> None:
    reports: list[tuple[int, int]] = []
    StoreTally(lambda count, nbytes: reports.append((count, nbytes))).seal(True)
    assert reports == [(0, 0)]


@pytest.mark.parametrize("failure", ["batch", "seal"])
def test_store_tally_reports_nothing_on_failure(failure: str) -> None:
    reports: list[tuple[int, int]] = []
    tally = StoreTally(lambda count, nbytes: reports.append((count, nbytes)))
    tally.add_batch()
    tally.batch_done({_key(0): _Sized(4096)}, failure != "batch")
    tally.seal(failure != "seal")
    assert reports == [(0, 0)]


# =============================================================================
# GPU tests against a real cache server
# =============================================================================


def _gpu_ready() -> bool:
    if not (torch_dev.is_available() and torch_device_type == "cuda"):
        return False
    try:
        buf = torch.empty(1024, device=torch_device_type)
        return buf.untyped_storage()._share_cuda_() is not None
    except Exception:
        return False


gpu = pytest.mark.skipif(not _gpu_ready(), reason="requires CUDA IPC")


def _run_server(l2_dir: str) -> None:
    """Server process entry: codec on, small L1, filesystem L2."""
    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
        FSNativeL2AdapterConfig,
    )
    from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
    from lmcache.v1.multiprocess.server import run_cache_server

    mp_config = MPServerConfig(
        transport="zmq",
        host=_HOST,
        port=_PORT,
        chunk_size=_CHUNK,
        lossless_kv=LosslessKVConfig(enabled=True, batch_size=4),
    )
    storage_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=_L1_BYTES, use_lazy=False
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[FSNativeL2AdapterConfig(base_path=l2_dir)]
        ),
    )
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_config,
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Generator:
    """A request client of a codec-enabled cache server process."""
    if not _gpu_ready():
        pytest.skip("requires CUDA IPC")
    # First Party
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

    l2_dir: Path = tmp_path_factory.mktemp("lossless_l2")
    mp.set_start_method("spawn", force=True)
    process = mp.Process(target=_run_server, args=(str(l2_dir),), daemon=True)
    process.start()
    request_client = RequestClientFactory.create(f"tcp://{_HOST}:{_PORT}")
    try:
        deadline = time.monotonic() + _TIMEOUT
        while True:
            try:
                assert request_client.get_chunk_size().result(timeout=5) == _CHUNK
                break
            except Exception:
                if time.monotonic() > deadline or not process.is_alive():
                    raise
        yield request_client
    finally:
        request_client.close()
        process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join()


def _compressible_kv(
    dtype: torch.dtype, seed: int, fused: bool = False
) -> list[torch.Tensor]:
    """KV layers with a narrow exponent range.

    Separate K/V layers are ``[2, pages, page, heads, head]``; fused ones are
    vLLM's ``[pages, page, heads, 2 * head]``.
    """
    generator = torch.Generator(device=torch_device_type).manual_seed(seed)
    shape = (
        (_NUM_PAGES, _PAGE, _HEADS, 2 * _HEAD_SIZE)
        if fused
        else (2, _NUM_PAGES, _PAGE, _HEADS, _HEAD_SIZE)
    )
    layers = []
    for _ in range(_LAYERS):
        values = torch.randn(shape, generator=generator, device=torch_device_type)
        if dtype == torch.uint8:
            layers.append(values.to(torch.float8_e4m3fn).view(torch.uint8))
        else:
            layers.append(values.to(dtype))
    return layers


def _register(client, instance_id: int, layers: list[torch.Tensor], model: str):
    # First Party
    from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper

    client.register_kv_cache(
        instance_id,
        [CudaIPCWrapper(layer) for layer in layers],
        model,
        1,
        EngineType.VLLM,
        {},
        [],
    ).result(timeout=_TIMEOUT)


def _event_handle(keepalive: list) -> bytes:
    # First Party
    from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

    backend = get_event_ipc_backend(0)
    event = backend.create_event(0)
    backend.record_event(event, None)
    keepalive.append(event)
    return backend.export_event(event, 0)


def _cache_key(
    model: str, seed: int, num_chunks: int, request_id: str
) -> IPCCacheServerKey:
    tokens = [seed * 1000 + i // _CHUNK for i in range(num_chunks * _CHUNK)]
    return IPCCacheServerKey.from_token_ids(
        model, 1, 0, tokens, start=0, end=len(tokens), request_id=request_id
    )


def _store(client, key, instance_id: int, blocks: list[int], keepalive) -> None:
    result = (
        client.store(key, instance_id, [blocks], _event_handle(keepalive))
        .to_device_future()
        .result(timeout=_TIMEOUT)
    )
    assert result is True


def _lookup(client, key: IPCCacheServerKey) -> int:
    """Look ``key`` up until the prefetch settles; return the hit chunks."""
    lookup_key = key.no_worker_id_version()
    client.lookup(lookup_key, 1).result(timeout=_TIMEOUT)
    while True:
        hit = client.query_prefetch_status(lookup_key.request_id).result(
            timeout=_TIMEOUT
        )
        if hit is not None:
            return hit


def _retrieve(
    client, key, instance_id: int, blocks: list[int], skip: int, keepalive
) -> bool:
    return (
        client.retrieve(key, instance_id, [blocks], _event_handle(keepalive), skip)
        .to_device_future()
        .result(timeout=_TIMEOUT)
    )


def _blocks(first_chunk: int, num_chunks: int) -> list[int]:
    start = first_chunk * _BLOCKS_PER_CHUNK
    return list(range(start, start + num_chunks * _BLOCKS_PER_CHUNK))


def _pages(layer: torch.Tensor) -> torch.Tensor:
    """View a layer with its page axis first (fused layers already are)."""
    return layer if layer.dim() == 4 else layer.transpose(0, 1)


def _assert_blocks_equal(
    layers: list[torch.Tensor], src: list[int], dst: list[int]
) -> None:
    src_ids = torch.tensor(src, device=layers[0].device)
    dst_ids = torch.tensor(dst, device=layers[0].device)
    for layer in layers:
        assert torch.equal(_pages(layer)[src_ids], _pages(layer)[dst_ids])


@gpu
@pytest.mark.parametrize(
    ("dtype", "fused", "instance_id"),
    [
        (torch.bfloat16, False, 7001),
        (torch.uint8, False, 7002),
        (torch.bfloat16, True, 7004),
        (torch.uint8, True, 7005),
    ],
)
def test_codec_round_trips_from_l1_bit_exact(
    client, dtype: torch.dtype, fused: bool, instance_id: int
) -> None:
    """A multi-batch store comes back bit-exact, honoring skip_first_n_tokens."""
    model = f"lossless-l1-{dtype}-{fused}"
    layers = _compressible_kv(dtype, seed=instance_id, fused=fused)
    keepalive: list = []
    _register(client, instance_id, layers, model)
    try:
        num_chunks = 6  # two codec batches of 4, the first split per launch
        _store(
            client,
            _cache_key(model, 1, num_chunks, "store"),
            instance_id,
            _blocks(0, num_chunks),
            keepalive,
        )
        torch_dev.synchronize()
        time.sleep(1.0)  # records commit on the codec worker
        key = _cache_key(model, 1, num_chunks, "retrieve")
        assert _lookup(client, key) == num_chunks

        dst = _blocks(100, num_chunks)
        for layer in layers:
            _pages(layer)[dst] = 0
        skip = _CHUNK // 2
        assert _retrieve(client, key, instance_id, dst, skip, keepalive)
        torch_dev.synchronize()

        skipped = skip // _PAGE
        _assert_blocks_equal(layers, _blocks(0, num_chunks)[skipped:], dst[skipped:])
        for layer in layers:
            assert not _pages(layer)[dst[:skipped]].any()
    finally:
        client.unregister_kv_cache(instance_id).result(timeout=_TIMEOUT)


@gpu
def test_codec_round_trips_from_l2_after_l1_eviction(client) -> None:
    """Records evicted from L1 are prefetched back from L2 and decode exactly."""
    instance_id, model = 7003, "lossless-l2"
    layers = _compressible_kv(torch.bfloat16, seed=instance_id)
    keepalive: list = []
    _register(client, instance_id, layers, model)
    try:
        num_early = 4
        for i in range(num_early):
            _store(
                client,
                _cache_key(model, i, 1, f"early-{i}"),
                instance_id,
                _blocks(i, 1),
                keepalive,
            )
        torch_dev.synchronize()
        time.sleep(2.0)  # records commit and reach L2
        # Fill L1 several times over so the early chunks are evicted. Stores
        # into a full L1 are skipped, so each round of half an L1 waits for
        # the once-a-second eviction loop before the next.
        raw_chunk_bytes = _CHUNK * _LAYERS * 2 * _HEADS * _HEAD_SIZE * 2
        per_round = _L1_BYTES // 2 // raw_chunk_bytes
        filler = 0
        for _ in range(8):
            for _ in range(per_round):
                _store(
                    client,
                    _cache_key(model, 100 + filler, 1, f"fill-{filler}"),
                    instance_id,
                    _blocks(num_early, 1),
                    keepalive,
                )
                filler += 1
            torch_dev.synchronize()
            time.sleep(1.5)

        dst_first = 100
        for i in range(num_early):
            key = _cache_key(model, i, 1, f"reload-{i}")
            assert _lookup(client, key) == 1
            dst = _blocks(dst_first + i, 1)
            for layer in layers:
                _pages(layer)[dst] = 0
            assert _retrieve(client, key, instance_id, dst, 0, keepalive)
            torch_dev.synchronize()
            _assert_blocks_equal(layers, _blocks(i, 1), dst)
    finally:
        client.unregister_kv_cache(instance_id).result(timeout=_TIMEOUT)


@gpu
@pytest.mark.parametrize(
    ("dtype", "instance_id"), [(torch.bfloat16, 7006), (torch.uint8, 7007)]
)
def test_codec_round_trips_through_engine_blocks(
    client, dtype: torch.dtype, instance_id: int
) -> None:
    """Chunk-sized pages in one allocation store and retrieve bit-exact.

    Such caches are packed from and decoded into the engine blocks directly;
    a raw chunk and a partly skipped chunk exercise the fallbacks.
    """
    model = f"lossless-engine-{dtype}"
    shape = (_LAYERS, 64, _CHUNK, _HEADS, 2 * _HEAD_SIZE)
    generator = torch.Generator(device=torch_device_type).manual_seed(instance_id)
    values = torch.randn(shape, generator=generator, device=torch_device_type)
    alloc = (
        values.to(torch.float8_e4m3fn).view(torch.uint8)
        if dtype == torch.uint8
        else values.to(dtype)
    )
    noise = torch.randint(
        0,
        256,
        alloc[:, 3].view(torch.uint8).shape,
        dtype=torch.uint8,
        generator=generator,
        device=torch_device_type,
    )
    alloc[:, 4].view(torch.uint8).copy_(noise)  # chunk 3 stays raw
    layers = [alloc[layer] for layer in range(_LAYERS)]
    keepalive: list = []
    _register(client, instance_id, layers, model)
    try:
        num_chunks = 6
        # Block 0 is the engine's null block, which stores are skipped for.
        src_blocks = list(range(1, 1 + num_chunks))
        _store(
            client,
            _cache_key(model, 1, num_chunks, "store"),
            instance_id,
            src_blocks,
            keepalive,
        )
        torch_dev.synchronize()
        time.sleep(1.0)  # records commit on the codec worker
        key = _cache_key(model, 1, num_chunks, "retrieve")
        assert _lookup(client, key) == num_chunks

        dst = list(range(40, 40 + num_chunks))
        alloc[:, dst] = 0
        # Skips are page-granular: chunk 0 is skipped, while the page of the
        # partly skipped chunk 1 is restored whole.
        skip = _CHUNK + _CHUNK // 2
        assert _retrieve(client, key, instance_id, dst, skip, keepalive)
        torch_dev.synchronize()

        src = alloc[:, src_blocks].view(torch.uint8)
        out = alloc[:, dst].view(torch.uint8)
        assert not out[:, 0].any()
        assert torch.equal(out[:, 1:], src[:, 1:])
    finally:
        client.unregister_kv_cache(instance_id).result(timeout=_TIMEOUT)

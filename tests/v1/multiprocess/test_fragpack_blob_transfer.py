# SPDX-License-Identifier: Apache-2.0
"""FragPack block blobs through a real cache server.

The test process plays the vLLM worker: it fills a FragPack store over its
paged KV cache, registers the cache with the ``kv_codec`` layout hint, and
shares the store tables. A cache server process then stores closed blocks as
blobs and installs them into other blocks on retrieve, which must gather the
written KV bit-exactly. Stores before the tables are shared, and stores of
open blocks, must commit nothing.
"""

# Standard
from typing import Generator
import multiprocessing as mp
import time

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.lossless_kv.fragpack.hbm_kv import page_bytes
from lmcache.v1.lossless_kv.fragpack.registry import KV_CODEC_BF16, KV_CODEC_FP8
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from tests.v1.lossless_kv.test_fragpack_hbm_store import H, T, _Harness, _prefill

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

_HOST = "localhost"
_PORT = 5617
_L1_BYTES = 64 * 1024 * 1024
_TIMEOUT = 600.0


def _run_server() -> None:
    """Server process entry: LMCache-driven transfer, chunk = one block."""
    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
    from lmcache.v1.multiprocess.config import LosslessKVConfig, MPServerConfig
    from lmcache.v1.multiprocess.server import run_cache_server

    run_cache_server(
        # Two-block batches, so retrieves span batches and their staging sets.
        mp_config=MPServerConfig(
            transport="zmq",
            host=_HOST,
            port=_PORT,
            chunk_size=T,
            lossless_kv=LosslessKVConfig(batch_size=2),
        ),
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=_L1_BYTES, use_lazy=False
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


@pytest.fixture(scope="module")
def client() -> Generator:
    """A request client of a cache server process."""
    # First Party
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

    mp.set_start_method("spawn", force=True)
    process = mp.Process(target=_run_server, daemon=True)
    process.start()
    request_client = RequestClientFactory.create(f"tcp://{_HOST}:{_PORT}")
    try:
        deadline = time.monotonic() + _TIMEOUT
        while True:
            try:
                assert request_client.get_chunk_size().result(timeout=5) == T
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


def _event_handle(keepalive: list) -> bytes:
    # First Party
    from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

    backend = get_event_ipc_backend(0)
    event = backend.create_event(0)
    backend.record_event(event, None)
    keepalive.append(event)
    return backend.export_event(event, 0)


def _key(model: str, seed: int, chunks: int, request_id: str) -> IPCCacheServerKey:
    tokens = [seed * 1000 + i // T for i in range(chunks * T)]
    return IPCCacheServerKey.from_token_ids(
        model, 1, 0, tokens, start=0, end=len(tokens), request_id=request_id
    )


def _store(client, key, instance_id: int, blocks: list[int], keepalive) -> None:
    ok = (
        client.store(key, instance_id, [blocks], _event_handle(keepalive))
        .to_device_future()
        .result(timeout=_TIMEOUT)
    )
    assert ok is True


def _lookup(client, key: IPCCacheServerKey) -> int:
    lookup_key = key.no_worker_id_version()
    client.lookup(lookup_key, 1).result(timeout=_TIMEOUT)
    while True:
        hit = client.query_prefetch_status(lookup_key.request_id).result(
            timeout=_TIMEOUT
        )
        if hit is not None:
            return hit


def _register(client, instance_id: int, h: _Harness, model: str) -> None:
    # First Party
    from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper

    # vLLM's view of a FragPack page: one head slot of page / T bytes a token.
    caches = [c.view(c.shape[0], T, 1, c.shape[1] // T) for c in h.caches]
    client.register_kv_cache(
        instance_id,
        [CudaIPCWrapper(c) for c in caches],
        model,
        1,
        EngineType.VLLM,
        {
            "kv_codec": KV_CODEC_FP8 if h.fp8 else KV_CODEC_BF16,
            "num_kv_heads": H,
            "kv_layout": "NHD",
        },
        [],
    ).result(timeout=_TIMEOUT)


def _share(client, instance_id: int, h: _Harness) -> None:
    # First Party
    from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper

    tables = h.store.blob_tables()
    client.register_fragpack_store(
        instance_id,
        [CudaIPCWrapper(t) for t in tables.tensors()],
        tables.arena_stride,
        tables.ovf_rec,
        tables.arena_rec,
        tables.digest,
    ).result(timeout=_TIMEOUT)


@pytest.mark.parametrize(
    ("fp8", "instance_id"), [(False, 7101), (True, 7102)], ids=["bf16", "fp8"]
)
def test_blobs_round_trip_through_server(client, fp8: bool, instance_id: int) -> None:
    model = f"fragpack-{fp8}"
    h = _Harness(np.random.default_rng(0), page_bytes(H, fp8), fp8=fp8)
    _prefill(h)  # closes blocks 0, 1, 4, 8, 9; block 10 stays open
    torch.cuda.synchronize()
    keepalive: list = []
    _register(client, instance_id, h, model)
    try:
        # Block 0 is the engine's null block, which LMCache never stores.
        src, dst = [1, 4, 8, 9], [6, 7, 3, 11]
        early = _key(model, 1, len(src), "early")
        _store(client, early, instance_id, src, keepalive)
        _share(client, instance_id, h)
        opened = _key(model, 2, 1, "open")
        _store(client, opened, instance_id, [10], keepalive)
        stored = _key(model, 3, len(src), "store")
        _store(client, stored, instance_id, src, keepalive)
        torch.cuda.synchronize()
        time.sleep(1.0)  # blobs commit on the record worker
        assert _lookup(client, early) == 0
        assert _lookup(client, opened) == 0
        # Repeated retrieves reinstall the same blobs; with two batches each,
        # the third wraps around the staging sets.
        for attempt in range(3):
            retrieve = _key(model, 3, len(src), f"retrieve-{attempt}")
            assert _lookup(client, retrieve) == len(src)
            ok = (
                client.retrieve(
                    retrieve, instance_id, [dst], _event_handle(keepalive), T
                )
                .to_device_future()
                .result(timeout=_TIMEOUT)
            )
            assert ok is True
        torch.cuda.synchronize()
        for s_b, d_b in zip(src[1:], dst[1:], strict=True):
            h.ref[:, :, d_b] = h.ref[:, :, s_b]
        h.check_gather({b: T for b in dst[1:]})
        # The skipped block was not installed: it still has no records.
        lengths, status = h.store.blob_port().export_blocks(
            [dst[0]],
            torch.empty(
                h.store.blob_port().blob_stride, dtype=torch.uint8, device="cuda"
            ),
            [0],
        )
        assert status.item() != 0
    finally:
        client.unregister_kv_cache(instance_id).result(timeout=_TIMEOUT)

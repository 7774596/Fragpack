# SPDX-License-Identifier: Apache-2.0

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.v1.lossless_kv.chunk_codec import (
    ChunkCapacityError,
    KVMajorChunkCodec,
    kv_major_to_physical,
)
from lmcache.v1.lossless_kv.calibration import (
    calibrate_codebooks,
    load_calibrated_codebooks,
    write_calibration_artifact,
)
from lmcache.v1.lossless_kv.codec import decode_slot
from lmcache.v1.lossless_kv.format import (
    IO_ALIGNMENT,
    CompressedStoreGeometry,
    SlotHeader,
    SlotMode,
)
from lmcache.v1.lossless_kv.fragpack.attn_tables import fp8_bytes
from lmcache.v1.lossless_kv.kv_layout import layer_major_kv_view

_LAYERS, _TOKENS, _HEADS, _HEAD_DIM = 4, 64, 4, 128


def _codec(
    max_chunks: int,
    capacity_ratio: float | None = 0.75,
    *,
    fused_kv: bool = False,
    kv_dtype: torch.dtype = torch.bfloat16,
    codebook_dir: str | None = None,
    kv_head_start: int = 0,
) -> KVMajorChunkCodec:
    return KVMajorChunkCodec(
        num_layers=_LAYERS,
        chunk_tokens=_TOKENS,
        num_kv_heads=_HEADS,
        head_dim=_HEAD_DIM,
        kv_dtype=kv_dtype,
        device=torch.device("cuda"),
        capacity_ratio=capacity_ratio,
        max_chunks=max_chunks,
        fused_kv=fused_kv,
        codebook_dir=codebook_dir,
        kv_head_start=kv_head_start,
    )


def _chunk(codec: KVMajorChunkCodec, seed: int, *, compressible: bool) -> torch.Tensor:
    """Return one contiguous CUDA uint8 kv-major chunk buffer."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, codec.raw_bytes, dtype=np.uint8)
    if compressible:
        common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
        high = common[rng.integers(0, len(common), codec.raw_bytes // 2)]
        high[::997] = 0x7E
        raw[1::2] = high
    return torch.from_numpy(raw).cuda()


def _fp8_chunk(codec: KVMajorChunkCodec, seed: int) -> torch.Tensor:
    """Return one FP8 E4M3 chunk with KV-like skewed sign+exponent fields."""
    rng = np.random.default_rng(seed)
    mantissa = rng.integers(0, 8, codec.raw_bytes, dtype=np.uint8)
    # Sign+exponent symbols concentrate on a few values, as in real KV.
    common = np.array([0x6, 0x7, 0x8, 0x16, 0x17], dtype=np.uint8)
    symbols = common[rng.integers(0, len(common), codec.raw_bytes)]
    symbols[::997] = 0x1E
    return torch.from_numpy((symbols << 3) | mantissa).cuda()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch", [1, 3])
def test_chunk_codec_round_trips_through_capacity_records(batch: int) -> None:
    codec = _codec(max_chunks=3)
    stream = torch.cuda.Stream()
    originals = [_chunk(codec, seed, compressible=True) for seed in range(batch)]
    chunks = [chunk.clone() for chunk in originals]
    lengths = codec.encode(chunks, stream)
    assert all(0 < length <= codec.capacity_bytes for length in lengths)
    # Engines move only the capacity prefix; the rest of the buffer is reused.
    records = [chunk[: codec.capacity_bytes].clone() for chunk in chunks]
    restored = [torch.full_like(chunk, 0xA5) for chunk in originals]
    for target, record in zip(restored, records, strict=True):
        target[: codec.capacity_bytes].copy_(record)
    codec.decode(restored, lengths, stream)
    stream.synchronize()
    for target, original in zip(restored, originals, strict=True):
        assert torch.equal(target, original)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_variable_chunk_codec_round_trips_stored_prefixes() -> None:
    codec = _codec(max_chunks=3, capacity_ratio=None)
    assert not codec.fixed_capacity
    assert codec.capacity_bytes == codec.raw_bytes
    stream = torch.cuda.Stream()
    originals = [
        _chunk(codec, 1, compressible=True),
        _chunk(codec, 2, compressible=False),
        _chunk(codec, 3, compressible=True),
    ]
    chunks = [chunk.clone() for chunk in originals]
    lengths = codec.encode(chunks, stream)
    # Compressible chunks store an aligned record; the noise chunk stays raw.
    assert lengths[1] == codec.raw_bytes
    for length in (lengths[0], lengths[2]):
        assert length < codec.raw_bytes * 0.75
        assert length % IO_ALIGNMENT == 0
    # Engines keep exactly the stored prefix of each buffer.
    restored = [torch.full_like(chunk, 0xA5) for chunk in originals]
    for target, chunk, length in zip(restored, chunks, lengths, strict=True):
        target[:length].copy_(chunk[:length])
    codec.decode(restored, lengths, stream)
    stream.synchronize()
    for target, original in zip(restored, originals, strict=True):
        assert torch.equal(target, original)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_variable_chunk_codec_skips_all_raw_batches() -> None:
    codec = _codec(max_chunks=2, capacity_ratio=None)
    stream = torch.cuda.Stream()
    originals = [_chunk(codec, seed, compressible=False) for seed in (7, 8)]
    chunks = [chunk.clone() for chunk in originals]
    lengths = codec.encode(chunks, stream)
    assert lengths == [codec.raw_bytes] * 2
    codec.decode(chunks, lengths, stream)
    stream.synchronize()
    for chunk, original in zip(chunks, originals, strict=True):
        assert torch.equal(chunk, original)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_codec_decode_reuses_rings_across_batches() -> None:
    codec = _codec(max_chunks=2)
    stream = torch.cuda.Stream()
    originals = [_chunk(codec, seed, compressible=True) for seed in range(6)]
    encoded = [chunk.clone() for chunk in originals]
    lengths: list[int] = []
    for start in range(0, 6, 2):
        lengths += codec.encode(encoded[start : start + 2], stream)
    for start in range(0, 6, 2):
        codec.decode(encoded[start : start + 2], lengths[start : start + 2], stream)
    stream.synchronize()
    for target, original in zip(encoded, originals, strict=True):
        assert torch.equal(target, original)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_codec_decodes_batches_filled_on_another_stream() -> None:
    codec = _codec(max_chunks=2, capacity_ratio=None)
    stream = torch.cuda.Stream()
    originals = [_chunk(codec, seed, compressible=seed != 3) for seed in range(1, 7)]
    records = [chunk.clone() for chunk in originals]
    lengths: list[int] = []
    for start in range(0, 6, 2):
        lengths += codec.encode(records[start : start + 2], stream)
    assert lengths[2] == codec.raw_bytes
    stream.synchronize()
    # Batches rotate through the record sets: each fill runs ahead on its own
    # stream and each decode, read here on the decode stream, waits for it.
    fill = torch.cuda.Stream()
    outputs: list[torch.Tensor] = []
    for start in range(0, 6, 2):
        batch = lengths[start : start + 2]
        buffers = codec.record_buffers(2, fill)
        with torch.cuda.stream(fill):
            for buf, record, length in zip(
                buffers, records[start : start + 2], batch, strict=True
            ):
                buf[:length].copy_(record[:length])
        decoded = codec.decode_records(batch, stream)
        with torch.cuda.stream(stream):
            outputs += [chunk.clone() for chunk in decoded]
    stream.synchronize()
    for output, original in zip(outputs, originals, strict=True):
        assert torch.equal(output, original)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_codec_fails_closed_when_record_exceeds_capacity() -> None:
    codec = _codec(max_chunks=2)
    stream = torch.cuda.Stream()
    chunks = [
        _chunk(codec, 1, compressible=True),
        _chunk(codec, 2, compressible=False),
    ]
    with pytest.raises(ChunkCapacityError):
        codec.encode(chunks, stream)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_codec_rejects_mismatched_buffers() -> None:
    codec = _codec(max_chunks=1)
    stream = torch.cuda.Stream()
    good = _chunk(codec, 1, compressible=True)
    with pytest.raises(ValueError):
        codec.encode([good, good.clone()], stream)
    with pytest.raises(ValueError):
        codec.encode([good[:-IO_ALIGNMENT]], stream)
    with pytest.raises(ValueError):
        codec.decode([good], [codec.capacity_bytes + IO_ALIGNMENT], stream)
    with pytest.raises(ValueError):
        codec.decode([good], [], stream)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_chunk_codec_matches_kv_major_records_and_round_trips() -> None:
    kv_major = _codec(max_chunks=2)
    fused = _codec(max_chunks=2, fused_kv=True)
    stream = torch.cuda.Stream()
    originals = [_chunk(kv_major, seed, compressible=True) for seed in (5, 6)]
    shape = (2, _LAYERS, _TOKENS, _HEADS, _HEAD_DIM)
    # The same KV staged K/V-fused per head, as LMCache stages kv_size == 1.
    fused_originals = [
        kv_major_to_physical(chunk.view(torch.bfloat16).view(shape))
        .contiguous()
        .view(-1)
        .view(torch.uint8)
        for chunk in originals
    ]
    fused_chunks = [chunk.clone() for chunk in fused_originals]
    kv_major_chunks = [chunk.clone() for chunk in originals]
    lengths = fused.encode(fused_chunks, stream)
    assert kv_major.encode(kv_major_chunks, stream) == lengths
    # Both layouts produce the same record format: the fused codec restores
    # records written from the kv-major chunks, and its own.
    fused.decode(kv_major_chunks, lengths, stream)
    fused.decode(fused_chunks, lengths, stream)
    stream.synchronize()
    for original, *targets in zip(
        fused_originals, kv_major_chunks, fused_chunks, strict=True
    ):
        for target in targets:
            assert torch.equal(target, original)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kv_dtype", [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("fused_kv", [False, True])
def test_fp8_chunk_codec_round_trips_variable_records(
    kv_dtype: torch.dtype, fused_kv: bool
) -> None:
    codec = _codec(
        max_chunks=3, capacity_ratio=None, fused_kv=fused_kv, kv_dtype=kv_dtype
    )
    # FP8 chunks hold one byte per scalar.
    assert codec.raw_bytes == 2 * _LAYERS * _TOKENS * _HEADS * _HEAD_DIM
    stream = torch.cuda.Stream()
    originals = [
        _fp8_chunk(codec, 1),
        _chunk(codec, 2, compressible=False),
        _fp8_chunk(codec, 3),
    ]
    chunks = [chunk.clone() for chunk in originals]
    lengths = codec.encode(chunks, stream)
    assert lengths[1] == codec.raw_bytes
    for length in (lengths[0], lengths[2]):
        assert length < codec.raw_bytes
        assert length % IO_ALIGNMENT == 0
    restored = [torch.full_like(chunk, 0xA5) for chunk in originals]
    for target, chunk, length in zip(restored, chunks, lengths, strict=True):
        target[:length].copy_(chunk[:length])
    codec.decode(restored, lengths, stream)
    stream.synchronize()
    for target, original in zip(restored, originals, strict=True):
        assert torch.equal(target, original)


def _layer_major_engine(
    num_blocks: int, dtype: torch.dtype
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return vLLM-style per-layer fused K/V tensors in one allocation."""
    alloc = torch.full(
        (_LAYERS, num_blocks, _TOKENS, _HEADS, 2 * _HEAD_DIM),
        0xA5,
        dtype=torch.uint8 if dtype is not torch.bfloat16 else torch.int16,
        device="cuda",
    ).view(dtype)
    return alloc, [alloc[layer] for layer in range(_LAYERS)]


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_chunk_codec_decodes_records_into_layer_major_engine_blocks(
    kv_dtype: torch.dtype,
) -> None:
    codec = _codec(max_chunks=3, capacity_ratio=None, fused_kv=True, kv_dtype=kv_dtype)
    alloc, layers = _layer_major_engine(9, kv_dtype)
    before = alloc.view(torch.uint8).clone()
    view = layer_major_kv_view(layers, _HEADS, _HEAD_DIM)
    assert view is not None
    codec.attach_engine_cache(view)
    stream = torch.cuda.Stream()
    make = (
        (lambda seed: _chunk(codec, seed, compressible=True))
        if kv_dtype is torch.bfloat16
        else (lambda seed: _fp8_chunk(codec, seed))
    )
    originals = [make(seed) for seed in (11, 12, 13)]
    chunks = [chunk.clone() for chunk in originals]
    lengths = codec.encode(chunks, stream)
    assert all(length < codec.raw_bytes for length in lengths)
    targets = [5, 2, 7]
    blocks = torch.tensor(targets, dtype=torch.int32, device="cuda")
    buffers = codec.record_buffers(3, stream)
    with torch.cuda.stream(stream):
        for buf, chunk, length in zip(buffers, chunks, lengths, strict=True):
            buf[:length].copy_(chunk[:length])
    assert codec.decode_records_to_engine(lengths, blocks, stream)
    stream.synchronize()
    after = alloc.view(torch.uint8)
    for block, original in zip(targets, originals, strict=True):
        # The fused chunk is [layers, tokens, heads, 2 * head_dim].
        expected = original.view(_LAYERS, -1)
        assert torch.equal(after[:, block].reshape(_LAYERS, -1), expected)
    untouched = [b for b in range(9) if b not in targets]
    assert torch.equal(after[:, untouched], before[:, untouched])


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_codec_engine_decode_leaves_raw_batches_to_scratch() -> None:
    codec = _codec(max_chunks=2, capacity_ratio=None, fused_kv=True)
    alloc, layers = _layer_major_engine(4, torch.bfloat16)
    before = alloc.clone()
    codec.attach_engine_cache(layer_major_kv_view(layers, _HEADS, _HEAD_DIM))
    stream = torch.cuda.Stream()
    chunks = [
        _chunk(codec, 21, compressible=True),
        _chunk(codec, 22, compressible=False),
    ]
    lengths = codec.encode(chunks, stream)
    assert lengths[1] == codec.raw_bytes
    codec.record_buffers(2, stream)
    blocks = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    assert not codec.decode_records_to_engine(lengths, blocks, stream)
    stream.synchronize()
    assert torch.equal(alloc, before)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_chunk_codec_encodes_engine_blocks_like_staged_chunks(
    kv_dtype: torch.dtype,
) -> None:
    codec = _codec(max_chunks=3, capacity_ratio=None, fused_kv=True, kv_dtype=kv_dtype)
    alloc, layers = _layer_major_engine(9, kv_dtype)
    codec.attach_engine_cache(layer_major_kv_view(layers, _HEADS, _HEAD_DIM))
    assert codec.engine_store
    stream = torch.cuda.Stream()
    compressible = (
        (lambda seed: _chunk(codec, seed, compressible=True))
        if kv_dtype is torch.bfloat16
        else (lambda seed: _fp8_chunk(codec, seed))
    )
    # The middle chunk is incompressible and stays raw.
    originals = [
        compressible(31),
        _chunk(codec, 32, compressible=False),
        compressible(33),
    ]
    targets = [6, 1, 4]
    engine = alloc.view(torch.uint8)
    for block, original in zip(targets, originals, strict=True):
        engine[:, block].copy_(original.view(_LAYERS, *engine.shape[2:]))
    staged = [chunk.clone() for chunk in originals]
    lengths = codec.encode(staged, stream)
    assert lengths[0] < codec.raw_bytes and lengths[2] < codec.raw_bytes
    assert lengths[1] == codec.raw_bytes

    staging = torch.full((3 * codec.raw_bytes,), 0x5A, dtype=torch.uint8, device="cuda")
    pending = codec.encode_engine_blocks(targets, staging, stream)
    # The pack read the blocks, so the engine may overwrite them.
    with torch.cuda.stream(stream):
        engine[:, targets] = 0
    records = codec.engine_records(pending, staging, stream)
    stream.synchronize()
    assert [record.numel() for record in records] == lengths
    # A raw chunk is stored in the chunk layout; record padding is unspecified.
    assert torch.equal(records[1], originals[1])

    # The records decode back like any stored bytes, whatever their padding.
    buffers = codec.record_buffers(3, stream)
    with torch.cuda.stream(stream):
        for buf, record in zip(buffers, records, strict=True):
            buf[: record.numel()].copy_(record)
    outputs = codec.decode_records(lengths, stream)
    stream.synchronize()
    for output, original in zip(outputs, originals, strict=True):
        assert torch.equal(output, original)


def test_layer_major_kv_view_rejects_scattered_layers() -> None:
    alloc = torch.zeros(3, 2, 4, 2, 8)
    adjacent = [alloc[i] for i in range(3)]
    view = layer_major_kv_view(adjacent, 2, 4)
    assert view is not None
    assert tuple(view.shape) == (2, 3, 2, 4, 2, 4)
    # Logical [block, layer, kv, token, head, dim] aliases the layers.
    assert view[1, 2, 1, 3, 0, 2].data_ptr() == adjacent[2][1, 3, 0, 4 + 2].data_ptr()
    assert layer_major_kv_view([adjacent[0], adjacent[2]], 2, 4) is None
    assert (
        layer_major_kv_view([torch.zeros(2, 4, 2, 8) for _ in range(2)], 2, 4) is None
    )


def _kv_dump(seed: int, slots: int, heads: int) -> np.ndarray:
    """Return BF16 bits ``[slots, layers, 2, tokens, heads, dim]`` whose
    heads and channels have different scales, as captured KV does."""
    rng = np.random.default_rng(seed)
    shape = (slots, _LAYERS, 2, _TOKENS, heads, _HEAD_DIM)
    scale = 2.0 ** rng.integers(-6, 6, (1, _LAYERS, 2, 1, heads, _HEAD_DIM))
    values = (rng.standard_normal(shape) * scale).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def _dump_chunk(slot: np.ndarray, fp8: bool) -> torch.Tensor:
    """Return one dump slot as a CUDA uint8 kv-major chunk."""
    kv_major = np.ascontiguousarray(slot.transpose(1, 0, 2, 3, 4))
    data = fp8_bytes(kv_major) if fp8 else kv_major.view(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(data).reshape(-1)).cuda()


def _reference_decode(
    record: torch.Tensor, codebooks: bytes, fp8: bool, max_chunks: int
) -> torch.Tensor:
    """Decode a GPU record with the CPU reference codec, as a kv-major chunk."""
    dtype_bytes = 1 if fp8 else 2
    geometry = CompressedStoreGeometry(
        num_slots=max_chunks,
        slot_size=2 * _LAYERS * _TOKENS * _HEADS * _HEAD_DIM * dtype_bytes,
        block_tokens=_TOKENS,
        num_layers=_LAYERS,
        num_kv_heads=_HEADS,
        head_dim=_HEAD_DIM,
        dtype_bytes=dtype_bytes,
    )
    payload = record.cpu().numpy().tobytes()
    header = SlotHeader.parse(payload, expected_geometry=geometry)
    slot = decode_slot(
        payload,
        mode=SlotMode.COMPRESSED,
        slot_id=header.slot_id,
        geometry=geometry,
        codebooks=codebooks,
        # The GPU packer leaves the raw digest unset; bytes are compared.
        verify_hash=False,
    )
    # Records hold the physical block [layers, 2, tokens, heads, dim].
    block = np.frombuffer(slot, dtype=np.uint8).reshape(
        _LAYERS, 2, _TOKENS, _HEADS, _HEAD_DIM * dtype_bytes
    )
    return torch.from_numpy(np.ascontiguousarray(block.transpose(1, 0, 2, 3, 4)))


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("kv_head_start", [0, _HEADS])
def test_calibrated_chunk_codec_matches_the_reference_codec(
    tmp_path, fp8: bool, kv_head_start: int
) -> None:
    """GPU records under per-head tables (and FP8 channel offsets) are the
    reference codec's format, for any TP rank's head shard."""
    dump = _kv_dump(seed=21, slots=6, heads=2 * _HEADS)
    write_calibration_artifact(
        calibrate_codebooks(dump[:4], model_id="m", fp8=fp8), tmp_path
    )
    kv_dtype = torch.uint8 if fp8 else torch.bfloat16
    codec = _codec(
        max_chunks=2,
        capacity_ratio=None,
        kv_dtype=kv_dtype,
        codebook_dir=str(tmp_path),
        kv_head_start=kv_head_start,
    )
    generic = _codec(max_chunks=2, capacity_ratio=None, kv_dtype=kv_dtype)
    heads = slice(kv_head_start, kv_head_start + _HEADS)
    originals = [_dump_chunk(dump[s][:, :, :, heads], fp8) for s in (4, 5)]
    stream = torch.cuda.Stream()
    chunks = [chunk.clone() for chunk in originals]
    lengths = codec.encode(chunks, stream)
    generic_lengths = generic.encode([c.clone() for c in originals], stream)
    stream.synchronize()
    assert all(length < codec.raw_bytes for length in lengths)
    # Calibrated tables beat the generic palette on skewed heads/channels.
    assert sum(lengths) < sum(generic_lengths)

    codebooks = load_calibrated_codebooks(
        tmp_path,
        CompressedStoreGeometry(
            num_slots=2,
            slot_size=codec.raw_bytes,
            block_tokens=_TOKENS,
            num_layers=_LAYERS,
            num_kv_heads=_HEADS,
            head_dim=_HEAD_DIM,
            dtype_bytes=1 if fp8 else 2,
        ),
        kv_head_start,
    )
    for chunk, length, original in zip(chunks, lengths, originals, strict=True):
        reference = _reference_decode(chunk[:length], codebooks, fp8, 2)
        assert torch.equal(reference.reshape(-1), original.cpu())

    restored = [torch.full_like(chunk, 0xA5) for chunk in originals]
    for target, chunk, length in zip(restored, chunks, lengths, strict=True):
        target[:length].copy_(chunk[:length])
    codec.decode(restored, lengths, stream)
    stream.synchronize()
    for target, original in zip(restored, originals, strict=True):
        assert torch.equal(target, original)

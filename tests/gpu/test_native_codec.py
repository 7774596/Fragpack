# SPDX-License-Identifier: Apache-2.0
"""Independent native-page codec: exact bytes, mapping, and rejection safety."""

from dataclasses import dataclass
import struct

import numpy as np
import pytest
import torch

from fragpack.compression import attn_page as ap
from fragpack.ops.native_codec import NativeCodec, NativeKVPages

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]
POISON = 0x5AA5


def _pages(
    layers: int, count: int, tokens: int, heads: int, strided: bool, reverse: bool
) -> tuple[NativeKVPages, list[torch.Tensor]]:
    """Keep guarded owners, including holes and a two-byte source offset."""
    shape = (2, count, tokens, heads, 128)
    caches, owners = [], []
    for layer in range(layers):
        order = (
            (1, 0, 3, 2, 4) if strided and (layer % 2 == reverse) else tuple(range(5))
        )
        strides = [0] * 5
        extent = 1
        for axis in reversed(order):
            strides[axis] = extent + (3 if strided and axis != 4 else 0)
            extent += (shape[axis] - 1) * strides[axis]
        offset = int(strided)
        owner = torch.full(
            (extent + offset + 7,), POISON, dtype=torch.int16, device="cuda"
        )
        cache = owner.as_strided(shape, strides, storage_offset=offset).view(
            torch.bfloat16
        )
        if strided:
            assert cache.data_ptr() % 4 == 2
        owners.append(owner)
        caches.append(cache)
    return NativeKVPages(caches), owners


def _expected(
    owners: list[torch.Tensor],
    pages: NativeKVPages,
    mapping: np.ndarray,
    bits: np.ndarray,
) -> list[torch.Tensor]:
    """Scatter the host reference while preserving every unselected byte."""
    output = [owner.cpu().clone() for owner in owners]
    tokens = pages.page_tokens
    for layer, cache in enumerate(pages.caches):
        view = output[layer].as_strided(
            cache.shape, cache.stride(), cache.storage_offset()
        )
        for block, row in enumerate(mapping):
            for chunk, page in enumerate(row):
                values = bits[block, layer, :, :, chunk * tokens : (chunk + 1) * tokens]
                values = values.transpose(0, 2, 1, 3).copy().view(np.int16)
                view[:, int(page)].copy_(torch.from_numpy(values))
    return output


@dataclass
class _Case:
    table: np.ndarray
    bits: np.ndarray
    source: NativeKVPages
    target: NativeKVPages
    source_owners: list[torch.Tensor]
    target_owners: list[torch.Tensor]
    source_map: torch.Tensor
    target_map: torch.Tensor
    expected: list[torch.Tensor]


def _case(
    tokens: int = 32,
    strided: bool = True,
    blocks: int = 3,
    layers: int = 3,
    heads: int = 2,
) -> _Case:
    rng = np.random.default_rng(731)
    table = np.empty((layers, 2, heads, ap.TABLE_ENTRIES), dtype=np.uint8)
    bits = np.empty((blocks, layers, 2, heads, 128, 128), dtype=np.uint16)
    for layer, kv, head in np.ndindex(layers, 2, heads):
        table[layer, kv, head] = rng.permutation(256)[: ap.TABLE_ENTRIES]
        for block in range(blocks):
            high = rng.choice(table[layer, kv, head, :7], (128, 128))
            escaped = rng.choice(128 * 128, 500, replace=False)
            high.flat[escaped] = rng.choice(table[layer, kv, head, 7:], 500)
            outside = np.setdiff1d(np.arange(256), table[layer, kv, head])[0]
            high[3, 5] = high[101, 77] = outside
            bits[block, layer, kv, head] = (high.astype(np.uint16) << 8) | rng.integers(
                0, 256, (128, 128), dtype=np.uint16
            )
    # Arbitrary BF16 payloads must retain NaNs, infinities, signed zero and subnormals.
    raw = bits[-1, -1, 1, -1]
    raw[:] = rng.integers(0, 65536, raw.shape, dtype=np.uint16)
    raw.flat[:9] = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 0xFFC1, 1, 0xFFFF]
    chunks = 128 // tokens
    total = blocks * chunks
    source_ids = rng.permutation(np.arange(1, 2 * total, 2)).reshape(blocks, chunks)
    target_ids = rng.permutation(np.arange(2, 2 * total + 1, 2)).reshape(blocks, chunks)
    source, source_owners = _pages(layers, 2 * total + 3, tokens, heads, strided, False)
    target, target_owners = _pages(layers, 2 * total + 3, tokens, heads, strided, True)
    populated = _expected(source_owners, source, source_ids, bits)
    for owner, data in zip(source_owners, populated, strict=True):
        owner.copy_(data)
    expected = _expected(target_owners, target, target_ids, bits)
    return _Case(
        table,
        bits,
        source,
        target,
        source_owners,
        target_owners,
        torch.tensor(source_ids, dtype=torch.int32, device="cuda"),
        torch.tensor(target_ids, dtype=torch.int32, device="cuda"),
        expected,
    )


def _check_owners(owners: list[torch.Tensor], expected: list[torch.Tensor]) -> None:
    for actual, reference in zip(owners, expected, strict=True):
        assert torch.equal(actual.cpu(), reference)


@pytest.mark.parametrize("tokens", [16, 32, 64, 128])
@pytest.mark.parametrize("strided", [False, True])
def test_native_codec_matches_cpu_records_and_restores_permuted_pages(
    tokens: int, strided: bool
) -> None:
    case = _case(tokens, strided)
    codec = NativeCodec(case.table, max_blocks=3, device="cuda")
    source_before = [owner.cpu().clone() for owner in case.source_owners]
    codec.buffer.fill_(0xA5)
    assert not bool(codec.encode(case.source, case.source_map).any())
    offsets = codec.offsets.cpu().numpy()
    lengths = codec.lengths.cpu().numpy()
    record_offsets = codec.rec_offsets.cpu().numpy()
    record_lengths = codec.rec_lens.cpu().numpy()
    records = codec.records.cpu().numpy()
    packed = codec.buffer[: int(offsets[-1])].cpu().numpy().tobytes()
    assert offsets[0] == 0
    np.testing.assert_array_equal(np.diff(offsets), lengths)
    assert np.all(offsets % ap.BLOB_ALIGN == 0)
    layers, _, heads, _ = case.table.shape
    header_bytes = ap.blob_head_bytes(layers, heads)
    raw_count = 0
    for block in range(3):
        blob = packed[int(offsets[block]) : int(offsets[block + 1])]
        cursor = header_bytes
        for layer, kv, head in np.ndindex(layers, 2, heads):
            original = case.bits[block, layer, kv, head]
            reference = ap.encode_record(original, case.table[layer, kv, head], kv)
            size = len(reference)
            raw_count += size == ap.RAW_RECORD_BYTES
            assert record_lengths[block, layer, kv, head] == size - ap.DENSE_BYTES
            assert record_offsets[block, layer, kv, head] == cursor
            assert records[block, layer, kv, head, :size].tobytes() == reference
            assert blob[cursor : cursor + size] == reference
            np.testing.assert_array_equal(
                ap.decode_record(
                    blob[cursor : cursor + size], case.table[layer, kv, head], kv
                ),
                original,
            )
            cursor += size
        assert struct.unpack_from("<4I2q", blob) == (
            ap.BLOB_MAGIC,
            1,
            layers,
            heads,
            cursor,
            codec.digest,
        )
        np.testing.assert_array_equal(
            np.frombuffer(blob, dtype="<i4", count=layers * 2 * heads, offset=64),
            record_lengths[block].reshape(-1),
        )
        assert not any(blob[32:64])
        assert not any(blob[64 + layers * 2 * heads * 4 : header_bytes])
        assert not any(blob[cursor:])
    assert raw_count == 1
    # Restore explicitly transferred copies, after invalidating the workspace payload.
    received = torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).copy()).cuda()
    received_offsets = torch.from_numpy(offsets.copy()).cuda()
    codec.buffer.fill_(0xA5)
    assert not bool(
        codec.decode(
            case.target, case.target_map, src=received, offsets=received_offsets
        ).any()
    )
    _check_owners(case.target_owners, case.expected)
    _check_owners(case.source_owners, source_before)

    raw_buffer = torch.empty(case.bits.nbytes, dtype=torch.uint8, device="cuda")
    assert not bool(codec.raw_gather(case.source, case.source_map, raw_buffer).any())
    np.testing.assert_array_equal(
        raw_buffer.cpu().numpy().view(np.uint16).reshape(case.bits.shape), case.bits
    )
    for owner in case.target_owners:
        owner.fill_(POISON)
    assert not bool(codec.raw_scatter(case.target, case.target_map, raw_buffer).any())
    _check_owners(case.target_owners, case.expected)


@pytest.mark.parametrize(
    "corruption,error_bit",
    [
        ("magic", 1),
        ("version", 1),
        ("digest", 1),
        ("truncated", 1),
        ("negative_offset", 1),
        ("length", 2),
        ("payload", 8),
        ("negative_page", 4),
        ("outside_page", 4),
        ("duplicate_page", 4),
    ],
)
def test_native_decode_rejects_invalid_input_without_writing_target(
    corruption: str, error_bit: int
) -> None:
    case = _case(blocks=1, layers=2, heads=1)
    codec = NativeCodec(case.table, max_blocks=1, device="cuda")
    assert not bool(codec.encode(case.source, case.source_map).any())
    count = int(codec.offsets[-1])
    blob = bytearray(codec.buffer[:count].cpu().numpy().tobytes())
    offsets = codec.offsets.clone()
    mapping = case.target_map.clone()
    if corruption in ("magic", "version", "digest"):
        blob[{"magic": 0, "version": 4, "digest": 24}[corruption]] ^= 0xFF
    elif corruption == "truncated":
        blob = blob[: struct.unpack_from("<q", blob, 16)[0] - 1]
    elif corruption == "negative_offset":
        offsets[0] = -16
    elif corruption == "length":
        struct.pack_into("<i", blob, 64, 17)
    elif corruption == "payload":
        # Impossible escape count in a coded record; not arbitrary data corruption.
        struct.pack_into(
            "<I", blob, ap.blob_head_bytes(2, 1) + ap.DENSE_BYTES, 0xFFFFFFFF
        )
    elif corruption == "negative_page":
        mapping[0, 0] = -1
    elif corruption == "outside_page":
        mapping[0, 0] = case.target.caches[0].shape[1]
    elif corruption == "duplicate_page":
        mapping[0, 1] = mapping[0, 0]
    received = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).cuda()
    before = [owner.cpu().clone() for owner in case.target_owners]
    status = codec.decode(case.target, mapping, src=received, offsets=offsets)
    assert int(status[0]) & error_bit
    _check_owners(case.target_owners, before)


def test_native_mapping_errors_and_bad_lengths_propagate_through_encoding() -> None:
    case = _case(blocks=1, layers=1, heads=1)
    codec = NativeCodec(case.table, max_blocks=1, device="cuda")
    bad_map = case.source_map.clone()
    bad_map[0, 1] = -1
    codec.buffer.fill_(0xA5)
    assert int(codec.encode(case.source, bad_map)[0]) & 4
    assert codec.offsets.cpu().tolist() == [0, 0]
    assert bool(torch.all(codec.buffer == 0xA5))
    codec.encode_records(case.source, case.source_map)
    codec.rec_lens[0, 0, 1, 0] = 17
    codec.plan()
    codec.pack()
    assert int(codec.status[0]) & 2
    assert codec.lengths.cpu().tolist() == [0]
    assert bool(torch.all(codec.buffer == 0xA5))
    # A later valid operation must not inherit the rejected batch's status.
    assert not bool(codec.encode(case.source, case.source_map).any())


def test_native_raw_scatter_rejects_cross_block_destination_aliases() -> None:
    case = _case(blocks=2, layers=1, heads=1)
    codec = NativeCodec(case.table, max_blocks=2, device="cuda")
    raw = torch.empty(case.bits.nbytes, dtype=torch.uint8, device="cuda")
    assert not bool(codec.raw_gather(case.source, case.source_map, raw).any())
    mapping = case.target_map.clone()
    mapping[1, 2] = mapping[0, 0]
    before = [owner.cpu().clone() for owner in case.target_owners]
    status = codec.raw_scatter(case.target, mapping, raw).cpu().tolist()
    assert all(value & 4 for value in status)
    _check_owners(case.target_owners, before)


def test_native_page_registration_rejects_overlapping_layer_storage() -> None:
    cache = torch.empty((2, 3, 32, 1, 128), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        NativeKVPages([cache, cache])


def test_native_record_prefix_crosses_scan_tile_and_rejects_late_bad_length() -> None:
    """The second 256-record tile contributes offsets and failure status."""
    case = _case(blocks=1, layers=33, heads=4)
    codec = NativeCodec(case.table, max_blocks=1, device="cuda")
    assert not bool(codec.encode(case.source, case.source_map).any())
    lengths = codec.rec_lens.cpu().numpy().reshape(-1)
    expected_offsets = ap.blob_head_bytes(33, 4) + np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(ap.DENSE_BYTES + lengths[:-1])]
    )
    np.testing.assert_array_equal(
        codec.rec_offsets.cpu().numpy().reshape(-1), expected_offsets
    )
    assert len(lengths) == 264
    total = int(codec.offsets[-1])
    packed = codec.buffer[:total].cpu().numpy().tobytes()
    # Inspect every packed record, including raw fallback at index 263.
    for record, (layer, kv, head) in enumerate(np.ndindex(33, 2, 4)):
        expected = ap.encode_record(
            case.bits[0, layer, kv, head], case.table[layer, kv, head], kv
        )
        offset = int(expected_offsets[record])
        assert packed[offset : offset + len(expected)] == expected
    assert not bool(codec.decode(case.target, case.target_map).any())
    _check_owners(case.target_owners, case.expected)

    # An invalid length in the second tile invalidates the entire blob before
    # any earlier layer may write. It must also survive the export-side OR.
    for owner in case.target_owners:
        owner.fill_(POISON)
    untouched = [owner.cpu().clone() for owner in case.target_owners]
    malformed = bytearray(packed)
    struct.pack_into("<i", malformed, ap.BLOB_FIXED_BYTES + 257 * 4, 17)
    received = torch.from_numpy(np.frombuffer(malformed, dtype=np.uint8).copy()).cuda()
    offsets = torch.tensor([0, total], dtype=torch.int64, device="cuda")
    status = codec.decode(case.target, case.target_map, src=received, offsets=offsets)
    assert int(status[0]) & 2
    _check_owners(case.target_owners, untouched)
    codec.encode_records(case.source, case.source_map)
    codec.rec_lens.view(-1)[257] = 17
    codec.buffer.fill_(0xA5)
    codec.plan()
    codec.pack()
    assert int(codec.status[0]) & 2
    assert codec.offsets.cpu().tolist() == [0, 0]
    assert bool(torch.all(codec.buffer == 0xA5))


def test_native_public_stages_reject_uninitialized_or_invalidated_plans() -> None:
    """Only record encoding can establish metadata usable by plan and pack."""
    case = _case(blocks=1, layers=1, heads=1)
    codec = NativeCodec(case.table, max_blocks=1, device="cuda")
    raw = torch.empty(case.bits.nbytes, dtype=torch.uint8, device="cuda")

    def reject_planning() -> None:
        for operation in (codec.plan, codec.pack):
            with pytest.raises(RuntimeError):
                operation()

    reject_planning()
    assert not bool(codec.raw_gather(case.source, case.source_map, raw).any())
    reject_planning()
    codec.encode_records(case.source, case.source_map)
    with pytest.raises(RuntimeError):
        codec.pack()
    codec.plan()
    codec.pack()
    assert not bool(codec.status.any())
    # Replanning/repacking initialized encoded records is explicitly permitted.
    codec.plan()
    codec.pack()
    codec.pack()
    assert not bool(codec.decode(case.target, case.target_map).any())
    reject_planning()
    assert not bool(codec.encode(case.source, case.source_map).any())
    assert not bool(codec.raw_scatter(case.target, case.target_map, raw).any())
    reject_planning()

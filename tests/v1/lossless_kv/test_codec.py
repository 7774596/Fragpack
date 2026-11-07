# SPDX-License-Identifier: Apache-2.0

# Third Party
import numpy as np
import pytest

# First Party
from lmcache.v1.lossless_kv import (
    CompressedStoreGeometry,
    SlotMode,
    decode_slot,
    default_online_codebooks,
    encode_slot,
)
from lmcache.v1.lossless_kv.format import (
    CODEBOOK_ENTRIES,
    FORMAT_VERSION,
    INDEX_GROUP_TILES,
    INDEX_GROUP_WORDS,
    IO_ALIGNMENT,
    ONLINE_TILE_SCALARS,
    SlotHeader,
    digest_bytes,
    pack_tile_index,
    slot_header_bytes,
    tile_index_bytes,
)


def _geometry(num_slots: int = 2) -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=num_slots,
        slot_size=2 * 2 * 128 * 4 * 127 * 2,
        block_tokens=128,
        num_layers=2,
        num_kv_heads=4,
        head_dim=127,
        tile_scalars=1024,
    )


def _slot(
    geometry: CompressedStoreGeometry,
    *,
    seed: int,
    escaped_high_byte: int | None = None,
) -> bytes:
    rng = np.random.default_rng(seed)
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = rng.integers(0, 256, geometry.slot_size // 2, dtype=np.uint8)
    common = np.array([0x3E, 0x3F, 0x40, 0xBF], dtype=np.uint8)
    raw[1::2] = common[
        rng.integers(0, len(common), geometry.slot_size // 2, dtype=np.uint8)
    ]
    if escaped_high_byte is not None:
        raw[1::2][::997] = escaped_high_byte
    return raw.tobytes()


def test_reference_codec_is_byte_exact_with_escapes_and_tail_tile() -> None:
    geometry = _geometry()
    evaluation = _slot(geometry, seed=2, escaped_high_byte=0x7E)
    codebooks = default_online_codebooks(geometry)

    encoded = encode_slot(
        evaluation,
        slot_id=1,
        geometry=geometry,
        codebooks=codebooks,
    )

    assert encoded.mode is SlotMode.COMPRESSED
    assert len(encoded.payload) < geometry.slot_size
    assert len(encoded.payload) % IO_ALIGNMENT == 0
    header = SlotHeader.parse(
        encoded.payload,
        expected_geometry=geometry,
        expected_slot_id=1,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert header.descriptors[0].scalar_count % geometry.tile_scalars == 512
    # Only the complete slot is an O_DIRECT transfer. Plane records are
    # intentionally packed back-to-back so their internal padding is not
    # counted as stored KV bytes.
    assert any(
        descriptor.record_offset % IO_ALIGNMENT for descriptor in header.descriptors[1:]
    )
    assert any(
        descriptor.record_length % IO_ALIGNMENT for descriptor in header.descriptors
    )
    assert sum(item.escape_count for item in header.descriptors) > 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=1,
            geometry=geometry,
            codebooks=codebooks,
        )
        == evaluation
    )


def test_tile_index_recovers_every_tile_prefix() -> None:
    """Each tile's prefixes are its group's base plus its relative counts."""
    rng = np.random.default_rng(7)
    tiles = 2 * INDEX_GROUP_TILES + 3
    escapes = rng.integers(0, ONLINE_TILE_SCALARS + 1, tiles)
    raws = rng.integers(0, 64, tiles)
    escape_prefix = np.concatenate([[0], np.cumsum(escapes)])
    raw_prefix = np.concatenate([[0], np.cumsum(raws)])

    index = pack_tile_index(escape_prefix, raw_prefix)

    assert len(index) == tile_index_bytes(tiles) == 3 * INDEX_GROUP_WORDS * 4
    words = np.frombuffer(index, dtype="<u4").reshape(-1, INDEX_GROUP_WORDS)
    for tile in range(tiles):
        group, offset = divmod(tile, INDEX_GROUP_TILES)
        relative = int(words[group, offset + 1]) if offset else 0
        assert int(words[group, 0]) + (relative & 0xFFFF) == escape_prefix[tile]
        assert int(words[group, 1]) + (relative >> 16) == raw_prefix[tile]


def test_three_bit_codec_treats_unrepresentable_symbols_as_escapes() -> None:
    """Three-bit packing must preserve codebook entries 7 through 14."""
    geometry = _geometry(num_slots=1)
    codebooks = default_online_codebooks(geometry)
    table = np.frombuffer(codebooks, dtype=np.uint8)[:CODEBOOK_ENTRIES]
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = 17
    high = np.full(geometry.slot_size // 2, table[0], dtype=np.uint8)
    high[::997] = table[10]
    raw[1::2] = high

    encoded = encode_slot(
        raw.tobytes(), slot_id=0, geometry=geometry, codebooks=codebooks
    )

    assert encoded.mode is SlotMode.COMPRESSED
    header = SlotHeader.parse(
        encoded.payload,
        expected_geometry=geometry,
        expected_slot_id=0,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert sum(item.escape_count for item in header.descriptors) > 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw.tobytes()
    )


def test_three_bit_escape_stream_round_trips_the_eighth_secondary_entry() -> None:
    """The narrow escape stream promotes codebook entry fourteen to raw bytes."""
    geometry = _geometry(num_slots=1)
    codebooks = default_online_codebooks(geometry)
    table = np.frombuffer(codebooks, dtype=np.uint8)[:CODEBOOK_ENTRIES]
    raw = np.empty(geometry.slot_size, dtype=np.uint8)
    raw[0::2] = 23
    high = np.full(geometry.slot_size // 2, table[0], dtype=np.uint8)
    high[::997] = table[14]
    raw[1::2] = high

    encoded = encode_slot(
        raw.tobytes(), slot_id=0, geometry=geometry, codebooks=codebooks
    )

    header = SlotHeader.parse(
        encoded.payload,
        expected_geometry=geometry,
        expected_slot_id=0,
        expected_codebook_hash=digest_bytes(codebooks),
    )
    assert sum(item.escape_count for item in header.descriptors) > 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw.tobytes()
    )


def test_online_codebook_is_generic_for_all_geometries() -> None:
    """The startup default is model-independent, one row per (plane, head)."""
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=36 * 2 * 128 * 8 * 128 * 2,
        block_tokens=128,
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
    )

    payload = np.frombuffer(default_online_codebooks(geometry), dtype=np.uint8)
    rows = geometry.plane_count * geometry.num_kv_heads
    tables = payload[: rows * CODEBOOK_ENTRIES].reshape(rows, CODEBOOK_ENTRIES)

    assert len(payload) == rows * (CODEBOOK_ENTRIES + geometry.head_dim)
    assert not payload[rows * CODEBOOK_ENTRIES :].any()
    assert tables.shape == (72 * 8, CODEBOOK_ENTRIES)
    assert all(len(np.unique(row)) == CODEBOOK_ENTRIES for row in tables)
    assert np.all(tables == tables[0])
    assert tables[0].tolist() == [
        63,
        191,
        62,
        190,
        64,
        192,
        61,
        189,
        60,
        188,
        59,
        187,
        58,
        186,
        57,
    ]


def test_incompressible_slot_uses_explicit_raw_mode() -> None:
    geometry = _geometry(num_slots=1)
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 256, geometry.slot_size, dtype=np.uint8).tobytes()
    codebooks = default_online_codebooks(geometry)

    encoded = encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks)

    assert encoded.mode is SlotMode.RAW
    assert encoded.payload == raw
    assert (
        decode_slot(
            encoded.payload,
            mode=SlotMode.RAW,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw
    )


def test_slot_header_rejects_corrupt_descriptor() -> None:
    geometry = _geometry(num_slots=1)
    slot = _slot(geometry, seed=30)
    codebooks = default_online_codebooks(geometry)
    encoded = encode_slot(slot, slot_id=0, geometry=geometry, codebooks=codebooks)
    assert encoded.mode is SlotMode.COMPRESSED
    payload = bytearray(encoded.payload)
    payload[128:136] = b"\xff" * 8

    with pytest.raises(ValueError):
        decode_slot(
            payload,
            mode=SlotMode.COMPRESSED,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )


def test_unknown_format_version_is_rejected() -> None:
    """Only current-version records parse; other header versions are refused."""
    geometry = _geometry(num_slots=1)
    codebooks = default_online_codebooks(geometry)
    raw = _slot(geometry, seed=45)
    payload = bytearray(
        encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks).payload
    )
    assert SlotHeader.parse(payload).format_version == FORMAT_VERSION
    payload[8] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="version"):
        SlotHeader.parse(payload, expected_geometry=geometry)


def _layer_geometry(num_layers: int) -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=1,
        slot_size=num_layers * 2 * 16 * 8 * 128 * 2,
        block_tokens=16,
        num_layers=num_layers,
        num_kv_heads=8,
        head_dim=128,
        tile_scalars=ONLINE_TILE_SCALARS,
    )


@pytest.mark.parametrize(
    ("num_layers", "header_pages"), [(41, 1), (42, 2), (48, 2), (64, 2)]
)
def test_deep_models_round_trip_with_multi_page_headers(
    num_layers: int, header_pages: int
) -> None:
    # One page holds 82 plane descriptors; deeper models spill to more pages.
    geometry = _layer_geometry(num_layers)
    assert geometry.header_bytes == slot_header_bytes(2 * num_layers)
    assert geometry.header_bytes == header_pages * IO_ALIGNMENT
    codebooks = default_online_codebooks(geometry)
    slot = _slot(geometry, seed=num_layers)
    encoded = encode_slot(slot, slot_id=0, geometry=geometry, codebooks=codebooks)
    assert encoded.mode is SlotMode.COMPRESSED
    header = SlotHeader.parse(encoded.payload, expected_geometry=geometry)
    assert header.descriptors[0].record_offset == geometry.header_bytes
    assert len(header.descriptors) == 2 * num_layers
    decoded = decode_slot(
        encoded.payload,
        mode=encoded.mode,
        slot_id=0,
        geometry=geometry,
        codebooks=codebooks,
    )
    assert decoded == slot


def test_parse_rejects_a_header_length_that_does_not_fit_the_planes() -> None:
    geometry = _layer_geometry(48)
    encoded = encode_slot(
        _slot(geometry, seed=1),
        slot_id=0,
        geometry=geometry,
        codebooks=default_online_codebooks(geometry),
    )
    payload = bytearray(encoded.payload)
    # header_bytes is the third little-endian u32 of the fixed header.
    payload[12:16] = IO_ALIGNMENT.to_bytes(4, "little")
    with pytest.raises(ValueError, match="header length"):
        SlotHeader.parse(bytes(payload), expected_geometry=geometry)


def _fp8_geometry() -> CompressedStoreGeometry:
    return CompressedStoreGeometry(
        num_slots=1,
        slot_size=2 * 2 * 16 * 8 * 128,
        block_tokens=16,
        num_layers=2,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=1,
        tile_scalars=ONLINE_TILE_SCALARS,
    )


def _fp8_slot(geometry: CompressedStoreGeometry, *, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    # Sign+exponent symbols concentrate near the bias; mantissas are uniform.
    symbols = np.array([7, 23, 6, 22, 8, 24, 5, 21, 9, 25, 4, 20, 3, 19, 10, 30])
    weights = np.array([30, 30, 12, 12, 6, 6, 1, 1, 0.5, 0.5, 0.3, 0.3] + [0.1] * 4)
    se = rng.choice(symbols, geometry.slot_size, p=weights / weights.sum())
    mantissa = rng.integers(0, 8, geometry.slot_size)
    return ((se << 3) | mantissa).astype(np.uint8).tobytes()


def test_fp8_codec_round_trips_and_marks_the_dtype() -> None:
    geometry = _fp8_geometry()
    codebooks = default_online_codebooks(geometry)
    raw = _fp8_slot(geometry, seed=3)

    encoded = encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks)

    assert encoded.mode is SlotMode.COMPRESSED
    assert len(encoded.payload) < geometry.slot_size
    header = SlotHeader.parse(encoded.payload, expected_geometry=geometry)
    assert header.dtype_bytes == 1
    descriptor = header.descriptors[0]
    # The low stream holds three mantissa bits per scalar.
    assert (
        descriptor.symbol_offset - descriptor.low_offset
        == (geometry.plane_scalars * 3 + 7) // 8
    )
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw
    )


def test_fp8_record_does_not_parse_under_bf16_geometry() -> None:
    geometry = _fp8_geometry()
    codebooks = default_online_codebooks(geometry)
    encoded = encode_slot(
        _fp8_slot(geometry, seed=5), slot_id=0, geometry=geometry, codebooks=codebooks
    )
    bf16 = CompressedStoreGeometry(
        num_slots=1,
        slot_size=geometry.slot_size,
        block_tokens=8,
        num_layers=2,
        num_kv_heads=8,
        head_dim=128,
        tile_scalars=ONLINE_TILE_SCALARS,
    )
    with pytest.raises(ValueError):
        SlotHeader.parse(encoded.payload, expected_geometry=bf16)


def _tables(
    geometry: CompressedStoreGeometry,
    books: np.ndarray,
    offsets: np.ndarray | None = None,
) -> bytes:
    """Return a tables payload of ``[rows, 15]`` books and optional offsets."""
    rows = geometry.plane_count * geometry.num_kv_heads
    if offsets is None:
        offsets = np.zeros((rows, geometry.head_dim), dtype=np.uint8)
    return books.astype(np.uint8).tobytes() + offsets.astype(np.uint8).tobytes()


def test_each_head_codes_with_its_own_codebook() -> None:
    """A head whose symbols are only in its own row stays compressed."""
    geometry = _fp8_geometry()
    rows = geometry.plane_count * geometry.num_kv_heads
    generic = np.frombuffer(default_online_codebooks(geometry), dtype=np.uint8)
    books = generic[: rows * CODEBOOK_ENTRIES].reshape(rows, CODEBOOK_ENTRIES).copy()
    # Head h's symbols are 2 * h and 2 * h + 1; its row lists them first.
    raw = np.empty(
        (geometry.num_layers * 2, geometry.block_tokens, geometry.num_kv_heads, 128),
        dtype=np.uint8,
    )
    rng = np.random.default_rng(4)
    for head in range(geometry.num_kv_heads):
        symbols = 2 * head + rng.integers(0, 2, raw[:, :, head].shape)
        raw[:, :, head] = (symbols << 3) | rng.integers(0, 8, symbols.shape)
        own = [2 * head, 2 * head + 1]
        rest = [value for value in range(32) if value not in own]
        books[head :: geometry.num_kv_heads] = own + rest[: CODEBOOK_ENTRIES - 2]
    codebooks = _tables(geometry, books)

    encoded = encode_slot(
        raw.tobytes(), slot_id=0, geometry=geometry, codebooks=codebooks
    )
    header = SlotHeader.parse(encoded.payload, expected_geometry=geometry)

    assert encoded.mode is SlotMode.COMPRESSED
    assert sum(item.escape_count for item in header.descriptors) == 0
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw.tobytes()
    )


def test_fp8_exponent_offsets_align_channels_losslessly() -> None:
    """Per-channel offsets turn shifted channels into coded symbols."""
    geometry = _fp8_geometry()
    rows = geometry.plane_count * geometry.num_kv_heads
    generic = np.frombuffer(default_online_codebooks(geometry), dtype=np.uint8)
    books = generic[: rows * CODEBOOK_ENTRIES].reshape(rows, CODEBOOK_ENTRIES)
    base = np.frombuffer(_fp8_slot(geometry, seed=11), dtype=np.uint8).reshape(
        -1, geometry.head_dim
    )
    rng = np.random.default_rng(12)
    channel_offset = rng.integers(0, 16, geometry.head_dim)
    # Raise each channel's exponent by its offset (mod 16), keeping sign and
    # mantissa, so only the offsets bring the symbols back into the table.
    exponent = ((base >> 3) + channel_offset) & 15
    raw = ((base & 0x87) | (exponent << 3)).astype(np.uint8).tobytes()
    offsets = np.broadcast_to(channel_offset, (rows, geometry.head_dim))
    plain = encode_slot(
        raw, slot_id=0, geometry=geometry, codebooks=_tables(geometry, books)
    )
    codebooks = _tables(geometry, books, offsets)

    encoded = encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks)

    assert encoded.mode is SlotMode.COMPRESSED
    assert len(encoded.payload) < len(plain.payload)
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw
    )


def test_bf16_tables_reject_exponent_offsets() -> None:
    geometry = _geometry(num_slots=1)
    rows = geometry.plane_count * geometry.num_kv_heads
    payload = bytearray(default_online_codebooks(geometry))
    payload[rows * CODEBOOK_ENTRIES] = 1
    with pytest.raises(ValueError, match="offset"):
        encode_slot(
            _slot(geometry, seed=1),
            slot_id=0,
            geometry=geometry,
            codebooks=bytes(payload),
        )

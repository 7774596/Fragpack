# SPDX-License-Identifier: Apache-2.0
"""CPU-only prefix identity, FPKB envelope and opaque-message contracts."""

from dataclasses import replace
import struct

import numpy as np
import pytest

from fragpack.compression import attn_page as ap
from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity
from fragpack.integrations.lmcache.protocol import (
    canonicalize_blob,
    decode_message,
    encode_message,
    validate_blob,
)


def _identity() -> CacheIdentity:
    return CacheIdentity("model@revision-bf16", -354, 2, 1)


def _blob(identity: CacheIdentity) -> tuple[bytes, list[bytes]]:
    records = []
    for i in range(2 * identity.num_layers * identity.kv_heads):
        if i % 2:
            records.append(bytes(ap.RAW_RECORD_BYTES))
        else:
            records.append(
                ap.encode_record(
                    np.zeros((128, 128), dtype=np.uint16),
                    np.arange(22, dtype=np.uint8),
                    i % 2,
                )
            )
    head = ap.blob_head_bytes(identity.num_layers, identity.kv_heads)
    declared = head + sum(map(len, records))
    size = (declared + ap.BLOB_ALIGN - 1) // ap.BLOB_ALIGN * ap.BLOB_ALIGN
    # Model native export into reused storage, whose padding may be arbitrary.
    output = bytearray([0xA5]) * size
    struct.pack_into(
        "<4Iqq",
        output,
        0,
        ap.BLOB_MAGIC,
        1,
        identity.num_layers,
        identity.kv_heads,
        declared,
        identity.table_digest,
    )
    offset = head
    for index, record in enumerate(records):
        struct.pack_into("<i", output, 64 + 4 * index, len(record) - ap.DENSE_BYTES)
        output[offset : offset + len(record)] = record
        offset += len(record)
    return bytes(output), records


def test_chained_prefix_keys_preserve_complete_prefix_and_exclude_tail() -> None:
    identity = _identity()
    tokens = list(range(400))
    keys = identity.block_keys(tokens, "tenant-one")
    assert len(keys) == 3
    assert identity.block_keys(tokens[:128], "tenant-one") == keys[:1]
    assert identity.block_keys(tokens[:255], "tenant-one") == keys[:1]
    assert identity.block_keys(tokens[:384], "tenant-one") == keys
    modified = tokens.copy()
    modified[128] = 999
    changed = identity.block_keys(modified, "tenant-one")
    assert changed[0] == keys[0]
    assert changed[1:] != keys[1:]
    assert all(a != b for a, b in zip(changed[1:], keys[1:], strict=True))
    assert identity.block_keys(tokens[:127], "tenant-one") == []


@pytest.mark.parametrize(
    "change",
    [
        {"model_revision": "other-revision"},
        {"table_digest": 42},
        {"num_layers": 3},
        {"kv_heads": 2},
        {"world_size": 2},
        {"world_size": 2, "rank": 1},
    ],
)
def test_incompatible_identity_never_reuses_keys(change: dict) -> None:
    identity = _identity()
    other = replace(identity, **change)
    assert identity.namespace != other.namespace
    assert identity.block_keys([1] * 128) != other.block_keys([1] * 128)


def test_tenant_salt_is_preserved_and_separates_keys() -> None:
    identity = _identity()
    a = identity.block_keys([4] * 128, "a")[0]
    b = identity.block_keys([4] * 128, "b")[0]
    assert a.namespace == b.namespace
    assert a.prefix_hash != b.prefix_hash
    assert a.cache_salt == "a"
    assert BlockKey.from_dict(a.to_dict()) == a
    with pytest.raises(ValueError):
        BlockKey.from_dict({**a.to_dict(), "surprise": "field"})
    with pytest.raises(ValueError):
        BlockKey.from_dict({**a.to_dict(), "prefix_hash": "xyz"})


@pytest.mark.parametrize("token", [True, -1, 1.5, 1 << 64])
def test_invalid_token_in_incomplete_tail_is_rejected(token: object) -> None:
    with pytest.raises(ValueError):
        _identity().block_keys([1] * 128 + [token])


@pytest.mark.parametrize("salt", ["a/b", "a\\b", "a@b", "a\x00b", "a" * 129])
def test_rejects_unsafe_salt(salt: str) -> None:
    with pytest.raises(ValueError):
        _identity().block_keys([1] * 128, salt)


def test_blob_validation_and_padding_sanitization_preserve_record_bytes() -> None:
    identity = _identity()
    blob, records = _blob(identity)
    validate_blob(blob, identity)
    clean = canonicalize_blob(blob, identity)
    validate_blob(clean, identity)
    assert clean != blob
    assert canonicalize_blob(clean, identity) == clean
    assert clean[32:64] == bytes(32)
    head = ap.blob_head_bytes(identity.num_layers, identity.kv_heads)
    assert clean[64 + 8 * identity.num_layers * identity.kv_heads : head] == bytes(
        head - 64 - 8 * identity.num_layers * identity.kv_heads
    )
    position = head
    for record in records:
        assert clean[position : position + len(record)] == record
        position += len(record)
    assert clean[position:] == bytes(len(clean) - position)


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (0, "<I", 0),
        (4, "<I", 2),
        (8, "<I", 3),
        (12, "<I", 4),
        (16, "<q", 1),
        (24, "<q", 0),
        (64, "<i", -1),
        (64, "<i", 49),
        (64, "<i", ap.VAR_CAP + 16),
        (64, "<i", ap.HEADER_BYTES + 16),
    ],
)
def test_rejects_bad_blob_identity_and_lengths(
    offset: int, fmt: str, value: int
) -> None:
    identity = _identity()
    blob = bytearray(_blob(identity)[0])
    struct.pack_into(fmt, blob, offset, value)
    with pytest.raises(ValueError):
        validate_blob(bytes(blob), identity)


def test_blob_requires_exact_padded_extent() -> None:
    identity = _identity()
    blob, _ = _blob(identity)
    for bad in (b"", blob[:32], blob[:-1], blob + bytes(ap.BLOB_ALIGN)):
        with pytest.raises(ValueError):
            validate_blob(bad, identity)


def test_binary_wire_roundtrip_and_receiver_bound() -> None:
    blob = canonicalize_blob(_blob(_identity())[0], _identity())
    header = {"key": _identity().block_keys([2] * 128)[0].to_dict(), "op": "put"}
    encoded = encode_message(header, blob)
    assert len(encoded) < len(blob) + 1000
    assert decode_message(encoded, len(blob)) == (header, blob)
    with pytest.raises(ValueError, match="bounds"):
        decode_message(encoded, len(blob) - 1)
    assert decode_message(encode_message({"ok": True}), 0) == ({"ok": True}, b"")


@pytest.mark.parametrize("data", [b"", b"FPLM", b"x" * 30])
def test_bad_wire_header(data: bytes) -> None:
    with pytest.raises(ValueError):
        decode_message(data, 100)


def test_wire_rejects_truncation_trailing_bytes_and_invalid_metadata() -> None:
    message = encode_message({"ok": True}, b"abc")
    for data in (message[:-1], message + b"junk"):
        with pytest.raises(ValueError):
            decode_message(data, 100)
    for raw in (b"[]", b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff"):
        frame = struct.pack("<4sIIQ", b"FPLM", 1, len(raw), 0) + raw
        with pytest.raises(ValueError):
            decode_message(frame, 0)
    for header in ({"x": float("inf")}, {1: "bad"}, {"x": object()}):
        with pytest.raises(ValueError):
            encode_message(header)

# SPDX-License-Identifier: Apache-2.0
"""Bounded RPC framing and structural validation of FPKB and native-raw blobs.

All functions operate on owned CPU bytes. They do not initialize CUDA, invoke
LMCache, or change the FPKB representation. Canonicalization erases only bytes
the native exporter does not define; encoded records remain byte-for-byte intact.
"""

from __future__ import annotations

import json
import struct
from typing import Any

from fragpack.compression import attn_page as ap
from fragpack.integrations.lmcache.identity import CacheIdentity

_BLOB_HEADER = struct.Struct("<4Iqq")
_FRAME = struct.Struct("<4sIIQ")
_WIRE_MAGIC = b"FPLM"
_WIRE_VERSION = 1
MAX_HEADER_BYTES = 64 * 1024
NATIVE_RAW_MAGIC = int.from_bytes(b"FPNR", "little")


def validate_blob(blob: bytes, identity: CacheIdentity) -> None:
    """Validate the identity-selected framing, record lengths and padded extent.

    Args:
        blob: One complete aligned host blob, without another object's bytes.
        identity: Expected model/shard geometry and codec-table digest.

    Raises:
        ValueError: The envelope or record-length table is inconsistent.

    This checks the native import envelope, not the correctness of arbitrary
    codec symbols or the semantic provenance of the KV. The GPU import status
    must still be checked before attention may read an installed block.
    """
    if not isinstance(blob, bytes):
        raise ValueError("blob must be owned immutable bytes")
    head = ap.blob_head_bytes(identity.num_layers, identity.kv_heads)
    if not head <= len(blob) <= identity.max_blob_bytes:
        raise ValueError("blob extent is outside the identity's bounds")
    magic, version, layers, heads, declared, digest = _BLOB_HEADER.unpack_from(blob)
    if (magic, version, layers, heads, digest) != (
        ap.BLOB_MAGIC if identity.representation == "fpkb" else NATIVE_RAW_MAGIC,
        1,
        identity.num_layers,
        identity.kv_heads,
        identity.table_digest,
    ):
        raise ValueError("blob format, geometry or codec-table identity mismatch")
    if declared < head or declared > len(blob):
        raise ValueError("FPKB declared length exceeds the object extent")
    if (declared + ap.BLOB_ALIGN - 1) // ap.BLOB_ALIGN * ap.BLOB_ALIGN != len(blob):
        raise ValueError("FPKB object has truncated or excess alignment padding")
    record_count = 2 * identity.num_layers * identity.kv_heads
    table_end = ap.BLOB_FIXED_BYTES + 4 * record_count
    raw_var_length = ap.RAW_RECORD_BYTES - ap.DENSE_BYTES
    total = head
    for (length,) in struct.iter_unpack("<i", blob[ap.BLOB_FIXED_BYTES : table_end]):
        if identity.representation == "native-raw" and length != raw_var_length:
            raise ValueError("native-raw requires all-raw token-major records")
        if length != raw_var_length and not (
            ap.HEADER_BYTES <= length <= ap.VAR_CAP and length % 16 == 0
        ):
            raise ValueError("FPKB record has an invalid variable length")
        total += ap.DENSE_BYTES + length
    if total != declared:
        raise ValueError("FPKB record lengths do not match the declared extent")


def canonicalize_blob(blob: bytes, identity: CacheIdentity) -> bytes:
    """Return validated wire bytes with undefined header/alignment bytes zeroed.

    Native export leaves reserved header bytes, head padding and tail padding
    unwritten. Clear them before crossing a process boundary so allocator reuse
    cannot disclose unrelated bytes or make equal records compare differently.
    Record contents and the core's format remain unchanged.
    """
    validate_blob(blob, identity)
    declared = _BLOB_HEADER.unpack_from(blob)[4]
    table_end = ap.BLOB_FIXED_BYTES + 8 * identity.num_layers * identity.kv_heads
    head = ap.blob_head_bytes(identity.num_layers, identity.kv_heads)
    output = bytearray(blob)
    output[_BLOB_HEADER.size : ap.BLOB_FIXED_BYTES] = bytes(
        ap.BLOB_FIXED_BYTES - _BLOB_HEADER.size
    )
    output[table_end:head] = bytes(head - table_end)
    output[declared:] = bytes(len(blob) - declared)
    return bytes(output)


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def encode_message(header: dict[str, Any], payload: bytes = b"") -> bytes:
    """Frame JSON metadata and raw bytes for one opaque LMCache module call.

    Headers are limited to 64 KiB; the caller must enforce its configured
    payload bound before submission. No payload base64 expansion is performed.
    The output owns its data and is safe to pass to another host thread.
    """
    if not isinstance(header, dict) or any(not isinstance(k, str) for k in header):
        raise ValueError("message header must be a dictionary with string keys")
    if not isinstance(payload, bytes):
        raise ValueError("message payload must be immutable bytes")
    try:
        encoded = json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("message header is not finite JSON metadata") from exc
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError("message header exceeds its byte bound")
    return _FRAME.pack(_WIRE_MAGIC, _WIRE_VERSION, len(encoded), len(payload)) + (
        encoded + payload
    )


def decode_message(data: bytes, max_payload_bytes: int) -> tuple[dict[str, Any], bytes]:
    """Parse one bounded frame, rejecting truncation, duplicates and trailing data.

    Args:
        data: Owned transport response/request bytes.
        max_payload_bytes: Nonnegative payload limit chosen by the receiving
            side; never trust a peer-provided limit.

    Returns:
        The metadata dictionary and raw immutable payload.
    """
    if type(max_payload_bytes) is not int or max_payload_bytes < 0:
        raise ValueError("max_payload_bytes must be a nonnegative integer")
    if not isinstance(data, bytes) or len(data) < _FRAME.size:
        raise ValueError("message frame is missing or truncated")
    magic, version, header_size, payload_size = _FRAME.unpack_from(data)
    if magic != _WIRE_MAGIC or version != _WIRE_VERSION:
        raise ValueError("unsupported FragPack LMCache message version")
    if header_size > MAX_HEADER_BYTES or payload_size > max_payload_bytes:
        raise ValueError("message exceeds receiver bounds")
    if len(data) != _FRAME.size + header_size + payload_size:
        raise ValueError("message has truncated or trailing bytes")
    end = _FRAME.size + header_size
    try:
        header = json.loads(
            data[_FRAME.size : end],
            object_pairs_hook=_json_object,
            parse_constant=_invalid_constant,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid message JSON metadata") from exc
    if not isinstance(header, dict):
        raise ValueError("message metadata must be a JSON object")
    return header, data[end:]

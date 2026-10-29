# SPDX-License-Identifier: Apache-2.0
"""Attention-resident lossless KV page records (research prototype).

A record holds one (page, kv, head) slice of ``PAGE_TOKENS`` tokens by
``HEAD_DIM`` BF16 scalars. The layout is shaped for a fused decode-attention
kernel with 256 threads that decodes straight into ``mma.m16n8k16`` B
fragments: the scalars are first permuted into kernel order, where thread
``t`` (warp ``w``, lane ``4 g + c``) owns positions ``[64 t, 64 t + 64)`` and
its scalar ``s = 8 st + 4 j + 2 f + e`` is

- K: token ``16 w + 8 j + g``, dim ``16 st + 8 f + 2 c + e``;
- V: token ``16 st + 8 f + 2 c + e``, dim ``16 w + 8 j + g``;

so word ``st`` (eight scalars) is the ``(b0, b1)`` pair of both n-blocks of
k-step ``st``. Kernel order then defines every stream below:

- dense low bytes: one raw low byte per scalar (16 KiB). Thread ``t``'s 64
  bytes are four 16-byte chunks stored chunk-major (chunk ``j`` at
  ``16 (256 j + t)``) so one warp load covers 512 contiguous bytes;
- dense symbols: per 32-scalar group three little-endian u32 words; thread
  ``t``'s six words are three 8-byte chunks, also chunk-major. Nibble
  ``j`` of word ``k`` holds the 3-bit primary code of scalar ``8 k + j`` in its
  low three bits; its top bit is bit ``k`` of scalar ``24 + j``'s code. The
  kernel masks the words with ``0x77777777`` and feeds them to ``prmt`` as
  selectors over an 8-byte table, and rebuilds the fourth selector word from the
  top bits;
- variable region: a 48-byte header, then one nibble per escape (primary code
  7) in scalar order, then (4-byte aligned) one little-endian u32 per raw
  scalar, ``ordinal | high << 16`` where ``ordinal`` is its escape index.
  Nibble 0-14 selects extended table entry 7-21; nibble 15 marks a raw scalar
  whose expanded escape byte the kernel patches from the raw list before the
  deposit. The header carries per-warp (2048-scalar) escape and raw prefix
  counts so a warp scan alone locates every thread's escapes and each warp its
  raw patches.

A record that does not fit (variable region over ``VAR_CAP`` or a warp with
more than ``MAX_WARP_RAW`` raw scalars) is stored raw: ``RAW_RECORD_BYTES`` of
little-endian BF16 in kernel order, thread ``t``'s word ``st`` at
``16 (256 st + t)``. Its length alone marks it, since every coded record is at
most ``DENSE_BYTES + VAR_CAP`` bytes.

The high-byte code is the online codec's (3-bit primary, 7 direct entries plus
escape), so bits per scalar match it; only the arrangement differs.

FP8 E4M3 records (:data:`FP8`) use the same kernel order with one byte per
scalar. The coded symbol is the byte's sign and exponent, ``byte & 0xF8``, so
table entries keep their low three bits clear and a decoded symbol ORs with the
mantissa. The 3-bit mantissas replace the low bytes: a 6 KiB stream packed
like the symbol words with each 8-scalar run interleaved (nibble ``2 e`` of
the run's word holds scalar ``e``'s mantissa and nibble ``2 e + 1`` scalar
``4 + e``'s, so each half's four mantissas mask out as bytes), so the dense
part is 12 KiB. The variable region is unchanged and capped at
``FP8.var_cap``; a raw FP8 record is 16 KiB with thread ``t``'s 8-byte word
``st`` at ``8 (256 st + t)``.

FP8 symbols are coded per channel: a scalar in head dim ``d`` codes the symbol
of its byte with the exponent field lowered by ``offset[d]`` mod 16
(:func:`exponent_shift`), so channels whose magnitudes sit at different scales
share the head table's frequent symbols. Decoders raise the exponent back;
mantissas and raw records hold the original bits. :func:`calibrate_fp8` fits
the offsets with the table.
"""

# Standard
from dataclasses import dataclass
import struct

# Third Party
from numpy.typing import NDArray
import numpy as np

PAGE_TOKENS = 128
HEAD_DIM = 128
RECORD_SCALARS = PAGE_TOKENS * HEAD_DIM
GROUP_SCALARS = 32
WARP_SCALARS = 2048
WARPS = RECORD_SCALARS // WARP_SCALARS
PRIMARY_DIRECT = 7
EXTENDED = 15
TABLE_ENTRIES = PRIMARY_DIRECT + EXTENDED
RAW_NIBBLE = 15
HEADER_BYTES = 48
LOW_BYTES = RECORD_SCALARS
SYMBOL_BYTES = RECORD_SCALARS * 3 // 8
DENSE_BYTES = LOW_BYTES + SYMBOL_BYTES
VAR_CAP = 2048
THREADS = 256
LOW_CHUNKS = 4  # 16-byte low-byte chunks per thread
SYMBOL_CHUNKS = 3  # 8-byte symbol chunks per thread
WORDS = 8  # 8-scalar words (mma k-steps) per thread
RAW_RECORD_BYTES = RECORD_SCALARS * 2
MAX_WARP_RAW = 32  # the kernel patches at most one raw scalar per lane
FP8_SYMBOL_MASK = 0xF8  # sign and exponent of an E4M3 byte
FP8_EXPONENTS = 16
# Calibration cost (bits) of a scalar whose symbol escapes to the extended
# table (its nibble) or past it (nibble and raw entry).
_ESCAPE_BITS = 4.0
_RAW_BITS = 36.0
_FIT_ROUNDS = 4
# Block blob (one closed block's records in an external tier, hbm_kv.cu): a
# fixed header (u32 magic, version, L, H; i64 bytes, digest) and the i32
# var-length table [L, 2, H] in a BLOB_ALIGN-padded head, then the records in
# (layer, kv, head) order, each DENSE_BYTES + var_len (raw records whole).
BLOB_MAGIC = 0x424B5046  # "FPKB"
BLOB_FIXED_BYTES = 64
BLOB_ALIGN = 4096

_HEADER = struct.Struct("<4I8H8H")


@dataclass(frozen=True)
class RecordFormat:
    """Byte geometry of one scalar type's records.

    Attributes:
        fp8: True for FP8 E4M3 scalars, False for BF16.
        low_bytes: Dense low stream (BF16 low bytes / FP8 mantissas), which
            the symbol words follow.
        dense_bytes: Dense part of a coded record.
        var_cap: Largest variable region of a coded record.
        raw_bytes: A raw record; its variable length ``raw_bytes -
            dense_bytes`` exceeds ``var_cap``, which marks it.
    """

    fp8: bool
    low_bytes: int
    dense_bytes: int
    var_cap: int
    raw_bytes: int

    @property
    def raw_var_len(self) -> int:
        """Return the ``rec_len`` of a raw record."""
        return self.raw_bytes - self.dense_bytes


BF16 = RecordFormat(
    fp8=False,
    low_bytes=LOW_BYTES,
    dense_bytes=DENSE_BYTES,
    var_cap=VAR_CAP,
    raw_bytes=RAW_RECORD_BYTES,
)
FP8 = RecordFormat(
    fp8=True,
    low_bytes=SYMBOL_BYTES,
    dense_bytes=2 * SYMBOL_BYTES,
    var_cap=RECORD_SCALARS - 2 * SYMBOL_BYTES - 16,
    raw_bytes=RECORD_SCALARS,
)


def record_format(fp8: bool) -> RecordFormat:
    """Return :data:`FP8` or :data:`BF16`.

    Args:
        fp8: Select the FP8 E4M3 format.

    Returns:
        The record format.

    Async/thread-safety:
        Pure function.
    """
    return FP8 if fp8 else BF16


def _kernel_order(kv: int) -> NDArray[np.int64]:
    """Token-major scalar index of each kernel-order position (module docstring)."""
    t = np.arange(THREADS)[:, None]
    s = np.arange(RECORD_SCALARS // THREADS)[None, :]
    w, g, c = t >> 5, (t >> 2) & 7, t & 3
    st, j, f, e = s >> 3, (s >> 2) & 1, (s >> 1) & 1, s & 1
    row, col = 16 * w + 8 * j + g, 16 * st + 8 * f + 2 * c + e
    tok, dim = (row, col) if kv == 0 else (col, row)
    return (tok * HEAD_DIM + dim).reshape(-1)


_ORDER = (_kernel_order(0), _kernel_order(1))
# FP8 mantissa interleave within each 8-scalar run: the scalar held by nibble
# ``n`` and the nibble holding scalar ``e``.
_MANTISSA_SCALARS = np.array([0, 4, 1, 5, 2, 6, 3, 7])
_MANTISSA_NIBBLES = np.argsort(_MANTISSA_SCALARS)


def calibrate_table(
    high_bytes: NDArray[np.uint8], fp8: bool = False
) -> NDArray[np.uint8]:
    """Pick the 22 most frequent symbols of one (layer, kv, head).

    Args:
        high_bytes: Calibration symbols of any shape: BF16 high bytes, or FP8
            bytes masked with ``FP8_SYMBOL_MASK``.
        fp8: Restrict entries to FP8 symbols (multiples of 8).

    Returns:
        uint8 array of ``TABLE_ENTRIES`` symbols, most frequent first.

    Async/thread-safety:
        Pure function.
    """
    counts = np.bincount(np.asarray(high_bytes, dtype=np.uint8).ravel(), minlength=256)
    candidates = np.arange(0, 256, 8) if fp8 else np.arange(256)
    order = candidates[np.argsort(-counts[candidates], kind="stable")]
    return order[:TABLE_ENTRIES].astype(np.uint8)


def exponent_shift(
    symbols: NDArray[np.uint8], offset: NDArray[np.integer] | int
) -> NDArray[np.uint8]:
    """Lower the exponent field of FP8 E4M3 symbols by ``offset`` mod 16.

    Args:
        symbols: FP8 bytes or symbols; bits other than the exponent pass
            through.
        offset: Exponent offsets broadcast against ``symbols`` (for a record,
            one per head dim on the last axis); a negative offset raises the
            exponent, so ``exponent_shift(exponent_shift(x, o), -o) == x``.

    Returns:
        uint8 array of the shifted bytes.

    Async/thread-safety:
        Pure function.
    """
    x = np.asarray(symbols, dtype=np.uint8)
    exp = (x.astype(np.int64) >> 3) - np.asarray(offset, dtype=np.int64)
    return ((x & 0x87) | ((exp & (FP8_EXPONENTS - 1)) << 3)).astype(np.uint8)


def calibrate_fp8(
    fp8: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Fit the table and per-channel exponent offsets of one FP8 (layer, kv, head).

    Starts from each channel's modal exponent aligned to the head's, then
    alternates a frequency table over the shifted symbols with a per-channel
    offset that minimizes estimated escape bits under that table, keeping the
    cheapest round.

    Args:
        fp8: Calibration FP8 E4M3 bytes ``[..., HEAD_DIM]`` (head dim last).

    Returns:
        ``(table, offset)``: the :func:`calibrate_table` table of the shifted
        symbols and uint8 ``[HEAD_DIM]`` offsets for :func:`encode_record`.

    Async/thread-safety:
        Pure function.
    """
    x = np.asarray(fp8, dtype=np.uint8).reshape(-1, HEAD_DIM)
    sym = (x >> 3).astype(np.int64)  # sign << 4 | exponent
    dims = np.arange(HEAD_DIM)
    # hist[d, z]: occurrences of symbol z in channel d.
    hist = np.bincount((dims * 32 + sym).ravel(), minlength=HEAD_DIM * 32)
    hist = hist.reshape(HEAD_DIM, 32)
    exp_hist = hist[:, :16] + hist[:, 16:]
    offset = (exp_hist.argmax(1) - exp_hist.sum(0).argmax()) % FP8_EXPONENTS
    z = np.arange(32)
    # shifted[o, z]: symbol z with its exponent lowered by o.
    shifted = (z & 16)[None, :] | ((z[None, :] - np.arange(16)[:, None]) & 15)
    best = None
    for _ in range(_FIT_ROUNDS):
        counts = np.zeros(32, np.int64)
        np.add.at(counts, shifted[offset], hist)
        order = np.argsort(-counts, kind="stable")
        rank = np.empty(32, np.int64)
        rank[order] = z
        cost_of = np.where(
            rank >= TABLE_ENTRIES,
            _RAW_BITS,
            np.where(rank >= PRIMARY_DIRECT, _ESCAPE_BITS, 0.0),
        )
        bits = float((hist * cost_of[shifted[offset]]).sum())
        if best is None or bits < best[0]:
            best = (bits, (order[:TABLE_ENTRIES] << 3).astype(np.uint8), offset)
        # cost[d, o]: channel d's bits under offset o.
        offset = (hist @ cost_of[shifted].T).argmin(1)
    return best[1], best[2].astype(np.uint8)


def primary_lut(table: NDArray[np.uint8]) -> tuple[int, int]:
    """Return the two u32 words of the 8-byte primary ``prmt`` table.

    Args:
        table: Head table from :func:`calibrate_table`.

    Returns:
        ``(lo, hi)`` little-endian words; byte 7 (escape) is zero.

    Async/thread-safety:
        Pure function.
    """
    entries = bytes(table[:PRIMARY_DIRECT]) + b"\x00"
    lo, hi = struct.unpack("<II", entries)
    return lo, hi


def secondary_lut(table: NDArray[np.uint8]) -> bytes:
    """Return the 16-byte extended table indexed by escape nibble.

    Args:
        table: Head table from :func:`calibrate_table`.

    Returns:
        Entries 7-21 followed by one zero byte.

    Async/thread-safety:
        Pure function.
    """
    return bytes(table[PRIMARY_DIRECT:TABLE_ENTRIES]) + b"\x00"


def encode_record(
    raw: NDArray[np.uint16] | NDArray[np.uint8],
    table: NDArray[np.uint8],
    kv: int,
    offset: NDArray[np.uint8] | None = None,
) -> bytes:
    """Encode one token-major ``[PAGE_TOKENS, HEAD_DIM]`` record.

    Args:
        raw: BF16 bit patterns as uint16, or FP8 E4M3 bytes as uint8 (the
            dtype selects the format).
        table: Head table from :func:`calibrate_table` or
            :func:`calibrate_fp8` (``fp8`` matching).
        kv: 0 for a K record, 1 for a V record (selects the kernel order).
        offset: FP8 only: uint8 ``[HEAD_DIM]`` exponent offsets of the
            table's calibration; None for zeros.

    Returns:
        Record bytes (16-byte aligned): coded, or the format's raw fallback
        when the variable region would exceed its ``var_cap`` or a warp would
        hold more than ``MAX_WARP_RAW`` raw scalars.

    Raises:
        ValueError: If ``raw`` has the wrong size or dtype.

    Async/thread-safety:
        Pure function.
    """
    arr = np.asarray(raw)
    if arr.dtype not in (np.uint8, np.uint16):
        raise ValueError("record must be uint16 (BF16) or uint8 (FP8)")
    fmt = record_format(arr.dtype == np.uint8)
    values = np.ascontiguousarray(arr).reshape(-1)
    if values.size != RECORD_SCALARS:
        raise ValueError("record must hold PAGE_TOKENS * HEAD_DIM scalars")
    values = values[_ORDER[kv]]
    if fmt.fp8:
        high = exponent_shift(values, _channel_offsets(offset, kv)) & FP8_SYMBOL_MASK
        low = _pack_codes((values & 7).reshape(-1, 8)[:, _MANTISSA_SCALARS])
    else:
        high = (values >> 8).astype(np.uint8)
        low = (values & 0xFF).astype(np.uint8)
    rank_of = np.full(256, 255, dtype=np.uint8)
    rank_of[table] = np.arange(TABLE_ENTRIES, dtype=np.uint8)
    rank = rank_of[high]
    words = _pack_codes(np.minimum(rank, PRIMARY_DIRECT))

    escape = rank >= PRIMARY_DIRECT
    is_raw = rank >= TABLE_ENTRIES
    esc_rank = rank[escape]
    nibbles = np.where(esc_rank < TABLE_ENTRIES, esc_rank - PRIMARY_DIRECT, RAW_NIBBLE)
    nibbles = nibbles.astype(np.uint8)
    raw_ordinal = np.flatnonzero(esc_rank >= TABLE_ENTRIES).astype(np.uint32)
    raw_list = (
        (raw_ordinal | (high[is_raw].astype(np.uint32) << 16)).astype("<u4").tobytes()
    )
    starts = np.arange(WARPS) * WARP_SCALARS
    esc_prefix = np.concatenate([[0], np.cumsum(escape)])[starts]
    raw_prefix = np.concatenate([[0], np.cumsum(is_raw)])[starts]
    if nibbles.size % 2:
        nibbles = np.append(nibbles, np.uint8(0))
    nibble_bytes = (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8).tobytes()
    raw_offset = (HEADER_BYTES + len(nibble_bytes) + 3) & ~3
    var_len = _align16(raw_offset + len(raw_list))
    warp_raw = is_raw.reshape(WARPS, WARP_SCALARS).sum(axis=1)
    if var_len > fmt.var_cap or int(warp_raw.max()) > MAX_WARP_RAW:
        return _raw_words(values).tobytes()
    header = _HEADER.pack(
        int(escape.sum()), int(is_raw.sum()), raw_offset, 0, *esc_prefix, *raw_prefix
    )
    var = header + nibble_bytes
    var += b"\x00" * (raw_offset - len(var)) + raw_list
    var += b"\x00" * (var_len - len(var))
    if fmt.fp8:
        dense = _chunk_major(low.astype("<u4"), SYMBOL_CHUNKS)
    else:
        dense = _chunk_major(low, LOW_CHUNKS)
    return dense + _chunk_major(words.astype("<u4"), SYMBOL_CHUNKS) + var


def decode_record(
    record: bytes,
    table: NDArray[np.uint8],
    kv: int,
    fp8: bool = False,
    offset: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint16] | NDArray[np.uint8]:
    """Reference decoder for :func:`encode_record`.

    Args:
        record: Encoded record.
        table: Head table used to encode it.
        kv: The ``kv`` the record was encoded with.
        fp8: Whether the record is in the FP8 format.
        offset: The FP8 exponent offsets it was encoded with.

    Returns:
        ``[PAGE_TOKENS, HEAD_DIM]`` BF16 bit patterns (uint16) or FP8 bytes
        (uint8).

    Async/thread-safety:
        Pure function.
    """
    fmt = record_format(fp8)
    dtype = np.uint8 if fp8 else np.uint16
    out = np.empty(RECORD_SCALARS, dtype=dtype)
    if len(record) == fmt.raw_bytes:
        words = np.frombuffer(record, dtype=np.dtype(dtype).newbyteorder("<"))
        out[_ORDER[kv]] = (
            words.reshape(WORDS, THREADS, 8).transpose(1, 0, 2).reshape(-1)
        )
        return out.reshape(PAGE_TOKENS, HEAD_DIM)
    if fp8:
        low = _unpack_codes(
            _thread_major(record[: fmt.low_bytes], "<u4", SYMBOL_CHUNKS)
        ).reshape(-1, 8)[:, _MANTISSA_NIBBLES].reshape(-1)
    else:
        low = _thread_major(record[: fmt.low_bytes], np.uint8, LOW_CHUNKS).astype(
            np.uint16
        )
    codes = _unpack_codes(
        _thread_major(record[fmt.low_bytes : fmt.dense_bytes], "<u4", SYMBOL_CHUNKS)
    )
    high = np.zeros(RECORD_SCALARS, dtype=np.uint16)
    direct = codes < PRIMARY_DIRECT
    high[direct] = table[codes[direct]]
    var = record[fmt.dense_bytes :]
    n_esc, n_raw, raw_offset, _ = struct.unpack_from("<4I", var)
    packed = np.frombuffer(
        var[HEADER_BYTES : HEADER_BYTES + (n_esc + 1) // 2], np.uint8
    )
    nibbles = np.stack([packed & 15, packed >> 4], axis=1).reshape(-1)[:n_esc]
    raws = np.frombuffer(var[raw_offset : raw_offset + 4 * n_raw], "<u4")
    values = np.zeros(n_esc, dtype=np.uint16)
    ext = nibbles < RAW_NIBBLE
    values[ext] = table[PRIMARY_DIRECT + nibbles[ext]]
    values[raws & 0xFFFF] = raws >> 16
    high[~direct] = values
    if fp8:
        high = exponent_shift(high.astype(np.uint8), -_channel_offsets(offset, kv))
        out[_ORDER[kv]] = low | high
    else:
        out[_ORDER[kv]] = low | (high << 8)
    return out.reshape(PAGE_TOKENS, HEAD_DIM)


def _channel_offsets(offset: NDArray[np.uint8] | None, kv: int) -> NDArray[np.int64]:
    """Exponent offset of each kernel-order scalar (head dim of ``_ORDER``)."""
    if offset is None:
        return np.zeros(RECORD_SCALARS, np.int64)
    offset = np.asarray(offset, dtype=np.int64)
    if offset.shape != (HEAD_DIM,):
        raise ValueError(f"offset must be [{HEAD_DIM}]")
    return offset[_ORDER[kv] % HEAD_DIM]


def _pack_codes(codes: NDArray[np.integer]) -> NDArray[np.uint32]:
    """Pack 3-bit codes into three nibble words per 32-scalar group."""
    codes = np.asarray(codes, dtype=np.uint32).reshape(-1, GROUP_SCALARS)
    words = np.zeros((codes.shape[0], 3), dtype=np.uint32)
    for k in range(3):
        for j in range(8):
            words[:, k] |= (codes[:, 8 * k + j] & 7) << (4 * j)
            words[:, k] |= ((codes[:, 24 + j] >> k) & 1) << (4 * j + 3)
    return words


def _unpack_codes(words: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Inverse of :func:`_pack_codes` for ``[groups * 3]`` words."""
    words = words.reshape(-1, 3)
    codes = np.zeros((words.shape[0], GROUP_SCALARS), dtype=np.uint32)
    for k in range(3):
        for j in range(8):
            codes[:, 8 * k + j] = (words[:, k] >> (4 * j)) & 7
            codes[:, 24 + j] |= ((words[:, k] >> (4 * j + 3)) & 1) << k
    return codes.reshape(-1)


def _chunk_major(stream: NDArray, chunks: int) -> bytes:
    """Store each thread's ``chunks`` equal chunks chunk-major (512 B warp loads)."""
    per_thread = stream.reshape(THREADS, chunks, -1).transpose(1, 0, 2)
    return np.ascontiguousarray(per_thread).tobytes()


def _thread_major(data: bytes, dtype: np.dtype | str, chunks: int) -> NDArray:
    """Inverse of :func:`_chunk_major`: the stream in thread order."""
    arr = np.frombuffer(data, dtype=dtype).reshape(chunks, THREADS, -1)
    return arr.transpose(1, 0, 2).reshape(-1)


def _raw_words(values: NDArray) -> NDArray:
    """Kernel-order scalars as 8-scalar thread words stored word-major."""
    dtype = np.dtype(values.dtype).newbyteorder("<")
    words = values.astype(dtype).reshape(THREADS, WORDS, 8).transpose(1, 0, 2)
    return np.ascontiguousarray(words)


def blob_head_bytes(num_layers: int, kv_heads: int) -> int:
    """Return the padded head (header + length table) of a block blob.

    Args:
        num_layers: Layers ``L``.
        kv_heads: KV heads ``H``.

    Returns:
        ``BLOB_FIXED_BYTES + 4 * L * 2 * H`` rounded up to ``BLOB_ALIGN``.

    Async/thread-safety:
        Pure function.
    """
    head = BLOB_FIXED_BYTES + 8 * num_layers * kv_heads
    return -(-head // BLOB_ALIGN) * BLOB_ALIGN


def blob_slot_bytes(num_layers: int, kv_heads: int, fp8: bool = False) -> int:
    """Return the largest block blob, rounded up to ``BLOB_ALIGN``.

    Every record raw: the external-tier slot size of FragPack storage.

    Args:
        num_layers: Layers ``L``.
        kv_heads: KV heads ``H``.
        fp8: Size for FP8 records.

    Returns:
        ``blob_head_bytes + 2 L H`` raw records (already aligned).

    Async/thread-safety:
        Pure function.
    """
    return blob_head_bytes(num_layers, kv_heads) + (
        2 * num_layers * kv_heads * record_format(fp8).raw_bytes
    )


def _align16(value: int) -> int:
    return (value + 15) & ~15


__all__ = [
    "BLOB_ALIGN",
    "BLOB_FIXED_BYTES",
    "BLOB_MAGIC",
    "BF16",
    "DENSE_BYTES",
    "FP8",
    "FP8_EXPONENTS",
    "FP8_SYMBOL_MASK",
    "HEAD_DIM",
    "PAGE_TOKENS",
    "RAW_RECORD_BYTES",
    "VAR_CAP",
    "RecordFormat",
    "blob_head_bytes",
    "blob_slot_bytes",
    "calibrate_fp8",
    "calibrate_table",
    "decode_record",
    "encode_record",
    "exponent_shift",
    "primary_lut",
    "record_format",
    "secondary_lut",
]

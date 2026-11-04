# SPDX-License-Identifier: Apache-2.0
"""Per-(layer, kv, head) FragPack lookup tables for the HBM-resident KV store.

The tables are calibrated offline from captured KV pages and saved as one
``.npz`` with ``table`` uint8 ``[L, 2, H, 22]`` (symbols, most frequent first),
``offset`` uint8 ``[L, 2, H, 128]`` (per-channel FP8 exponent offsets, zero for
BF16; an absent ``offset`` means zeros) and ``fp8`` (the record format). BF16
symbols are high bytes; FP8 E4M3 symbols are bytes with the mantissa masked off
(``attn_page.FP8_SYMBOL_MASK``) after the exponent shift of
:func:`attn_page.calibrate_fp8`. :func:`load_tables` derives the device-side
forms used by the kernels.

They are fitted by :func:`lmcache.v1.lossless_kv.calibration.calibrate_codebooks`
(``benchmarks/lossless_kv/calibrate.py``), which also derives the tier codec's
tables from them, and written as an artifact's ``tables.npz``. Captured K sits
at the capture's (short) positions, but RoPE rotates K channel pairs with
position, so FP8 K exponent offsets fitted there miss at long positions;
:func:`calibrate` with ``rope_theta`` fits FP8 K over the captured K
re-rotated to positions spanning ``[0, max_positions)`` (NeoX
``rotate_half`` pairs).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass

# Third Party
from numpy.typing import NDArray
import numpy as np
import torch

# First Party
from lmcache.v1.lossless_kv.fragpack import attn_page as ap

# Positions FP8 K calibration spans when re-rotating captured K.
ROPE_CALIBRATION_POSITIONS = 16


def rerotate_k(
    bf16: NDArray[np.uint16], delta: int, theta: float
) -> NDArray[np.uint16]:
    """Advance post-RoPE K by ``delta`` positions (NeoX ``rotate_half`` pairs).

    Args:
        bf16: BF16 bit patterns ``[..., HEAD_DIM]`` of post-RoPE K.
        delta: Positions to add.
        theta: RoPE base.

    Returns:
        BF16 bit patterns of K as if each token sat ``delta`` positions later.

    Async/thread-safety:
        Pure function (CPU).
    """
    half = ap.HEAD_DIM // 2
    inv = theta ** (-torch.arange(0, ap.HEAD_DIM, 2, dtype=torch.float64) / ap.HEAD_DIM)
    x = torch.from_numpy(np.ascontiguousarray(bf16).view(np.int16))
    x = x.view(torch.bfloat16).double()
    cos, sin = torch.cos(delta * inv), torch.sin(delta * inv)
    a, b = x[..., :half], x[..., half:]
    x = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)
    return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


@dataclass(frozen=True)
class DeviceTables:
    """Kernel lookup tables for every layer.

    Attributes:
        rank_of: uint8 ``[L, 2, H, 256]`` table rank of each high byte, 255 if
            absent (encoder).
        plut: int32 ``[L, 2, H, 2]`` primary ``prmt`` words (decoder).
        slut: uint8 ``[L, 2, H, 16]`` extended table (decoder).
        offset: uint8 ``[L, 2, H, 128]`` FP8 exponent offsets, pre-shifted
            to the exponent field (``offset << 3``); zeros for BF16.
        fp8: Tables of FP8 E4M3 records.
    """

    rank_of: torch.Tensor
    plut: torch.Tensor
    slut: torch.Tensor
    offset: torch.Tensor
    fp8: bool = False


def fp8_bytes(bf16: NDArray[np.uint16], scale: float = 1.0) -> NDArray[np.uint8]:
    """Quantize BF16 bit patterns to FP8 E4M3 bytes like an FP8 cache write.

    Args:
        bf16: BF16 bit patterns of any shape.
        scale: Cache scale; values are divided by it before rounding.

    Returns:
        uint8 E4M3 bytes, same shape (saturated to +-448, rounded to nearest).

    Async/thread-safety:
        Pure function (CPU).
    """
    x = torch.from_numpy(np.ascontiguousarray(bf16).view(np.int16))
    x = x.view(torch.bfloat16).float() / scale
    x = x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x.view(torch.uint8).numpy()


def calibrate(
    dump: NDArray[np.uint16],
    fp8: bool = False,
    rope_theta: float | None = None,
    max_positions: int = 0,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Calibrate one table per (layer, kv, head) from captured pages.

    Args:
        dump: uint16 ``[S, L, 2, 128, H, 128]`` BF16 bit patterns.
        fp8: Calibrate FP8 records of the unscaled FP8 quantization (tables
            and exponent offsets from :func:`attn_page.calibrate_fp8`).
        rope_theta: RoPE base of the model. With ``fp8``, K is calibrated on
            the captured K re-rotated (:func:`rerotate_k`) to
            ``ROPE_CALIBRATION_POSITIONS`` positions spanning
            ``[0, max_positions)``; None calibrates K at the captured
            positions only.
        max_positions: Longest context the tables serve (with ``rope_theta``).

    Returns:
        ``(table, offset)``: uint8 ``[L, 2, H, 22]`` tables and uint8
        ``[L, 2, H, 128]`` exponent offsets (zeros for BF16).

    Async/thread-safety:
        Pure function.
    """
    _, n_layers, kv, _, heads, dim = dump.shape
    if fp8 and rope_theta is not None and max_positions <= 0:
        raise ValueError("rope_theta needs max_positions > 0")
    shifts = (
        np.linspace(0, max_positions, ROPE_CALIBRATION_POSITIONS, endpoint=False)
        .astype(np.int64)
        .tolist()
        if fp8 and rope_theta is not None
        else None
    )
    table = np.empty((n_layers, kv, heads, ap.TABLE_ENTRIES), dtype=np.uint8)
    offset = np.zeros((n_layers, kv, heads, dim), dtype=np.uint8)
    for layer in range(n_layers):
        block = np.asarray(dump[:, layer])
        symbols = fp8_bytes(block) if fp8 else (block >> 8).astype(np.uint8)
        if shifts is not None:
            keys = np.concatenate(
                [fp8_bytes(rerotate_k(block[:, 0], d, rope_theta)) for d in shifts]
            )
        for k in range(kv):
            for h in range(heads):
                head = symbols[:, k, :, h, :]
                if shifts is not None and k == 0:
                    head = keys[:, :, h, :]
                if fp8:
                    table[layer, k, h], offset[layer, k, h] = ap.calibrate_fp8(head)
                else:
                    table[layer, k, h] = ap.calibrate_table(head)
    return table, offset


_ALL_HEADS = slice(None)


def load_tables(
    path: str, device: torch.device, heads: slice = _ALL_HEADS
) -> DeviceTables:
    """Load a calibrated ``.npz`` and build the kernel tables on ``device``.

    Args:
        path: An artifact's ``tables.npz``.
        device: Target CUDA device.
        heads: KV heads this worker holds (a TP rank's contiguous shard of
            the calibrated heads); all heads by default.

    Returns:
        Device tables of the selected heads.

    Raises:
        ValueError: if the selection holds no calibrated head.

    Async/thread-safety:
        Blocking file read and host-to-device copies.
    """
    data = np.load(path)
    table = data["table"][:, :, heads]
    if table.shape[2] == 0:
        raise ValueError(f"FragPack tables {path} hold no head in {heads}")
    fp8 = bool(data["fp8"]) if "fp8" in data else False
    offset = data["offset"][:, :, heads] if "offset" in data else None
    return device_tables(np.ascontiguousarray(table), device, fp8, offset)


def device_tables(
    table: NDArray[np.uint8],
    device: torch.device,
    fp8: bool = False,
    offset: NDArray[np.uint8] | None = None,
) -> DeviceTables:
    """Build the kernel tables from calibrated tables.

    Args:
        table: uint8 ``[L, 2, H, 22]`` from :func:`calibrate`.
        device: Target CUDA device.
        fp8: The tables code FP8 E4M3 records.
        offset: uint8 ``[L, 2, H, 128]`` FP8 exponent offsets from
            :func:`calibrate`; None for zeros.

    Returns:
        Device tables.

    Raises:
        ValueError: If the shapes do not match, or BF16 tables carry offsets.

    Async/thread-safety:
        Blocking host-to-device copies.
    """
    n_layers, kv, heads, entries = table.shape
    if entries != ap.TABLE_ENTRIES:
        raise ValueError(f"expected {ap.TABLE_ENTRIES} table entries, got {entries}")
    if offset is None:
        offset = np.zeros((n_layers, kv, heads, ap.HEAD_DIM), dtype=np.uint8)
    offset = np.asarray(offset, dtype=np.uint8)
    if offset.shape != (n_layers, kv, heads, ap.HEAD_DIM):
        raise ValueError(f"offset {offset.shape} does not match tables {table.shape}")
    if not fp8 and offset.any():
        raise ValueError("exponent offsets apply to FP8 tables only")
    rank_of = np.full((n_layers, kv, heads, 256), 255, dtype=np.uint8)
    plut = np.empty((n_layers, kv, heads, 2), dtype=np.uint32)
    slut = np.empty((n_layers, kv, heads, 16), dtype=np.uint8)
    for idx in np.ndindex(n_layers, kv, heads):
        t = table[idx]
        rank_of[idx][t] = np.arange(ap.TABLE_ENTRIES, dtype=np.uint8)
        plut[idx] = ap.primary_lut(t)
        slut[idx] = np.frombuffer(ap.secondary_lut(t), dtype=np.uint8)
    return DeviceTables(
        rank_of=torch.from_numpy(rank_of).to(device),
        plut=torch.from_numpy(plut.view(np.int32)).to(device),
        slut=torch.from_numpy(slut).to(device),
        offset=torch.from_numpy(((offset % ap.FP8_EXPONENTS) << 3).astype(np.uint8)).to(
            device
        ),
        fp8=fp8,
    )


__all__ = [
    "ROPE_CALIBRATION_POSITIONS",
    "DeviceTables",
    "calibrate",
    "device_tables",
    "fp8_bytes",
    "load_tables",
    "rerotate_k",
]

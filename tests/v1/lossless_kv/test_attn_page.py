# SPDX-License-Identifier: Apache-2.0

# Third Party
import numpy as np
import pytest

# First Party
from lmcache.v1.lossless_kv.fragpack import attn_page as ap


def _bf16_like(rng: np.random.Generator, scale: float) -> np.ndarray:
    values = (rng.standard_normal((ap.PAGE_TOKENS, ap.HEAD_DIM)) * scale).astype(
        np.float32
    )
    return (values.view(np.uint32) >> 16).astype(np.uint16)


@pytest.mark.parametrize("kv", [0, 1])
def test_round_trip_with_escapes_and_raw_bytes(kv: int) -> None:
    rng = np.random.default_rng(0)
    cal = _bf16_like(rng, 0.5)
    table = ap.calibrate_table((cal >> 8).astype(np.uint8))
    record = _bf16_like(rng, 0.5)
    # Force a few bytes outside the 22-entry table so raw escapes occur.
    record[3, 5] = 0x7F00 | (record[3, 5] & 0xFF)
    record[100, 127] = 0x0100
    encoded = ap.encode_record(record, table, kv)
    assert len(encoded) % 16 == 0
    assert len(encoded) <= ap.DENSE_BYTES + ap.VAR_CAP
    assert np.array_equal(ap.decode_record(encoded, table, kv), record)


def test_compressible_record_is_smaller_than_raw() -> None:
    rng = np.random.default_rng(1)
    table = ap.calibrate_table((_bf16_like(rng, 0.1) >> 8).astype(np.uint8))
    encoded = ap.encode_record(_bf16_like(rng, 0.1), table, 0)
    assert len(encoded) < 0.75 * ap.RAW_RECORD_BYTES


@pytest.mark.parametrize("kv", [0, 1])
def test_incompressible_record_round_trips_raw(kv: int) -> None:
    rng = np.random.default_rng(2)
    table = np.arange(22, dtype=np.uint8)
    record = rng.integers(
        0, 1 << 16, size=(ap.PAGE_TOKENS, ap.HEAD_DIM), dtype=np.uint16
    )
    encoded = ap.encode_record(record, table, kv)
    assert len(encoded) == ap.RAW_RECORD_BYTES
    assert np.array_equal(ap.decode_record(encoded, table, kv), record)


def test_rejects_wrong_record_size() -> None:
    with pytest.raises(ValueError):
        ap.encode_record(
            np.zeros(10, dtype=np.uint16), np.arange(22, dtype=np.uint8), 0
        )


def _fp8_like(rng: np.random.Generator, scale: float) -> np.ndarray:
    # Third Party
    import torch

    values = torch.from_numpy(
        (rng.standard_normal((ap.PAGE_TOKENS, ap.HEAD_DIM)) * scale).astype(np.float32)
    )
    return values.to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()


@pytest.mark.parametrize("kv", [0, 1])
def test_fp8_round_trip_with_escapes_and_raw_symbols(kv: int) -> None:
    rng = np.random.default_rng(3)
    cal = _fp8_like(rng, 0.5)
    table = ap.calibrate_table(cal & ap.FP8_SYMBOL_MASK, fp8=True)
    assert not np.any(table & 7)
    record = _fp8_like(rng, 0.5)
    # Exponents far outside the calibration range code as raw escapes.
    record[3, 5] = 0x7E
    record[100, 127] = 0xF1
    encoded = ap.encode_record(record, table, kv)
    assert len(encoded) % 16 == 0
    assert len(encoded) <= ap.FP8.dense_bytes + ap.FP8.var_cap
    decoded = ap.decode_record(encoded, table, kv, fp8=True)
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, record)


def test_fp8_compressible_record_is_smaller_than_raw() -> None:
    rng = np.random.default_rng(4)
    table = ap.calibrate_table(_fp8_like(rng, 0.1) & ap.FP8_SYMBOL_MASK, fp8=True)
    encoded = ap.encode_record(_fp8_like(rng, 0.1), table, 1)
    assert len(encoded) < 0.9 * ap.FP8.raw_bytes


@pytest.mark.parametrize("kv", [0, 1])
def test_fp8_incompressible_record_round_trips_raw(kv: int) -> None:
    rng = np.random.default_rng(5)
    table = np.arange(0, 8 * 22, 8, dtype=np.uint8)
    record = rng.integers(0, 256, size=(ap.PAGE_TOKENS, ap.HEAD_DIM), dtype=np.uint8)
    encoded = ap.encode_record(record, table, kv)
    assert len(encoded) == ap.FP8.raw_bytes
    assert np.array_equal(ap.decode_record(encoded, table, kv, fp8=True), record)


def test_blob_slot_bytes_scale_with_record_format() -> None:
    bf16 = ap.blob_slot_bytes(4, 2) - ap.blob_head_bytes(4, 2)
    fp8 = ap.blob_slot_bytes(4, 2, fp8=True) - ap.blob_head_bytes(4, 2)
    assert bf16 == 2 * fp8 == 2 * 4 * 2 * ap.RAW_RECORD_BYTES


def _skewed_fp8(rng: np.random.Generator, pages: int) -> np.ndarray:
    """FP8 pages whose channels sit at magnitudes 2^-6 .. 2^6."""
    # Third Party
    import torch

    chan = np.exp2(np.random.default_rng(9).integers(-6, 7, ap.HEAD_DIM))
    values = rng.standard_normal((pages, ap.PAGE_TOKENS, ap.HEAD_DIM)) * chan
    x = torch.from_numpy(values.astype(np.float32)).to(torch.float8_e4m3fn)
    return x.view(torch.uint8).numpy().copy()


def test_exponent_shift_inverts_and_keeps_sign_and_mantissa() -> None:
    x = np.arange(256, dtype=np.uint8)
    for o in range(ap.FP8_EXPONENTS):
        y = ap.exponent_shift(x, o)
        assert np.array_equal(y & 0x87, x & 0x87)
        assert np.array_equal(ap.exponent_shift(y, -o), x)


@pytest.mark.parametrize("kv", [0, 1])
def test_fp8_offsets_round_trip(kv: int) -> None:
    rng = np.random.default_rng(6)
    table, offset = ap.calibrate_fp8(_skewed_fp8(rng, 4))
    assert offset.shape == (ap.HEAD_DIM,) and offset.any()
    record = _skewed_fp8(rng, 1)[0]
    record[7, 9] = 0x7E  # out-of-table symbol: raw escape
    encoded = ap.encode_record(record, table, kv, offset)
    assert len(encoded) < ap.FP8.raw_bytes
    assert np.array_equal(ap.decode_record(encoded, table, kv, True, offset), record)


def test_fp8_offsets_shrink_skewed_channels() -> None:
    rng = np.random.default_rng(7)
    cal, record = _skewed_fp8(rng, 4), _skewed_fp8(rng, 1)[0]
    plain = ap.calibrate_table(cal & ap.FP8_SYMBOL_MASK, fp8=True)
    table, offset = ap.calibrate_fp8(cal)
    for kv in (0, 1):
        assert len(ap.encode_record(record, table, kv, offset)) < len(
            ap.encode_record(record, plain, kv)
        )

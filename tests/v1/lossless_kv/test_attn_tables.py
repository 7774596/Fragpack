# SPDX-License-Identifier: Apache-2.0
"""FragPack table calibration over RoPE positions (CPU)."""

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.v1.lossless_kv.fragpack import attn_page as ap
from lmcache.v1.lossless_kv.fragpack.attn_tables import calibrate, fp8_bytes, rerotate_k

THETA = 1e7
MAX_POSITIONS = 262144


def _bits(x: np.ndarray) -> np.ndarray:
    return (
        torch.from_numpy(x).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    )


def _floats(bits: np.ndarray) -> np.ndarray:
    return (
        torch.from_numpy(np.ascontiguousarray(bits).view(np.int16))
        .view(torch.bfloat16)
        .double()
        .numpy()
    )


def _dump(pages: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic ``[S, 1, 2, 128, 1, 128]`` dump whose K is RoPE'd at short positions.

    Returns the dump and its post-RoPE K ``[S, 128, 1, 128]``. Pre-RoPE K has a
    per-channel bias spanning eight binades, like real K.
    """
    rng = np.random.default_rng(0)
    t, d = ap.PAGE_TOKENS, ap.HEAD_DIM
    scale = 2.0 ** rng.uniform(-4, 4, d)
    pre = scale * rng.choice([-1, 1], d) * 2 + scale * 0.3 * rng.standard_normal(
        (pages, t, 1, d)
    )
    k = np.stack(
        [
            np.concatenate(
                [
                    rerotate_k(
                        _bits(pre[s, i : i + 1].astype(np.float32)), s * t + i, THETA
                    )
                    for i in range(t)
                ]
            )
            for s in range(pages)
        ]
    )
    v = _bits(rng.standard_normal((pages, t, 1, d)).astype(np.float32))
    return np.stack([k, v], 1)[:, None], k


def _raw_records(k: np.ndarray, table: np.ndarray, offset: np.ndarray) -> int:
    fp8 = fp8_bytes(k)
    return sum(
        len(ap.encode_record(fp8[s, :, 0, :], table[0, 0, 0], 0, offset[0, 0, 0]))
        == ap.RECORD_SCALARS  # an FP8 raw record stores one byte per scalar
        for s in range(k.shape[0])
    )


def test_rerotate_k_composes_and_zero_is_identity() -> None:
    rng = np.random.default_rng(1)
    k = _bits(rng.standard_normal((8, 4, ap.HEAD_DIM)).astype(np.float32))
    assert np.array_equal(rerotate_k(k, 0, THETA), k)
    two = _floats(rerotate_k(rerotate_k(k, 30000, THETA), 50000, THETA))
    one = _floats(rerotate_k(k, 80000, THETA))
    np.testing.assert_allclose(two, one, atol=0.05, rtol=0.02)
    # A rotation keeps each channel pair's norm.
    pairs = lambda x: np.hypot(x[..., :64], x[..., 64:])  # noqa: E731
    np.testing.assert_allclose(pairs(one), pairs(_floats(k)), rtol=0.02, atol=0.01)


def test_rope_calibration_keeps_k_offsets_valid_at_long_positions() -> None:
    dump, k = _dump()
    short = calibrate(dump, True)
    aug = calibrate(dump, True, THETA, MAX_POSITIONS)
    far = rerotate_k(k, 100000, THETA)
    # Offsets fitted at the captured positions only: far K falls back to raw.
    assert _raw_records(far, *short) == k.shape[0]
    assert _raw_records(far, *aug) == 0
    assert _raw_records(k, *aug) == 0
    # V carries no position: both calibrations give it the same tables.
    assert np.array_equal(short[0][:, 1], aug[0][:, 1])
    assert np.array_equal(short[1][:, 1], aug[1][:, 1])


def test_rope_calibration_needs_max_positions() -> None:
    dump, _ = _dump(1)
    with pytest.raises(ValueError):
        calibrate(dump, True, THETA, 0)

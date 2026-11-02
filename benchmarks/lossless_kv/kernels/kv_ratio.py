# SPDX-License-Identifier: Apache-2.0
"""Stored ratios of a captured KV dump under both lossless formats.

Calibrates on the first ``min(8, slots / 4)`` slots and evaluates the rest:

- ``chunk_generic`` / ``chunk_calibrated``: the chunk codec of LMCache's
  tiers (stored bytes over raw bytes per slot) with the generic and the
  calibrated codebooks;
- ``fragpack``: FragPack attention records, mean record over raw record,
  plus the share of records left raw;
- ``order0_plane`` / ``order0_global``: the order-0 entropy bound with one
  high-byte table per (layer, K/V, head) plane, or one table overall.

Usage::

    python -m benchmarks.lossless_kv.kernels.kv_ratio KV.npy MODEL_NAME
"""

from __future__ import annotations

# Standard
import json
import sys

# Third Party
import numpy as np

# First Party
from lmcache.v1.lossless_kv.calibration import calibrate_codebooks
from lmcache.v1.lossless_kv.codec import default_online_codebooks, encode_slot
from lmcache.v1.lossless_kv.format import CompressedStoreGeometry, SlotMode
from lmcache.v1.lossless_kv.fragpack import attn_page as ap
from lmcache.v1.lossless_kv.fragpack.attn_tables import calibrate


def _entropy(hist: np.ndarray) -> float:
    """Return the count-weighted order-0 entropy (bits) of histogram rows."""
    p = hist / hist.sum(-1, keepdims=True)
    return float(-(hist * np.log2(np.where(p > 0, p, 1))).sum() / hist.sum())


def main() -> None:
    """CLI entry point; prints one JSON object."""
    path, model = sys.argv[1], sys.argv[2]
    kv = np.load(path, mmap_mode="r")
    slots, layers, two, tokens, heads, head_dim = kv.shape
    geometry = CompressedStoreGeometry(
        num_slots=slots,
        slot_size=kv[0].nbytes,
        block_tokens=tokens,
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
        tile_scalars=256,
    )
    raw = [np.ascontiguousarray(kv[s]).tobytes() for s in range(slots)]
    ncal = min(8, slots // 4)
    books = calibrate_codebooks(raw[:ncal], model_id=model, geometry=geometry)
    evals = range(ncal, slots)
    res: dict[str, object] = {"model": model, "slots_eval": len(evals)}
    for name, codebooks in (
        ("chunk_generic", default_online_codebooks(geometry)),
        ("chunk_calibrated", books.codebooks),
    ):
        ratios = []
        for s in evals:
            enc = encode_slot(raw[s], slot_id=s, geometry=geometry, codebooks=codebooks)
            coded = enc.mode is SlotMode.COMPRESSED
            ratios.append(len(enc.payload) / geometry.slot_size if coded else 1.0)
        res[name] = round(float(np.mean(ratios)), 4)
    tables, _offsets = calibrate(np.ascontiguousarray(kv[:ncal]))
    lens = []
    for s in evals:
        block = np.ascontiguousarray(kv[s])
        for layer in range(layers):
            for k in range(2):
                for h in range(heads):
                    record = ap.encode_record(
                        block[layer, k, :, h, :], tables[layer, k, h], k
                    )
                    lens.append(len(record))
    lens_a = np.array(lens)
    res["fragpack"] = round(float(lens_a.mean() / ap.RAW_RECORD_BYTES), 4)
    res["fragpack_raw_records"] = round(
        float((lens_a >= ap.RAW_RECORD_BYTES).mean()), 5
    )
    x = np.ascontiguousarray(kv[ncal:]).reshape(
        len(evals), layers * two, tokens, heads, head_dim
    )
    x = x.transpose(0, 1, 3, 2, 4).reshape(len(evals), layers * two * heads, -1)
    hi, lo = (x >> 8).astype(np.int64), (x & 0xFF).astype(np.int64)
    plane = np.stack(
        [np.bincount(hi[:, p].ravel(), minlength=256) for p in range(hi.shape[1])]
    )
    low = np.bincount(lo.ravel(), minlength=256)[None]
    res["order0_plane"] = round((_entropy(plane) + _entropy(low)) / 16, 4)
    res["order0_global"] = round(
        (_entropy(plane.sum(0, keepdims=True)) + _entropy(low)) / 16, 4
    )
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

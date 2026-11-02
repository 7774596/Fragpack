# SPDX-License-Identifier: Apache-2.0
"""Fit the codec codebooks used by the ``tier`` arm from a captured KV dump.

The dump is ``[slots, layers, 2, tokens, heads, head_dim]`` uint16 BF16 bits
(``capture_kv.py``). ``--fp8`` first casts it to FP8 E4M3 (unit scale,
saturating, as vLLM's ``--kv-cache-dtype fp8`` stores KV), for FP8 arms.
The output directory holds ``codebooks.bin`` and ``metadata.json`` and is
passed to ``arms.py --codebooks``.

The ``pool`` arms' FragPack tables come from the same dump (for FP8, pass
the model's RoPE base and longest context so K is calibrated across
positions)::

    python -m lmcache.v1.lossless_kv.fragpack.attn_tables KV.npy OUT.npz \\
        [--fp8 --rope-theta THETA --max-positions N]

Usage::

    python -m benchmarks.lossless_kv.calibrate KV.npy OUT_DIR --model-id NAME
"""

from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import hashlib

# Third Party
import numpy as np

# First Party
from lmcache.v1.lossless_kv.calibration import (
    calibrate_codebooks,
    write_calibration_artifact,
)
from lmcache.v1.lossless_kv.format import CompressedStoreGeometry
from lmcache.v1.lossless_kv.fragpack.attn_tables import fp8_bytes


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dump", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--model-id", required=True, help="bound into the metadata")
    p.add_argument("--fp8", action="store_true", help="FP8 E4M3 KV cache")
    args = p.parse_args()
    kv = np.load(args.dump)
    slots, layers, _kv, tokens, heads, head_dim = kv.shape
    if args.fp8:
        kv = fp8_bytes(kv)
    geometry = CompressedStoreGeometry(
        num_slots=slots,
        slot_size=kv[0].nbytes,
        block_tokens=tokens,
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
        tile_scalars=256,
        dtype_bytes=kv.dtype.itemsize,
    )
    artifact = calibrate_codebooks(
        (np.ascontiguousarray(kv[s]).tobytes() for s in range(slots)),
        model_id=args.model_id,
        geometry=geometry,
        source_sha256=hashlib.sha256(args.dump.read_bytes()).hexdigest(),
    )
    write_calibration_artifact(artifact, args.out)
    print(f"{args.out}: {slots} slots, {layers} layers, {heads} KV heads")


if __name__ == "__main__":
    main()

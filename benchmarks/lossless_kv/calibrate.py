# SPDX-License-Identifier: Apache-2.0
"""Fit the lossless KV tables of one model from a captured KV dump.

The dump is ``[slots, layers, 2, tokens, heads, head_dim]`` uint16 BF16 bits
of every KV head (``capture_kv.py``). ``--fp8`` fits tables for the unscaled
FP8 E4M3 cast vLLM's ``--kv-cache-dtype fp8`` stores, with per-channel
exponent offsets; pass the model's RoPE base and longest context so K is
fitted across positions.

The output directory holds ``tables.npz`` and ``metadata.json``. Both arms
use it: ``tier`` passes the directory as the codec's codebooks, ``pool``
points FragPack at ``OUT_DIR/tables.npz``.

Usage::

    python -m benchmarks.lossless_kv.calibrate KV.npy OUT_DIR --model-id NAME \\
        [--fp8 --rope-theta THETA --max-positions N]
"""

from __future__ import annotations

# Standard
from pathlib import Path
import argparse

# Third Party
import numpy as np

# First Party
from lmcache.v1.lossless_kv.calibration import (
    calibrate_codebooks,
    file_sha256,
    write_calibration_artifact,
)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dump", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--model-id", required=True, help="bound into the metadata")
    p.add_argument("--fp8", action="store_true", help="FP8 E4M3 KV cache")
    p.add_argument("--rope-theta", type=float, default=None, help="RoPE base (FP8 K)")
    p.add_argument(
        "--max-positions", type=int, default=0, help="longest served context"
    )
    args = p.parse_args()
    artifact = calibrate_codebooks(
        np.load(args.dump, mmap_mode="r"),
        model_id=args.model_id,
        fp8=args.fp8,
        rope_theta=args.rope_theta,
        max_positions=args.max_positions,
        source_sha256=file_sha256(args.dump),
    )
    write_calibration_artifact(artifact, args.out)
    print(
        f"{args.out}: {artifact.sample_slots} slots, {artifact.num_layers} layers, "
        f"{artifact.num_kv_heads} KV heads, fp8={artifact.fp8}"
    )


if __name__ == "__main__":
    main()

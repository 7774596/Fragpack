# SPDX-License-Identifier: Apache-2.0
"""Write the cold-then-warm restore trace (W2) in Mooncake format.

``--rounds`` rounds of one prompt per ``--lengths`` entry, each prompt with
fresh block hash IDs (cold: prefill), followed by the same prompts again in
the same order (warm: every prefix comes back from the KV cache tiers). With
a small L1 and this cyclic order, warm prefixes load from L2.

Usage::

    python -m benchmarks.lossless_kv.w2_trace OUT.jsonl \\
        --lengths 8192 32768 65536 131072 --rounds 12
"""

from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import json

BLOCK_TOKENS = 512


def build_trace(
    lengths: list[int], rounds: int, output_tokens: int
) -> list[dict[str, object]]:
    """Return the cold rows followed by the identical warm rows.

    Args:
        lengths: Prompt lengths of one round, in tokens.
        rounds: Rounds in the cold pass.
        output_tokens: Tokens decoded per request.

    Returns:
        Trace rows with zero timestamps (admission order is the file order).
    """
    cold: list[dict[str, object]] = []
    next_id = 0
    for _ in range(rounds):
        for length in lengths:
            blocks = -(-length // BLOCK_TOKENS)
            cold.append(
                {
                    "timestamp": 0,
                    "input_length": length,
                    "output_length": output_tokens,
                    "hash_ids": list(range(next_id, next_id + blocks)),
                }
            )
            next_id += blocks
    return cold + cold


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--lengths", type=int, nargs="+", default=[8192, 32768, 65536, 131072]
    )
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--output-tokens", type=int, default=64)
    args = p.parse_args()
    rows = build_trace(args.lengths, args.rounds, args.output_tokens)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()

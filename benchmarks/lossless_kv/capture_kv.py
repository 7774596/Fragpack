# SPDX-License-Identifier: Apache-2.0
"""Capture a model's KV cache on real text as codec calibration input.

Runs one Hugging Face forward pass over ``--tokens`` tokens of a corpus and
saves the KV as ``[blocks, layers, 2, 128, heads, head_dim]`` uint16 BF16
bits (whole 128-token blocks). FP8-quantized checkpoints are dequantized to
BF16 for the forward pass.

Corpora: a plain text file (``--text``), or OpenHands-style agent
trajectories in a parquet file whose rows hold a ``trajectory`` list of
``{"content" | "message"}`` steps (``--trajectories``).

Usage::

    python -m benchmarks.lossless_kv.capture_kv --model M \\
        --trajectories train.parquet --out KV.npy [--tokens 4096]
"""

from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import json

# Third Party
import numpy as np
import torch

BLOCK_TOKENS = 128


def _trajectory_text(path: Path, chars: int) -> str:
    """Concatenate agent trajectory steps up to about ``chars`` characters."""
    # Third Party
    import pyarrow.parquet as pq

    parts: list[str] = []
    for row in pq.read_table(path).to_pylist():
        for step in row.get("trajectory") or []:
            body = step.get("content") or step.get("message") or ""
            if len(body) > 40:
                parts.append(body)
        if sum(map(len, parts)) > chars:
            break
    return "\n".join(parts)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    corpus = p.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--text", type=Path)
    corpus.add_argument("--trajectories", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=4096)
    args = p.parse_args()
    # Third Party
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    chars = args.tokens * 8
    text = (
        args.text.read_text()[:chars]
        if args.text
        else _trajectory_text(args.trajectories, chars)
    )
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens]
    config = AutoConfig.from_pretrained(args.model)
    kwargs = {}
    if getattr(config, "quantization_config", None):
        # Third Party
        from transformers import FineGrainedFP8Config

        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", **kwargs
    ).eval()
    with torch.inference_mode():
        out = model(ids.to(model.device), use_cache=True, logits_to_keep=1)
    pairs = [
        (layer.keys[0].transpose(0, 1), layer.values[0].transpose(0, 1))
        for layer in out.past_key_values.layers
    ]
    tokens = ids.shape[1] // BLOCK_TOKENS * BLOCK_TOKENS
    layers = len(pairs)
    heads, head_dim = pairs[0][0].shape[1:]
    kv = torch.stack(
        [torch.stack([k[:tokens].cpu(), v[:tokens].cpu()]) for k, v in pairs]
    ).to(torch.bfloat16)
    blocks = kv.view(layers, 2, tokens // BLOCK_TOKENS, BLOCK_TOKENS, heads, head_dim)
    blocks = blocks.permute(2, 0, 1, 3, 4, 5).contiguous()
    np.save(args.out, blocks.view(torch.uint16).numpy())
    print(
        json.dumps(
            {
                "model": Path(args.model).name,
                "blocks": int(blocks.shape[0]),
                "layers": layers,
                "kv_heads": int(heads),
                "head_dim": int(head_dim),
            }
        )
    )


if __name__ == "__main__":
    main()

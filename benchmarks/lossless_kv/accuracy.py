# SPDX-License-Identifier: Apache-2.0
"""Cold-then-warm RULER-style accuracy through a KV-cache-backed vLLM.

Tasks (greedy, 64 output tokens; answers are kept for agreement checks):

- ``kv``: RULER niah_multikey_3. The context is only "magic uuid for <key>
  is <value>" lines; the question asks for one key's value. Score 1 if the
  value appears in the answer.
- ``vt``: RULER variable tracking. A 5-hop chain "VAR A = <n>", "VAR B =
  VAR A", ... hidden at random depths in a haystack text; the question asks
  for every variable equal to <n>. Score: share of the chain named.

All prompts are sent once (cold: prefill), then again in the same order
(warm: prefixes come back from the KV cache tiers; with a small L1 and this
cyclic order they load from L2). Per cell it prints cold and warm scores and
how many warm answers differ from cold; records go to ``accuracy.jsonl``.

Harness-compatible, so ``arms.py --agentx-bin`` can run it::

    accuracy.py profile --url U --model M --output-artifact-dir D \\
        --tokenizer T --haystack wikitext.txt [--n 40]
"""

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import argparse
import asyncio
import json
import random
import re
import string
import time
import uuid

# Third Party
import aiohttp

LENGTHS = (32768, 65536, 131072)
TASKS = ("kv", "vt")
# RULER's one-shot example; without it the model names only the direct
# assignment.
_VT_SHOT = (
    'Example: in the text "... VAR QPXZT = 12345. ... VAR MBKRE = VAR QPXZT. '
    '... VAR LOWND = VAR MBKRE. ..." the variables assigned the value 12345 '
    "are: QPXZT, MBKRE, LOWND.\n\n"
)


def _uuid(rng: random.Random) -> str:
    """Return a deterministic UUID string."""
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def build_kv(tok: Any, n_tok: int, rng: random.Random) -> dict[str, Any]:
    """Build one niah_multikey_3 prompt of about ``n_tok`` tokens.

    Args:
        tok: Hugging Face tokenizer.
        n_tok: Target prompt length.
        rng: Seeded generator.

    Returns:
        ``{"task", "len", "prompt", "want"}``.
    """
    line = " One of the special magic uuids for {} is: {}."
    sample = "".join(line.format(_uuid(rng), _uuid(rng)) for _ in range(64))
    n = (n_tok - 200) * 64 // len(tok.encode(sample))
    pairs = [(_uuid(rng), _uuid(rng)) for _ in range(n)]
    key, value = pairs[rng.randrange(n)]
    text = (
        "Some special magic uuids are hidden within the following text. "
        "Make sure to memorize them. I will quiz you about the uuids "
        "afterwards.\n"
        + "".join(line.format(a, b) for a, b in pairs)
        + f"\nWhat is the special magic uuid for {key} mentioned in the "
        "provided text? Answer with the uuid only."
    )
    return {"task": "kv", "len": n_tok, "prompt": text, "want": [value]}


def build_vt(
    tok: Any, hay: list[int], n_tok: int, rng: random.Random
) -> dict[str, Any]:
    """Build one variable-tracking prompt of about ``n_tok`` tokens.

    Args:
        tok: Hugging Face tokenizer.
        hay: Tokenized haystack text.
        n_tok: Target prompt length.
        rng: Seeded generator.

    Returns:
        ``{"task", "len", "prompt", "want"}``.
    """
    names = ["".join(rng.choices(string.ascii_uppercase, k=5)) for _ in range(5)]
    value = rng.randrange(10_000, 99_999)
    stmts = [f" VAR {names[0]} = {value}. "] + [
        f" VAR {names[i]} = VAR {names[i - 1]}. " for i in range(1, 5)
    ]
    body = n_tok - 250 - sum(len(tok.encode(s)) for s in stmts)
    start = rng.randrange(0, len(hay) - body)
    chunk = hay[start : start + body]
    cuts = sorted(rng.randrange(0, body) for _ in stmts)
    parts, prev = [], 0
    for cut, stmt in zip(cuts, stmts, strict=True):
        parts += [tok.decode(chunk[prev:cut]), stmt]
        prev = cut
    parts.append(tok.decode(chunk[prev:]))
    text = (
        _VT_SHOT + "Memorize and track the chain(s) of variable assignment hidden in "
        "the following text.\n"
        + "".join(parts)
        + f"\nQuestion: Find all variables that are assigned the value {value} "
        "in the text above, directly or through a chain of assignments. "
        "Answer with all such variable names only, separated by commas."
    )
    return {"task": "vt", "len": n_tok, "prompt": text, "want": names}


async def _counter(sess: aiohttp.ClientSession, base: str, name: str) -> float:
    """Sum one Prometheus counter of the server over label sets."""
    async with sess.get(f"{base}/metrics") as response:
        text = await response.text()
    pattern = rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)"
    return sum(float(m[1]) for m in re.finditer(pattern, text, re.M))


async def _ask(
    sess: aiohttp.ClientSession,
    url: str,
    model: str,
    item: dict[str, Any],
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Send one greedy chat request and score it."""
    async with sem:
        started = time.monotonic()
        body = {
            "model": model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 0,
            "max_tokens": 64,
        }
        async with sess.post(url, json=body) as response:
            out = await response.json()
        latency = time.monotonic() - started
    answer = out["choices"][0]["message"]["content"]
    score = sum(str(v) in answer for v in item["want"]) / len(item["want"])
    return {
        "task": item["task"],
        "len": item["len"],
        "score": score,
        "answer": answer,
        "want": item["want"],
        "prompt_tokens": out["usage"]["prompt_tokens"],
        "latency_s": latency,
    }


def _line(label: str, cold: list[dict], warm: list[dict]) -> str:
    """Format one cell's cold/warm scores and changed answers."""
    changed = sum(a["answer"] != b["answer"] for a, b in zip(cold, warm, strict=True))
    return (
        f"{label} cold {sum(r['score'] for r in cold) / len(cold):.3f} "
        f"warm {sum(r['score'] for r in warm) / len(warm):.3f} "
        f"(n={len(cold)}, warm answers changed {changed})"
    )


async def run(args: argparse.Namespace) -> None:
    """Build prompts, run both passes, write and print the results.

    Args:
        args: Parsed arguments.
    """
    # Third Party
    from transformers import AutoTokenizer

    out = Path(args.output_artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    hay = tok.encode(Path(args.haystack).read_text())
    items = []
    for length in LENGTHS:
        for task in TASKS:
            rng = random.Random(f"{length}-{task}")
            items += [
                build_kv(tok, length, rng)
                if task == "kv"
                else build_vt(tok, hay, length, rng)
                for _ in range(args.n)
            ]
    base = args.url.rstrip("/")
    sem = asyncio.Semaphore(args.inflight)
    res: dict[str, list[dict[str, Any]]] = {}
    timeout = aiohttp.ClientTimeout(total=7200)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        for phase in ("cold", "warm"):
            hits0 = await _counter(sess, base, "vllm:external_prefix_cache_hits_total")
            prompt0 = await _counter(sess, base, "vllm:prompt_tokens_total")
            started = time.monotonic()
            res[phase] = await asyncio.gather(
                *(
                    _ask(sess, f"{base}/v1/chat/completions", args.model, it, sem)
                    for it in items
                )
            )
            hits = await _counter(sess, base, "vllm:external_prefix_cache_hits_total")
            prompt = await _counter(sess, base, "vllm:prompt_tokens_total")
            share = (hits - hits0) / max(prompt - prompt0, 1)
            print(
                f"{phase}: {time.monotonic() - started:.0f} s, "
                f"external hit share {share:.3f}",
                flush=True,
            )
    with (out / "accuracy.jsonl").open("w") as f:
        for phase in ("cold", "warm"):
            for record in res[phase]:
                f.write(json.dumps({"phase": phase, **record}) + "\n")
    for length in LENGTHS:
        for task in TASKS:

            def cell(r: dict, length: int = length, task: str = task) -> bool:
                return r["len"] == length and r["task"] == task

            cold = [r for r in res["cold"] if cell(r)]
            warm = [r for r in res["warm"] if cell(r)]
            print(_line(f"{length:>7} {task:<4}", cold, warm))
    for task in TASKS:
        cold = [r for r in res["cold"] if r["task"] == task]
        warm = [r for r in res["warm"] if r["task"] == task]
        print(_line(f"all     {task:<4}", cold, warm))


def main() -> None:
    """CLI entry point (harness-compatible argument names)."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("cmd", help="'profile' (accepted for harness compatibility)")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-artifact-dir", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--haystack", required=True, help="text file for the vt task")
    p.add_argument("--n", type=int, default=40, help="prompts per cell")
    p.add_argument("--inflight", type=int, default=8)
    args, _ = p.parse_known_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

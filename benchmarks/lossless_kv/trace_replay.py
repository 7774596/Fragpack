# SPDX-License-Identifier: Apache-2.0
"""Replay a Mooncake-format trace against one vLLM server.

Each row is ``{"timestamp", "input_length", "output_length", "hash_ids"}``;
``hash_ids[i]`` names the i-th 512-token block of the prompt, so equal hash
prefixes give equal token prefixes. Prompts are deterministic token IDs: the
first tokens of a block encode its hash ID, the rest come from a
hash-seeded LCG over the tokenizer's ordinary tokens.

Requests are admitted in trace order (after their timestamp) with at most
``--max-inflight`` in flight, decode exactly ``output_length`` tokens
greedily, and are written to ``requests.jsonl`` (TTFT, latency, prompt and
cached tokens, a digest of the output) plus ``summary.json``. SIGTERM stops
the replay after writing what finished.
"""

from __future__ import annotations

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import argparse
import asyncio
import hashlib
import json
import signal
import statistics
import time

# Third Party
import httpx

BLOCK_TOKENS = 512
_ID_DIGITS = 8
_LCG_MUL = 6364136223846793005
_LCG_ADD = 1442695040888963407
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class TraceRequest:
    """One validated trace row.

    Attributes:
        index: Zero-based row number.
        timestamp_ms: Arrival offset in milliseconds.
        input_length: Prompt tokens.
        output_length: Tokens to decode.
        hash_ids: Block hash IDs, ``ceil(input_length / 512)`` of them.
    """

    index: int
    timestamp_ms: float
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]


def load_trace(path: Path) -> list[TraceRequest]:
    """Load and validate a Mooncake-format JSONL trace.

    Args:
        path: Trace path.

    Returns:
        Requests in file order.

    Raises:
        ValueError: If a row is malformed or timestamps decrease.
    """
    requests: list[TraceRequest] = []
    for index, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        request = TraceRequest(
            index=index,
            timestamp_ms=float(row["timestamp"]),
            input_length=int(row["input_length"]),
            output_length=int(row["output_length"]),
            hash_ids=tuple(int(h) for h in row["hash_ids"]),
        )
        blocks = -(-request.input_length // BLOCK_TOKENS)
        if len(request.hash_ids) != blocks or request.output_length < 1:
            raise ValueError(f"row {index}: hash_ids do not cover the prompt")
        if requests and request.timestamp_ms < requests[-1].timestamp_ms:
            raise ValueError(f"row {index}: timestamps must not decrease")
        requests.append(request)
    return requests


class PromptBuilder:
    """Build deterministic token-ID prompts from block hash IDs.

    Args:
        vocab_size: Tokenizer vocabulary size.
        special_token_ids: Token IDs never used in prompts.

    Thread-safety:
        Single event-loop use; the block LRU is not thread-safe.
    """

    def __init__(self, vocab_size: int, special_token_ids: Sequence[int]) -> None:
        """Index the ordinary tokens."""
        excluded = set(special_token_ids)
        self._tokens = tuple(t for t in range(vocab_size) if t not in excluded)
        self._blocks: OrderedDict[int, tuple[int, ...]] = OrderedDict()

    def build(self, request: TraceRequest) -> list[int]:
        """Return the ``input_length`` prompt tokens of ``request``.

        Args:
            request: Trace row.

        Returns:
            Token IDs whose 512-token blocks follow ``hash_ids``.
        """
        prompt: list[int] = []
        for hash_id in request.hash_ids:
            prompt.extend(self._block(hash_id))
        del prompt[request.input_length :]
        return prompt

    def _block(self, hash_id: int) -> tuple[int, ...]:
        """Return (and cache) the tokens of one block."""
        cached = self._blocks.get(hash_id)
        if cached is not None:
            self._blocks.move_to_end(hash_id)
            return cached
        base = len(self._tokens)
        if hash_id >= base**_ID_DIGITS:
            raise ValueError(f"hash ID {hash_id} exceeds the encoding")
        block, value = [], hash_id
        for _ in range(_ID_DIGITS):
            block.append(self._tokens[value % base])
            value //= base
        state = (hash_id + 1) & _MASK64
        for position in range(_ID_DIGITS, BLOCK_TOKENS):
            state = (state * _LCG_MUL + _LCG_ADD + position) & _MASK64
            block.append(self._tokens[state % base])
        result = tuple(block)
        self._blocks[hash_id] = result
        if len(self._blocks) > 8192:
            self._blocks.popitem(last=False)
        return result


async def _stream(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    request: TraceRequest,
    prompt: list[int],
    seed: int,
) -> dict[str, Any]:
    """Send one streaming completion and measure it.

    Args:
        client: Shared HTTP client.
        url: Server base URL.
        model: Served model name.
        request: Trace row.
        prompt: Prompt token IDs.
        seed: Sampling seed.

    Returns:
        Result record (without timing offsets).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": request.output_length,
        "temperature": 0.0,
        "ignore_eos": True,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        # A generated special token has empty text; token IDs still mark it.
        "return_token_ids": True,
        "request_id": f"trace-{request.index}",
    }
    text: list[str] = []
    usage: dict[str, Any] = {}
    first: float | None = None
    sent = time.perf_counter()
    error = None
    try:
        async with client.stream(
            "POST", f"{url}/v1/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                data = line.removeprefix("data: ").strip()
                if not line.startswith("data: ") or not data:
                    continue
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    fragment = str(choice.get("text", ""))
                    if first is None and (fragment or choice.get("token_ids")):
                        first = time.perf_counter()
                    text.append(fragment)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    done = time.perf_counter()
    if error is None and first is None:
        # A KV load failure ends the request with an empty stream.
        error = "stream completed without generated tokens"
    details = usage.get("prompt_tokens_details") or {}
    return {
        "sample_id": request.index,
        "status": "completed" if error is None else "failed",
        "error": error,
        "input_length": request.input_length,
        "output_length": request.output_length,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "cached_tokens": details.get("cached_tokens"),
        "ttft_ms": ((first or done) - sent) * 1000.0,
        "latency_ms": (done - sent) * 1000.0,
        # Comparable across arms (exactness) without storing the text.
        "generated_text_sha1": hashlib.sha1("".join(text).encode()).hexdigest(),
    }


async def replay(
    requests: list[TraceRequest],
    builder: PromptBuilder,
    url: str,
    model: str,
    output_dir: Path,
    max_inflight: int,
    timeout: float,
    time_scale: float,
    seed: int,
) -> tuple[list[dict[str, Any]], float, bool]:
    """Replay ``requests`` with ordered admission.

    Args:
        requests: Trace rows in order.
        builder: Prompt builder.
        url: Server base URL.
        model: Served model name.
        output_dir: Receives ``requests.jsonl``.
        max_inflight: Requests in flight at most.
        timeout: Per-request timeout in seconds.
        time_scale: Multiplier on trace timestamps (0 sends all at once).
        seed: Sampling seed.

    Returns:
        ``(records, wall seconds, stopped early)``.

    Async-safety:
        Owns the event loop's SIGTERM handler while running.
    """
    sem = asyncio.Semaphore(max_inflight)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    limits = httpx.Limits(max_connections=max_inflight)
    out = (output_dir / "requests.jsonl").open("w", encoding="utf-8")

    async def run(
        client: httpx.AsyncClient,
        request: TraceRequest,
        prompt: list[int],
        admitted: float,
    ) -> None:
        try:
            record = await _stream(client, url, model, request, prompt, seed)
        finally:
            sem.release()
        record["admit_offset_s"] = admitted - started
        record["completion_offset_s"] = time.perf_counter() - started
        records.append(record)
        out.write(json.dumps(record, sort_keys=True) + "\n")
        out.flush()

    tasks: list[asyncio.Task[None]] = []

    async def admit(client: httpx.AsyncClient) -> None:
        for request in requests:
            due = started + request.timestamp_ms / 1000.0 * time_scale
            await asyncio.sleep(max(0.0, due - time.perf_counter()))
            prompt = builder.build(request)
            await sem.acquire()
            admitted = time.perf_counter()
            tasks.append(asyncio.create_task(run(client, request, prompt, admitted)))
        await asyncio.gather(*tasks)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout), limits=limits
        ) as client:
            main = asyncio.create_task(admit(client))
            waiter = asyncio.create_task(stop.wait())
            await asyncio.wait({main, waiter}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            if not main.done():
                # SIGTERM: keep what finished, drop what is in flight.
                for task in [main, *tasks]:
                    task.cancel()
            await asyncio.gather(main, *tasks, return_exceptions=True)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        out.close()
    return records, time.perf_counter() - started, stop.is_set()


def summarise(
    requests: list[TraceRequest],
    records: list[dict[str, Any]],
    wall_s: float,
    stopped: bool,
) -> dict[str, Any]:
    """Return the run summary.

    Args:
        requests: Trace rows.
        records: Finished request records.
        wall_s: Replay wall time.
        stopped: Whether SIGTERM ended the replay.

    Returns:
        JSON-serialisable summary.
    """
    ok = [r for r in records if r["error"] is None]
    ttft = sorted(r["ttft_ms"] for r in ok)

    def pct(q: float) -> float:
        return ttft[min(len(ttft) - 1, int(q * len(ttft)))] if ttft else 0.0

    prompt = sum(r["prompt_tokens"] for r in ok)
    cached = sum(r["cached_tokens"] or 0 for r in ok)
    return {
        "trace_requests": len(requests),
        "completed_requests": len(ok),
        "failed_requests": len(records) - len(ok),
        "stopped": stopped,
        "wall_seconds": wall_s,
        "prompt_tokens": prompt,
        "cached_prompt_share": cached / prompt if prompt else 0.0,
        "input_tokens_per_s": prompt / wall_s if wall_s else 0.0,
        "ttft_ms": {
            "mean": statistics.fmean(ttft) if ttft else 0.0,
            "p50": pct(0.5),
            "p90": pct(0.9),
            "p99": pct(0.99),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse replay arguments.

    Args:
        argv: Arguments without the program name.

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8047")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point; exits 1 if any request failed or the run stopped."""
    args = parse_args(argv)
    # Third Party
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    builder = PromptBuilder(len(tokenizer), tokenizer.all_special_ids)
    requests = load_trace(args.trace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, wall_s, stopped = asyncio.run(
        replay(
            requests,
            builder,
            args.url.rstrip("/"),
            args.model,
            args.output_dir,
            args.max_inflight,
            args.timeout,
            args.time_scale,
            args.seed,
        )
    )
    summary = summarise(requests, records, wall_s, stopped)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    raise SystemExit(1 if summary["failed_requests"] or stopped else 0)


if __name__ == "__main__":
    main()

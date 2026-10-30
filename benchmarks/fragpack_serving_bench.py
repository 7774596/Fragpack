# SPDX-License-Identifier: Apache-2.0
"""Measure real streaming requests against an already running vLLM service.

Use a fresh, exclusively allocated server with prefix caching disabled. The
saved plan supplies identical token-ID inputs and request order across arms.
All timing is client-observed: TTFT ends at the first nonempty generated
token-ID delta, TPOT uses first/last token arrivals, and latency ends at DONE.
Setup, tokenization, JSON request construction, and warmups are excluded.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import threading
import time
from typing import Any
import urllib.error
import urllib.request

from fragpack.logging import init_logger

logger = init_logger(__name__)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _persist(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or min(values) <= 0 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("provide distinct positive integers")
    return values


def sse_events(lines: Iterable[bytes]) -> Iterator[str]:
    """Yield complete SSE data events from byte lines, ignoring comments.

    The caller timestamps each yielded event before JSON decoding. No clock,
    I/O, or shared state is owned here; unterminated event data is rejected.
    """
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        raise ValueError("stream ended inside an SSE event")


@dataclass
class StreamRecord:
    """Accumulate one completion without counting prompt or usage-only chunks.

    ``started_ns`` and event times use the same monotonic client clock. Token
    events retain chunk sizes because network coalescing can merge arrivals.
    This object belongs to one request thread and performs no device work.
    """

    started_ns: int
    prompt: list[int]
    output_tokens: int
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    done_ns: int | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    prompt_verified: bool = False
    generated: list[int] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None

    def consume(self, data: str, arrived_ns: int) -> None:
        """Process an event received at ``arrived_ns``; reject stream errors."""
        if data == "[DONE]":
            self.done_ns = arrived_ns
            return
        event = json.loads(data)
        if not isinstance(event, dict) or "error" in event:
            raise ValueError(f"completion stream error: {event!r}")
        event_id = event.get("id")
        if event_id is not None:
            if self.request_id is not None and event_id != self.request_id:
                raise ValueError("request identity changed during the stream")
            self.request_id = str(event_id)
        usage = event.get("usage")
        if isinstance(usage, dict):
            self.usage = usage
        choices = event.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("expected one completion choice")
        for choice in choices:
            if choice.get("index") != 0:
                raise ValueError("unexpected completion choice index")
            prompt = choice.get("prompt_token_ids")
            if prompt is not None:
                if prompt != self.prompt:
                    raise ValueError("returned prompt IDs differ from the saved plan")
                self.prompt_verified = True
            tokens = choice.get("token_ids")
            if tokens is not None and (
                not isinstance(tokens, list)
                or any(not isinstance(token, int) or token < 0 for token in tokens)
            ):
                raise ValueError("generated token IDs must be a list of integers")
            if tokens:
                if self.first_token_ns is None:
                    self.first_token_ns = arrived_ns
                self.last_token_ns = arrived_ns
                self.generated.extend(tokens)
                self.events.append(
                    {
                        "elapsed_ms": (arrived_ns - self.started_ns) / 1e6,
                        "token_ids": tokens,
                        "text": choice.get("text", ""),
                    }
                )
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]

    def result(self) -> dict[str, Any]:
        """Return checks and client latencies; absent evidence stays invalid."""
        usage = self.usage or {}
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        checks = {
            "done_received": self.done_ns is not None,
            "prompt_ids_equal": self.prompt_verified,
            "prompt_token_count": usage.get("prompt_tokens") == len(self.prompt),
            "usage_completion_count": usage.get("completion_tokens")
            == self.output_tokens,
            "received_completion_count": len(self.generated) == self.output_tokens,
            "cached_tokens_zero": cached == 0,
            "finished_at_length": self.finish_reason == "length",
            "generated_token_seen": self.first_token_ns is not None,
        }
        first, last, done = self.first_token_ns, self.last_token_ns, self.done_ns
        return {
            "valid": all(checks.values()),
            "checks": checks,
            "server_request_id": self.request_id,
            "usage": self.usage,
            "cached_tokens": cached,
            "finish_reason": self.finish_reason,
            "generated_token_ids": self.generated,
            "generated_sha256": _digest(self.generated),
            "token_events": self.events,
            "first_chunk_tokens": len(self.events[0]["token_ids"])
            if self.events
            else 0,
            "ttft_ms": (first - self.started_ns) / 1e6 if first is not None else None,
            "tpot_ms": (last - first) / (len(self.generated) - 1) / 1e6
            if first is not None and last is not None and len(self.generated) > 1
            else None,
            "latency_ms": (done - self.started_ns) / 1e6 if done is not None else None,
        }


def _json_request(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _corpus(seed: int) -> str:
    rng = random.Random(seed)
    subjects = ("storage engine", "query planner", "cache manager", "network worker")
    actions = ("inspects", "updates", "compares", "validates", "records", "loads")
    objects = ("page metadata", "request history", "queue statistics", "model outputs")
    reasons = (
        "before the next batch arrives",
        "after checking the available memory",
        "while preserving the original values",
        "to measure the response time precisely",
    )
    return "\n".join(
        f"Observation {index}: The {rng.choice(subjects)} {rng.choice(actions)} "
        f"the {rng.choice(objects)} {rng.choice(reasons)}. "
        f"The measured interval contains {rng.randrange(3, 997)} requests."
        for index in range(1024)
    )


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    signature = {
        "version": 1,
        "model": args.model,
        "seed": args.seed,
        "input_tokens": args.input_tokens,
        "concurrency": args.concurrency,
        "output_tokens": args.output_tokens,
        "requests_per_case": args.requests_per_case,
        "warmup_batches": args.warmup_batches,
    }
    if args.plan.exists():
        saved = json.loads(args.plan.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise ValueError("saved prompt plan differs from requested configuration")
        return saved
    corpus = _corpus(args.seed)
    tokenized = _json_request(
        args.base_url + "/tokenize",
        {"model": args.model, "prompt": corpus, "add_special_tokens": False},
        args.timeout,
    )
    pool = tokenized.get("tokens", [])
    if len(pool) < max(args.input_tokens):
        raise ValueError("tokenized corpus is shorter than the longest test input")
    rng = random.Random(args.seed)
    cases = []
    for tokens in args.input_tokens:
        for concurrency in args.concurrency:
            name = f"input{tokens}-c{concurrency}"
            case: dict[str, Any] = {
                "name": name,
                "input_tokens": tokens,
                "concurrency": concurrency,
            }
            for phase, batches in (
                ("warmup", args.warmup_batches),
                ("measured", args.requests_per_case // concurrency),
            ):
                rows = []
                for batch in range(batches):
                    requests = []
                    for member in range(concurrency):
                        offset = rng.randrange(len(pool) - tokens + 1)
                        prompt = pool[offset : offset + tokens]
                        requests.append(
                            {
                                "id": f"{name}-{phase}-{batch}-{member}",
                                "prompt": prompt,
                                "prompt_sha256": _digest(prompt),
                            }
                        )
                    rows.append(requests)
                case[phase] = rows
            cases.append(case)
    saved = {
        "signature": signature,
        "corpus_text": corpus,
        "corpus_token_sha256": _digest(pool),
        "corpus_token_count": len(pool),
        "cases": cases,
    }
    _persist(args.plan, saved)
    return saved


def _stream_request(
    args: argparse.Namespace, item: dict[str, Any], gate: threading.Barrier
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "prompt": item["prompt"],
        "max_tokens": args.output_tokens,
        "min_tokens": args.output_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "add_special_tokens": False,
        "echo": False,
        "n": 1,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        args.base_url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "close",
        },
    )
    gate.wait(timeout=args.timeout)
    started = time.perf_counter_ns()
    record = StreamRecord(started, item["prompt"], args.output_tokens)
    status, error = None, None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            for data in sse_events(response):
                arrived = time.perf_counter_ns()
                record.consume(data, arrived)
                if record.done_ns is not None:
                    break
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read(4096).decode("utf-8", errors="replace")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = record.done_ns or time.perf_counter_ns()
    result = record.result()
    result.update(
        id=item["id"],
        prompt_sha256=item["prompt_sha256"],
        requested_input_tokens=len(item["prompt"]),
        requested_output_tokens=args.output_tokens,
        http_status=status,
        error=error,
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
    )
    result["valid"] = result["valid"] and status == 200 and error is None
    return result


def _batch(
    args: argparse.Namespace,
    pool: ThreadPoolExecutor,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    gate = threading.Barrier(len(items) + 1)
    futures = [pool.submit(_stream_request, args, item, gate) for item in items]
    gate.wait(timeout=args.timeout)
    samples = [future.result() for future in futures]
    starts = [sample["started_monotonic_ns"] for sample in samples]
    ends = [sample["ended_monotonic_ns"] for sample in samples]
    seconds = (max(ends) - min(starts)) / 1e9
    valid = all(sample["valid"] for sample in samples)
    output = sum(len(sample["generated_token_ids"]) for sample in samples)
    prompt = sum(sample["requested_input_tokens"] for sample in samples)
    return {
        "valid": valid,
        "samples": samples,
        "elapsed_s": seconds,
        "launch_spread_ms": (max(starts) - min(starts)) / 1e6,
        "output_tokens": output,
        "input_tokens": prompt,
        "output_tokens_per_second": output / seconds if valid else None,
        "total_tokens_per_second": (prompt + output) / seconds if valid else None,
        "requests_per_second": len(samples) / seconds if valid else None,
    }


def _summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p10": quantile(0.1),
        "p90": quantile(0.9),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def _case_summary(batches: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [sample for batch in batches for sample in batch["samples"]]
    seconds = sum(batch["elapsed_s"] for batch in batches)
    return {
        "valid": all(sample["valid"] for sample in samples),
        "requests": len(samples),
        "ttft_ms": _summary([sample["ttft_ms"] for sample in samples]),
        "tpot_ms": _summary([sample["tpot_ms"] for sample in samples]),
        "latency_ms": _summary([sample["latency_ms"] for sample in samples]),
        "active_batch_seconds": seconds,
        "output_tokens_per_second": sum(b["output_tokens"] for b in batches) / seconds,
        "total_tokens_per_second": sum(
            b["input_tokens"] + b["output_tokens"] for b in batches
        )
        / seconds,
        "output_sequence_sha256": _digest(
            [sample["generated_token_ids"] for sample in samples]
        ),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("staged", "direct"), required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--input-tokens", type=_ints, default=[128, 1024, 4096])
    parser.add_argument("--concurrency", type=_ints, default=[1, 4])
    parser.add_argument("--requests-per-case", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if (
        args.requests_per_case <= 0
        or any(args.requests_per_case % c for c in args.concurrency)
        or args.warmup_batches < 1
        or args.output_tokens < 2
        or args.timeout <= 0
    ):
        parser.error(
            "positive divisible request count, warmups, and >=2 outputs required"
        )
    report: dict[str, Any] = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Real full-model streaming HTTP; cold KV with APC disabled",
        "arm": args.arm,
        "replicate": args.replicate,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement": {
            "ttft": "submission to first nonempty generated token-ID delta",
            "tpot": "(last-token arrival - first-token arrival)/(received tokens - 1)",
            "latency": "submission to SSE DONE marker",
            "throughput": "tokens / batch makespan; aggregate sums active batches",
            "excluded": "setup, tokenization, payload construction, warmups",
            "transport": "HTTP connection/request; returns delta token IDs and usage",
            "replication": "in-process samples are not independent process replicates",
        },
        "cases": [],
    }
    _persist(args.out, report)
    try:
        report["models"] = _json_request(
            args.base_url + "/v1/models", None, args.timeout
        )
        plan = _plan(args)
        report["plan_sha256"] = _digest(plan)
        report["case_order"] = [case["name"] for case in plan["cases"]]
        _persist(args.out, report)
        with ThreadPoolExecutor(max_workers=max(args.concurrency)) as pool:
            for case in plan["cases"]:
                result = {
                    "name": case["name"],
                    "input_tokens": case["input_tokens"],
                    "concurrency": case["concurrency"],
                    "warmup": [],
                    "measured": [],
                }
                report["cases"].append(result)
                for phase in ("warmup", "measured"):
                    for items in case[phase]:
                        batch = _batch(args, pool, items)
                        result[phase].append(batch)
                        _persist(args.out, report)
                        if not batch["valid"]:
                            raise RuntimeError(
                                f"invalid {phase} batch in {case['name']}"
                            )
                result["summary"] = _case_summary(result["measured"])
                _persist(args.out, report)
                logger.info(
                    "%s %s: TTFT %.2f ms, TPOT %.2f ms, output %.2f token/s",
                    args.arm,
                    case["name"],
                    result["summary"]["ttft_ms"]["median"],
                    result["summary"]["tpot_ms"]["median"],
                    result["summary"]["output_tokens_per_second"],
                )
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _persist(args.out, report)


if __name__ == "__main__":
    _main()

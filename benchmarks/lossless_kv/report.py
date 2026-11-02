# SPDX-License-Identifier: Apache-2.0
"""Summaries of finished arms.

Subcommands (each takes arm artifact directories, ``<artifact>/<arm>``):

``agentx ARM...``
    Closed-loop AgentX arms behind ``slotgate.py``, profiling phase only:
    requests, input tokens/s, mean proxy wait, engine TTFT (harness TTFT
    minus the proxy wait; mean/p50/p90/p99), ITL (mean/p90), goodput
    (requests/s with engine TTFT <= ``--ttft-slo`` and ITL <= ``--itl-slo``),
    vLLM queue time per request, GPU and external prefix-hit shares of all
    prompt tokens, mean admitted requests and the share with TTFT <= 2 s.

``cached ARM...``
    Token-weighted cached share (GPU prefix cache plus LMCache) of the
    harness's warmup and profiling phases, from per-request usage.

``queue METRICS.tsv ARM``
    vLLM running/waiting queue and KV usage over the profiling window,
    from ``metrics_sampler.py`` samples.

``w2 ARM...``
    Cold/warm trace-replay arms (``w2_trace.py``): TTFT per prompt length,
    warm cached share, and output digests equal between passes and arms.
"""

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import argparse
import json
import re
import statistics

_PROM = re.compile(r"(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)")


def _prom(path: Path) -> dict[str, float]:
    """Sum every ``vllm:`` metric of a Prometheus text file over labels."""
    totals: dict[str, float] = {}
    for line in path.read_text().splitlines():
        match = _PROM.match(line)
        if match:
            totals[match[1]] = totals.get(match[1], 0.0) + float(match[3])
    return totals


def _pct(values: list[float], q: float) -> float:
    """Return the ``q`` quantile (nearest rank) or NaN for no values."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _profile_rows(arm: Path, phase: str = "profiling") -> list[dict[str, Any]]:
    """Return the harness's finished request records of one phase."""
    rows = [
        json.loads(line)
        for line in (arm / "agentx" / "profile_export.jsonl").read_text().splitlines()
    ]
    return [
        r
        for r in rows
        if r["metadata"].get("benchmark_phase") == phase
        and "time_to_first_token" in r["metrics"]
    ]


def _metric(row: dict[str, Any], name: str) -> float:
    """Return one per-request harness metric value, 0 when absent."""
    return (row["metrics"].get(name) or {}).get("value", 0) or 0


def agentx(arms: list[Path], ttft_slo: float, itl_slo_ms: float) -> None:
    """Print the closed-loop summary line of each arm.

    Args:
        arms: Arm artifact directories.
        ttft_slo: Engine TTFT bound for goodput, seconds.
        itl_slo_ms: ITL bound for goodput, milliseconds.
    """
    print(
        f"{'arm':28} {'req':>4} {'in tok/s':>9} {'wait':>5} {'eTTFT':>6} "
        f"{'p50':>5} {'p90':>5} {'p99':>5} {'ITL':>5} {'ip90':>5} {'good/s':>6} "
        f"{'q/req':>5} {'gpu':>5} {'ext':>5} {'infl':>5} {'le2':>5}"
    )
    for arm in arms:
        gate = {}
        for line in (arm / "agentx" / "slotgate.jsonl").read_text().splitlines():
            record = json.loads(line)
            gate[record["id"]] = record
        rows = _profile_rows(arm)
        wait, ttft, itl, events = [], [], [], []
        prompt_tokens = good = 0.0
        for row in rows:
            g = gate.get(row["metadata"]["x_request_id"])
            w = (g["admit"] - g["arrive"]) / 1e9 if g else 0.0
            wait.append(w)
            ttft.append(_metric(row, "time_to_first_token") / 1e3 - w)
            prompt_tokens += _metric(row, "usage_prompt_tokens")
            out = _metric(row, "output_token_count")
            decode = row["metrics"].get("decode_duration")
            # The harness reports durations in milliseconds.
            i = decode["value"] / (out - 1) if out > 1 and decode else None
            if i is not None:
                itl.append(i)
            good += ttft[-1] <= ttft_slo and (i is None or i <= itl_slo_ms)
            if g:
                events += [(g["admit"], 1), (g["end"], -1)]
        t0 = min(r["metadata"]["request_start_ns"] for r in rows)
        t1 = max(r["metadata"]["request_end_ns"] for r in rows)
        span = (t1 - t0) / 1e9
        events.sort()
        cur, last, area = 0, events[0][0], 0.0
        for t, d in events:
            area += cur * (t - last)
            last, cur = t, cur + d
        inflight = area / (events[-1][0] - events[0][0])
        before = _prom(arm / "vllm_metrics_before.prom")
        after = _prom(arm / "vllm_metrics_after.prom")

        def delta(key: str, before: dict = before, after: dict = after) -> float:
            return after.get(key, 0.0) - before.get(key, 0.0)

        done = delta("vllm:request_success_total")
        prompt = delta("vllm:prompt_tokens_total")
        name = "/".join(arm.resolve().parts[-2:])
        print(
            f"{name:28} {len(rows):4d} {prompt_tokens / span:9.0f} "
            f"{statistics.fmean(wait):5.1f} {statistics.fmean(ttft):6.2f} "
            f"{_pct(ttft, 0.5):5.2f} {_pct(ttft, 0.9):5.2f} {_pct(ttft, 0.99):5.2f} "
            f"{statistics.fmean(itl) if itl else 0:5.1f} "
            f"{_pct(itl, 0.9):5.1f} {good / span:6.3f} "
            f"{delta('vllm:request_queue_time_seconds_sum') / done:5.2f} "
            f"{delta('vllm:prefix_cache_hits_total') / prompt:5.3f} "
            f"{delta('vllm:external_prefix_cache_hits_total') / prompt:5.3f} "
            f"{inflight:5.1f} {sum(x <= 2 for x in ttft) / len(ttft):5.2f}"
        )


def cached(arms: list[Path]) -> None:
    """Print the token-weighted cached share per harness phase.

    Args:
        arms: Arm artifact directories.
    """
    for arm in arms:
        parts = []
        for phase in ("warmup", "profiling"):
            rows = _profile_rows(arm, phase)
            prompt = sum(_metric(r, "usage_prompt_tokens") for r in rows)
            hit = sum(_metric(r, "usage_prompt_cache_read_tokens") for r in rows)
            share = hit / prompt if prompt else float("nan")
            parts.append(f"{phase} n={len(rows):4d} cached {share:.2f}")
        print(f"{arm}: " + " | ".join(parts))


def queue(tsv: Path, arm: Path) -> None:
    """Print queue statistics over the profiling window.

    Args:
        tsv: ``metrics_sampler.py`` output.
        arm: Arm artifact directory.
    """
    rows = [r["metadata"] for r in _profile_rows(arm)]
    t0 = min(m["request_start_ns"] for m in rows) / 1e9
    t1 = max(m["request_end_ns"] for m in rows) / 1e9
    header, *data = [line.split("\t") for line in tsv.read_text().splitlines()]
    ix = {k: i for i, k in enumerate(header)}
    s = [r for r in data if r[0][0].isdigit() and t0 <= float(r[0]) <= t1]

    def col(key: str) -> list[float]:
        return [float(r[ix[key]]) for r in s]

    run, wait = col("num_requests_running"), col("num_requests_waiting")
    kv, pre = col("kv_cache_usage_perc"), col("num_preemptions_total")
    print(
        f"running {statistics.fmean(run):.1f}/{_pct(run, 0.9):.0f}/{max(run):.0f} "
        f"waiting {statistics.fmean(wait):.1f}/{_pct(wait, 0.9):.0f}/{max(wait):.0f} "
        f"kv {statistics.fmean(kv):.2f} preempt {pre[-1] - pre[0]:.0f} "
        f"(mean/p90/max, {len(s)} samples)"
    )


def w2(arms: list[Path]) -> None:
    """Print per-length cold/warm TTFT and output equality.

    Args:
        arms: Arm artifact directories holding ``requests.jsonl``.
    """
    loaded: dict[Path, list[dict[str, Any]]] = {}
    for arm in arms:
        lines = (arm / "requests.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        loaded[arm] = sorted(records, key=lambda r: r["sample_id"])
    for arm, records in loaded.items():
        n = len(records) // 2
        cold, warm = records[:n], records[n:]
        same = sum(
            a["generated_text_sha1"] == b["generated_text_sha1"]
            for a, b in zip(cold, warm, strict=True)
        )
        errors = sum(r["error"] is not None for r in records)
        print(f"{arm}: {len(records)} req, errors {errors}, warm == cold {same}/{n}")
        for k in sorted({r["prompt_tokens"] // 1024 for r in cold}):
            c = [r for r in cold if r["prompt_tokens"] // 1024 == k]
            w = [r for r in warm if r["prompt_tokens"] // 1024 == k]
            share = statistics.fmean(
                (r["cached_tokens"] or 0) / r["prompt_tokens"] for r in w
            )
            print(
                f"  {k:4d}K n {len(c):2d} "
                f"cold TTFT {statistics.fmean(r['ttft_ms'] for r in c):8.1f} ms  "
                f"warm {statistics.fmean(r['ttft_ms'] for r in w):7.1f} ms  "
                f"p50 {statistics.median(r['ttft_ms'] for r in w):7.1f}  "
                f"warm cached {share:.3f}"
            )
    names = list(loaded)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n = len(loaded[a]) // 2
            pairs = list(zip(loaded[a], loaded[b], strict=True))
            equal = [
                x["generated_text_sha1"] == y["generated_text_sha1"] for x, y in pairs
            ]
            print(
                f"{a} vs {b}: cold equal {sum(equal[:n])}/{n}, "
                f"warm equal {sum(equal[n:])}/{n}"
            )


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("agentx")
    a.add_argument("arms", type=Path, nargs="+")
    a.add_argument("--ttft-slo", type=float, default=3.0, help="seconds")
    a.add_argument("--itl-slo", type=float, default=40.0, help="milliseconds")
    sub.add_parser("cached").add_argument("arms", type=Path, nargs="+")
    q = sub.add_parser("queue")
    q.add_argument("tsv", type=Path)
    q.add_argument("arm", type=Path)
    sub.add_parser("w2").add_argument("arms", type=Path, nargs="+")
    args = p.parse_args()
    if args.cmd == "agentx":
        agentx(args.arms, args.ttft_slo, args.itl_slo)
    elif args.cmd == "cached":
        cached(args.arms)
    elif args.cmd == "queue":
        queue(args.tsv, args.arm)
    else:
        w2(args.arms)


if __name__ == "__main__":
    main()

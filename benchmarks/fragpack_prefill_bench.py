# SPDX-License-Identifier: Apache-2.0
"""Compare raw FA3, gather+FA3, unsplit fused prefill, and split-KV fused prefill.

This is a single-layer, single-request microbenchmark with synthetic BF16 KV.
Every arm uses the same physical pages and right-aligned causal queries. Setup,
encoding, and validation are excluded from timing. Eager CUDA-event samples
include host launch gaps; --cuda-graph measures captured GPU work instead.
Round order alternates forward/reverse and all samples are retained as JSON.
Split selection defaults to the serving heuristic. --splits 1,2,4,8,16,32
repeats the four arms for each forced split count using the same query/KV inputs.
Run only on an explicitly allocated, exclusively available SM90 GPU.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any

import numpy as np
import torch

from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import calibrate, device_tables
from fragpack.logging import init_logger
from fragpack.ops.fused_attn_decode import RecordTables
from fragpack.ops.fused_attn_prefill import (
    prefill_attention,
    prefill_split_count,
    prefill_workspace_elems,
)
from fragpack.ops.hbm_kv import HbmKvStore, page_bytes

logger = init_logger(__name__)
T, D = ap.PAGE_TOKENS, ap.HEAD_DIM
ARMS = ("raw_fa3", "gather_fa3", "fused_prefill", "fused_splitkv")
ATOL = RTOL = 2e-2


def _ints(value: str) -> list[int]:
    """Parse positive comma-separated integer sizes for the CLI."""
    values = [int(part) for part in value.split(",")]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("sizes must be positive integers")
    return values


def _splits(value: str) -> list[int | None]:
    """Select the automatic heuristic or an explicit power-of-two sweep."""
    if value == "auto":
        return [None]
    choices = _ints(value)
    if any(choice not in (1, 2, 4, 8, 16, 32) for choice in choices):
        raise argparse.ArgumentTypeError(
            "splits must be auto or values from 1,2,4,8,16,32"
        )
    return choices


def _summary(samples: list[float]) -> dict[str, Any]:
    """Describe per-call CUDA-event microseconds while retaining raw samples."""
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "p10_us": float(np.percentile(samples, 10)),
        "p90_us": float(np.percentile(samples, 90)),
        "stdev_us": statistics.pstdev(samples),
    }


def _fill(store: HbmKvStore, kv: torch.Tensor) -> None:
    """Encode complete source pages in bounded chunks through public methods."""
    for first in range(0, kv.shape[1], 32):
        count = min(32, kv.shape[1] - first)
        slots = torch.arange(
            first * T, (first + count) * T, dtype=torch.int64, device=kv.device
        )
        store.prepare(slots, slots.numel(), np.zeros(0, np.int32))
        key, value = kv[:, first : first + count].reshape(2, count * T, -1, D)
        store.write(0, key, value, slots)
        store.encode(0, count)
        store.commit()
    torch.cuda.synchronize()


def _make_store(
    blocks: int, heads: int, seed: int
) -> tuple[HbmKvStore, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Create one compressed layer plus identical raw K/V for the FA3 baseline."""
    generator = torch.Generator().manual_seed(seed)
    calibration = torch.randn(16, 1, 2, T, heads, D, generator=generator) * 0.5
    bits = calibration.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    table = calibrate(bits)
    source = (torch.randn(2, blocks, T, heads, D, generator=generator) * 0.5).to(
        torch.bfloat16
    )
    input_digest = hashlib.sha256(
        source.view(torch.int16).numpy().tobytes()
    ).hexdigest()
    source = source.to("cuda")
    arena = torch.empty(blocks, page_bytes(heads), dtype=torch.uint8, device="cuda")
    store = HbmKvStore(
        [arena],
        heads,
        device_tables(table, torch.device("cuda")),
        max_tokens=32 * T,
        staging_slots=1,
        overflow_records=blocks * 2 * heads,
    )
    _fill(store, source)
    stats = store.stats()
    if stats.err_staging or stats.err_overflow or stats.encoded <= stats.fallbacks:
        raise RuntimeError("benchmark store must contain successfully compressed KV")
    evidence = {
        "input_sha256": input_digest,
        "table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "store_stats": asdict(stats),
        "store_allocated_bytes": store.memory_usage(),
        "source_kind": "synthetic normal BF16, independent calibration samples",
    }
    return store, arena, source, evidence


def _time_arms(
    functions: list[Callable[[], None]],
    outputs: list[torch.Tensor],
    expected: torch.Tensor,
    rounds: int,
    inner: int,
    warmup_ms: float,
    cuda_graph: bool,
) -> tuple[list[list[float]], list[list[str]], list[float]]:
    """Validate and time alternating arm orders on the current CUDA stream."""
    for function in functions:
        for _ in range(3):
            function()
    torch.cuda.synchronize()
    runners = functions
    graphs = []
    if cuda_graph:
        for function in functions:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(inner):
                    function()
            graphs.append(graph)
        runners = [graph.replay for graph in graphs]

    def run(arm: int) -> None:
        for _ in range(1 if cuda_graph else inner):
            runners[arm]()

    def validate(arm: int) -> float:
        outputs[arm].fill_(float("nan"))
        run(arm)
        torch.testing.assert_close(outputs[arm], expected, atol=ATOL, rtol=RTOL)
        return float((outputs[arm].float() - expected.float()).abs().max())

    for arm in range(len(functions)):
        validate(arm)
        elapsed = 0.0
        while elapsed < warmup_ms:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(5):
                run(arm)
            end.record()
            end.synchronize()
            elapsed += start.elapsed_time(end)

    samples: list[list[float]] = [[] for _ in functions]
    orders = []
    for repeat in range(rounds):
        order = list(range(len(functions)))
        if repeat % 2:
            order.reverse()
        orders.append([ARMS[arm] for arm in order])
        for arm in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            run(arm)
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000 / inner)
    errors = [validate(arm) for arm in range(len(functions))]
    return samples, orders, errors


def _case(
    store: HbmKvStore,
    source: torch.Tensor,
    context: int,
    queries: int,
    args: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
    """Run four arms for each split choice with identical queries and KV pages."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    pages = math.ceil(context / T)
    ids = torch.randperm(source.shape[1], dtype=torch.int32, device="cuda")[:pages]
    table = ids[None]
    workspace_table = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
    offsets = torch.tensor([0, queries], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([context], dtype=torch.int32, device="cuda")
    q = torch.randn(
        queries, args.heads * args.group, D, dtype=torch.bfloat16, device="cuda"
    )
    outputs = [torch.empty_like(q) for _ in ARMS]
    wk = torch.empty(pages, T, args.heads, D, dtype=torch.bfloat16, device="cuda")
    wv = torch.empty_like(wk)
    tables = RecordTables(*store.layer_tables(0))

    def attention(
        key: torch.Tensor,
        value: torch.Tensor,
        blocks: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        flash_attn_varlen_func(
            q=q,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=offsets,
            max_seqlen_q=queries,
            seqused_k=lengths,
            max_seqlen_k=context,
            softmax_scale=D**-0.5,
            causal=True,
            block_table=blocks,
            fa_version=3,
        )

    def raw() -> None:
        attention(source[0], source[1], table, outputs[0])

    def gathered() -> None:
        store.gather(0, ids, wk, wv)
        attention(wk, wv, workspace_table, outputs[1])

    def fused() -> None:
        prefill_attention(
            q, tables, table, offsets, lengths, queries, D**-0.5, outputs[2]
        )

    def splitkv() -> None:
        prefill_attention(
            q,
            tables,
            table,
            offsets,
            lengths,
            queries,
            D**-0.5,
            outputs[3],
            num_splits=selected_splits,
            workspace=partial_workspace,
        )

    raw()
    expected = outputs[0].clone()
    automatic = prefill_split_count(
        [queries],
        [context],
        args.heads,
        args.group,
        torch.cuda.get_device_properties(q.device).multi_processor_count,
    )
    for requested in args.splits:
        selected_splits = automatic if requested is None else requested
        counts = prefill_workspace_elems(q.shape[0], q.shape[1], selected_splits)
        partial_workspace = (
            tuple(torch.empty(n, dtype=torch.float32, device=q.device) for n in counts)
            if selected_splits > 1
            else None
        )
        samples, orders, errors = _time_arms(
            [raw, gathered, fused, splitkv],
            outputs,
            expected,
            args.rounds,
            args.inner,
            args.warmup_ms,
            args.cuda_graph,
        )
        timing = {
            name: _summary(values) for name, values in zip(ARMS, samples, strict=True)
        }
        # Yield each completed choice so an interrupted sweep retains earlier
        # measurements rather than discarding the entire geometry.
        yield {
            "context_tokens": context,
            "query_tokens": queries,
            "prefix_tokens": context - queries,
            "kv_heads": args.heads,
            "gqa_group": args.group,
            "physical_pages": pages,
            "split_selection": "auto" if requested is None else "forced",
            "automatic_splits": automatic,
            "selected_splits": selected_splits,
            "splitkv_partial_workspace_bytes": sum(counts) * 4,
            "gather_workspace_bytes": 2 * wk.numel() * wk.element_size(),
            "fused_expanded_kv_workspace_bytes": 0,
            "validation_reference": "FA3 over the original BF16 KV",
            "max_abs_error": dict(zip(ARMS, errors, strict=True)),
            "validation_atol": ATOL,
            "validation_rtol": RTOL,
            "round_order": orders,
            "timing": timing,
            "fused_over_gather_fa3": timing["fused_prefill"]["median_us"]
            / timing["gather_fa3"]["median_us"],
            "fused_over_raw_fa3": timing["fused_prefill"]["median_us"]
            / timing["raw_fa3"]["median_us"],
            "splitkv_over_unsplit": timing["fused_splitkv"]["median_us"]
            / timing["fused_prefill"]["median_us"],
            "splitkv_over_gather_fa3": timing["fused_splitkv"]["median_us"]
            / timing["gather_fa3"]["median_us"],
            "splitkv_over_raw_fa3": timing["fused_splitkv"]["median_us"]
            / timing["raw_fa3"]["median_us"],
        }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Persist completed cases and status beside the requested result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=_ints, default=[1024, 4096, 8192])
    parser.add_argument("--queries", type=_ints, default=[16, 128, 512])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--group", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--warmup-ms", type=float, default=20.0)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--splits",
        type=_splits,
        default=[None],
        help="auto (default), or a comma-separated sweep from 1,2,4,8,16,32",
    )
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.heads < 1 or args.rounds < 2 or args.inner < 1 or args.warmup_ms <= 0:
        parser.error(
            "heads/inner/warmup must be positive and rounds must be at least 2"
        )
    if max(args.queries) > min(args.contexts):
        parser.error("every query size must fit every context size")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("this benchmark requires an exclusively allocated SM90 GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode
        != 0
    )
    report: dict[str, Any] = {
        "status": "running",
        "scope": "single-layer synthetic cached-context prefill; not serving TTFT",
        "revision": revision,
        "tracked_source_dirty": dirty,
        "kernel_source_sha256": hashlib.sha256(
            (root / "fragpack/ops/fused_attn_prefill.cu").read_bytes()
        ).hexdigest(),
        "device_name": props.name,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "timing_mode": "cuda_graph" if args.cuda_graph else "eager",
        "cache_policy": "warm repeated pages; no hardware-cache flush between arms",
        "rounds": args.rounds,
        "inner": args.inner,
        "warmup_ms": args.warmup_ms,
        "seed": args.seed,
        "requested_splits": ["auto" if s is None else s for s in args.splits],
        "cases": [],
    }
    _write_report(args.out, report)
    try:
        store, arena, source, evidence = _make_store(
            math.ceil(max(args.contexts) / T), args.heads, args.seed
        )
        report.update(evidence)
        for context in args.contexts:
            for queries in args.queries:
                for case in _case(store, source, context, queries, args):
                    report["cases"].append(case)
                    _write_report(args.out, report)
                    logger.info(
                        "context=%d q=%d splits=%d medians_us=%s",
                        context,
                        queries,
                        case["selected_splits"],
                        {arm: case["timing"][arm]["median_us"] for arm in ARMS},
                    )
        # Arena ownership must outlive every asynchronous store/attention use.
        torch.cuda.synchronize()
        del store, arena, source
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_report(args.out, report)


if __name__ == "__main__":
    _main()

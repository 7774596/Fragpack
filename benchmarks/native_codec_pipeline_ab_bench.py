# SPDX-License-Identifier: Apache-2.0
"""Compare original enqueue, serial double buffering and H2D overlap."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback
from typing import Any

import torch

from benchmarks import native_codec_budget_bench as budget
from benchmarks import native_codec_dispatch_ab_bench as shared
from benchmarks.native_codec_pipeline import MODES, PipelineRuntime
from fragpack.ops.hbm_kv import page_bytes


def orders(
    rounds: int, reverse: bool, *, profiled: bool = False
) -> list[tuple[str, ...]]:
    """Rotate palindromic three-mode pairs; twelve rounds cover two full cycles."""
    if rounds < 1 or (not profiled and (rounds < 6 or rounds % 6)):
        raise ValueError("rounds must be a positive multiple of six")
    result = []
    for index in range(rounds):
        base = MODES[index % 3 :] + MODES[: index % 3]
        if reverse ^ bool(index % 2):
            base = base[::-1]
        result.append(base + base[::-1])
    return result


def source_identity() -> dict[str, Any]:
    """Extend the frozen evaluator provenance with both new pipeline modules."""
    source = shared.source_identity()
    for name in ("native_codec_pipeline.py", "native_codec_pipeline_ab_bench.py"):
        source["files_sha256"][f"benchmarks/{name}"] = shared.hash_file(
            Path(__file__).with_name(name)
        )
    source["digest"] = hashlib.sha256(
        json.dumps(source["files_sha256"], sort_keys=True).encode()
    ).hexdigest()
    return source


def slot_storage(runtime: PipelineRuntime) -> dict[str, Any]:
    """Identify slot/status storage without retaining extra tensor references."""
    return {
        str(index): {
            **{
                name: shared.storage_signature(getattr(slot, name))
                for name in ("receive", "mapping", "offsets")
            },
            "statuses": {
                str(count): shared.storage_signature(graph.status)
                for count, graph in slot.graphs.items()
            },
        }
        for index, slot in enumerate(runtime.slots)
    }


def fit_runtime(
    arm: str,
    case: dict,
    corpus: budget.Corpus,
    sessions: list[budget.Session],
    queries: torch.Tensor,
) -> tuple[PipelineRuntime | None, dict]:
    """Fit the complete double-slot runtime and prove the next-block boundary."""
    minimum = max(session.blocks for session in sessions)
    stride = (
        budget.RAW_BLOCK_BYTES
        if arm in ("raw_fa3", "native_fa3")
        else budget.L * page_bytes(budget.H)
    )
    limit, upper = case["budget_bytes"], case["budget_bytes"] // stride
    info = {
        "budget_bytes": limit,
        "minimum_blocks": minimum,
        "attempts": [],
        "scope": "actual double-slot construction/cold peak; no single-slot fallback",
    }

    def probe(blocks: int) -> tuple[PipelineRuntime, int]:
        shared.cleanup_gpu()
        torch.cuda.reset_peak_memory_stats()
        runtime = PipelineRuntime(arm, blocks, case, corpus, sessions, queries)
        largest = max(range(len(sessions)), key=lambda index: sessions[index].blocks)
        try:
            runtime.step(largest)
        except BaseException:
            runtime.close()
            raise
        peak = torch.cuda.max_memory_allocated()
        info["attempts"].append(
            {
                "blocks": blocks,
                "peak": peak,
                "within_budget": peak <= limit,
                "components": runtime.memory_usage(),
            }
        )
        return runtime, peak

    if upper < minimum:
        return None, {
            **info,
            "status": "infeasible",
            "reason": "resident KV alone exceeds budget",
        }
    runtime, peak = probe(minimum)
    runtime.close()
    del runtime
    shared.cleanup_gpu()
    if peak > limit:
        return None, {
            **info,
            "status": "infeasible",
            "reason": "minimum double-slot runtime exceeds budget",
        }
    fit = min(upper, minimum + (limit - peak) // stride)
    while True:
        runtime, peak = probe(fit)
        runtime.close()
        del runtime
        shared.cleanup_gpu()
        if peak <= limit:
            break
        fit -= max(1, math.ceil((peak - limit) / stride))
        if fit < minimum:
            raise AssertionError("fit fell below a proven feasible minimum")
    while fit < upper:
        runtime, peak = probe(fit + 1)
        runtime.close()
        del runtime
        shared.cleanup_gpu()
        if peak > limit:
            break
        fit += 1
    runtime, peak = probe(fit)
    if peak > limit:
        runtime.close()
        raise RuntimeError("recreated double-slot fit exceeded budget")
    info.update(
        status="admitted",
        physical_blocks=fit,
        final_setup_peak=peak,
        next_block_boundary_verified=True,
    )
    return runtime, info


def run_arm(
    args: argparse.Namespace,
    case: dict,
    corpus: budget.Corpus,
    sessions: list,
    queries: torch.Tensor,
    references: list,
    arm: str,
    row: dict,
    report: dict,
) -> None:
    """Validate each mode, then pair full trace wall with complete ownership guards."""
    runtime, row["capacity"] = fit_runtime(arm, case, corpus, sessions, queries)
    if runtime is None:
        row["status"] = "infeasible"
        return
    row["components"] = runtime.memory_usage()
    ownership, extra = shared.runtime_storage(runtime), slot_storage(runtime)
    live, blocks = torch.cuda.memory_allocated(), runtime.blocks
    row.update(
        persistent_storage=ownership,
        slot_storage=extra,
        persistent_live_bytes=live,
        validation={},
        cold_probes={},
        warmup={},
        paired_rounds=[],
    )
    expected = None

    def measured(mode: str) -> dict:
        runtime.select_mode(mode)
        result = shared.measure_trace(runtime, case, ownership, live, blocks, expected)
        result["pipeline_mode"] = mode
        if slot_storage(runtime) != extra:
            raise shared.TraceInvariantError("pipeline slot storage changed", result)
        return result

    try:
        for mode in MODES:
            runtime.select_mode(mode)
            row["validation"][mode] = {
                "pipeline_probes": runtime.validate_pipeline(),
                "trace": shared.run_trace(
                    runtime, case["trace"], validate=True, references=references
                ),
            }
            shared.cleanup_gpu()
            shared.check_runtime(runtime, ownership, live, blocks)
            if slot_storage(runtime) != extra:
                raise AssertionError("validation changed pipeline ownership")
            shared.write_result(args.out, report)
        for mode in MODES:
            result = measured(mode)
            expected = shared.policy_signature(result)
            row["cold_probes"][mode] = result
            row["warmup"][mode] = [measured(mode) for _ in range(args.warmup)]
        for index, order in enumerate(
            orders(
                args.rounds, args.order == "reverse", profiled=args.profile_round >= 0
            )
        ):
            paired = {"round": index, "order": order, "traces": []}
            row["paired_rounds"].append(paired)
            if index == args.profile_round:
                torch.cuda.profiler.start()
            try:
                for mode in order:
                    if index == args.profile_round:
                        torch.cuda.nvtx.range_push(f"pipeline/{mode}")
                    try:
                        paired["traces"].append(measured(mode))
                    finally:
                        if index == args.profile_round:
                            torch.cuda.nvtx.range_pop()
            finally:
                if index == args.profile_round:
                    torch.cuda.profiler.stop()
            means = {
                mode: sum(
                    t["trace_wall_us"]
                    for t in paired["traces"]
                    if t["pipeline_mode"] == mode
                )
                / 2
                for mode in MODES
            }
            paired["ratios"] = {
                "net_original_over_overlap": means["original"] / means["overlap"],
                "serial_double_over_overlap": means["serial-double"] / means["overlap"],
                "original_over_serial_double": means["original"]
                / means["serial-double"],
            }
            shared.write_result(args.out, report)
        row["summary"] = {
            name: shared.summary([r["ratios"][name] for r in row["paired_rounds"]])
            for name in row["paired_rounds"][0]["ratios"]
        }
        row["status"] = "passed"
    except BaseException as error:
        row["status"] = "failed"
        if isinstance(error, shared.TraceInvariantError):
            row["failed_trace"] = error.trace
        raise
    finally:
        runtime.close()
        runtime = None
        shared.cleanup_gpu()
        row["remaining_allocated_bytes_after_release"] = torch.cuda.memory_allocated()
        shared.write_result(args.out, report)


def main() -> None:
    """Execute only under the external exclusive-GPU campaign supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dump", "cases", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(budget.ARMS))
    for name, default in (
        ("calib-pages", 8),
        ("eval-start", 8),
        ("eval-pages", 24),
        ("rounds", 12),
        ("warmup", 1),
        ("seed", 314),
        ("profile-round", -1),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--input-description", required=True)
    args = parser.parse_args()
    args.arms = args.arms.split(",")
    orders(args.rounds, args.order == "reverse", profiled=args.profile_round >= 0)
    if (
        min(args.calib_pages, args.eval_pages) < 1
        or args.warmup < 0
        or args.profile_round < -1
        or args.profile_round >= args.rounds
        or len(set(args.arms)) != len(args.arms)
        or any(a not in budget.ARMS for a in args.arms)
    ):
        raise ValueError("invalid configuration")
    cases = budget.load_cases(args.cases)
    report = {
        "schema": "fragpack.native-codec-copy-pipeline.v1",
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_identity(),
        "cases_sha256": shared.hash_file(args.cases),
        "cases": [],
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "profiled_process": args.profile_round >= 0,
        "protocol": {
            "allocation": (
                "same double-slot allocation; original binds unchanged base enqueue "
                "and only uses slot0; no added ready/done events in that control"
            ),
            "work": (
                "48-layer synthetic repeated-capture KV/attention, real pinned H2D, "
                "immutable-session LRU; not serving"
            ),
            "pairing": (
                "rotated/mirrored ABCCBA; two samples/mode/round; paired mean ratios; "
                "no independent-process CI"
            ),
            "wall": (
                "original full trace: policy/preparation/transfers/status gate/"
                "stock attention/completion"
            ),
            "profile_scope": (
                "if enabled, entire output is profiled evidence; separate from "
                "normal timing"
            ),
            "boundary": (
                "verified archives, compressed receive capacity, raw exact extent; "
                "no single-slot fallback"
            ),
            "resources": (
                "supervisor may NUMA-bind; no dedicated CPU cores or machine isolation"
            ),
        },
    }
    shared.write_result(args.out, report)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("select one explicitly authorized exclusive GPU")
        props = torch.cuda.get_device_properties(0)
        report["environment"] = {
            "torch": torch.__version__,
            "gpu_name": props.name,
            "gpu_uuid": str(props.uuid),
            "gpu_memory_bytes": props.total_memory,
        }
        corpus = budget.prepare_corpus(args)
        report["input"] = corpus.provenance
        shared.cleanup_gpu()
        for case in cases:
            sessions = budget.prepare_sessions(corpus, case["contexts"])
            queries = torch.randn(
                budget.L,
                case["query_tokens"],
                budget.HQ,
                budget.D,
                generator=torch.Generator().manual_seed(args.seed),
            ).to(torch.bfloat16)
            references = budget.prepare_reference(corpus, sessions, queries)
            shared.cleanup_gpu()
            if torch.cuda.memory_allocated():
                raise RuntimeError("reference/source setup retained GPU owners")
            row = {
                "case": case,
                "arms": {},
                "status": "running",
                "query_sha256": shared.hash_array(queries.view(torch.int16).numpy()),
                "archives": [
                    {
                        "indices": s.indices.tolist(),
                        "raw_sha256": shared.hash_array(s.raw.numpy()),
                        "compressed_sha256": shared.hash_array(s.compressed.numpy()),
                    }
                    for s in sessions
                ],
            }
            report["cases"].append(row)
            for arm in args.arms:
                row["arms"][arm] = {"status": "preparing"}
                run_arm(
                    args,
                    case,
                    corpus,
                    sessions,
                    queries,
                    references,
                    arm,
                    row["arms"][arm],
                    report,
                )
                print(
                    json.dumps(
                        {
                            "case": case["name"],
                            "arm": arm,
                            "status": row["arms"][arm]["status"],
                        }
                    ),
                    flush=True,
                )
            row["status"] = "passed"
            del sessions, references, queries
        report["status"] = "passed"
    except BaseException as error:
        report.update(
            status="failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        report["source_after"] = source_identity()
        report["source_unchanged"] = report["source_after"] == report["source"]
        report["cases_unchanged"] = (
            shared.hash_file(args.cases) == report["cases_sha256"]
        )
        report["dump_unchanged"] = (
            shared.hash_file(args.dump) == report["input"]["dump_sha256"]
            if "input" in report
            else None
        )
        if (
            not report["source_unchanged"]
            or not report["cases_unchanged"]
            or report["dump_unchanged"] is False
        ):
            report.update(status="failed", integrity_error="source or input changed")
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        shared.write_result(args.out, report)
    if report["status"] != "passed":
        raise RuntimeError(f"pipeline benchmark failed; inspect {args.out}")


if __name__ == "__main__":
    main()

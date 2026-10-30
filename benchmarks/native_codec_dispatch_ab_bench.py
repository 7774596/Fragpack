# SPDX-License-Identifier: Apache-2.0
"""Pair four restore policies on one fixed-capacity, superset-allocated runtime.

This imports the frozen budget evaluator. All trace work and validation retain
that evaluator's boundaries; pairing and allocation invariants are external.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback
from typing import Any

import numpy as np
import torch

from benchmarks import native_codec_budget_bench as budget
from benchmarks.native_codec_budget_bench import (
    _cleanup_gpu as cleanup_gpu,
)
from benchmarks.native_codec_budget_bench import (
    _run_trace as run_trace,
)
from benchmarks.native_codec_budget_bench import (
    _source_identity as evaluator_source_identity,
)
from benchmarks.native_codec_prefill_bench import (
    _hash_array as hash_array,
)
from benchmarks.native_codec_prefill_bench import (
    _hash_file as hash_file,
)
from benchmarks.native_codec_prefill_bench import (
    _summary as summary,
)
from benchmarks.native_codec_prefill_bench import (
    _write as write_result,
)

VARIANTS = {
    "eager_capacity_chunk": ("eager-capacity", "chunk"),
    "graph_chunk": ("graph", "chunk"),
    "eager_capacity_request": ("eager-capacity", "request"),
    "graph_request": ("graph", "request"),
}
CONTRASTS = {
    "dispatch_with_chunk_wait": ("eager_capacity_chunk", "graph_chunk"),
    "dispatch_with_request_wait": ("eager_capacity_request", "graph_request"),
    "request_wait_with_eager": ("eager_capacity_chunk", "eager_capacity_request"),
    "request_wait_with_graph": ("graph_chunk", "graph_request"),
    "combined": ("eager_capacity_chunk", "graph_request"),
}


class TraceInvariantError(RuntimeError):
    """Retain the complete offending trace when a measured invariant fails."""

    def __init__(self, message: str, trace: dict[str, Any]) -> None:
        super().__init__(message)
        self.trace = trace


def variant_orders(rounds: int, reverse: bool) -> list[tuple[str, ...]]:
    """Balance each variant's position across complete four-round Latin cycles."""
    if rounds < 4 or rounds % 4:
        raise ValueError("rounds must be a positive multiple of four")
    base = tuple(VARIANTS)
    if reverse:
        base = base[::-1]
    return [base[index % 4 :] + base[: index % 4] for index in range(rounds)]


def paired_ratios(traces: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Return same-round baseline/candidate wall ratios, without pooling rounds."""
    return {
        name: traces[first]["trace_wall_us"] / traces[second]["trace_wall_us"]
        for name, (first, second) in CONTRASTS.items()
    }


def policy_signature(trace: dict[str, Any]) -> dict[str, Any]:
    """Keep exact policy/transfer observations; exclude timing and intended waits."""
    totals = (
        "hits",
        "misses",
        "evictions",
        "payload_h2d_bytes",
        "metadata_h2d_bytes",
        "restore_chunks",
        "status_d2h_bytes",
        "peak_resident_sessions",
        "peak_resident_blocks",
    )
    requests = (
        "session",
        "context",
        "admitted",
        "hit",
        "evicted",
        "slots",
        "resident_sessions",
        "resident_blocks",
        "payload_h2d_bytes",
        "restore_metadata_h2d_bytes",
        "status_d2h_bytes",
        "chunks",
        "read_metadata_h2d_bytes",
    )
    return {
        "totals": {name: trace[name] for name in totals},
        "requests": [
            {name: row[name] for name in requests} for row in trace["samples"]
        ],
    }


def storage_signature(tensor: torch.Tensor) -> dict[str, Any]:
    """Identify underlying storage, allowing active tensor-view shapes to differ."""
    storage = tensor.untyped_storage()
    return {
        "address": storage.data_ptr(),
        "bytes": storage.nbytes(),
        "device": str(tensor.device),
    }


def runtime_storage(runtime: budget.Runtime) -> dict[str, dict[str, Any]]:
    """Inspect public runtime/core tensor owners without retaining extra tensors."""
    found = {}

    def add(name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            found[name] = storage_signature(value)

    for name in (
        "receive",
        "offsets",
        "mapping",
        "status_host",
        "offsets_host",
        "mapping_host",
        "request_mapping_host",
        "request_offsets_host",
        "request_status_host",
        "read_ids_host",
        "read_ids",
        "length_host",
        "length",
        "cu_q",
        "gather_table",
        "q",
        "output",
        "gather_k",
        "gather_v",
    ):
        add(name, getattr(runtime, name))
    for index, value in enumerate(runtime.fused_workspace or ()):
        add(f"fused_workspace/{index}", value)
    for index, value in enumerate(runtime.arenas):
        add(f"arena/{index}", value)
    for count, captured in runtime.restore_graphs.items():
        add(f"restore_graph/{count}/status", captured.status)
    if runtime.pages is not None:
        add("pages/descriptors", runtime.pages.descriptors)
        for index, value in enumerate(runtime.pages.caches):
            add(f"pages/cache/{index}", value)
    if runtime.codec is not None:
        add("codec/status", runtime.codec.status)
        if runtime.codec.workspace != "raw":
            add("codec/record_offsets", runtime.codec.rec_offsets)
        if runtime.codec.workspace == "all":
            for name in ("buffer", "records", "rec_lens", "offsets", "lengths"):
                add(f"codec/{name}", getattr(runtime.codec, name))
        for name in ("rank_of", "plut", "slut"):
            add(f"codec/tables/{name}", getattr(runtime.codec.tables, name))
    if runtime.store is not None:
        for name in (
            "staging",
            "scratch",
            "overflow",
            "counters",
            "wslot",
            "tags",
            "free_stack",
            "ovf_bits",
            "claim_list",
            "close_list",
            "null_record",
            "rec_addr",
            "rec_len",
            "fill",
            "live",
            "arena_base_dev",
            "arena_stride_dev",
        ):
            add(f"store/{name}", getattr(runtime.store, name))
        for name in ("rank_of", "plut", "slut"):
            add(f"store/tables/{name}", getattr(runtime.store.tables, name))
    return found


def source_identity() -> dict[str, Any]:
    """Record driver, frozen evaluator and all core source identities."""
    source = evaluator_source_identity()
    source["files_sha256"][f"benchmarks/{Path(__file__).name}"] = hash_file(
        Path(__file__)
    )
    source["digest"] = hashlib.sha256(
        json.dumps(source["files_sha256"], sort_keys=True).encode()
    ).hexdigest()
    return source


def select_variant(runtime: budget.Runtime, name: str) -> None:
    """Switch host policy only; construction always owns the complete superset."""
    runtime.restore_dispatch, runtime.restore_wait = VARIANTS[name]


def check_runtime(
    runtime: budget.Runtime, ownership: dict, live_bytes: int, physical_blocks: int
) -> None:
    """Fail if persistent storage, capacity or live allocation changes."""
    if runtime.blocks != physical_blocks or runtime_storage(runtime) != ownership:
        raise AssertionError("paired variants changed capacity or owned storage")
    if torch.cuda.memory_allocated() != live_bytes:
        raise AssertionError("paired variants changed persistent live GPU allocation")


def measure_trace(
    runtime: budget.Runtime,
    case: dict,
    ownership: dict,
    live_bytes: int,
    physical_blocks: int,
    expected_policy: dict | None,
) -> dict[str, Any]:
    """Run the unchanged complete-wall trace and verify all matched invariants."""
    torch.cuda.reset_peak_memory_stats()
    result = run_trace(runtime, case["trace"], validate=False)
    result["runtime_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["runtime_reserved_bytes"] = torch.cuda.memory_reserved()
    result["persistent_live_allocated_bytes"] = torch.cuda.memory_allocated()
    result["restore_dispatch"] = runtime.restore_dispatch
    result["restore_wait"] = runtime.restore_wait
    if result["runtime_peak_allocated_bytes"] > case["budget_bytes"]:
        raise TraceInvariantError(
            f"variant peak {result['runtime_peak_allocated_bytes']} exceeds budget "
            f"{case['budget_bytes']}; paired capacity not refit",
            result,
        )
    try:
        check_runtime(runtime, ownership, live_bytes, physical_blocks)
    except AssertionError as error:
        raise TraceInvariantError(str(error), result) from error
    if expected_policy is not None and policy_signature(result) != expected_policy:
        raise TraceInvariantError("paired policy decisions or bytes changed", result)
    expected_waits = (
        result["restore_chunks"]
        if runtime.restore_wait == "chunk"
        else result["misses"]
    )
    if result["restore_host_waits"] != expected_waits:
        raise TraceInvariantError("completion count differs from wait policy", result)
    result["matched_invariants_passed"] = True
    return result


def run_case(
    args: argparse.Namespace, case: dict, corpus: budget.Corpus, row: dict, report: dict
) -> None:
    """Fit each arm once, validate each variant, then collect balanced paired rounds."""
    sessions = budget.prepare_sessions(corpus, case["contexts"])
    queries = torch.randn(
        budget.L,
        case["query_tokens"],
        budget.HQ,
        budget.D,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(torch.bfloat16)
    references = budget.prepare_reference(corpus, sessions, queries)
    cleanup_gpu()
    if torch.cuda.memory_allocated():
        raise RuntimeError("source/reference preparation retained GPU owners")
    row["setup"] = {
        "query_sha256": hash_array(queries.view(torch.int16).numpy()),
        "host_raw_history_bytes": sum(session.raw.numel() for session in sessions),
        "host_compressed_history_bytes": sum(s.compressed.numel() for s in sessions),
        "sessions": [
            {
                "context": session.context,
                "blocks": session.blocks,
                "capture_pool_indices": session.indices.tolist(),
                "raw_sha256": hash_array(session.raw.numpy()),
                "compressed_sha256": hash_array(session.compressed.numpy()),
                "reference_sha256": hash_array(ref.view(torch.int16).numpy()),
            }
            for session, ref in zip(sessions, references, strict=True)
        ],
    }
    row["arms"] = {}
    fit_case = dict(case, restore_dispatch="graph", restore_wait="request")
    row["superset_fit_case"] = fit_case
    for arm in args.arms:
        arm_row = {"status": "fitting", "variants": {}, "paired_rounds": []}
        row["arms"][arm] = arm_row
        write_result(args.out, report)
        runtime, capacity = budget.fit_runtime(arm, fit_case, corpus, sessions, queries)
        arm_row["capacity"] = capacity
        if runtime is None:
            arm_row["status"] = "infeasible"
            continue
        arm_row["superset_components"] = runtime.memory_usage()
        ownership = runtime_storage(runtime)
        live_bytes, physical_blocks = torch.cuda.memory_allocated(), runtime.blocks
        arm_row["persistent_storage"] = ownership
        arm_row["persistent_live_allocated_bytes"] = live_bytes
        try:
            for name in VARIANTS:
                select_variant(runtime, name)
                variant = {
                    "dispatch": runtime.restore_dispatch,
                    "wait": runtime.restore_wait,
                }
                arm_row["variants"][name] = variant
                variant["dispatch_validation"] = runtime.validate_restore_dispatch()
                variant["status_retention_validation"] = (
                    runtime.validate_request_status_retention()
                )
                variant["validation_trace"] = run_trace(
                    runtime, case["trace"], validate=True, references=references
                )
                cleanup_gpu()
                check_runtime(runtime, ownership, live_bytes, physical_blocks)
                write_result(args.out, report)
            expected_policy = None
            for name in VARIANTS:
                select_variant(runtime, name)
                result = measure_trace(
                    runtime,
                    case,
                    ownership,
                    live_bytes,
                    physical_blocks,
                    expected_policy,
                )
                expected_policy = policy_signature(result)
                arm_row["variants"][name]["cold_budget_probe"] = result
            for name in VARIANTS:
                select_variant(runtime, name)
                arm_row["variants"][name]["warmup_traces"] = [
                    measure_trace(
                        runtime,
                        case,
                        ownership,
                        live_bytes,
                        physical_blocks,
                        expected_policy,
                    )
                    for _ in range(args.warmup)
                ]
            for index, order in enumerate(
                variant_orders(args.rounds, args.order == "reverse")
            ):
                paired = {"round": index, "order": list(order), "traces": {}}
                arm_row["paired_rounds"].append(paired)
                for position, name in enumerate(order):
                    select_variant(runtime, name)
                    result = measure_trace(
                        runtime,
                        case,
                        ownership,
                        live_bytes,
                        physical_blocks,
                        expected_policy,
                    )
                    result["position_in_round"] = position
                    paired["traces"][name] = result
                paired["baseline_over_candidate_ratios"] = paired_ratios(
                    paired["traces"]
                )
                write_result(args.out, report)
            arm_row["summary"] = {
                "trace_wall_us": {
                    name: summary(
                        [
                            r["traces"][name]["trace_wall_us"]
                            for r in arm_row["paired_rounds"]
                        ]
                    )
                    for name in VARIANTS
                },
                "paired_baseline_over_candidate_ratios": {
                    name: summary(
                        [
                            r["baseline_over_candidate_ratios"][name]
                            for r in arm_row["paired_rounds"]
                        ]
                    )
                    for name in CONTRASTS
                },
                "scope": "paired within-process rounds; no independent-process CI",
            }
            arm_row["status"] = "passed"
        except BaseException as error:
            arm_row["status"] = "failed"
            arm_row["error"] = {"type": type(error).__name__, "message": str(error)}
            if isinstance(error, TraceInvariantError):
                arm_row["failed_trace"] = error.trace
            raise
        finally:
            del runtime
            cleanup_gpu()
            arm_row["remaining_allocated_bytes_after_release"] = (
                torch.cuda.memory_allocated()
            )
            write_result(args.out, report)
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "arm": arm,
                    "status": "passed",
                    "rounds": args.rounds,
                    "physical_blocks": physical_blocks,
                }
            ),
            flush=True,
        )
    row["status"] = "passed"


def main() -> None:
    """Run the paired experiment only on a separately reserved exclusive GPU."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("dump", "cases", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(budget.ARMS))
    parser.add_argument("--calib-pages", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=8)
    parser.add_argument("--eval-pages", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--input-description", required=True)
    args = parser.parse_args()
    args.arms = args.arms.split(",")
    orders = variant_orders(args.rounds, args.order == "reverse")
    if (
        min(args.calib_pages, args.eval_pages) < 1
        or args.warmup < 0
        or not args.arms
        or len(set(args.arms)) != len(args.arms)
        or any(arm not in budget.ARMS for arm in args.arms)
    ):
        raise ValueError("invalid arm selection or sample counts")
    cases = budget.load_cases(args.cases)
    report = {
        "schema": "fragpack.native-codec-paired-dispatch.v1",
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "source": source_identity(),
        "cases_sha256": hash_file(args.cases),
        "cases": [],
        "variant_order_by_round": orders,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "vllm": importlib.metadata.version("vllm"),
        },
        "protocol": {
            "allocation": (
                "same superset graph+request runtime/capacity/storage for every "
                "variant; not per-mode minimal deployment memory"
            ),
            "work": (
                "48-layer immutable-session KV/attention trace with synthetic "
                "repeated capture contents; not model serving or TTFT"
            ),
            "extent": (
                "compressed eager-capacity and graph use full receive capacity on "
                "verified immutable archives; raw uses exact bytes"
            ),
            "wall": (
                "unchanged budget evaluator trace wall: policy, metadata, H2D, "
                "restores, status handling, attention/completion, result collection"
            ),
            "outside_wall": (
                "mode switch, invariant checking, output writing and source hashes"
            ),
            "ordering": (
                "balanced four-position cyclic Latin order; reverse initial order "
                "for a separate process; no CPU affinity/isolation or drift immunity"
            ),
            "pairing": (
                "within-round baseline/candidate trace-wall ratios; rounds are not "
                "independent processes; no confidence interval"
            ),
            "source_guard": (
                "driver/helper/core/case/input hashes rechecked after execution"
            ),
            "failure": (
                "retain failures; no mid-pair refit or failed-LRU-admission recovery"
            ),
        },
    }
    write_result(args.out, report)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("select one explicitly authorized exclusive GPU")
        props = torch.cuda.get_device_properties(0)
        report["environment"].update(
            {
                "gpu_name": props.name,
                "gpu_uuid": str(props.uuid),
                "sm_count": props.multi_processor_count,
                "gpu_memory_bytes": props.total_memory,
            }
        )
        corpus = budget.prepare_corpus(args)
        cleanup_gpu()
        if torch.cuda.memory_allocated():
            raise RuntimeError("source preparation retained GPU owners")
        report["input"] = corpus.provenance
        report["status"] = "running"
        for case in cases:
            row = {"case": case, "status": "preparing"}
            report["cases"].append(row)
            write_result(args.out, report)
            run_case(args, case, corpus, row, report)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        final_source = source_identity()
        report["source_after"] = final_source
        report["source_unchanged"] = final_source == report["source"]
        report["cases_unchanged"] = hash_file(args.cases) == report["cases_sha256"]
        report["dump_unchanged"] = (
            hash_file(args.dump) == report["input"]["dump_sha256"]
            if "input" in report
            else None
        )
        if (
            not report["source_unchanged"]
            or not report["cases_unchanged"]
            or report["dump_unchanged"] is False
        ):
            report["status"] = "failed"
            report["integrity_error"] = "source or input changed during execution"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_result(args.out, report)
    if report["status"] != "passed":
        raise RuntimeError(f"paired benchmark failed; inspect {args.out}")


if __name__ == "__main__":
    main()

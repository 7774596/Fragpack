# SPDX-License-Identifier: Apache-2.0
"""Measure isolated deployment tensor memory at fixed resident KV capacities.

Separate full/decode codec workspace and bounded chunk size, keeping resident
pages fixed. Each row constructs one complete reader, restores actual host KV,
and reads 48 layers. Bit/output validation runs after the measured peak window.
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

from benchmarks.native_codec_budget_bench import (
    ARMS,
    HQ,
    Corpus,
    D,
    H,
    L,
    Runtime,
    Session,
    SessionLRU,
    T,
    prepare_corpus,
    prepare_reference,
    prepare_sessions,
)
from benchmarks.native_codec_budget_bench import (
    _cleanup_gpu as cleanup_gpu,
)
from benchmarks.native_codec_budget_bench import (
    _source_identity as budget_source_identity,
)
from benchmarks.native_codec_prefill_bench import (
    ATOL,
    RTOL,
    _hash_array,
    _hash_file,
    _write,
)
from fragpack.ops.native_codec import native_codec_workspace_bytes
from fragpack.ops.native_codec_pool import NativeCodecPool, native_codec_pool_bytes


def parse_sizes(value: str) -> list[int]:
    """Read distinct positive integer sizes from a comma-separated argument."""
    values = [int(part) for part in value.split(",")]
    if not values or min(values) < 1 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected distinct positive integer sizes")
    return values


def plan_cases(
    contexts: list[int],
    queries: list[int],
    chunks: list[int],
    arms: list[str],
    workspaces: list[str],
    resident_blocks: int | None,
    receive_capacity: str = "format-max",
) -> list[dict[str, Any]]:
    """Expand independent workspace/chunk axes while preserving fixed residency."""
    if (
        not contexts
        or not queries
        or not chunks
        or not arms
        or not workspaces
        or min(contexts + queries + chunks) < 1
        or max(queries) > min(contexts)
        or len(set(arms)) != len(arms)
        or any(arm not in ARMS for arm in arms)
        or len(set(workspaces)) != len(workspaces)
        or any(mode not in ("all", "decode") for mode in workspaces)
        or receive_capacity not in ("format-max", "observed")
        or (
            resident_blocks is not None
            and resident_blocks < max((c + T - 1) // T for c in contexts)
        )
    ):
        raise ValueError(
            "invalid independent size/arm/workspace/residency configuration"
        )
    rows = []
    for context in contexts:
        blocks = resident_blocks or (context + T - 1) // T
        for query in queries:
            for chunk in chunks:
                for arm in arms:
                    for workspace in workspaces if arm == "native_fa3" else ["decode"]:
                        rows.append(
                            {
                                "name": (
                                    f"c{context}-q{query}-n{blocks}-k{chunk}-"
                                    f"{arm}-{workspace}"
                                ),
                                "arm": arm,
                                "resident_blocks": blocks,
                                "contexts": [context],
                                "query_tokens": query,
                                "chunk_blocks": chunk,
                                "native_workspace": workspace,
                                "receive_capacity": receive_capacity,
                            }
                        )
    return rows


def source_identity() -> dict[str, Any]:
    """Record imported evaluator and production source identities with this driver."""
    identity = budget_source_identity()
    identity["files_sha256"]["benchmarks/native_codec_memory_bench.py"] = _hash_file(
        Path(__file__)
    )
    identity["digest"] = hashlib.sha256(
        json.dumps(identity["files_sha256"], sort_keys=True).encode()
    ).hexdigest()
    return identity


def measure_case(
    case: dict[str, Any],
    corpus: Corpus,
    sessions: list[Session],
    queries: torch.Tensor,
    reference: torch.Tensor,
    row: dict[str, Any],
) -> None:
    """Measure complete ownership and cold use, then validate outside peak scope."""
    cleanup_gpu()
    if torch.cuda.memory_allocated() != 0:
        raise RuntimeError("another row or preparation still owns GPU tensors")
    torch.cuda.reset_peak_memory_stats()
    runtime = Runtime(
        case["arm"], case["resident_blocks"], case, corpus, sessions, queries
    )
    after_construction = {
        "current_allocated_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    sample = runtime.step(0)
    after_cold = {
        "current_allocated_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    row.update(
        {
            "construction": after_construction,
            "construction_and_cold_read": after_cold,
            "components": runtime.memory_usage(),
            "cold_probe": sample,
            "cold_probe_scope": (
                "one allocation/use probe, not a latency distribution or "
                "performance comparison"
            ),
            "source_pages_gpu_remaining_before_row": 0,
        }
    )
    # Reset logical residency so validation must actually restore every chunk.
    # This poisons selected destinations and checks every layer against CPU bits.
    runtime.lru = SessionLRU(runtime.blocks)
    validation_sample = runtime.step(0, validate=True)
    actual = runtime.output.cpu()
    torch.testing.assert_close(actual, reference, atol=ATOL, rtol=RTOL)
    exact = torch.equal(actual.view(torch.int16), reference.view(torch.int16))
    if case["arm"] != "fused_hbm" and not exact:
        raise AssertionError("native/gather FA3 differs from source reference bits")
    row["validation"] = {
        "restored_chunks_bitchecked_after_poison": validation_sample["chunks"],
        "layers_checked_per_chunk": L,
        "attention_output_bitwise_equal": exact,
        "max_abs_attention_error": float(
            (actual.float() - reference.float()).abs().max()
        ),
        "atol": ATOL,
        "rtol": RTOL,
        "scope": "validation-only allocations excluded from preceding measured peaks",
    }
    del runtime
    cleanup_gpu()
    row["remaining_allocated_bytes_after_release"] = torch.cuda.memory_allocated()
    if row["remaining_allocated_bytes_after_release"] != 0:
        raise RuntimeError("deployment runtime owners survived release")
    row["status"] = "passed"
    print(
        json.dumps(
            {
                "name": case["name"],
                "current_bytes": after_cold["current_allocated_bytes"],
                "peak_bytes": after_cold["peak_allocated_bytes"],
                "codec_bytes": sum(row["components"]["codec"].values()),
                "receive_bytes": row["components"]["receive"],
            }
        ),
        flush=True,
    )


def measure_encode_pool(corpus: Corpus, chunk: int, slots: int) -> dict[str, Any]:
    """Measure an actual standalone encode pool, never alongside reader ownership."""
    cleanup_gpu()
    if torch.cuda.memory_allocated() != 0:
        raise RuntimeError(
            "reader tensors remain before separate encode-pool measurement"
        )
    torch.cuda.reset_peak_memory_stats()
    pool = NativeCodecPool(
        corpus.table, max_blocks=chunk, slots=slots, workspace="encode", device="cuda"
    )
    torch.cuda.current_stream().synchronize()
    result = {
        "chunk_blocks": chunk,
        "slots": slots,
        "workspace": "encode",
        "components": pool.memory_usage(),
        "current_allocated_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "scope": (
            "standalone encode pool allocation; no native KV or encoding "
            "operation; not co-resident with any restore reader row"
        ),
    }
    expected = native_codec_pool_bytes(L, H, chunk, slots, workspace="encode")
    if result["components"] != expected:
        raise AssertionError(
            "public encode pool allocation differs from byte estimator"
        )
    del pool
    cleanup_gpu()
    result["remaining_allocated_bytes_after_release"] = torch.cuda.memory_allocated()
    return result


def main() -> None:
    """Run fixed-capacity ownership measurements on one exclusively selected GPU."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", type=parse_sizes, default=[32768])
    parser.add_argument("--queries", type=parse_sizes, default=[1, 128])
    parser.add_argument("--chunks", type=parse_sizes, default=[1, 4, 16, 256])
    parser.add_argument("--resident-blocks", type=int)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--native-workspaces", default="all,decode")
    parser.add_argument(
        "--receive-capacity", choices=("format-max", "observed"), default="format-max"
    )
    parser.add_argument("--measure-encode-pool", action="store_true")
    parser.add_argument("--encode-pool-slots", type=int, default=1)
    parser.add_argument("--calib-pages", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=8)
    parser.add_argument("--eval-pages", type=int, default=24)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--input-description", required=True)
    args = parser.parse_args()
    args.arms = args.arms.split(",")
    args.native_workspaces = args.native_workspaces.split(",")
    if min(args.calib_pages, args.eval_pages, args.encode_pool_slots) < 1:
        raise ValueError("positive calibration/evaluation and pool sizes required")
    cases = plan_cases(
        args.contexts,
        args.queries,
        args.chunks,
        args.arms,
        args.native_workspaces,
        args.resident_blocks,
        args.receive_capacity,
    )
    report: dict[str, Any] = {
        "schema": "fragpack.native-codec-deployment-memory.v1",
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": source_identity(),
        "groups": [],
        "encode_pool_measurements": [],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "vllm": importlib.metadata.version("vllm"),
        },
        "protocol": {
            "axes": (
                "fixed resident physical blocks N; independent restore chunk K "
                "and native workspace all/decode axes; raw always lean raw-only "
                "workspace"
            ),
            "memory": (
                "actual absolute torch tensor allocations, one complete reader at "
                "a time, including codec/receive/KV/Q/output/attention scratch "
                "and captured graph buffers; CUDA context outside torch and "
                "allocator reserved are not available KV"
            ),
            "peak": (
                "construction plus actual bounded-chunk cold restore and48layer "
                "read; separate current/peak and reserved/peakreserved; source "
                "preparation GPU tensors must be freed before each row"
            ),
            "validation": (
                "after measured peak, reset residency, poison every restored "
                "chunk, compare every layer KV bits and attention output with "
                "independent native FA3 reference from CPU source bits"
            ),
            "receive": (
                "format-max default reserves safe FPKB capacity including raw "
                "fallback; observed explicit finite-corpus ablation is not "
                "general deployment capacity"
            ),
            "pool": (
                "reader uses one fixed-capacity codec plus receive set on one "
                "serialized stream; separate public encode-pool allocation can be "
                "measured but is never silently counted as co-resident"
            ),
            "scope": (
                "deployment tensor-memory measurement and functional read replay; "
                "no full model, connector service, cache policy performance, TTFT "
                "or throughput"
            ),
        },
        "legacy_raw_codec_estimates": [
            {
                "chunk_blocks": chunk,
                "full_workspace_components": native_codec_workspace_bytes(
                    L, H, chunk, workspace="all"
                ),
                "lean_raw_components": native_codec_workspace_bytes(
                    L, H, chunk, workspace="raw"
                ),
                "scope": (
                    "CPU tensor-size estimator only; not an actual legacy deployment "
                    "measurement; excludes native pages, transfer/metadata and "
                    "attention allocations"
                ),
            }
            for chunk in args.chunks
        ],
        "encode_pool_estimates": [
            {
                "chunk_blocks": chunk,
                "slots": args.encode_pool_slots,
                "components": native_codec_pool_bytes(
                    L, H, chunk, args.encode_pool_slots, workspace="encode"
                ),
                "scope": (
                    "separate optional save-side pool; not co-resident with measured "
                    "restore readers"
                ),
            }
            for chunk in args.chunks
        ],
    }
    _write(args.out, report)
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
        corpus = prepare_corpus(args)
        cleanup_gpu()
        if torch.cuda.memory_allocated() != 0:
            raise RuntimeError("archive preparation retained GPU tensors")
        report["input"] = corpus.provenance
        report["status"] = "running"
        for context in args.contexts:
            sessions = prepare_sessions(corpus, [context])
            for query in args.queries:
                generator = torch.Generator().manual_seed(args.seed)
                queries = torch.randn(L, query, HQ, D, generator=generator).to(
                    torch.bfloat16
                )
                references = prepare_reference(corpus, sessions, queries)
                cleanup_gpu()
                if torch.cuda.memory_allocated() != 0:
                    raise RuntimeError("reference preparation retained GPU tensors")
                group: dict[str, Any] = {
                    "context": context,
                    "query_tokens": query,
                    "query_sha256": _hash_array(queries.view(torch.int16).numpy()),
                    "reference_sha256": _hash_array(
                        references[0].view(torch.int16).numpy()
                    ),
                    "capture_pool_indices": sessions[0].indices.tolist(),
                    "host_raw_archive_bytes": sessions[0].raw.numel(),
                    "host_compressed_archive_bytes": sessions[0].compressed.numel(),
                    "raw_archive_sha256": _hash_array(sessions[0].raw.numpy()),
                    "compressed_archive_sha256": _hash_array(
                        sessions[0].compressed.numpy()
                    ),
                    "rows": [],
                }
                report["groups"].append(group)
                _write(args.out, report)
                for case in cases:
                    if case["contexts"][0] != context or case["query_tokens"] != query:
                        continue
                    row = {"case": case, "status": "preparing"}
                    group["rows"].append(row)
                    _write(args.out, report)
                    measure_case(case, corpus, sessions, queries, references[0], row)
                    _write(args.out, report)
        if args.measure_encode_pool:
            for chunk in args.chunks:
                report["encode_pool_measurements"].append(
                    measure_encode_pool(corpus, chunk, args.encode_pool_slots)
                )
                _write(args.out, report)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write(args.out, report)


if __name__ == "__main__":
    main()

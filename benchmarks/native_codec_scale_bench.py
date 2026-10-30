# SPDX-License-Identifier: Apache-2.0
"""Scale the native codec roundtrip with explicitly synthetic capture replication.

This driver reuses the original benchmark's timing and validation paths. Physical
pages are distinct even when their BF16 content repeats. Forced-fallback inputs
replace whole records with finite values whose high byte is outside that record's
calibrated table. They measure failure-to-compress cost, not model accuracy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import traceback
from typing import Any

import numpy as np
import torch

from benchmarks.native_codec_bench import (
    Inputs,
)
from benchmarks.native_codec_bench import (
    _file_hash as file_hash,
)
from benchmarks.native_codec_bench import (
    _parser as base_parser,
)
from benchmarks.native_codec_bench import (
    _run as run_roundtrip,
)
from benchmarks.native_codec_bench import (
    _source_identity as base_source_identity,
)
from benchmarks.native_codec_bench import (
    _write as write_report,
)
from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import calibrate


def array_hash(values: np.ndarray) -> str:
    """Hash contiguous input bytes without materializing a second bytes copy."""
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(values)).cast("B")
    ).hexdigest()


def expand_capture(
    pool: np.ndarray, table: np.ndarray, blocks: int, pattern: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replicate a CPU BF16 pool and optionally force complete records to raw.

    Inputs have the capture layout ``[N,L,2,128,H,128]`` and the calibrated
    table layout ``[L,2,H,22]``. The source pool remains unchanged. All modified
    BF16 values are finite. This CPU-only preparation does not initialize CUDA.
    """
    if pattern not in {"capture-repeat", "fallback25", "fallback100"}:
        raise ValueError("unknown source pattern")
    if blocks < 1 or pool.shape[0] < 1 or pool.dtype != np.uint16:
        raise ValueError("expected positive block count and nonempty uint16 pool")
    if pool.shape[1:] != (
        table.shape[0],
        2,
        ap.PAGE_TOKENS,
        table.shape[2],
        ap.HEAD_DIM,
    ):
        raise ValueError("capture geometry does not match the calibrated tables")
    output = np.empty((blocks, *pool.shape[1:]), dtype=np.uint16)
    for block in range(blocks):
        output[block] = pool[block % len(pool)]
    absent = np.empty(table.shape[:3], dtype=np.uint16)
    for index in np.ndindex(table.shape[:3]):
        # Excluding these high bytes avoids exponent=255 for every low byte.
        absent[index] = next(
            high
            for high in range(256)
            if high not in {0x7F, 0xFF} and high not in table[index]
        )
    forced = 0
    if pattern != "capture-repeat":
        for block in range(blocks):
            for layer, kv, head in np.ndindex(table.shape[:3]):
                if pattern == "fallback25" and (block + layer + kv + head) % 4:
                    continue
                record = output[block, layer, kv, :, head, :]
                np.bitwise_and(record, np.uint16(0xFF), out=record)
                np.bitwise_or(record, absent[layer, kv, head] << 8, out=record)
                forced += 1
    count = blocks * int(np.prod(table.shape[:3]))
    return output, {
        "pattern": pattern,
        "synthetic_blocks": blocks,
        "source_pool_blocks": len(pool),
        "source_pool_index_per_synthetic_block": [i % len(pool) for i in range(blocks)],
        "forced_raw_records": forced,
        "record_count": count,
        "forced_raw_fraction": forced / count,
        "selection": "all records"
        if pattern == "fallback100"
        else (
            "(block + layer + kv + head) % 4 == 0"
            if pattern == "fallback25"
            else "none"
        ),
        "fallback_scope": (
            "Whole-record finite out-of-table high bytes; low bytes retained."
        ),
        "scale_scope": (
            "Distinct physical pages with repeated captured content; not a real "
            "long-context forward or additional independent model captures."
        ),
    }


def load_inputs(args: argparse.Namespace) -> Inputs:
    """Load disjoint calibration/source positions and expand the source on CPU."""
    dump = np.load(args.dump, mmap_mode="r", allow_pickle=False)
    if (
        dump.dtype != np.uint16
        or dump.ndim != 6
        or dump.shape[2] != 2
        or dump.shape[3] != ap.PAGE_TOKENS
        or dump.shape[5] != ap.HEAD_DIM
    ):
        raise ValueError("dump must be uint16 [blocks,layers,2,128,heads,128]")
    if not 0 < args.calib_pages <= args.eval_start:
        raise ValueError("calibration must precede the source evaluation pool")
    if args.eval_count < 1 or args.eval_start + args.eval_count > len(dump):
        raise ValueError("source evaluation pool lies outside the capture")
    calibration = np.asarray(dump[: args.calib_pages])
    pool = np.asarray(dump[args.eval_start : args.eval_start + args.eval_count])
    table = calibrate(calibration)
    evaluation, construction = expand_capture(
        pool, table, max(args.blocks), args.source_pattern
    )
    return Inputs(
        table,
        evaluation,
        {
            "dump": str(args.dump.resolve()),
            "dump_basename": args.dump.name,
            "dump_sha256": file_hash(args.dump),
            "dump_shape": list(dump.shape),
            "dump_dtype": str(dump.dtype),
            "calibration_range": [0, args.calib_pages],
            "calibration_sha256": array_hash(calibration),
            "source_pool_range": [args.eval_start, args.eval_start + args.eval_count],
            "source_pool_sha256": array_hash(pool),
            "evaluation_sha256": array_hash(evaluation),
            "table_sha256": array_hash(table),
            "construction": construction,
            "case_evaluation_index_scope": (
                "Case evaluation_block_indices refer to synthetic block positions, "
                "not positions in the original capture. Original position = "
                "source_pool_range[0] + synthetic_index % source_pool_blocks."
            ),
            "input_description": args.input_description,
            "split_scope": (
                "Disjoint positions in one capture; not independent documents."
            ),
        },
    )


def main() -> None:
    """Prepare or measure on the caller's explicitly reserved CUDA device."""
    parser = base_parser()
    parser.description = __doc__.split("\n\n")[0]
    parser.set_defaults(blocks=[1, 16, 64, 256])
    parser.add_argument(
        "--source-pattern",
        choices=["capture-repeat", "fallback25", "fallback100"],
        default="capture-repeat",
    )
    args = parser.parse_args()
    if args.rounds < 2 or args.warmup < 1:
        raise ValueError("rounds must be >=2 and warmup >=1")
    source = base_source_identity()
    root = Path(__file__).resolve().parents[1]
    source["files_sha256"][str(Path(__file__).resolve().relative_to(root))] = file_hash(
        Path(__file__)
    )
    source["source_digest"] = hashlib.sha256(
        json.dumps(source["files_sha256"], sort_keys=True).encode()
    ).hexdigest()
    report: dict[str, Any] = {
        "schema": "fragpack.native-codec-scale-roundtrip.v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preparing",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": source,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "cases": [],
    }
    write_report(args.out, report)
    try:
        inputs = load_inputs(args)
        report["inputs"] = inputs.provenance
        if args.prepare_only:
            tables_path = args.out.with_suffix(".tables.npz")
            np.savez(tables_path, table=inputs.table)
            report["tables_file"] = str(tables_path.resolve())
            report["status"] = "inputs_prepared_cpu_only"
        else:
            run_args = argparse.Namespace(**vars(args))
            run_args.eval_start = 0  # Existing evaluator indexes the synthetic pool.
            run_args.eval_count = len(inputs.evaluation)
            run_roundtrip(run_args, inputs, report)
            if args.source_pattern == "fallback25":
                for row in report["cases"]:
                    forced = sum(
                        (block + layer + kv + head) % 4 == 0
                        for block, layer, kv, head in np.ndindex(
                            row["blocks"], *inputs.table.shape[:3]
                        )
                    )
                    row["format"]["forced_raw_records_minimum"] = forced
                    if row["format"]["raw_fallback_records"] < forced:
                        raise AssertionError(
                            "fallback25 did not preserve every forced raw record"
                        )
            if args.source_pattern == "fallback100":
                if any(
                    row["format"]["raw_fallback_fraction"] != 1.0
                    for row in report["cases"]
                ):
                    raise AssertionError(
                        "fallback100 did not produce entirely raw records"
                    )
            report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(args.out, report)


if __name__ == "__main__":
    main()

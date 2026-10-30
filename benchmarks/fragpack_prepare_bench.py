# SPDX-License-Identifier: Apache-2.0
"""Measure fresh staged/direct prepare calls with realistic store metadata.

Every invocation uses a new host step stamp and runs prepare then commit.
This prepare-only workload neither writes nor encodes KV. Public host-submit
time and CUDA-event stream span are reported separately; neither is presented
as kernel-only time. Profile with Nsight Systems --trace=cuda,nvtx and match
prepare_kernel to the per-sample NVTX ranges for actual kernel durations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import numpy as np
import torch

from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import device_tables
from fragpack.logging import init_logger
from fragpack.ops.hbm_kv import HbmKvStore, page_bytes

logger = init_logger(__name__)
T = ap.PAGE_TOKENS
ARMS = ("staged", "direct")
PATTERNS = ("aligned", "ragged", "permuted")


def _ints(value: str) -> list[int]:
    items = [int(part) for part in value.split(",")]
    if not items or min(items) <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return items


def _patterns(value: str) -> list[str]:
    items = value.split(",")
    if not items or any(item not in PATTERNS for item in items):
        raise argparse.ArgumentTypeError(f"patterns must be from {PATTERNS}")
    return items


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "p10_us": float(np.percentile(samples, 10)),
        "p90_us": float(np.percentile(samples, 90)),
    }


def _persist(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _slots(
    tokens: int, pattern: str
) -> tuple[np.ndarray, dict[int, int], list[int], list[int]]:
    """Return slots, direct source rows, closed blocks and open tails."""
    lengths = [tokens]
    if pattern == "ragged":
        if tokens < 128:
            raise ValueError("ragged benchmark requires at least 128 tokens")
        quarter = tokens // 4
        lengths = [quarter - 31, quarter + 1, quarter + 17]
        lengths.append(tokens - sum(lengths))
    result: list[int] = []
    direct: dict[int, int] = {}
    closed: list[int] = []
    tails: list[int] = []
    block = 0
    for length in lengths:
        row = len(result)
        result.extend(range(block * T, block * T + length))
        for offset in range(length // T):
            closed.append(block + offset)
            direct[block + offset] = row + offset * T
        if length % T:
            tails.append(block + length // T)
        block += math.ceil(length / T)
    slots = np.array(result, dtype=np.int64)
    if pattern == "permuted":
        # Preserve every input token and both endpoints; break only the
        # interior order, so a full 128-slot validation must reject each block.
        for row in direct.values():
            slots[row + 31], slots[row + 32] = slots[row + 32], slots[row + 31]
        direct = {}
    return slots, direct, closed, tails


class _Arm:
    def __init__(
        self, mode: str, layers: int, slots: np.ndarray, args: argparse.Namespace
    ) -> None:
        self.mode = mode
        self.arenas = [
            torch.empty(
                (args.blocks, page_bytes(args.heads)), dtype=torch.uint8, device="cuda"
            )
            for _ in range(layers)
        ]
        # Prepare never decodes values, so table contents do not enter this
        # measurement; real layer/head table shapes and allocations are retained.
        table = np.broadcast_to(
            np.arange(ap.TABLE_ENTRIES, dtype=np.uint8),
            (layers, 2, args.heads, ap.TABLE_ENTRIES),
        ).copy()
        self.store = HbmKvStore(
            self.arenas,
            args.heads,
            device_tables(table, torch.device("cuda")),
            max_tokens=len(slots),
            staging_slots=args.max_seqs,
            overflow_records=0,
            direct_new_blocks=mode == "direct",
        )
        self.slots = torch.from_numpy(slots.copy()).cuda()
        self.empty_live = np.empty(0, dtype=np.int32)
        count = args.rounds * args.inner
        i32 = {"dtype": torch.int32, "device": "cuda"}
        self.maps = torch.empty((count, args.blocks), **i32)
        self.counters = torch.empty((count, self.store.counters.numel()), **i32)
        self.committed_counters = torch.empty_like(self.counters)
        self.committed_slots = torch.empty_like(self.maps)
        self.committed_maps = torch.empty_like(self.maps)

    def prepare(self) -> None:
        # Empty live-set makes tails from the previous simulated step eligible
        # for GC. No unchanged step stamp or replayed captured prepare is used.
        self.store.prepare(self.slots, self.slots.numel(), self.empty_live)

    def validate(
        self,
        direct: dict[int, int],
        closed: list[int],
        tails: list[int],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """Inspect all samples, outside timing, to reject no-op or leaking steps."""
        expected = np.full(args.blocks, -1, dtype=np.int32)
        if self.mode == "direct":
            for block, row in direct.items():
                expected[block] = row
        maps = self.maps.cpu().numpy()
        np.testing.assert_array_equal(maps, np.broadcast_to(expected, maps.shape))
        before = self.counters.cpu().numpy()
        after = self.committed_counters.cpu().numpy()
        slots = self.committed_slots.cpu().numpy()
        # Native counter ABI: free-stack depth, claims, closes. Capturing the
        # whole counter vector makes every invocation auditable in the JSON.
        assert np.all(before[:, 2] == len(closed) + len(tails))
        assert np.all(before[:, 3] == len(closed))
        assert np.all(after[:, 0] == args.max_seqs - len(tails))
        assert np.all(after[:, 3] == 0)
        assert np.all(self.committed_maps.cpu().numpy() == -1)
        inactive = np.ones(args.blocks, dtype=bool)
        inactive[tails] = False
        assert np.all(slots[:, inactive] == -1)
        for sample in slots[:, tails]:
            assert np.all((0 <= sample) & (sample < args.max_seqs))
            assert len(set(sample.tolist())) == len(tails)
        stats = self.store.stats()
        assert stats.encoded == stats.fallbacks == 0
        return {
            "samples_checked": len(maps),
            "all_direct_maps_match": True,
            "all_claim_close_counts_match": True,
            "all_commits_clear_direct_maps": True,
            "all_persistent_slots_accounted": True,
            "counters_after_prepare": before.tolist(),
            "counters_after_commit": after.tolist(),
            "stats": asdict(stats),
        }


def _case(
    layers: int, tokens: int, pattern: str, args: argparse.Namespace
) -> dict[str, Any]:
    slots, direct, closed, tails = _slots(tokens, pattern)
    if len(tails) > args.max_seqs:
        raise ValueError("max-seqs must cover the pattern's open tails")
    if int(slots.max()) // T >= args.blocks:
        raise ValueError("blocks is too small for the selected input")
    arms = [_Arm(mode, layers, slots, args) for mode in ARMS]
    case_id = f"L={layers}|N={tokens}|NB={args.blocks}|pattern={pattern}"
    host_samples: dict[str, list[float]] = {mode: [] for mode in ARMS}
    stream_samples: dict[str, list[float]] = {mode: [] for mode in ARMS}
    labels: dict[str, list[str]] = {mode: [] for mode in ARMS}
    # Initialize the extension before opening measured NVTX ranges.
    for arm in arms:
        for _ in range(args.warmup):
            with torch.cuda.nvtx.range(
                f"fragpack.prepare|{case_id}|mode={arm.mode}|phase=warmup"
            ):
                arm.prepare()
            arm.store.commit()
    torch.cuda.synchronize()
    orders = []
    for repeat in range(args.rounds):
        order = list(range(len(arms)))
        if repeat % 2:
            order.reverse()
        orders.append([ARMS[index] for index in order])
        for index in order:
            arm = arms[index]
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(args.inner)
            ]
            for inner, (start, stop) in enumerate(events):
                sample = repeat * args.inner + inner
                label = (
                    f"fragpack.prepare|{case_id}|mode={arm.mode}|phase=measure|"
                    f"round={repeat}|sample={inner}"
                )
                with torch.cuda.nvtx.range(label):
                    start.record()
                    host_start = time.perf_counter_ns()
                    arm.prepare()
                    host_samples[arm.mode].append(
                        (time.perf_counter_ns() - host_start) / 1000
                    )
                    stop.record()
                labels[arm.mode].append(label)
                # Snapshot every invocation asynchronously, outside the prepare
                # window. This catches stale claim stamps and skipped work.
                arm.maps[sample].copy_(arm.store.direct_rows)
                arm.counters[sample].copy_(arm.store.counters)
                arm.store.commit()
                arm.committed_maps[sample].copy_(arm.store.direct_rows)
                arm.committed_slots[sample].copy_(arm.store.wslot)
                arm.committed_counters[sample].copy_(arm.store.counters)
            events[-1][1].synchronize()
            stream_samples[arm.mode].extend(
                start.elapsed_time(stop) * 1000 for start, stop in events
            )
    validation = {arm.mode: arm.validate(direct, closed, tails, args) for arm in arms}
    return {
        "case_id": case_id,
        "layers": layers,
        "tokens": tokens,
        "blocks": args.blocks,
        "pattern": pattern,
        "closed_blocks": closed,
        "open_tail_blocks": tails,
        "expected_direct_rows": direct,
        "slot_mapping_sha256": hashlib.sha256(slots.tobytes()).hexdigest(),
        "public_host_submit": {mode: _summary(x) for mode, x in host_samples.items()},
        "public_cuda_event_span": {
            mode: _summary(x) for mode, x in stream_samples.items()
        },
        "native_kernel_duration": None,
        "native_kernel_source": "Nsight CUDA kernel timestamps correlated with NVTX",
        "nvtx_ranges": labels,
        "round_order": orders,
        "validation": validation,
        "allocated_store_bytes": {arm.mode: arm.store.memory_usage() for arm in arms},
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=_ints, default=[128, 1024, 4096])
    parser.add_argument("--layers", type=_ints, default=[1, 48])
    parser.add_argument("--blocks", type=int, default=469)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-seqs", type=int, default=4)
    parser.add_argument("--patterns", type=_patterns, default=list(PATTERNS))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cuda_graph:
        parser.error("prepare has host step stamps; use ordinary launches and Nsight")
    if min(args.blocks, args.heads, args.max_seqs, args.inner, args.warmup) < 1:
        parser.error("geometry, inner and warmup must be positive")
    if args.rounds < 2:
        parser.error("at least two alternating rounds are required")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("this benchmark requires an exclusively allocated SM90 GPU")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "status": "running",
        "scope": "prepare-only metadata workload; no KV writes, encoding or inference",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "fragpack/ops/hbm_kv.cu",
                "fragpack/ops/hbm_kv.py",
                "benchmarks/fragpack_prepare_bench.py",
            )
        },
        "geometry": {
            "heads": args.heads,
            "head_dim": 128,
            "max_seqs": args.max_seqs,
            "default_blocks_basis": "floor(4294967296 / (48 * page_bytes(4))) = 469",
            "default_model_geometry": "Qwen3-30B-A3B: 48 layers, 4 KV and 32 Q heads",
        },
        "gpu": str(props),
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rounds": args.rounds,
        "inner": args.inner,
        "warmup_calls_per_arm": args.warmup,
        "timing_contract": (
            "Host perf_counter measures public prepare submission. CUDA events "
            "measure the stream span around that call, including possible host "
            "gaps; these are not pure kernel timings. Actual prepare_kernel "
            "durations require Nsight Systems --trace=cuda,nvtx and NVTX/CUDA "
            "correlation. Each invocation increments the public host step stamp, "
            "then commits; per-sample state snapshots occur outside timing. "
            "All claims are fresh; old simulated tails are reclaimed with an "
            "empty live-set. Arena/staging/metadata dimensions are real, but no "
            "KV payload is written, encoded or read. This is not serving E2E."
        ),
        "cases": [],
    }
    _persist(args.out, report)
    try:
        for layers in args.layers:
            for tokens in args.tokens:
                for pattern in args.patterns:
                    case = _case(layers, tokens, pattern, args)
                    report["cases"].append(case)
                    _persist(args.out, report)
                    logger.info(
                        "%s public event staged=%.2f us direct=%.2f us",
                        case["case_id"],
                        case["public_cuda_event_span"]["staged"]["median_us"],
                        case["public_cuda_event_span"]["direct"]["median_us"],
                    )
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _persist(args.out, report)


if __name__ == "__main__":
    _main()

# SPDX-License-Identifier: Apache-2.0
"""Compare staged and direct complete-new-block writes on one allocated GPU.

Two single-layer cold-prefill windows use identical synthetic BF16 K/V:
write + encode, and write + gather + FA3 + encode. Prepare and commit execute
once for every sample but outside the reported layer window. Prepare is timed
separately; an additional total-step window includes prepare, layer work and
commit, along with host submission gaps and the intervening measurement events.
CUDA Graph mode captures only one layer invocation; each replay is
surrounded by a fresh ordinary prepare/commit to reset page allocation and
step stamps. This is not serving TTFT, throughput, or total GPU capacity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
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
from fragpack.ops.hbm_kv import HbmKvStore, page_bytes

logger = init_logger(__name__)
T, D = ap.PAGE_TOKENS, ap.HEAD_DIM
ARMS = ("staged", "direct")
WINDOWS = ("write_encode", "write_gather_fa3_encode")


def _ints(value: str) -> list[int]:
    sizes = [int(part) for part in value.split(",")]
    if not sizes or min(sizes) <= 0:
        raise argparse.ArgumentTypeError("sizes must be positive integers")
    return sizes


def _summary(samples: list[float]) -> dict[str, Any]:
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


def _persist(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class _Arm:
    """Own a store, layer buffers, and an optional captured layer invocation."""

    def __init__(
        self,
        direct: bool,
        table: np.ndarray,
        key: torch.Tensor,
        value: torch.Tensor,
        query: torch.Tensor,
        window: str,
    ) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        self.key, self.value = key, value
        self.query, self.window = query, window
        self.attention = flash_attn_varlen_func
        self.tokens, self.heads = key.shape[:2]
        self.blocks = math.ceil(self.tokens / T)
        self.closed = self.tokens // T
        self.arena = torch.empty(
            (self.blocks, page_bytes(self.heads)), dtype=torch.uint8, device="cuda"
        )
        self.store = HbmKvStore(
            [self.arena],
            self.heads,
            device_tables(table, torch.device("cuda")),
            max_tokens=self.tokens,
            staging_slots=1,
            overflow_records=self.blocks * 2 * self.heads,
            direct_new_blocks=direct,
        )
        self.slots = torch.arange(self.tokens, dtype=torch.int64, device="cuda")
        self.pages = torch.arange(self.blocks, dtype=torch.int32, device="cuda")
        self.block_table = self.pages[None]
        self.empty_live = np.empty(0, dtype=np.int32)
        self.kwargs = {"key": key, "value": value} if direct else {}
        self.wk = torch.empty(
            (self.blocks, T, self.heads, D), dtype=torch.bfloat16, device="cuda"
        )
        self.wv = torch.empty_like(self.wk)
        self.output = torch.empty_like(query)
        self.offsets = torch.tensor([0, self.tokens], dtype=torch.int32, device="cuda")
        self.lengths = torch.tensor([self.tokens], dtype=torch.int32, device="cuda")
        self.graph: torch.cuda.CUDAGraph | None = None

    def layer(self) -> None:
        """Submit one layer without retaining a closure cycle around the graph."""
        self.store.write(0, self.key, self.value, self.slots)
        if self.window == "write_gather_fa3_encode":
            self.store.gather(0, self.pages, self.wk, self.wv, **self.kwargs)
            self.attention(
                q=self.query,
                k=self.wk,
                v=self.wv,
                out=self.output,
                cu_seqlens_q=self.offsets,
                max_seqlen_q=self.tokens,
                seqused_k=self.lengths,
                max_seqlen_k=self.tokens,
                softmax_scale=D**-0.5,
                causal=True,
                block_table=self.block_table,
                fa_version=3,
            )
        self.store.encode(0, self.closed, **self.kwargs)

    def prepare(self) -> None:
        self.store.prepare(self.slots, self.tokens, self.empty_live)

    def run(self) -> None:
        if self.graph is None:
            self.layer()
        else:
            self.graph.replay()

    def step(self) -> None:
        self.prepare()
        self.run()
        self.store.commit()

    def capture(self) -> None:
        """Capture one layer only; host step stamps stay outside the graph."""
        # A previous case's graph must never be destroyed by cyclic GC while
        # this stream is capturing. Case cleanup also resets every owned graph.
        gc.collect()
        self.prepare()
        graph = torch.cuda.CUDAGraph()
        restore_gc = gc.isenabled()
        gc.disable()
        try:
            with torch.cuda.graph(graph):
                self.layer()
        finally:
            if restore_gc:
                gc.enable()
        self.store.commit()
        self.graph = graph

    def close(self) -> None:
        """Release captured GPU resources before any subsequent case captures."""
        if self.graph is not None:
            self.graph.reset()
            self.graph = None

    def validate(self, expected: torch.Tensor | None) -> dict[str, Any]:
        """Validate current graph/eager work and reconstructed KV outside timing."""
        self.output.fill_(float("nan"))
        self.step()
        error = None
        if expected is not None:
            torch.testing.assert_close(self.output, expected, atol=2e-2, rtol=2e-2)
            error = float((self.output.float() - expected.float()).abs().max())
        self.store.gather(0, self.pages, self.wk, self.wv)
        for workspace, original in ((self.wk, self.key), (self.wv, self.value)):
            reconstructed = workspace.reshape(-1, self.heads, D)[: self.tokens]
            assert torch.equal(
                reconstructed.view(torch.int16), original.view(torch.int16)
            )
        stats = self.store.stats()
        if stats.fallbacks:
            raise RuntimeError("normal-input benchmark unexpectedly used raw overflow")
        return {
            "kv_bitwise_equal": True,
            "attention_max_abs_error": error,
            "store_stats": asdict(stats),
        }


def _measure(
    arms: list[_Arm], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[list[str]]]:
    """Time layer, prepare and the submitted step in alternating arm order."""
    for arm in arms:
        elapsed = 0.0
        while elapsed < args.warmup_ms:
            start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            arm.prepare()
            start.record()
            arm.run()
            stop.record()
            arm.store.commit()
            stop.synchronize()
            elapsed += start.elapsed_time(stop)
    layer_samples: dict[str, list[float]] = {name: [] for name in ARMS}
    prepare_samples: dict[str, list[float]] = {name: [] for name in ARMS}
    total_samples: dict[str, list[float]] = {name: [] for name in ARMS}
    orders = []
    for repeat in range(args.rounds):
        order = list(range(len(arms)))
        if repeat % 2:
            order.reverse()
        orders.append([ARMS[index] for index in order])
        for index in order:
            events = [
                tuple(torch.cuda.Event(enable_timing=True) for _ in range(5))
                for _ in range(args.inner)
            ]
            arm = arms[index]
            for prep_start, prep_end, start, stop, step_end in events:
                prep_start.record()
                arm.prepare()
                prep_end.record()
                start.record()
                arm.run()
                stop.record()
                arm.store.commit()
                step_end.record()
            events[-1][-1].synchronize()
            layer_samples[ARMS[index]].extend(
                start.elapsed_time(stop) * 1000 for _, _, start, stop, _ in events
            )
            prepare_samples[ARMS[index]].extend(
                start.elapsed_time(stop) * 1000 for start, stop, _, _, _ in events
            )
            total_samples[ARMS[index]].extend(
                start.elapsed_time(stop) * 1000 for start, _, _, _, stop in events
            )
    return (
        {name: _summary(values) for name, values in layer_samples.items()},
        {name: _summary(values) for name, values in prepare_samples.items()},
        {name: _summary(values) for name, values in total_samples.items()},
        orders,
    )


def _case(tokens: int, window: str, args: argparse.Namespace) -> dict[str, Any]:
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    generator = torch.Generator().manual_seed(args.seed + tokens)
    calibration = torch.randn(16, 1, 2, T, args.heads, D, generator=generator) * 0.5
    calibration_bits = (
        calibration.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    )
    table = calibrate(calibration_bits)
    source_cpu = (torch.randn(2, tokens, args.heads, D, generator=generator) * 0.5).to(
        torch.bfloat16
    )
    source_hash = hashlib.sha256(
        source_cpu.view(torch.int16).numpy().tobytes()
    ).hexdigest()
    source = source_cpu.cuda()
    # Model-like QKV stride: token rows include Q as well as K/V heads.
    qkv = torch.empty(
        (tokens, args.heads * (args.group + 2), D), dtype=torch.bfloat16, device="cuda"
    )
    query = qkv[:, : args.heads * args.group]
    query.copy_(
        torch.randn(tokens, args.heads * args.group, D, generator=generator).to(
            device="cuda", dtype=torch.bfloat16
        )
    )
    key = qkv[:, args.heads * args.group : args.heads * (args.group + 1)]
    value = qkv[:, args.heads * (args.group + 1) :]
    key.copy_(source[0])
    value.copy_(source[1])
    arms = [_Arm(direct, table, key, value, query, window) for direct in (False, True)]
    try:
        expected = None
        if window == "write_gather_fa3_encode":
            padded = torch.zeros(
                (2, arms[0].blocks * T, args.heads, D),
                dtype=torch.bfloat16,
                device="cuda",
            )
            padded[:, :tokens].copy_(source)
            expected = torch.empty_like(query)
            flash_attn_varlen_func(
                q=query,
                k=padded[0].view(arms[0].blocks, T, args.heads, D),
                v=padded[1].view(arms[0].blocks, T, args.heads, D),
                out=expected,
                cu_seqlens_q=arms[0].offsets,
                max_seqlen_q=tokens,
                seqused_k=arms[0].lengths,
                max_seqlen_k=tokens,
                softmax_scale=D**-0.5,
                causal=True,
                block_table=arms[0].block_table,
                fa_version=3,
            )
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for arm in arms:
                for _ in range(3):
                    arm.step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        if args.cuda_graph:
            for arm in arms:
                arm.capture()
        before = {
            name: arm.validate(expected) for name, arm in zip(ARMS, arms, strict=True)
        }
        timing, prepare, total_step, orders = _measure(arms, args)
        after = {
            name: arm.validate(expected) for name, arm in zip(ARMS, arms, strict=True)
        }
        required_encodes = (
            (args.rounds * args.inner + 1) * (tokens // T) * 2 * args.heads
        )
        encoded_records_added = {
            name: after[name]["store_stats"]["encoded"]
            - before[name]["store_stats"]["encoded"]
            for name in ARMS
        }
        for name, count in encoded_records_added.items():
            assert count >= required_encodes, (
                f"{name} encoded {count} records; expected at least {required_encodes} "
                "from measured invocations and the final validation"
            )
        bytes_per_block = 2 * T * args.heads * D * 2
        return {
            "tokens": tokens,
            "complete_new_blocks": tokens // T,
            "tail_tokens": tokens % T,
            "window": window,
            "input_sha256": source_hash,
            "table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
            "kv_input_shape": list(key.shape),
            "kv_input_stride": list(key.stride()),
            "logical_scratch_bytes_not_written": (tokens // T) * bytes_per_block,
            "timing": timing,
            "prepare_timing": prepare,
            "total_step_timing": total_step,
            "encoded_records_added": encoded_records_added,
            "minimum_measured_and_postcheck_records": required_encodes,
            "round_order": orders,
            "validation_before": before,
            "validation_after": after,
            "allocated_bytes": {
                name: {
                    **arm.store.memory_usage(),
                    "prefill_workspace": 2 * arm.wk.numel() * arm.wk.element_size(),
                }
                for name, arm in zip(ARMS, arms, strict=True)
            },
            "direct_over_staged": timing["direct"]["median_us"]
            / timing["staged"]["median_us"],
        }
    finally:
        for arm in arms:
            arm.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=_ints, default=[128, 1024, 4096])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--group", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--warmup-ms", type=float, default=20.0)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.heads < 1 or args.rounds < 2 or args.inner < 1 or args.warmup_ms <= 0:
        parser.error("heads/inner/warmup must be positive; rounds must be at least 2")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("this benchmark requires an exclusively allocated SM90 GPU")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "status": "running",
        "scope": "single-layer synthetic cold prefill; not serving TTFT",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "fragpack/ops/hbm_kv.cu",
                "fragpack/ops/hbm_kv.py",
                "benchmarks/fragpack_write_bench.py",
            )
        },
        "gpu": str(props),
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_graph": args.cuda_graph,
        "heads": args.heads,
        "gqa_group": args.group,
        "seed": args.seed,
        "rounds": args.rounds,
        "inner": args.inner,
        "warmup_ms": args.warmup_ms,
        "timing_contract": (
            "CUDA events around one layer invocation; fresh prepare and commit "
            "outside every layer window, including each Graph replay. Prepare "
            "events reported separately. Prepare and commit remain eagerly "
            "submitted in both modes. Total-step events cover prepare through "
            "commit, including host launch gaps and intervening measurement "
            "events; this is submitted host/GPU-stream step time, not pure "
            "kernel time or serving end-to-end latency. Eager layer timings "
            "also include host launch gaps. "
            "Inputs and pages reused warm; no hardware-cache flush or clock lock. "
            "Codebooks and inputs allocated/validated outside timing."
        ),
        "cases": [],
    }
    _persist(args.out, report)
    try:
        for tokens in args.tokens:
            for window in WINDOWS:
                case = _case(tokens, window, args)
                report["cases"].append(case)
                _persist(args.out, report)
                logger.info(
                    "%s N=%d staged=%.2f us direct=%.2f us ratio=%.3f",
                    window,
                    tokens,
                    case["timing"]["staged"]["median_us"],
                    case["timing"]["direct"]["median_us"],
                    case["direct_over_staged"],
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

# SPDX-License-Identifier: Apache-2.0
"""Profile real native-codec stages without changing production validation.

The mapping bounds-only arm is a diagnostic ablation, not a usable restore or
competitive baseline. Actual restore paths retain all mandatory checks. GPU
events inside captured graphs exclude Python submission gaps; eager event spans
include them. Run only on an explicitly reserved, exclusive CUDA device.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import functools
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback
from typing import Any, Callable

import numpy as np
import torch

from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import calibrate
from fragpack.ops.native_codec import NativeCodec, NativeKVPages

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CUDA_SHA256 = (
    "27a5f604f81a16d14a9849a3bb24fc61c0506c6355ac22645e79aa3007466130"
)
STAGES = {
    "mapping_bounds_only_diagnostic": 0,
    "mapping_bounds_and_alias": 1,
    "headers_lengths_and_mapping": 2,
    "record_payload_validation": 3,
    "decode_scatter_after_validation": 4,
}
LAYERS, HEADS, QUERY_HEADS, D, T = 48, 4, 32, 128, 128
POISON = 0x5AA5


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": min(values),
        "max": max(values),
    }


def _ints(value: str) -> list[int]:
    result = [int(part) for part in value.split(",")]
    if not result or len(set(result)) != len(result) or min(result) < 1:
        raise argparse.ArgumentTypeError("expected distinct positive sizes")
    return result


def _source_identity() -> dict[str, Any]:
    files = [
        Path(__file__),
        ROOT / "benchmarks/native_codec_stage_ext.cu",
        ROOT / "fragpack/ops/native_codec.cu",
        ROOT / "fragpack/ops/fragpack_common.cuh",
        ROOT / "fragpack/ops/native_codec.py",
        ROOT / "fragpack/ops/native_codec_ext.cpp",
        ROOT / "fragpack/ops/native_codec_ext.py",
        ROOT / "fragpack/compression/attn_tables.py",
        ROOT / "fragpack/compression/attn_page.py",
    ]
    hashes = {str(path.relative_to(ROOT)): _hash_file(path) for path in files}
    if hashes["fragpack/ops/native_codec.cu"] != BASELINE_CUDA_SHA256:
        raise RuntimeError(
            "stage profiler requires the recorded static-shared baseline CUDA; "
            "update and validate the profiler launch contract for other sources"
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {"git_head": head.stdout.strip(), "files_sha256": hashes}


def _extension() -> Any:
    from torch.utils.cpp_extension import load

    return load(
        name="fragpack_native_codec_stage_diagnostic_cuda",
        sources=[str(ROOT / "benchmarks/native_codec_stage_ext.cu")],
        extra_include_paths=[str(ROOT / "fragpack/ops")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )


def _load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    capture = np.load(args.dump, mmap_mode="r", allow_pickle=False)
    if capture.dtype != np.uint16 or capture.shape[1:] != (LAYERS, 2, T, HEADS, D):
        raise ValueError("expected uint16 [blocks,48,2,128,4,128] captured BF16")
    if (
        args.calib_pages < 1
        or args.eval_start < args.calib_pages
        or args.eval_pages < 1
        or args.eval_start + args.eval_pages > len(capture)
    ):
        raise ValueError("calibration/evaluation ranges must be valid and disjoint")
    calibration = np.asarray(capture[: args.calib_pages])
    table = calibrate(calibration)
    pool = np.array(capture[args.eval_start : args.eval_start + args.eval_pages])
    return (
        table,
        pool,
        {
            "dump": str(args.dump.resolve()),
            "dump_sha256": _hash_file(args.dump),
            "capture_shape": list(capture.shape),
            "calibration_range": [0, args.calib_pages],
            "calibration_sha256": _hash_array(calibration),
            "evaluation_range": [args.eval_start, args.eval_start + args.eval_pages],
            "evaluation_sha256": _hash_array(pool),
            "table_sha256": _hash_array(table),
            "description": args.input_description,
            "scope": (
                "Held-out positions of one capture; repeated physical copies at scale, "
                "not a fresh longer-context model execution. Stored KV is BF16."
            ),
        },
    )


def _make_archive(
    table: np.ndarray, expected: np.ndarray, page_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build immutable FPKB inputs outside timing, then release source/encoder."""
    blocks = len(expected)
    parts = T // page_tokens
    count = blocks * parts
    caches = [
        torch.empty(
            (2, count + 2, page_tokens, HEADS, D), dtype=torch.bfloat16, device="cuda"
        )
        for _ in range(LAYERS)
    ]
    for layer, cache in enumerate(caches):
        values = np.ascontiguousarray(expected[:, layer].transpose(1, 0, 2, 3, 4))
        values = values.reshape(2, count, page_tokens, HEADS, D)
        cache[:, 1 : count + 1].copy_(
            torch.from_numpy(values.view(np.int16)).to("cuda").view(torch.bfloat16)
        )
    source = NativeKVPages(caches)
    mapping = torch.arange(1, count + 1, dtype=torch.int32, device="cuda").reshape(
        blocks, parts
    )
    encoder = NativeCodec(table, max_blocks=blocks, device="cuda", workspace="encode")
    if bool(encoder.encode(source, mapping).any()):
        raise RuntimeError("archive encoding failed")
    offsets = encoder.offsets.clone()
    size = int(offsets[-1])
    payload = encoder.buffer[:size].clone()
    lengths = encoder.rec_lens.cpu().numpy()
    metadata = {
        "payload_bytes": size,
        "raw_bytes": int(expected.nbytes),
        "fallback_records": int(
            np.count_nonzero(lengths == ap.RAW_RECORD_BYTES - ap.DENSE_BYTES)
        ),
        "records": int(lengths.size),
        "encoder_workspace_released_before_timing": encoder.memory_usage(),
    }
    torch.cuda.current_stream().synchronize()
    return payload, offsets, metadata


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    start: torch.cuda.Event
    end: torch.cuda.Event


class _Context:
    """One immutable native-only restore plus stock FA3, with no external tier."""

    def __init__(
        self,
        args: argparse.Namespace,
        table: np.ndarray,
        pool: np.ndarray,
        blocks: int,
        page_tokens: int,
        extension: Any,
    ) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        self.blocks, self.page_tokens = blocks, page_tokens
        self.parts = T // page_tokens
        indices = np.arange(blocks) % len(pool)
        self.expected = np.ascontiguousarray(pool[indices])
        self.input = {
            "physical_blocks": blocks,
            "page_tokens": page_tokens,
            "capture_indices_in_eval_pool": indices.tolist(),
            "synthetic_repetition": blocks > len(pool),
            "materialized_sha256": _hash_array(self.expected),
        }
        self.payload, self.offsets, self.archive = _make_archive(
            table, self.expected, page_tokens
        )
        gc.collect()
        torch.cuda.empty_cache()
        self.page_count = blocks * self.parts + 2
        self.caches = [
            torch.empty(
                (2, self.page_count, page_tokens, HEADS, D),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(LAYERS)
        ]
        self.native = NativeKVPages(self.caches)
        rng = np.random.default_rng(args.seed + blocks)
        self.ids = rng.permutation(np.arange(1, self.page_count - 1)).reshape(
            blocks, self.parts
        )
        self.mapping = torch.tensor(self.ids, dtype=torch.int32, device="cuda")
        self.decoder = NativeCodec(
            table, max_blocks=blocks, device="cuda", workspace="decode"
        )
        self.extension = extension
        self.restore()
        if bool(self.decoder.status.any()):
            raise RuntimeError("initial restore failed")
        self.args = (
            self.native.descriptors,
            self.mapping,
            self.decoder.tables.plut,
            self.decoder.tables.slut,
            self.payload,
            self.offsets,
            self.decoder.rec_offsets,
            self.decoder.status,
            page_tokens,
            HEADS,
            self.decoder.digest,
        )
        self.fa3 = flash_attn_varlen_func
        generator = torch.Generator().manual_seed(args.seed)
        host_q = torch.randn(
            LAYERS, args.query, QUERY_HEADS, D, generator=generator
        ).to(torch.bfloat16)
        self.input["query_sha256"] = _hash_array(host_q.view(torch.int16).numpy())
        self.q = host_q.to("cuda")
        self.output = torch.empty_like(self.q)
        self.q_layers = list(self.q.unbind(0))
        self.output_layers = list(self.output.unbind(0))
        self.keys = [cache[0] for cache in self.caches]
        self.values = [cache[1] for cache in self.caches]
        self.block_table = self.mapping.reshape(1, -1)
        self.cu_q = torch.tensor([0, args.query], dtype=torch.int32, device="cuda")
        self.seq_lens = torch.tensor([blocks * T], dtype=torch.int32, device="cuda")
        self.query = args.query
        self.operations: dict[str, Callable[[], None]] = {
            name: functools.partial(extension.stage, index, *self.args)
            for name, index in STAGES.items()
        }
        self.operations.update(
            {
                "direct_validated_extension_restore": functools.partial(
                    extension.full_restore, *self.args
                ),
                "public_validated_codec_restore": self.restore,
                "native_fa3_48_layers": self.read,
                "public_restore_and_fa3": self.restore_read,
            }
        )
        self.graphs: dict[str, _Graph] = {}
        self.eager_start = torch.cuda.Event(enable_timing=True)
        self.eager_end = torch.cuda.Event(enable_timing=True)
        self.read()
        self.reference = self.output.clone()
        self.validation = self.validate()
        self.validation["mapping_failures"] = self.validate_mapping_failures()
        for operation in self.operations.values():
            for _ in range(args.warmup):
                operation()
        torch.cuda.current_stream().synchronize()
        for name, operation in self.operations.items():
            graph = torch.cuda.CUDAGraph()
            # Explicit graph event nodes retain timestamps on every replay.
            start, end = (
                torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)
            )
            with torch.cuda.graph(graph):
                start.record()
                operation()
                end.record()
            self.graphs[name] = _Graph(graph, start, end)
        self.memory = {
            "native_pages": sum(
                cache.numel() * cache.element_size() for cache in self.caches
            ),
            "decoder_workspace": self.decoder.memory_usage(),
            "received_payload": self.payload.numel(),
            "received_offsets": self.offsets.numel() * self.offsets.element_size(),
            "queries": self.q.numel() * self.q.element_size(),
            "output": self.output.numel() * self.output.element_size(),
            "validation_reference": self.reference.numel()
            * self.reference.element_size(),
            "torch_allocated_after_graphs": torch.cuda.memory_allocated(),
            "torch_reserved_after_graphs": torch.cuda.memory_reserved(),
            "scope": (
                "Diagnostic context tensor accounting; not a deployment memory peak."
            ),
        }

    def restore(self) -> None:
        """Enqueue the complete public decode path on the current CUDA stream."""
        self.decoder.decode(
            self.native, self.mapping, src=self.payload, offsets=self.offsets
        )

    def read(self) -> None:
        """Read all 48 native KV layers with the unchanged public FA3 function."""
        for layer in range(LAYERS):
            self.fa3(
                q=self.q_layers[layer],
                k=self.keys[layer],
                v=self.values[layer],
                out=self.output_layers[layer],
                cu_seqlens_q=self.cu_q,
                max_seqlen_q=self.query,
                seqused_k=self.seq_lens,
                max_seqlen_k=self.blocks * T,
                softmax_scale=D**-0.5,
                causal=True,
                block_table=self.block_table,
                fa_version=3,
                num_splits=0,
            )

    def restore_read(self) -> None:
        """Restore once, then consume those pages in stream order without a sync."""
        self.restore()
        self.read()

    def poison(self) -> None:
        """Overwrite every destination page outside measured intervals."""
        for cache in self.caches:
            cache.view(torch.int16).fill_(POISON)

    def validate(self, graph: bool = False) -> dict[str, Any]:
        """Check fresh restoration, all selected BF16 bits, canaries and FA3 output."""
        self.poison()
        if graph:
            self.graphs["public_restore_and_fa3"].graph.replay()
        else:
            self.restore_read()
        if bool(self.decoder.status.any()):
            raise AssertionError("validated restore reported an error")
        index = self.mapping.flatten().to(torch.int64)
        for layer, cache in enumerate(self.caches):
            actual = (
                cache.index_select(1, index)
                .view(torch.int16)
                .cpu()
                .numpy()
                .view(np.uint16)
            )
            reference = np.ascontiguousarray(
                self.expected[:, layer].transpose(1, 0, 2, 3, 4)
            )
            reference = reference.reshape(
                2, self.blocks * self.parts, self.page_tokens, HEADS, D
            )
            if not np.array_equal(actual, reference):
                raise AssertionError(f"restored BF16 differs at layer {layer}")
            if not bool(
                torch.all(
                    cache[:, [0, self.page_count - 1]].view(torch.int16) == POISON
                )
            ):
                raise AssertionError("unused native page canary changed")
        if not torch.equal(
            self.output.view(torch.int16), self.reference.view(torch.int16)
        ):
            raise AssertionError("native FA3 output differs bitwise from reference")
        return {
            "kv_bitwise_exact": True,
            "unused_pages_intact": True,
            "fa3_bitwise_exact": True,
            "poisoned_before_restore": True,
        }

    def validate_mapping_failures(self) -> dict[str, Any]:
        """Probe alias/range rejection and no-write behavior outside timing."""
        results = {}
        for kind in ("duplicate", "negative", "out_of_range"):
            if kind == "duplicate" and self.blocks * self.parts < 2:
                continue
            bad = self.mapping.clone()
            if kind == "duplicate":
                bad.flatten()[1] = bad.flatten()[0]
            else:
                bad.flatten()[0] = -1 if kind == "negative" else self.page_count
            cpu = bad.cpu().numpy()
            flat = cpu.reshape(-1)
            unique, counts = np.unique(flat, return_counts=True)
            duplicates = unique[counts > 1]
            invalid = np.any(
                (cpu < 0) | (cpu >= self.page_count) | np.isin(cpu, duplicates), axis=1
            )
            arguments = (self.args[0], bad, *self.args[2:])
            # This ablation writes status only: repeated destination pages are
            # deliberately accepted, but range errors remain rejected. Never
            # follow it with an unchecked decoder launch on this invalid map.
            bounds_invalid = np.any((cpu < 0) | (cpu >= self.page_count), axis=1)
            self.extension.stage(0, *arguments)
            bounds_status = self.decoder.status.cpu().numpy().copy()
            np.testing.assert_array_equal((bounds_status & 4) != 0, bounds_invalid)
            self.extension.stage(1, *arguments)
            status = self.decoder.status.cpu().numpy()
            np.testing.assert_array_equal((status & 4) != 0, invalid)
            self.poison()
            self.decoder.decode(
                self.native, bad, src=self.payload, offsets=self.offsets
            )
            status = self.decoder.status.cpu().numpy()
            np.testing.assert_array_equal((status & 4) != 0, invalid)
            rejected_pages = np.unique(cpu[invalid])
            rejected_pages = rejected_pages[
                (rejected_pages >= 0) & (rejected_pages < self.page_count)
            ]
            for cache in self.caches:
                if not bool(
                    torch.all(
                        cache[:, rejected_pages.tolist()].view(torch.int16) == POISON
                    )
                ):
                    raise AssertionError(
                        "rejected mapping wrote native destination bytes"
                    )
            results[kind] = {
                "rejected_blocks": np.flatnonzero(invalid).tolist(),
                "bounds_only_diagnostic_status": bounds_status.tolist(),
                "status": status.tolist(),
                "rejected_destinations_intact": True,
            }
        self.restore_read()
        return results

    def sample(self, name: str, mode: str) -> dict[str, Any]:
        """Submit one warmed operation and record graph/span and host boundaries."""
        if mode == "graph":
            graph = self.graphs[name]
            started = time.perf_counter_ns()
            graph.graph.replay()
            enqueued = time.perf_counter_ns()
            graph.end.synchronize()
            finished = time.perf_counter_ns()
            elapsed = graph.start.elapsed_time(graph.end) * 1000
            key = "gpu_graph_us"
        else:
            self.eager_start.record()
            started = time.perf_counter_ns()
            self.operations[name]()
            enqueued = time.perf_counter_ns()
            self.eager_end.record()
            self.eager_end.synchronize()
            finished = time.perf_counter_ns()
            elapsed = self.eager_start.elapsed_time(self.eager_end) * 1000
            key = "gpu_stream_span_us"
        return {
            "operation": name,
            "mode": mode,
            key: elapsed,
            "host_enqueue_us": (enqueued - started) / 1000,
            "host_until_completion_us": (finished - started) / 1000,
        }

    def nsight_cycle(self) -> None:
        """One annotated eager cycle, bounded by CUDA profiler start/stop APIs."""
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:
            for name, operation in self.operations.items():
                with torch.cuda.nvtx.range(f"native-codec-stage/{name}"):
                    operation()
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=_ints, default=[16, 256])
    parser.add_argument("--page-tokens", type=_ints, default=[16, 128])
    parser.add_argument("--query", type=int, default=128)
    parser.add_argument("--calib-pages", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=8)
    parser.add_argument("--eval-pages", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--input-description", required=True)
    parser.add_argument("--nsight-capture", action="store_true")
    return parser


def main() -> None:
    """Run the explicitly selected diagnostic matrix and retain every sample."""
    args = _parser().parse_args()
    if (
        min(args.rounds, args.warmup, args.query) < 1
        or any(page not in (16, 32, 64, 128) for page in args.page_tokens)
        or args.query > min(args.blocks) * T
    ):
        raise ValueError("invalid rounds, page size, or query/context geometry")
    if args.nsight_capture and len(args.blocks) * len(args.page_tokens) != 1:
        raise ValueError("bounded Nsight capture requires exactly one N/P setting")
    source = _source_identity()
    report: dict[str, Any] = {
        "schema": "fragpack.native-codec-stage-profile.v1",
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cases": [],
        "protocol": {
            "cuda_sources": (
                "Diagnostic extension includes the actual native_codec.cu; exported "
                "host symbols are renamed, translation-local kernel code is reused."
            ),
            "cuda_build_flags": ["-O3", "--use_fast_math", "-lineinfo"],
            "scope": (
                "Resident received FPKB to native BF16 and 48-layer stock FA3; "
                "no H2D, new KV, model or serving."
            ),
            "production_paths": [
                "direct_validated_extension_restore",
                "public_validated_codec_restore",
                "public_restore_and_fa3",
            ],
            "mapping_ablation": (
                "Bounds-only omits alias rejection for diagnosis; it is not a valid "
                "restore path or competitive baseline."
            ),
            "overlapping_scopes": (
                "Header stage includes mapping. Independent mapping times are not "
                "additive to header/record/decode times."
            ),
            "graph_gpu_timing": (
                "CUDA timing events are nodes INSIDE each captured graph, excluding "
                "Python submission gaps."
            ),
            "eager_gpu_timing": (
                "CUDA event stream span includes GPU idle time caused by CPU "
                "submission gaps; Nsight supplies individual kernel durations."
            ),
            "host_timing": (
                "Host enqueue excludes completion wait; until-completion includes "
                "that wait. Neither includes setup or validation readback."
            ),
            "interface_comparison": (
                "Public codec calls include bind/status-zero; direct extension retains "
                "all CUDA validation and C++ tensor checks. Isolated stages are "
                "diagnostic launches."
            ),
            "ordering": (
                "Alternate forward/reversed operation order each round; a second "
                "process may reverse its starting order. Warmups excluded."
            ),
        },
    }
    _write(args.out, report)
    try:
        table, pool, report["input"] = _load_inputs(args)
        report["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": importlib.metadata.version("vllm"),
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_properties(0).name,
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "cuda_runtime": torch.version.cuda,
        }
        extension = _extension()
        report["status"] = "running"
        _write(args.out, report)
        for blocks in args.blocks:
            for page_tokens in args.page_tokens:
                row: dict[str, Any] = {
                    "blocks": blocks,
                    "page_tokens": page_tokens,
                    "query": args.query,
                    "status": "preparing",
                }
                report["cases"].append(row)
                _write(args.out, report)
                ctx = _Context(args, table, pool, blocks, page_tokens, extension)
                row.update(
                    {
                        "input": ctx.input,
                        "archive": ctx.archive,
                        "memory": ctx.memory,
                        "validation": ctx.validation,
                        "samples": [],
                        "summary": {},
                        "status": "running",
                    }
                )
                for mode in ("graph", "eager"):
                    names = list(ctx.operations)
                    for round_id in range(args.rounds):
                        order = (
                            names
                            if (round_id + (args.order == "reverse")) % 2 == 0
                            else list(reversed(names))
                        )
                        for position, name in enumerate(order):
                            sample = ctx.sample(name, mode)
                            sample.update(
                                {
                                    "round": round_id,
                                    "position": position,
                                    "order": order,
                                }
                            )
                            row["samples"].append(sample)
                    mode_summary = {}
                    for name in names:
                        samples = [
                            sample
                            for sample in row["samples"]
                            if sample["mode"] == mode and sample["operation"] == name
                        ]
                        mode_summary[name] = {
                            key: _summary([sample[key] for sample in samples])
                            for key in samples[0]
                            if key.endswith("_us")
                        }
                    row["summary"][mode] = mode_summary
                    _write(args.out, report)
                row["post_timing_validation"] = ctx.validate(graph=True)
                if args.nsight_capture:
                    ctx.nsight_cycle()
                    row["nsight_capture"] = (
                        "one warmed eager NVTX cycle inside CUDAProfilerApi range"
                    )
                row["status"] = "passed"
                _write(args.out, report)
                print(
                    json.dumps(
                        {
                            "blocks": blocks,
                            "page_tokens": page_tokens,
                            "graph_median_us": {
                                name: values["gpu_graph_us"]["median"]
                                for name, values in row["summary"]["graph"].items()
                            },
                        }
                    ),
                    flush=True,
                )
                del ctx
                gc.collect()
                torch.cuda.empty_cache()
        if _source_identity() != source:
            raise RuntimeError("source changed while profiling")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write(args.out, report)


if __name__ == "__main__":
    main()

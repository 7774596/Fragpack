# SPDX-License-Identifier: Apache-2.0
"""Compare native BF16 KV roundtrips through raw and compressed host buffers.

Input is captured uint16 BF16 bits [blocks, layers, 2, 128, heads, 128].
Calibration and evaluation block positions must be disjoint. This is a serial
codec/PCIe microbenchmark, not attention, model serving, or external-cache RPC.
Use an explicitly reserved GPU and keep outputs on the experiment filesystem.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback
from typing import Any

import numpy as np
import torch

from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import calibrate


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _ints(value: str) -> list[int]:
    values = [int(part) for part in value.split(",")]
    if (
        not values
        or any(part < 1 for part in values)
        or len(set(values)) != len(values)
    ):
        raise argparse.ArgumentTypeError(
            "expected distinct positive comma-separated integers"
        )
    return values


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples": values,
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


@dataclass
class Inputs:
    """CPU input bits and codebooks; no CUDA work is performed when loading."""

    table: np.ndarray
    evaluation: np.ndarray
    provenance: dict[str, Any]


def _load_inputs(args: argparse.Namespace) -> Inputs:
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
        raise ValueError("calibration [0,calib_pages) must precede evaluation")
    if args.eval_count < 1 or args.eval_start + args.eval_count > dump.shape[0]:
        raise ValueError("evaluation block range lies outside dump")
    if max(args.blocks) > args.eval_count:
        raise ValueError(
            "requested blocks exceed unique evaluation blocks; replication is disabled"
        )
    calibration = np.asarray(dump[: args.calib_pages])
    table = calibrate(calibration)
    evaluation = np.array(dump[args.eval_start : args.eval_start + args.eval_count])
    provenance = {
        "dump": str(Path(args.dump).resolve()),
        "dump_basename": Path(args.dump).name,
        "dump_sha256": _file_hash(Path(args.dump)),
        "dump_shape": list(dump.shape),
        "dump_dtype": str(dump.dtype),
        "dump_bytes": int(dump.nbytes),
        "calibration_range": [0, args.calib_pages],
        "calibration_sha256": _array_hash(calibration),
        "evaluation_range": [args.eval_start, args.eval_start + args.eval_count],
        "evaluation_sha256": _array_hash(evaluation),
        "table_shape": list(table.shape),
        "table_sha256": _array_hash(table),
        "input_description": args.input_description,
        "split_scope": (
            "Disjoint positions in supplied capture; not independent-document sampling."
        ),
    }
    return Inputs(table, evaluation, provenance)


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    files = sorted(
        path
        for path in (root / "fragpack").rglob("*")
        if path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}
    )
    files.append(Path(__file__).resolve())
    hashes = {str(path.relative_to(root)): _file_hash(path) for path in files}
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "files_sha256": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    }


class _Case:
    """Preallocated, stream-ordered arms with independent poisoned destinations."""

    def __init__(self, args: argparse.Namespace, inputs: Inputs, blocks: int) -> None:
        from fragpack.ops.native_codec import NativeCodec, NativeKVPages

        self.blocks = blocks
        self.expected = np.ascontiguousarray(inputs.evaluation[:blocks])
        _, self.layers, _, _, self.heads, _ = self.expected.shape
        self.page_tokens = args.page_tokens
        self.parts = ap.PAGE_TOKENS // self.page_tokens
        physical_pages = blocks * self.parts + 3
        rng = np.random.default_rng(args.seed + blocks)
        self.source_ids = rng.permutation(physical_pages)[: blocks * self.parts]
        self.target_ids = rng.permutation(physical_pages)[: blocks * self.parts]
        self.source_map = torch.tensor(
            self.source_ids.reshape(blocks, self.parts),
            dtype=torch.int32,
            device="cuda",
        )
        self.target_map = torch.tensor(
            self.target_ids.reshape(blocks, self.parts),
            dtype=torch.int32,
            device="cuda",
        )
        self.unselected_target = np.setdiff1d(
            np.arange(physical_pages), self.target_ids
        ).tolist()
        shape = (2, physical_pages, self.page_tokens, self.heads, ap.HEAD_DIM)

        def allocate_pages() -> list[torch.Tensor]:
            caches = [
                torch.empty(shape, dtype=torch.bfloat16, device="cuda")
                for _ in range(self.layers)
            ]
            for cache in caches:
                cache.view(torch.int16).fill_(0x7FA5)
            return caches

        self.source_caches = allocate_pages()
        self.destinations = {arm: allocate_pages() for arm in ("raw", "compressed")}
        source_index = self.source_map.flatten().to(torch.int64)
        for layer, cache in enumerate(self.source_caches):
            values = self.expected[:, layer].reshape(
                blocks, 2, self.parts, self.page_tokens, self.heads, ap.HEAD_DIM
            )
            values = np.ascontiguousarray(values.transpose(1, 0, 2, 3, 4, 5)).reshape(
                2, blocks * self.parts, self.page_tokens, self.heads, ap.HEAD_DIM
            )
            cache.index_copy_(
                1,
                source_index,
                torch.from_numpy(values.view(np.int16)).to("cuda").view(torch.bfloat16),
            )
        self.source = NativeKVPages(self.source_caches)
        self.targets = {
            arm: NativeKVPages(caches) for arm, caches in self.destinations.items()
        }
        self.codec = NativeCodec(
            inputs.table, max_blocks=blocks, device=self.source.device
        )
        self.raw_bytes = int(self.expected.nbytes)
        self.raw_send = torch.empty(self.raw_bytes, dtype=torch.uint8, device="cuda")
        self.raw_recv = torch.empty_like(self.raw_send)
        self.raw_host = torch.empty(self.raw_bytes, dtype=torch.uint8, pin_memory=True)
        self.comp_recv = torch.empty_like(self.codec.buffer)
        self.comp_host = torch.empty(
            self.codec.buffer.numel(), dtype=torch.uint8, pin_memory=True
        )
        # Public workspace views are empty until the first batch is bound.
        self.offsets_host = torch.empty(blocks + 1, dtype=torch.int64, pin_memory=True)
        self.offsets_recv = torch.empty(blocks + 1, dtype=torch.int64, device="cuda")
        self.status_host = torch.empty(blocks, dtype=torch.int32, pin_memory=True)
        # Events are reusable: each serial sample completes before the next starts.
        self.phase_events = [torch.cuda.Event(enable_timing=True) for _ in range(12)]
        self.completion_events = [torch.cuda.Event() for _ in range(12)]
        self.events = self.completion_events
        self.profiling = False
        self.required_events: tuple[int, ...] = ()
        self.stream = torch.cuda.current_stream()
        self.stream.synchronize()
        self.allocations = {
            "native_source_bytes": sum(
                t.numel() * t.element_size() for t in self.source_caches
            ),
            "native_destination_bytes_per_arm": sum(
                t.numel() * t.element_size() for t in self.destinations["raw"]
            ),
            "native_descriptor_bytes_each": self.source.memory_usage()["descriptors"],
            "selected_raw_payload_bytes": self.raw_bytes,
            "page_mapping_bytes": (self.source_map.numel() + self.target_map.numel())
            * 4,
            "raw_gpu_transfer_buffers_bytes": 2 * self.raw_bytes,
            "raw_pinned_payload_bytes": self.raw_bytes,
            "codec_owned": self.codec.memory_usage(),
            "compressed_gpu_receive_capacity_bytes": self.comp_recv.numel(),
            "compressed_pinned_payload_capacity_bytes": self.comp_host.numel(),
            "compressed_gpu_receive_offsets_bytes": self.offsets_recv.numel()
            * self.offsets_recv.element_size(),
            "compressed_pinned_offsets_bytes": self.offsets_host.numel()
            * self.offsets_host.element_size(),
            "pinned_status_bytes_shared_between_arms": self.status_host.numel()
            * self.status_host.element_size(),
            "actual_torch_allocated_bytes_both_arms": torch.cuda.memory_allocated(),
            "actual_torch_reserved_bytes_both_arms": torch.cuda.memory_reserved(),
            "accounting_scope": (
                "Both arms coexist for timing. Per-arm budgets use one source "
                "and one destination, plus their listed buffers. CUDA context/JIT, "
                "allocator reservation, Python state and CUDA events are outside "
                "tensor totals; descriptors are reported separately when available."
            ),
        }

    def _check_status(self) -> None:
        if bool(torch.any(self.status_host != 0)):
            raise RuntimeError(f"native codec status {self.status_host.tolist()}")

    def _status_copy(self, status: torch.Tensor) -> None:
        self.status_host.copy_(status, non_blocking=True)

    def _event(self, index: int) -> None:
        if self.profiling or index in self.required_events:
            self.events[index].record(self.stream)

    def _select_events(self, required: tuple[int, ...]) -> None:
        self.required_events = required
        self.events = self.phase_events if self.profiling else self.completion_events

    def _elapsed(self, start: int, end: int) -> float:
        return self.events[start].elapsed_time(self.events[end]) * 1000

    def raw(self) -> dict[str, Any]:
        """Time real raw gather, pinned D2H/H2D, scatter and status completion."""
        self._select_events((3, 7))
        started = time.perf_counter_ns()
        self._event(0)
        status = self.codec.raw_gather(self.source, self.source_map, self.raw_send)
        self._event(1)
        self.raw_host.copy_(self.raw_send, non_blocking=True)
        self._event(2)
        self._status_copy(status)
        self._event(3)
        self.events[3].synchronize()
        self._check_status()
        stored = time.perf_counter_ns()
        self._event(4)
        self.raw_recv.copy_(self.raw_host, non_blocking=True)
        self._event(5)
        status = self.codec.raw_scatter(
            self.targets["raw"], self.target_map, self.raw_recv
        )
        self._event(6)
        self._status_copy(status)
        self._event(7)
        self.events[7].synchronize()
        self._check_status()
        finished = time.perf_counter_ns()
        return {
            "arm": "raw",
            "timing_mode": "phase_diagnostic" if self.profiling else "complete",
            "payload_bytes_each_direction": self.raw_bytes,
            "metadata_d2h_bytes": 2
            * self.status_host.numel()
            * self.status_host.element_size(),
            "metadata_h2d_bytes": 0,
            "store_wall_us": (stored - started) / 1000,
            "load_wall_us": (finished - stored) / 1000,
            "roundtrip_wall_us": (finished - started) / 1000,
            "cuda_spans_us": {
                "gather": self._elapsed(0, 1),
                "payload_d2h": self._elapsed(1, 2),
                "store_status_d2h": self._elapsed(2, 3),
                "payload_h2d": self._elapsed(4, 5),
                "scatter": self._elapsed(5, 6),
                "load_status_d2h": self._elapsed(6, 7),
                "roundtrip_stream_span": self._elapsed(0, 7),
            }
            if self.profiling
            else {},
        }

    def compressed(self) -> dict[str, Any]:
        """Include variable length observation and only actual payload transfers."""
        self._select_events((4, 6, 11))
        started = time.perf_counter_ns()
        self._event(0)
        self.codec.encode_records(self.source, self.source_map)
        self._event(1)
        self.codec.plan()
        self._event(2)
        self.codec.pack()
        self._event(3)
        self.offsets_host.copy_(self.codec.offsets, non_blocking=True)
        self._status_copy(self.codec.status)
        self._event(4)
        self.events[4].synchronize()
        observation_started = time.perf_counter_ns()
        self._check_status()
        offsets = self.offsets_host.tolist()
        total = int(offsets[-1])
        if offsets[0] != 0 or any(
            b <= a for a, b in zip(offsets, offsets[1:], strict=False)
        ):
            raise RuntimeError(f"invalid encoded offsets {offsets}")
        if not 0 < total <= self.codec.buffer.numel():
            raise RuntimeError(f"encoded length {total} exceeds output capacity")
        observed = time.perf_counter_ns()
        self._event(5)
        self.comp_host[:total].copy_(self.codec.buffer[:total], non_blocking=True)
        self._event(6)
        self.events[6].synchronize()
        stored = time.perf_counter_ns()
        self._event(7)
        self.comp_recv[:total].copy_(self.comp_host[:total], non_blocking=True)
        self._event(8)
        self.offsets_recv.copy_(self.offsets_host, non_blocking=True)
        self._event(9)
        status = self.codec.decode(
            self.targets["compressed"],
            self.target_map,
            src=self.comp_recv[:total],
            offsets=self.offsets_recv,
        )
        self._event(10)
        self._status_copy(status)
        self._event(11)
        self.events[11].synchronize()
        self._check_status()
        finished = time.perf_counter_ns()
        return {
            "arm": "compressed",
            "timing_mode": "phase_diagnostic" if self.profiling else "complete",
            "payload_bytes_each_direction": total,
            "blob_sizes": [b - a for a, b in zip(offsets, offsets[1:], strict=False)],
            "metadata_d2h_bytes": self.offsets_host.numel()
            * self.offsets_host.element_size()
            + 2 * self.status_host.numel() * self.status_host.element_size(),
            "metadata_h2d_bytes": self.offsets_host.numel()
            * self.offsets_host.element_size(),
            "length_observation_cpu_us": (observed - observation_started) / 1000,
            "store_wall_us": (stored - started) / 1000,
            "load_wall_us": (finished - stored) / 1000,
            "roundtrip_wall_us": (finished - started) / 1000,
            "cuda_spans_us": {
                "encode_records": self._elapsed(0, 1),
                "plan": self._elapsed(1, 2),
                "pack": self._elapsed(2, 3),
                "lengths_status_d2h": self._elapsed(3, 4),
                "payload_d2h": self._elapsed(5, 6),
                "payload_h2d": self._elapsed(7, 8),
                "offsets_h2d": self._elapsed(8, 9),
                "decode_validate_scatter": self._elapsed(9, 10),
                "load_status_d2h": self._elapsed(10, 11),
                "roundtrip_stream_span": self._elapsed(0, 11),
            }
            if self.profiling
            else {},
        }

    def validate(self, arm: str) -> dict[str, Any]:
        """Check BF16 bits and untouched native pages outside measured windows."""
        for layer, cache in enumerate(self.destinations[arm]):
            host = cache.view(torch.int16).cpu().numpy().view(np.uint16)
            selected = (
                np.ascontiguousarray(host[:, self.target_ids])
                .reshape(
                    2,
                    self.blocks,
                    self.parts,
                    self.page_tokens,
                    self.heads,
                    ap.HEAD_DIM,
                )
                .transpose(1, 0, 2, 3, 4, 5)
                .reshape(self.blocks, 2, ap.PAGE_TOKENS, self.heads, ap.HEAD_DIM)
            )
            if not np.array_equal(selected, self.expected[:, layer]):
                raise AssertionError(
                    f"{arm} reconstructed BF16 bits differ at layer {layer}"
                )
            if not np.all(host[:, self.unselected_target] == 0x7FA5):
                raise AssertionError(
                    f"{arm} overwrote an unselected native page at layer {layer}"
                )
        return {
            "bitwise_equal": True,
            "unselected_pages_unchanged": True,
            "checked_layers": self.layers,
        }

    def format_accounting(self) -> dict[str, Any]:
        """Read record lengths outside timing to distinguish payload from capacity."""
        lengths = self.codec.rec_lens.cpu().numpy()
        raw = lengths == ap.RAW_RECORD_BYTES - ap.DENSE_BYTES
        head = (
            (ap.BLOB_FIXED_BYTES + 8 * self.layers * self.heads + ap.BLOB_ALIGN - 1)
            // ap.BLOB_ALIGN
            * ap.BLOB_ALIGN
        )
        record_bytes = (
            (lengths.astype(np.int64) + ap.DENSE_BYTES)
            .reshape(self.blocks, -1)
            .sum(axis=1)
        )
        stored = (
            (record_bytes + head + ap.BLOB_ALIGN - 1) // ap.BLOB_ALIGN * ap.BLOB_ALIGN
        )
        return {
            "record_count": int(lengths.size),
            "raw_fallback_records": int(raw.sum()),
            "raw_fallback_fraction": float(raw.mean()),
            "raw_fallback_by_layer_kv_head": raw.sum(axis=0).tolist(),
            "record_variable_lengths": lengths.tolist(),
            "blob_header_bytes_each": head,
            "record_bytes_total": int(record_bytes.sum()),
            "blob_payload_bytes_total": int(stored.sum()),
            "header_and_padding_bytes_total": int(stored.sum() - record_bytes.sum()),
            "scope": "Actual FPKB bytes, not the old arena reservation ratio.",
        }


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for arm in ("raw", "compressed"):
        selected = [row for row in samples if row["arm"] == arm]
        output[arm] = {
            key: _summary([row[key] for row in selected])
            for key in ("store_wall_us", "load_wall_us", "roundtrip_wall_us")
        }
        output[arm]["cuda_spans_us"] = {
            key: _summary([row["cuda_spans_us"][key] for row in selected])
            for key in selected[0]["cuda_spans_us"]
        }
        output[arm]["payload_bytes_each_direction"] = sorted(
            {row["payload_bytes_each_direction"] for row in selected}
        )
    return output


def _run(args: argparse.Namespace, inputs: Inputs, report: dict[str, Any]) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark requires exactly one explicitly selected CUDA device"
        )
    props = torch.cuda.get_device_properties(0)
    report["environment"].update(
        {
            "gpu_name": props.name,
            "gpu_uuid": str(props.uuid),
            "gpu_total_memory": props.total_memory,
            "compute_capability": [props.major, props.minor],
        }
    )
    report["timing_scope"] = {
        "host_wall": (
            "Serial preallocated roundtrip: gather or encode/plan/pack, "
            "status/length observation, actual-size pinned D2H, completion, H2D, "
            "scatter or validated decode/scatter, status observation and completion. "
            "Main samples record only required completion events: raw 2, "
            "compressed 3. These events have timing disabled."
        ),
        "cuda_spans": (
            "Separate profiling_samples record all phase timing events after "
            "an excluded diagnostic warmup. Their CUDA-event stream spans include "
            "Python launch gaps; not isolated kernel durations. Their host wall "
            "includes instrumentation and is not pooled with main samples. "
            "roundtrip_stream_span also contains host control gaps."
        ),
        "excluded": (
            "Initialization, calibration, allocation, input upload, warmups, "
            "correctness checks, table initialization, RPC, SSD, attention, "
            "and model execution."
        ),
        "replication": (
            "Repeated serial warm-buffer samples within one process; "
            "not independent model requests or independent process repetitions."
        ),
        "metadata": (
            "Compressed offsets travel D2H and H2D in separate buffers. "
            "Status is checked before publication/reuse. Tables are already "
            "resident for both directions."
        ),
    }
    for blocks in args.blocks:
        case = _Case(args, inputs, blocks)
        row: dict[str, Any] = {
            "blocks": blocks,
            "layers": case.layers,
            "heads": case.heads,
            "page_tokens": args.page_tokens,
            "evaluation_block_indices": list(
                range(args.eval_start, args.eval_start + blocks)
            ),
            "source_page_map": case.source_map.cpu().tolist(),
            "target_page_map": case.target_map.cpu().tolist(),
            "allocations": case.allocations,
            "warmup_samples": [],
            "samples": [],
            "profiling_warmup_samples": [],
            "profiling_samples": [],
            "validation": {},
        }
        report["cases"].append(row)
        _write(args.out, report)
        functions = {"raw": case.raw, "compressed": case.compressed}
        for index in range(args.warmup):
            for arm in ("raw", "compressed"):
                sample = functions[arm]()
                sample["warmup_index"] = index
                row["warmup_samples"].append(sample)
        row["validation"]["after_warmup"] = {
            arm: case.validate(arm) for arm in functions
        }
        for index in range(args.rounds):
            order = ["raw", "compressed"] if index % 2 == 0 else ["compressed", "raw"]
            if args.order == "reverse":
                order.reverse()
            for position, arm in enumerate(order):
                sample = functions[arm]()
                sample.update({"round": index, "position": position, "order": order})
                row["samples"].append(sample)
            _write(args.out, report)
        row["validation"]["after_measurement"] = {
            arm: case.validate(arm) for arm in functions
        }
        row["summary"] = _aggregate(row["samples"])
        case.profiling = True
        for arm in functions:
            row["profiling_warmup_samples"].append(functions[arm]())
        for index in range(3):
            order = ["raw", "compressed"] if index % 2 == 0 else ["compressed", "raw"]
            if args.order == "reverse":
                order.reverse()
            for position, arm in enumerate(order):
                sample = functions[arm]()
                sample.update({"round": index, "position": position, "order": order})
                row["profiling_samples"].append(sample)
        row["profiling_summary"] = _aggregate(row["profiling_samples"])
        row["validation"]["after_profiling"] = {
            arm: case.validate(arm) for arm in functions
        }
        row["format"] = case.format_accounting()
        payloads = row["summary"]["compressed"]["payload_bytes_each_direction"]
        if payloads != [row["format"]["blob_payload_bytes_total"]]:
            raise AssertionError(
                "measured payload lengths disagree with record accounting"
            )
        row["payload_ratio_raw_over_compressed"] = (
            case.raw_bytes
            / row["summary"]["compressed"]["payload_bytes_each_direction"][0]
        )
        print(
            json.dumps(
                {
                    "blocks": blocks,
                    "summary": {
                        arm: row["summary"][arm]["roundtrip_wall_us"]["median"]
                        for arm in functions
                    },
                    "payload_ratio": row["payload_ratio_raw_over_compressed"],
                }
            ),
            flush=True,
        )
        _write(args.out, report)
        del functions, case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=_ints, default=[1, 4, 16])
    parser.add_argument("--calib-pages", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument(
        "--page-tokens", type=int, choices=[16, 32, 64, 128], default=128
    )
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--order", choices=["mixed", "reverse"], default="mixed")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument(
        "--input-description",
        default="User-supplied captured BF16 KV; see dump provenance.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="CPU only: save split hashes and calibrated .tables.npz",
    )
    return parser


def main() -> None:
    """Prepare inputs or run on the explicitly selected CUDA device."""
    args = _parser().parse_args()
    if args.rounds < 2 or args.warmup < 1:
        raise ValueError("rounds must be >=2 and warmup >=1")
    report: dict[str, Any] = {
        "schema": "fragpack.native-codec-roundtrip.v2",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preparing",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": _source_identity(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "cases": [],
    }
    _write(args.out, report)
    try:
        inputs = _load_inputs(args)
        report["inputs"] = inputs.provenance
        if args.prepare_only:
            tables_path = args.out.with_suffix(".tables.npz")
            np.savez(tables_path, table=inputs.table)
            report["tables_file"] = str(tables_path.resolve())
            report["status"] = "inputs_prepared_cpu_only"
        else:
            _run(args, inputs, report)
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
        _write(args.out, report)


if __name__ == "__main__":
    main()

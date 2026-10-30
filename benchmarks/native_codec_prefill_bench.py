# SPDX-License-Identifier: Apache-2.0
"""Replay 48 read-only attention layers over identical captured BF16 KV.

Compare native FA3 after one-time restoration, per-layer gather plus FA3, and
automatic split-KV fused prefill. Resident and actual pinned-host cold-read
windows are separate. This is not model inference or a new-KV write benchmark:
the context already includes the right-aligned query suffix, and all KV exists
before timing. Run only on an explicitly reserved exclusive CUDA device.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
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
from fragpack.compression.attn_tables import calibrate, device_tables
from fragpack.ops.fused_attn_decode import RecordTables
from fragpack.ops.fused_attn_prefill import (
    prefill_attention,
    prefill_split_count,
    prefill_workspace_elems,
)
from fragpack.ops.hbm_kv import HbmKvStore, page_bytes
from fragpack.ops.native_codec import NativeCodec, NativeKVPages

ARMS = ("native_fa3", "gather_fa3", "fused_prefill")
ATOL = RTOL = 2e-2
LAYERS, HEADS, QUERY_HEADS = 48, 4, 32
T, D = ap.PAGE_TOKENS, ap.HEAD_DIM
CANARY = 0x5A5A


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _sizes(value: str) -> list[int]:
    result = [int(part) for part in value.split(",")]
    if not result or min(result) < 1 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected distinct positive sizes")
    return result


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples": values,
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": min(values),
        "max": max(values),
    }


def _orders(rounds: int, reverse: bool) -> list[tuple[str, ...]]:
    # Every six rounds gives every arm every position twice and every pair
    # both relative orders three times. Reversal also applies to warmups.
    cycle = list(itertools.permutations(ARMS))
    return [
        tuple(reversed(cycle[index % 6])) if reverse else cycle[index % 6]
        for index in range(rounds)
    ]


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        path
        for path in (root / "fragpack").rglob("*")
        if path.suffix in {".py", ".cpp", ".cu", ".cuh", ".h"}
    ) + [Path(__file__).resolve()]
    hashes = {str(path.relative_to(root)): _hash_file(path) for path in paths}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return {
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "files_sha256": hashes,
        "digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    }


def _load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    dump = np.load(args.dump, mmap_mode="r", allow_pickle=False)
    if dump.dtype != np.uint16 or dump.shape[1:] != (LAYERS, 2, T, HEADS, D):
        raise ValueError("expected uint16 [blocks,48,2,128,4,128] capture")
    if any(context % T for context in args.contexts):
        raise ValueError("contexts must contain complete unique 128-token blocks")
    count = max(args.contexts) // T
    if args.calib_pages < 1 or args.eval_start < args.calib_pages:
        raise ValueError("calibration and evaluation positions must be disjoint")
    if args.eval_start + count > dump.shape[0]:
        raise ValueError("not enough unique evaluation blocks; duplication is disabled")
    if min(args.contexts) < max(args.queries):
        raise ValueError("every context must include its complete query suffix")
    table = calibrate(np.asarray(dump[: args.calib_pages]))
    content = np.array(dump[args.eval_start : args.eval_start + count])
    return (
        table,
        content,
        {
            "dump": str(args.dump.resolve()),
            "dump_basename": args.dump.name,
            "dump_sha256": _hash_file(args.dump),
            "shape": list(dump.shape),
            "dtype": str(dump.dtype),
            "calibration_range": [0, args.calib_pages],
            "evaluation_range": [args.eval_start, args.eval_start + count],
            "calibration_sha256": _hash_array(np.asarray(dump[: args.calib_pages])),
            "evaluation_sha256": _hash_array(content),
            "table_sha256": _hash_array(table),
            "description": args.input_description,
            "split_scope": (
                "Disjoint positions of one capture, not independent documents."
            ),
        },
    )


def _overflow_quota(lengths: np.ndarray) -> tuple[int, int, int]:
    """Derive live overflow and twice-live replacement capacity from record sizes."""
    raw_length = ap.RAW_RECORD_BYTES - ap.DENSE_BYTES
    needed = spills = 0
    for layer_records in lengths.reshape(-1, 2 * HEADS):
        filled = 0
        for variable in layer_records:
            size = ap.DENSE_BYTES + int(variable)
            if variable != raw_length and filled + size <= page_bytes(HEADS):
                filled += size
            else:
                needed += 1
                spills += int(variable != raw_length)
    # Existing import allocates replacement slots before freeing old records.
    return needed, spills, max(1, 2 * needed)


class _Context:
    """Own the common immutable payload, original/restored pages and old store."""

    def __init__(self, context: int, table: np.ndarray, content: np.ndarray) -> None:
        self.context = context
        self.blocks = context // T
        self.expected = np.ascontiguousarray(content[: self.blocks])
        self.ids = torch.arange(1, self.blocks + 1, dtype=torch.int32, device="cuda")
        self.mapping = self.ids[:, None].clone()
        shape = (2, self.blocks + 2, T, HEADS, D)

        def pages() -> NativeKVPages:
            tensors = [
                torch.empty(shape, dtype=torch.bfloat16, device="cuda")
                for _ in range(LAYERS)
            ]
            for tensor in tensors:
                tensor.view(torch.int16).fill_(CANARY)
            return NativeKVPages(tensors)

        self.source = pages()
        self.native = pages()
        for layer, cache in enumerate(self.source.caches):
            values = np.ascontiguousarray(
                self.expected[:, layer].transpose(1, 0, 2, 3, 4)
            )
            cache[:, 1 : self.blocks + 1].copy_(
                torch.from_numpy(values.view(np.int16)).to("cuda").view(torch.bfloat16)
            )
        self.codec = NativeCodec(table, max_blocks=self.blocks, device="cuda")
        self.codec.encode(self.source, self.mapping)
        torch.cuda.current_stream().synchronize()
        if bool(torch.any(self.codec.status.cpu() != 0)):
            raise RuntimeError("initial native encoding failed")
        offsets = self.codec.offsets.cpu().numpy().copy()
        self.record_lengths = self.codec.rec_lens.cpu().numpy().copy()
        self.payload_bytes = int(offsets[-1])
        if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise RuntimeError("invalid initial blob offsets")
        self.host_payload = torch.empty(
            self.payload_bytes, dtype=torch.uint8, pin_memory=True
        )
        self.host_offsets = torch.empty(
            self.blocks + 1, dtype=torch.int64, pin_memory=True
        )
        self.host_payload.copy_(
            self.codec.buffer[: self.payload_bytes], non_blocking=True
        )
        self.host_offsets.copy_(self.codec.offsets, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        needed, spills, quota = _overflow_quota(self.record_lengths)
        self.arenas = [
            torch.full(
                (self.blocks + 2, page_bytes(HEADS)),
                0xA5,
                dtype=torch.uint8,
                device="cuda",
            )
            for _ in range(LAYERS)
        ]
        self.store = HbmKvStore(
            self.arenas,
            HEADS,
            device_tables(table, torch.device("cuda")),
            max_tokens=T,
            staging_slots=1,
            overflow_records=quota,
        )
        if self.codec.digest != self.store.table_digest:
            raise RuntimeError("native and HBM codecs disagree on table identity")
        self.receive = {
            arm: torch.empty(self.payload_bytes, dtype=torch.uint8, device="cuda")
            for arm in ARMS
        }
        self.offsets_receive = {
            arm: torch.empty(self.blocks + 1, dtype=torch.int64, device="cuda")
            for arm in ARMS
        }
        self.status_host = {
            arm: torch.empty(self.blocks, dtype=torch.int32, pin_memory=True)
            for arm in ARMS
        }
        self.gather_k = torch.empty(
            self.blocks, T, HEADS, D, dtype=torch.bfloat16, device="cuda"
        )
        self.gather_v = torch.empty_like(self.gather_k)
        self.tables = [
            RecordTables(*self.store.layer_tables(layer)) for layer in range(LAYERS)
        ]
        self.ready = torch.cuda.Event()
        self.stream = torch.cuda.current_stream()
        self.stream.synchronize()
        for arm in ARMS:
            self.transfer(arm)
            status = self.restore(arm)
            self.status_host[arm].copy_(status, non_blocking=True)
            self.ready.record(self.stream)
            self.ready.synchronize()
            self.check_status(arm)
        validation = self.validate_kv()
        self.metadata = {
            "context": context,
            "unique_blocks": self.blocks,
            "input_bits_sha256": _hash_array(self.expected),
            "payload_sha256": _hash_array(self.host_payload.numpy()),
            "payload_bytes": self.payload_bytes,
            "raw_payload_bytes": self.expected.nbytes,
            "payload_ratio_raw_over_compressed": self.expected.nbytes
            / self.payload_bytes,
            "offsets": offsets.tolist(),
            "record_variable_lengths": self.record_lengths.tolist(),
            "raw_fallback_records": int(
                (self.record_lengths == ap.RAW_RECORD_BYTES - ap.DENSE_BYTES).sum()
            ),
            "old_store_expected_overflow_live_records": needed,
            "old_store_expected_arena_spill_records": spills,
            "old_store_overflow_capacity_records": quota,
            "kv_validation": validation,
            "memory_bytes": {
                "source_native": self.source.memory_usage(),
                "restored_native": self.native.memory_usage(),
                "native_codec_all_owned": self.codec.memory_usage(),
                "hbm_store": self.store.memory_usage(),
                "gather_workspace_reused_across_layers": _tensor_bytes(self.gather_k)
                + _tensor_bytes(self.gather_v),
                "receive_payload_per_arm": self.payload_bytes,
                "receive_offsets_per_arm": _tensor_bytes(self.offsets_receive[ARMS[0]]),
                "shared_pinned_payload": _tensor_bytes(self.host_payload),
                "shared_pinned_offsets": _tensor_bytes(self.host_offsets),
                "pinned_status_per_arm": _tensor_bytes(self.status_host[ARMS[0]]),
                "page_ids": _tensor_bytes(self.ids),
                "page_mapping": _tensor_bytes(self.mapping),
                "actual_torch_allocated_all_arms_setup": torch.cuda.memory_allocated(),
                "actual_torch_reserved_all_arms_setup": torch.cuda.memory_reserved(),
                "scope": (
                    "Arms coexist. Codebooks/store buffers and source encode scratch "
                    "remain allocated and counted, including unused encode buffers."
                ),
            },
        }

    def transfer(self, arm: str) -> None:
        """Enqueue real H2D payload and framing metadata on the current stream."""
        self.receive[arm].copy_(self.host_payload, non_blocking=True)
        self.offsets_receive[arm].copy_(self.host_offsets, non_blocking=True)

    def restore(self, arm: str) -> torch.Tensor:
        """Restore from this arm's received bytes; caller waits and checks status."""
        if arm == "native_fa3":
            return self.codec.decode(
                self.native,
                self.mapping,
                src=self.receive[arm],
                offsets=self.offsets_receive[arm],
            )
        return self.store.import_blocks(
            self.receive[arm], self.offsets_receive[arm][:-1], self.ids
        )

    def check_status(self, arm: str) -> None:
        if bool(torch.any(self.status_host[arm] != 0)):
            raise RuntimeError(f"{arm} restore status {self.status_host[arm].tolist()}")

    def validate_kv(self) -> dict[str, Any]:
        """Validate every selected BF16 bit and untouched pages outside timing."""
        for layer in range(LAYERS):
            native = (
                self.native.caches[layer]
                .view(torch.int16)
                .cpu()
                .numpy()
                .view(np.uint16)
            )
            expected = self.expected[:, layer].transpose(1, 0, 2, 3, 4)
            if not np.array_equal(native[:, 1 : self.blocks + 1], expected):
                raise AssertionError(f"native restore bits differ at layer {layer}")
            if not np.all(native[:, [0, self.blocks + 1]] == CANARY):
                raise AssertionError(
                    f"native restore overwrote unused layer {layer} pages"
                )
            self.gather_k.view(torch.int16).fill_(CANARY)
            self.gather_v.view(torch.int16).fill_(CANARY)
            self.store.gather(layer, self.ids, self.gather_k, self.gather_v)
            for kv, value in enumerate((self.gather_k, self.gather_v)):
                bits = value.view(torch.int16).cpu().numpy().view(np.uint16)
                if not np.array_equal(bits, self.expected[:, layer, kv]):
                    raise AssertionError(
                        f"old gather bits differ at layer {layer} kv {kv}"
                    )
            if not bool(
                torch.all(self.arenas[layer][[0, self.blocks + 1]] == 0xA5).cpu()
            ):
                raise AssertionError(f"old import overwrote unused layer {layer} pages")
        return {
            "layers_checked": LAYERS,
            "native_exact_bits": True,
            "old_gather_exact_bits": True,
            "unused_pages_unchanged": True,
            "store_stats": asdict(self.store.stats()),
        }


class _Attention:
    """Own queries, outputs and captured 48-layer readers for one C/Q case."""

    def __init__(
        self, context: _Context, queries: int, args: argparse.Namespace
    ) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        self.ctx = context
        self.queries = queries
        self.fa3 = flash_attn_varlen_func
        generator = torch.Generator().manual_seed(args.seed)
        host_q = torch.randn(LAYERS, queries, QUERY_HEADS, D, generator=generator).to(
            torch.bfloat16
        )
        self.query_sha256 = _hash_array(host_q.view(torch.int16).numpy())
        self.q = host_q.to("cuda")
        self.q_layers = list(self.q.unbind(0))
        self.outputs = {arm: torch.empty_like(self.q) for arm in ARMS}
        self.output_layers = {
            arm: list(value.unbind(0)) for arm, value in self.outputs.items()
        }
        self.native_key = [cache[0] for cache in context.native.caches]
        self.native_value = [cache[1] for cache in context.native.caches]
        self.table = context.ids[None]
        self.gather_table = torch.arange(
            context.blocks, dtype=torch.int32, device="cuda"
        )[None]
        self.cu_q = torch.tensor([0, queries], dtype=torch.int32, device="cuda")
        self.lengths = torch.tensor([context.context], dtype=torch.int32, device="cuda")
        self.splits = prefill_split_count(
            [queries],
            [context.context],
            HEADS,
            QUERY_HEADS // HEADS,
            torch.cuda.get_device_properties(0).multi_processor_count,
        )
        counts = prefill_workspace_elems(queries, QUERY_HEADS, self.splits)
        self.fused_workspace = (
            tuple(torch.empty(n, dtype=torch.float32, device="cuda") for n in counts)
            if self.splits > 1
            else None
        )
        self.done = torch.cuda.Event()
        self.status_ready = torch.cuda.Event()
        self.phase_events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        self.graph_start = torch.cuda.Event(enable_timing=True)
        self.graph_end = torch.cuda.Event(enable_timing=True)
        self.stream = torch.cuda.current_stream()
        self.stream.synchronize()
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}
        for arm in ARMS:
            for _ in range(args.warmup):
                self.read(arm)
        self.stream.synchronize()
        self.reference = self.outputs["native_fa3"].clone()
        self.validation = {"eager_before_capture": self.validate_outputs(graph=False)}
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        for arm in ARMS:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.read(arm)
            self.graphs[arm] = graph
        self.validation["graph_after_capture"] = self.validate_outputs(graph=True)
        self.memory = {
            "queries": _tensor_bytes(self.q),
            "output_per_arm": _tensor_bytes(self.outputs[ARMS[0]]),
            "validation_reference": _tensor_bytes(self.reference),
            "fused_workspace_reused_across_layers": sum(
                _tensor_bytes(t) for t in self.fused_workspace or ()
            ),
            "attention_metadata": sum(
                _tensor_bytes(t) for t in (self.gather_table, self.cu_q, self.lengths)
            ),
            "torch_allocated_before_graphs": allocated_before,
            "torch_allocated_after_graphs": torch.cuda.memory_allocated(),
            "torch_reserved_before_graphs": reserved_before,
            "torch_reserved_after_graphs": torch.cuda.memory_reserved(),
            "scope": (
                "Graph-private FA3 intermediates are included in observed "
                "allocated/reserved bytes; explicit buffers above are not a "
                "total allocator/context budget."
            ),
        }

    def _fa3(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        arm: str,
    ) -> None:
        self.fa3(
            q=self.q_layers[layer],
            k=key,
            v=value,
            out=self.output_layers[arm][layer],
            cu_seqlens_q=self.cu_q,
            max_seqlen_q=self.queries,
            seqused_k=self.lengths,
            max_seqlen_k=self.ctx.context,
            softmax_scale=D**-0.5,
            causal=True,
            block_table=table,
            fa_version=3,
            num_splits=0,
        )

    def read(self, arm: str) -> None:
        """Enqueue the whole 48-layer reader without allocating a KV workspace."""
        for layer in range(LAYERS):
            if arm == "native_fa3":
                self._fa3(
                    layer,
                    self.native_key[layer],
                    self.native_value[layer],
                    self.table,
                    arm,
                )
            elif arm == "gather_fa3":
                self.ctx.store.gather(
                    layer, self.ctx.ids, self.ctx.gather_k, self.ctx.gather_v
                )
                self._fa3(
                    layer, self.ctx.gather_k, self.ctx.gather_v, self.gather_table, arm
                )
            else:
                prefill_attention(
                    self.q_layers[layer],
                    self.ctx.tables[layer],
                    self.table,
                    self.cu_q,
                    self.lengths,
                    self.queries,
                    D**-0.5,
                    self.output_layers[arm][layer],
                    num_splits=self.splits,
                    workspace=self.fused_workspace,
                )

    def validate_outputs(self, *, graph: bool) -> dict[str, Any]:
        """Check numerical agreement and that every output row is overwritten."""
        results = {}
        for arm in ARMS:
            self.outputs[arm].fill_(float("nan"))
            if graph:
                self.graphs[arm].replay()
            else:
                self.read(arm)
            torch.testing.assert_close(
                self.outputs[arm], self.reference, atol=ATOL, rtol=RTOL
            )
            equal = torch.equal(
                self.outputs[arm].view(torch.int16), self.reference.view(torch.int16)
            )
            if arm != "fused_prefill" and not equal:
                raise AssertionError(f"{arm} output is not bitwise equal to native FA3")
            results[arm] = {
                "bitwise_equal": equal,
                "max_abs": float(
                    (self.outputs[arm].float() - self.reference.float()).abs().max()
                ),
                "atol": ATOL,
                "rtol": RTOL,
            }
        return results

    def sample(
        self, arm: str, mode: str, *, diagnostic: bool = False
    ) -> dict[str, Any]:
        """Time one complete 48-layer call, optionally including a cold restore."""
        cold = mode.startswith("cold")
        graph = mode.endswith("graph")
        events = self.phase_events
        started = time.perf_counter_ns()
        if diagnostic:
            events[0].record(self.stream)
        if cold:
            self.ctx.transfer(arm)
            if diagnostic:
                events[1].record(self.stream)
            status = self.ctx.restore(arm)
            if diagnostic:
                events[2].record(self.stream)
            self.ctx.status_host[arm].copy_(status, non_blocking=True)
            if diagnostic:
                events[3].record(self.stream)
            self.status_ready.record(self.stream)
            waiting = time.perf_counter_ns()
            self.status_ready.synchronize()
            self.ctx.check_status(arm)
            ready = time.perf_counter_ns()
        else:
            waiting = ready = started
        if diagnostic:
            events[4].record(self.stream)
        if graph and not cold and not diagnostic:
            self.graph_start.record(self.stream)
        if graph:
            self.graphs[arm].replay()
        else:
            self.read(arm)
        if diagnostic:
            events[5].record(self.stream)
        if graph and not cold and not diagnostic:
            self.graph_end.record(self.stream)
            self.graph_end.synchronize()
        else:
            self.done.record(self.stream)
            self.done.synchronize()
        finished = time.perf_counter_ns()
        result: dict[str, Any] = {
            "arm": arm,
            "mode": mode,
            "diagnostic": diagnostic,
            "wall_us": (finished - started) / 1000,
        }
        if graph and not cold and not diagnostic:
            result["gpu_us"] = self.graph_start.elapsed_time(self.graph_end) * 1000
        if cold:
            result.update(
                {
                    "h2d_restore_ready_wall_us": (ready - started) / 1000,
                    "status_barrier_wait_check_us": (ready - waiting) / 1000,
                    "read_wall_us": (finished - ready) / 1000,
                    "payload_h2d_bytes": self.ctx.payload_bytes,
                    "offsets_h2d_bytes": _tensor_bytes(self.ctx.host_offsets),
                    "status_d2h_bytes": _tensor_bytes(self.ctx.status_host[arm]),
                }
            )
        if diagnostic:
            spans = {
                "read": events[4].elapsed_time(events[5]) * 1000,
                "full_stream_span": events[0].elapsed_time(events[5]) * 1000,
            }
            if cold:
                spans.update(
                    {
                        "h2d_payload_offsets": events[0].elapsed_time(events[1]) * 1000,
                        "restore": events[1].elapsed_time(events[2]) * 1000,
                        "status_d2h": events[2].elapsed_time(events[3]) * 1000,
                    }
                )
            result["cuda_spans_us"] = spans
        return result


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ARMS:
        rows = [row for row in samples if row["arm"] == arm]
        keys = (
            "wall_us",
            "gpu_us",
            "h2d_restore_ready_wall_us",
            "status_barrier_wait_check_us",
            "read_wall_us",
        )
        result[arm] = {
            key: _summary([row[key] for row in rows]) for key in keys if key in rows[0]
        }
        if "cuda_spans_us" in rows[0]:
            result[arm]["cuda_spans_us"] = {
                key: _summary([row["cuda_spans_us"][key] for row in rows])
                for key in rows[0]["cuda_spans_us"]
            }
    return result


def _run(
    args: argparse.Namespace,
    table: np.ndarray,
    content: np.ndarray,
    report: dict[str, Any],
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one explicitly authorized CUDA device")
    properties = torch.cuda.get_device_properties(0)
    report["environment"].update(
        {
            "gpu_name": properties.name,
            "gpu_uuid": str(properties.uuid),
            "sm_count": properties.multi_processor_count,
            "compute_capability": [properties.major, properties.minor],
            "gpu_memory_bytes": properties.total_memory,
        }
    )
    modes = ["resident_eager", "resident_graph", "cold_graph"]
    if args.cold_eager:
        modes.append("cold_eager")
    for context in args.contexts:
        context_row: dict[str, Any] = {
            "context": context,
            "status": "preparing",
            "cases": [],
        }
        report["contexts"].append(context_row)
        _write(args.out, report)
        ctx = _Context(context, table, content)
        context_row["setup"] = ctx.metadata
        for queries in args.queries:
            row: dict[str, Any] = {
                "queries": queries,
                "status": "preparing",
                "measurements": {},
            }
            context_row["cases"].append(row)
            _write(args.out, report)
            attention = _Attention(ctx, queries, args)
            row.update(
                {
                    "query_sha256": attention.query_sha256,
                    "automatic_fused_splits": attention.splits,
                    "memory_bytes": attention.memory,
                    "validation": attention.validation,
                }
            )
            for mode in modes:
                measurement: dict[str, Any] = {
                    "warmup_samples": [],
                    "samples": [],
                    "diagnostic_warmup_samples": [],
                    "diagnostic_samples": [],
                }
                row["measurements"][mode] = measurement
                for order in _orders(args.warmup, args.order == "reverse"):
                    for arm in order:
                        measurement["warmup_samples"].append(
                            attention.sample(arm, mode)
                        )
                for index, order in enumerate(
                    _orders(args.rounds, args.order == "reverse")
                ):
                    for position, arm in enumerate(order):
                        sample = attention.sample(arm, mode)
                        sample.update(
                            {"round": index, "position": position, "order": order}
                        )
                        measurement["samples"].append(sample)
                measurement["summary"] = _aggregate(measurement["samples"])
                for arm in ARMS:
                    measurement["diagnostic_warmup_samples"].append(
                        attention.sample(arm, mode, diagnostic=True)
                    )
                for index, order in enumerate(_orders(3, args.order == "reverse")):
                    for position, arm in enumerate(order):
                        sample = attention.sample(arm, mode, diagnostic=True)
                        sample.update(
                            {"round": index, "position": position, "order": order}
                        )
                        measurement["diagnostic_samples"].append(sample)
                measurement["diagnostic_summary"] = _aggregate(
                    measurement["diagnostic_samples"]
                )
                measurement["output_validation"] = attention.validate_outputs(
                    graph=mode.endswith("graph")
                )
                if mode.startswith("cold"):
                    measurement["restored_kv_validation"] = ctx.validate_kv()
            row["status"] = "passed"
            print(
                json.dumps(
                    {
                        "context": context,
                        "queries": queries,
                        "splits": attention.splits,
                        "medians": {
                            mode: {
                                arm: row["measurements"][mode]["summary"][arm][
                                    "wall_us"
                                ]["median"]
                                for arm in ARMS
                            }
                            for mode in modes
                        },
                    }
                ),
                flush=True,
            )
            _write(args.out, report)
            del attention
        context_row["status"] = "passed"
        del ctx


def main() -> None:
    """Run an authorized replay and retain partial measurements on failure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", type=_sizes, default=[512, 2048])
    parser.add_argument("--queries", type=_sizes, default=[16, 128, 512])
    parser.add_argument("--calib-pages", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--order", choices=("mixed", "reverse"), default="mixed")
    parser.add_argument("--cold-eager", action="store_true")
    parser.add_argument(
        "--input-description",
        default="Captured BF16 KV; specify the capture and model provenance.",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 1:
        raise ValueError("positive rounds and warmup are required")
    report: dict[str, Any] = {
        "schema": "fragpack.native-codec-prefill-replay.v1",
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": _source_identity(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "vllm": importlib.metadata.version("vllm"),
        },
        "protocol": {
            "work": (
                "One combined ordered replay of 48 attention layers, independent "
                "synthetic per-layer queries and captured BF16 KV. C includes "
                "the right-aligned Q suffix. All KV is already complete."
            ),
            "arms": {
                "native_fa3": "Restore into raw native pages once, then original FA3",
                "gather_fa3": (
                    "Restore compressed records into HbmKvStore; each layer "
                    "gathers BF16 then calls identical FA3"
                ),
                "fused_prefill": (
                    "Same HbmKvStore; current automatic split-KV "
                    "prefill_attention directly decodes inside attention"
                ),
            },
            "resident_eager": (
                "Complete host wall for 48 enqueued layer calls plus completion; "
                "only one completion event in primary samples."
            ),
            "resident_graph": (
                "CUDA-event time for one captured full 48-layer graph plus "
                "separately labeled host wall; no sum of layer medians."
            ),
            "cold": (
                "Identical compact FPKB already pinned on host, real payload "
                "and offsets H2D into separate arm buffers, restore, status "
                "D2H/event wait/check, then complete 48-layer attention and "
                "completion. Default attention uses its captured graph. "
                "Primary cold samples have only required completion events."
            ),
            "diagnostics": (
                "Three separate phase-instrumented calls per arm/mode after an "
                "excluded diagnostic warmup. CUDA spans include host launch gaps; "
                "status barrier host wait overlaps preceding queued GPU work "
                "and is not additive."
            ),
            "validation_asymmetry": (
                "Trusted identical valid FPKB. Native decode validates complete "
                "record payload structure; old import validates blob header/record "
                "lengths. Old import allocates its returned status tensor; native "
                "codec reuses owned status. These public API costs remain in "
                "cold timing."
            ),
            "excluded": (
                "Initial encoding, D2H, calibration, allocation, query generation, "
                "KV initialization, correctness checks, QKV projections, "
                "KV writes, model layer dependencies, serving/scheduler/RPC/SSD."
            ),
            "statistics": (
                "Warm-buffer repeated samples in one process, not independent "
                "requests or process repetitions. Each six rounds balances all "
                "three arm positions and pair orders. Below 30 rounds is "
                "diagnostic only."
            ),
            "memory_ownership": (
                "Source and restored native pages, common old store, three "
                "independent receive buffers, shared pinned source payload, "
                "per-arm outputs, and layer-reused gather/fused workspaces "
                "coexist. Only allocation addresses are reused between layers; "
                "KV contents remain distinct."
            ),
        },
        "contexts": [],
    }
    _write(args.out, report)
    try:
        table, content, provenance = _load_inputs(args)
        report["inputs"] = provenance
        _run(args, table, content, report)
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

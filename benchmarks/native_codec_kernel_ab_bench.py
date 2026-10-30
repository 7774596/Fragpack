# SPDX-License-Identifier: Apache-2.0
"""Compare two real, fully validated native-codec CUDA implementations.

Both modules restore identical resident FPKB bytes into the same native pages.
Host exports are uniquely renamed in generated translation units to prevent ELF
symbol interposition. Graph-internal timing events measure GPU restore kernels,
optionally followed by all 48 stock FA3 layers; never Python calls, transfers or
end-to-end serving. GPU execution needs
an explicitly reserved exclusive device; --prepare-only performs no CUDA work.
Forced-fallback patterns reuse the scale benchmark's finite out-of-table records.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import traceback
from typing import Any, Callable

import numpy as np
import torch

from benchmarks.native_codec_scale_bench import expand_capture

# Reuse only the audited CPU capture/calibration and summary helpers. The stage
# profiler's source-hash gate, stage ablations, FA3 context and JIT are not used.
from benchmarks.native_codec_stage_profiler import (
    HEADS,
    LAYERS,
    POISON,
    QUERY_HEADS,
    D,
    T,
    _hash_array,
    _hash_file,
    _ints,
    _load_inputs,
    _summary,
    _write,
)
from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import DeviceTables, device_tables
from fragpack.ops.native_codec import NativeKVPages

ROOT = Path(__file__).resolve().parents[1]
HOST_EXPORTS = (
    "encode_records",
    "plan_blobs",
    "pack_blobs",
    "decode_blobs",
    "raw_gather",
    "raw_scatter",
)
FLAGS = ["-O3", "--use_fast_math", "-lineinfo"]


def _pairs(value: str) -> list[tuple[int, int]]:
    """Parse exact FPKB-block/native-page-size pairs without a Cartesian product."""
    try:
        result = [
            tuple(int(part) for part in pair.split(":")) for pair in value.split(",")
        ]
        if not result or any(len(pair) != 2 for pair in result):
            raise ValueError
        return [(pair[0], pair[1]) for pair in result]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected N:P pairs, e.g. 1:128,256:16"
        ) from error


def _case_specs(args: argparse.Namespace) -> list[tuple[int, int]]:
    """Resolve legacy defaults or the optional read subset, rejecting ambiguity."""
    explicit_geometry = args.blocks is not None or args.page_tokens is not None
    if args.case_pairs is not None and explicit_geometry:
        raise ValueError("--case-pairs cannot be combined with --blocks/--page-tokens")
    if args.case_pairs is not None:
        pairs = args.case_pairs
    elif args.read_queries and not explicit_geometry:
        pairs = [(1, 128), (16, 128), (256, 128), (256, 16)]
    else:
        blocks = args.blocks if args.blocks is not None else [1, 4, 16, 64, 256]
        pages = args.page_tokens if args.page_tokens is not None else [16, 128]
        pairs = [(count, page) for count in blocks for page in pages]
    if (
        not pairs
        or len(set(pairs)) != len(pairs)
        or any(count < 1 or page not in (16, 32, 64, 128) for count, page in pairs)
    ):
        raise ValueError("invalid or duplicated block/native-page geometry")
    queries = args.read_queries or []
    if len(set(queries)) != len(queries) or any(
        query < 1 or query > min(count for count, _ in pairs) * T for query in queries
    ):
        raise ValueError("read queries must be unique, positive and no longer than KV")
    return pairs


def _fa3_identity() -> dict[str, Any]:
    """Identify the actual imported stock FA3 wrapper and native implementation."""
    names = (
        "vllm.vllm_flash_attn",
        "vllm.vllm_flash_attn.flash_attn_interface",
        "vllm.vllm_flash_attn._vllm_fa3_C",
    )
    files = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        files[str(path)] = _hash_file(path)
    return {
        "vllm_version": importlib.metadata.version("vllm"),
        "files_sha256": files,
        "api": "vllm.vllm_flash_attn.flash_attn_varlen_func",
        "fa_version": 3,
        "num_splits": 0,
        "causal": True,
    }


def _describe_sources(cuda: Path, role: str, build: Path) -> dict[str, Any]:
    """Write a source-preserving, uniquely renamed TU without compiling it."""
    cuda = cuda.resolve(strict=True)
    files = {
        "cuda": cuda,
        "binding": cuda.with_name("native_codec_ext.cpp"),
        "common_header": cuda.with_name("fragpack_common.cuh"),
    }
    hashes = {name: _hash_file(path) for name, path in files.items()}
    binding = files["binding"].read_text(encoding="utf-8")
    for name in HOST_EXPORTS:
        if f'"{name}"' not in binding or f"&{name}" not in binding:
            raise ValueError(f"missing native-codec binding {name}: {files['binding']}")
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    module = f"fragpack_native_codec_ab_{role}_{identity[:16]}"
    prefix = f"{role}_{identity[:16]}"
    source = "\n".join(
        [
            "// SPDX-License-Identifier: Apache-2.0",
            "// Generated TU: original CUDA and binding, renamed host exports.",
            *(f"#define {name} {prefix}_{name}" for name in HOST_EXPORTS),
            f"#include {json.dumps(str(files['cuda']))}",
            f"#include {json.dumps(str(files['binding']))}",
            *(f"#undef {name}" for name in HOST_EXPORTS),
            "",
        ]
    )
    generated = build / "generated" / f"{module}.cu"
    generated.parent.mkdir(parents=True, exist_ok=True)
    if generated.exists():
        if generated.read_text(encoding="utf-8") != source:
            raise RuntimeError(f"generated source identity collision: {generated}")
    else:
        generated.write_text(source, encoding="utf-8")
    return {
        "role": role,
        "module_name": module,
        "host_symbol_prefix": prefix,
        "paths": {name: str(path) for name, path in files.items()},
        "files_sha256": hashes,
        "generated_tu": str(generated),
        "generated_tu_sha256": _hash_file(generated),
        "identity": identity,
    }


def _source_pair(args: argparse.Namespace) -> dict[str, Any]:
    """Require identical binding/header contracts while allowing different CUDA."""
    build = args.build_root.resolve()
    if build.is_relative_to(ROOT):
        raise ValueError("--build-root must be outside the source worktree")
    pair = {
        role: _describe_sources(getattr(args, f"{role}_cuda"), role, build)
        for role in ("baseline", "candidate")
    }
    for name in ("binding", "common_header"):
        if (
            pair["baseline"]["files_sha256"][name]
            != pair["candidate"]["files_sha256"][name]
        ):
            raise ValueError(
                f"baseline/candidate {name} differ; ABI/format not matched"
            )
    files = [
        Path(__file__),
        ROOT / "benchmarks/native_codec_scale_bench.py",
        ROOT / "benchmarks/native_codec_bench.py",
        ROOT / "benchmarks/native_codec_stage_profiler.py",
        ROOT / "fragpack/ops/native_codec.py",
        ROOT / "fragpack/compression/attn_tables.py",
        ROOT / "fragpack/compression/attn_page.py",
    ]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "git_head": head,
        "driver_files_sha256": {
            str(path.relative_to(ROOT)): _hash_file(path) for path in files
        },
        "modules": pair,
        "binding_and_header_identical": True,
        "distinct_cuda_sources": pair["baseline"]["files_sha256"]["cuda"]
        != pair["candidate"]["files_sha256"]["cuda"],
    }


def _load_module(identity: dict[str, Any], build: Path) -> Any:
    """Compile only the generated actual-source TU, with production CUDA flags."""
    from torch.utils.cpp_extension import load

    directory = build / "jit" / identity["module_name"]
    directory.mkdir(parents=True, exist_ok=True)
    return load(
        name=identity["module_name"],
        sources=[identity["generated_tu"]],
        extra_cuda_cflags=FLAGS,
        build_directory=str(directory),
        verbose=False,
    )


def _verify_sources(source: dict[str, Any]) -> None:
    """Reject source changes between identification, compilation and completion."""
    for identity in source["modules"].values():
        for name, path in identity["paths"].items():
            if _hash_file(Path(path)) != identity["files_sha256"][name]:
                raise RuntimeError(f"A/B source changed during the run: {path}")
        if (
            _hash_file(Path(identity["generated_tu"]))
            != identity["generated_tu_sha256"]
        ):
            raise RuntimeError("generated A/B translation unit changed during the run")
    for path, digest in source["driver_files_sha256"].items():
        if _hash_file(ROOT / path) != digest:
            raise RuntimeError(f"A/B driver dependency changed during the run: {path}")
    for path, digest in source.get("attention", {}).get("files_sha256", {}).items():
        if _hash_file(Path(path)) != digest:
            raise RuntimeError(f"stock FA3 dependency changed during the run: {path}")


def _archive(
    expected: np.ndarray,
    page_tokens: int,
    tables: DeviceTables,
    digest: int,
    baseline: Any,
    raw_control: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Encode shared inputs using the renamed baseline module, outside timing."""
    blocks, parts = len(expected), T // page_tokens
    count = blocks * parts
    caches = []
    for layer in range(LAYERS):
        values = np.ascontiguousarray(expected[:, layer].transpose(1, 0, 2, 3, 4))
        values = values.reshape(2, count, page_tokens, HEADS, D)
        caches.append(
            torch.from_numpy(values.view(np.int16)).to("cuda").view(torch.bfloat16)
        )
    source = NativeKVPages(caches)
    mapping = torch.arange(count, dtype=torch.int32, device="cuda").reshape(
        blocks, parts
    )
    shape = (blocks, LAYERS, 2, HEADS)
    records = torch.empty(
        (*shape, ap.RAW_RECORD_BYTES), dtype=torch.uint8, device="cuda"
    )
    lengths = torch.empty(shape, dtype=torch.int32, device="cuda")
    record_offsets = torch.empty(shape, dtype=torch.int64, device="cuda")
    offsets = torch.empty(blocks + 1, dtype=torch.int64, device="cuda")
    blob_lengths = torch.empty(blocks, dtype=torch.int64, device="cuda")
    status = torch.empty(blocks, dtype=torch.int32, device="cuda")
    output = torch.empty(
        blocks * ap.blob_slot_bytes(LAYERS, HEADS), dtype=torch.uint8, device="cuda"
    )
    baseline.encode_records(
        source.descriptors,
        mapping,
        tables.rank_of,
        records,
        lengths,
        status,
        page_tokens,
        HEADS,
    )
    baseline.plan_blobs(
        lengths, status, offsets, blob_lengths, record_offsets, LAYERS, HEADS
    )
    baseline.pack_blobs(
        records,
        lengths,
        status,
        offsets,
        blob_lengths,
        record_offsets,
        output,
        LAYERS,
        HEADS,
        digest,
    )
    if bool(status.any()):
        raise RuntimeError("baseline archive encoding failed")
    size = int(offsets[-1])
    payload = output[:size].clone()
    raw = None
    if raw_control:
        raw = torch.empty(expected.nbytes, dtype=torch.uint8, device="cuda")
        baseline.raw_gather(
            source.descriptors, mapping, raw, status, page_tokens, HEADS
        )
        if bool(status.any()):
            raise RuntimeError("baseline raw archive gather failed")
    lens = lengths.cpu().numpy()
    fallback_records = int(
        np.count_nonzero(lens == ap.RAW_RECORD_BYTES - ap.DENSE_BYTES)
    )
    archive = {
        "encoder": "renamed_baseline_module",
        "payload_bytes": size,
        "raw_bytes": int(expected.nbytes),
        "fallback_records": fallback_records,
        "fallback_fraction": fallback_records / lens.size,
        "records": int(lens.size),
        "offsets_sha256": _hash_array(offsets.cpu().numpy()),
        "record_lengths_sha256": _hash_array(lens),
        "raw_source": "baseline raw_gather; checked by restore against prepared bits"
        if raw_control
        else None,
        "setup_source_and_encoder_buffers_released_before_timing": True,
    }
    torch.cuda.current_stream().synchronize()
    return payload, offsets, raw, archive


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    start: torch.cuda.Event
    end: torch.cuda.Event


def _native_source_layer(
    source: np.ndarray, layer: int, page_tokens: int
) -> np.ndarray:
    """Return logical native pages directly from CPU capture bits, without codec."""
    if page_tokens not in (16, 32, 64, 128):
        raise ValueError("unsupported native page size")
    values = np.ascontiguousarray(source[:, layer].transpose(1, 0, 2, 3, 4))
    return values.reshape(
        2, len(source) * T // page_tokens, page_tokens, *source.shape[-2:]
    )


def _assert_bf16_bits(actual: torch.Tensor, reference: torch.Tensor) -> None:
    """Compare BF16 storage bits, including signed zero and NaN payloads."""
    if actual.dtype != torch.bfloat16 or reference.dtype != torch.bfloat16:
        raise AssertionError("attention comparison requires BF16 tensors")
    if actual.shape != reference.shape or not torch.equal(
        actual.view(torch.int16), reference.view(torch.int16)
    ):
        raise AssertionError("attention output bits differ from original-source FA3")


class _Read:
    """One query shape's shared FA3 storage and independent CPU output oracle.

    Both A/B graphs use these same Q/output addresses and the caller's original
    mapping storage. Reference pages are populated from CPU bits and released
    before capture; references remain on CPU. All GPU work uses the current
    stream, with blocking CPU copies only during untimed preparation/validation.
    """

    def __init__(
        self,
        expected: np.ndarray,
        caches: list[torch.Tensor],
        mapping: torch.Tensor,
        query: int,
        seed: int,
        page_tokens: int,
    ) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        self.fa3 = flash_attn_varlen_func
        self.query, self.context = query, len(expected) * T
        host_q = torch.randn(
            LAYERS, query, QUERY_HEADS, D, generator=torch.Generator().manual_seed(seed)
        ).to(torch.bfloat16)
        self.q = host_q.to("cuda")
        self.output = torch.empty_like(self.q)
        self.poison()
        self.q_layers = list(self.q.unbind(0))
        self.output_layers = list(self.output.unbind(0))
        self.keys = [cache[0] for cache in caches]
        self.values = [cache[1] for cache in caches]
        # Alias the actual restore mapping: logical page order must not be
        # replaced by physical-address order, especially for P=16 subpages.
        self.block_table = mapping.reshape(1, -1)
        self.cu_q = torch.tensor([0, query], dtype=torch.int32, device="cuda")
        self.length = torch.tensor([self.context], dtype=torch.int32, device="cuda")
        self.reference = torch.empty_like(host_q)
        reference_cache = torch.empty_like(caches[0])
        reference_table = torch.arange(
            1, mapping.numel() + 1, dtype=torch.int32, device="cuda"
        )[None]
        for layer in range(LAYERS):
            bits = _native_source_layer(expected, layer, page_tokens)
            reference_cache[:, 1:-1].copy_(
                torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
            )
            self.invoke(layer, reference_cache[0], reference_cache[1], reference_table)
            self.reference[layer].copy_(self.output_layers[layer].cpu())
        if not bool(torch.isfinite(self.reference).all()):
            raise AssertionError("original-source attention reference is not finite")
        self.identity = {
            "query_tokens": query,
            "kv_tokens": self.context,
            "attention_layers": LAYERS,
            "query_heads": QUERY_HEADS,
            "query_sha256": _hash_array(host_q.view(torch.int16).numpy()),
            "reference_output_sha256": _hash_array(
                self.reference.view(torch.int16).numpy()
            ),
            "reference": "stock FA3 on original CPU source bits; no codec in oracle",
            "same_storage_for_both_roles": {
                "query": self.q.data_ptr(),
                "output": self.output.data_ptr(),
                "block_table": self.block_table.data_ptr(),
                "restore_mapping": mapping.data_ptr(),
                "native_caches": [cache.data_ptr() for cache in caches],
            },
            "block_table": "view of actual permutation mapping.flatten(), not arange",
        }
        self.memory = {
            "shared_query": self.q.numel() * self.q.element_size(),
            "shared_output": self.output.numel() * self.output.element_size(),
            "query_length_metadata": (self.cu_q.numel() + self.length.numel()) * 4,
            "block_table_extra_allocation": 0,
            "cpu_reference": self.reference.numel() * self.reference.element_size(),
            "reference_gpu_pages_released_before_capture": True,
        }

    def invoke(
        self, layer: int, key: torch.Tensor, value: torch.Tensor, table: torch.Tensor
    ) -> None:
        self.fa3(
            q=self.q_layers[layer],
            k=key,
            v=value,
            out=self.output_layers[layer],
            cu_seqlens_q=self.cu_q,
            max_seqlen_q=self.query,
            seqused_k=self.length,
            max_seqlen_k=self.context,
            softmax_scale=D**-0.5,
            causal=True,
            block_table=table,
            fa_version=3,
            num_splits=0,
        )

    def run(self) -> None:
        """Enqueue all 48 stock attention calls after restore on the same stream."""
        for layer in range(LAYERS):
            self.invoke(layer, self.keys[layer], self.values[layer], self.block_table)

    def poison(self) -> None:
        self.output.view(torch.int16).fill_(
            0x7FC0
        )  # BF16 quiet NaN, not a valid oracle.

    def validate(self) -> dict[str, Any]:
        """Check every layer's output storage bits against the independent oracle."""
        _assert_bf16_bits(self.output.cpu(), self.reference)
        return {
            "all_layer_output_bits_exact": True,
            "all_outputs_poisoned_before_restore": True,
            "compared_output_bytes": self.reference.numel()
            * self.reference.element_size(),
            "attention_layers": LAYERS,
            "query_tokens": self.query,
            "max_abs_error": 0.0,
        }


class _Case:
    """Shared pages, payload, mapping and scratch, sequentially used by both arms."""

    def __init__(
        self,
        args: argparse.Namespace,
        tables: DeviceTables,
        digest: int,
        pool: np.ndarray,
        table: np.ndarray,
        blocks: int,
        page_tokens: int,
        modules: dict[str, Any],
    ) -> None:
        self.blocks, self.parts = blocks, T // page_tokens
        self.page_tokens = page_tokens
        indices = np.arange(blocks) % len(pool)
        self.expected, construction = expand_capture(
            pool, table, blocks, args.source_pattern
        )
        self.input = {
            "physical_blocks": blocks,
            "page_tokens": page_tokens,
            "capture_indices_in_eval_pool": indices.tolist(),
            "synthetic_repetition": blocks > len(pool),
            "materialized_sha256": _hash_array(self.expected),
            "construction": construction,
        }
        self.payload, self.offsets, self.raw, self.archive = _archive(
            self.expected,
            page_tokens,
            tables,
            digest,
            modules["baseline"],
            args.raw_scatter_control,
        )
        forced = construction["forced_raw_records"]
        self.archive["forced_raw_records_minimum"] = forced
        if self.archive["fallback_records"] < forced or (
            args.source_pattern == "fallback100"
            and self.archive["fallback_fraction"] != 1.0
        ):
            raise AssertionError(
                "baseline archive did not preserve the forced fallback count: "
                f"actual={self.archive['fallback_records']}, forced={forced}"
            )
        gc.collect()
        torch.cuda.empty_cache()
        if args.read_queries:
            torch.cuda.reset_peak_memory_stats()
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
        self.input["mapping_sha256"] = _hash_array(self.ids.astype(np.int32))
        self.record_offsets = torch.empty(
            (blocks, LAYERS, 2, HEADS), dtype=torch.int64, device="cuda"
        )
        self.status = torch.empty(blocks, dtype=torch.int32, device="cuda")
        arguments = (
            self.native.descriptors,
            self.mapping,
            tables.plut,
            tables.slut,
            self.payload,
            self.offsets,
            self.record_offsets,
            self.status,
            page_tokens,
            HEADS,
            digest,
        )
        self.operations: dict[str, Callable[[], None]] = {}
        for role, module in modules.items():
            self.operations[f"{role}_decode"] = self._bind(
                module.decode_blobs, arguments
            )
            if args.raw_scatter_control:
                self.operations[f"{role}_raw_scatter"] = self._bind(
                    module.raw_scatter,
                    (
                        self.native.descriptors,
                        self.mapping,
                        self.raw,
                        self.status,
                        page_tokens,
                        HEADS,
                    ),
                )
        self.reads: dict[int, _Read] = {}
        self.operation_reads: dict[str, _Read] = {}
        for query in args.read_queries or []:
            reader = _Read(
                self.expected, self.caches, self.mapping, query, args.seed, page_tokens
            )
            self.reads[query] = reader
            for role in modules:
                name = f"{role}_decode_fa3_q{query}"
                self.operations[name] = self._bind_read(
                    self.operations[f"{role}_decode"], reader
                )
                self.operation_reads[name] = reader
        if self.reads:
            self.input["attention"] = {
                str(query): reader.identity for query, reader in self.reads.items()
            }
        self.validation = {name: self.validate(name) for name in self.operations}
        self.rejection = {
            role: self.validate_rejection(f"{role}_decode") for role in modules
        }
        for operation in self.operations.values():
            for _ in range(args.warmup):
                operation()
        torch.cuda.current_stream().synchronize()
        before_graphs = torch.cuda.memory_allocated()
        graph_allocations = {}
        self.graphs: dict[str, _Graph] = {}
        for name, operation in self.operations.items():
            before = torch.cuda.memory_allocated()
            graph = torch.cuda.CUDAGraph()
            start, end = (
                torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)
            )
            with torch.cuda.graph(graph):
                start.record()
                operation()
                end.record()
            self.graphs[name] = _Graph(graph, start, end)
            graph_allocations[name] = torch.cuda.memory_allocated() - before
        self.graph_validation = {
            name: self.validate(name, graph=True) for name in self.operations
        }
        for graph in self.graphs.values():
            for _ in range(args.warmup):
                graph.graph.replay()
        torch.cuda.current_stream().synchronize()
        self.memory = {
            "native_pages_and_descriptors": self.native.memory_usage(),
            "received_payload": self.payload.numel(),
            "received_offsets": self.offsets.numel() * self.offsets.element_size(),
            "raw_control_payload": self.raw.numel() if self.raw is not None else 0,
            "shared_record_offsets": self.record_offsets.numel()
            * self.record_offsets.element_size(),
            "shared_status": self.status.numel() * self.status.element_size(),
            "shared_tables": sum(
                t.numel() * t.element_size()
                for t in (tables.rank_of, tables.plut, tables.slut)
            ),
            "torch_allocated_after_graphs": torch.cuda.memory_allocated(),
            "torch_reserved_after_graphs": torch.cuda.memory_reserved(),
            "scope": "A/B diagnostic context; not a deployment memory measurement",
        }
        if self.reads:
            graph_pools = {}
            for name, measured in self.graphs.items():
                pool_id = measured.graph.pool()
                segments = torch.cuda.memory_snapshot(
                    mempool_id=pool_id, include_traces=False
                )
                graph_pools[name] = {
                    "pool_id": list(pool_id),
                    "reserved_bytes": sum(s["total_size"] for s in segments),
                    "allocated_bytes": sum(s["allocated_size"] for s in segments),
                    "active_bytes": sum(s["active_size"] for s in segments),
                }
            self.memory.update(
                {
                    "attention": {
                        str(q): reader.memory for q, reader in self.reads.items()
                    },
                    "torch_allocated_before_graphs": before_graphs,
                    "graph_capture_allocated_delta_by_operation": graph_allocations,
                    "graph_private_pools": graph_pools,
                    "torch_peak_since_archive_preparation": (
                        torch.cuda.max_memory_allocated()
                    ),
                    "peak_scope": (
                        "shared A/B buffers, reference preparation, validation "
                        "and graphs; excludes archive construction"
                    ),
                }
            )

    @staticmethod
    def _bind(
        function: Callable[..., None], arguments: tuple[Any, ...]
    ) -> Callable[[], None]:
        return lambda: function(*arguments)

    @staticmethod
    def _bind_read(restore: Callable[[], None], reader: _Read) -> Callable[[], None]:
        def restore_read() -> None:
            restore()
            reader.run()

        return restore_read

    def poison(self) -> None:
        for cache in self.caches:
            cache.view(torch.int16).fill_(POISON)

    def validate(self, name: str, *, graph: bool = False) -> dict[str, Any]:
        """Fresh poison, complete selected-byte oracle and unused-page canaries."""
        self.poison()
        reader = self.operation_reads.get(name)
        if reader is not None:
            reader.poison()
        if graph:
            self.graphs[name].graph.replay()
        else:
            self.operations[name]()
        if bool(self.status.any()):
            raise AssertionError(f"{name}: nonzero status on valid input")
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
                raise AssertionError(f"{name}: restored bits differ at layer {layer}")
            if not bool(
                torch.all(
                    cache[:, [0, self.page_count - 1]].view(torch.int16) == POISON
                )
            ):
                raise AssertionError(f"{name}: unused-page canary changed")
        result = {
            "all_selected_kv_bits_exact": True,
            "unused_pages_intact": True,
            "status_all_zero": True,
            "poisoned_before_restore": True,
            "compared_kv_bytes": int(self.expected.nbytes),
            "oracle": "held-out captured BF16 bits",
        }
        if reader is not None:
            result["attention"] = reader.validate()
        return result

    def validate_rejection(self, name: str) -> dict[str, Any]:
        """Confirm full decode still rejects a bad header and target alias."""
        result = {}
        original = int(self.payload[0])
        self.poison()
        self.payload[0].fill_(original ^ 255)
        try:
            self.operations[name]()
            status = self.status.cpu().numpy().copy()
            if not (status[0] & 1):
                raise AssertionError(f"{name}: malformed FPKB magic was accepted")
            self._assert_poisoned(self.ids[0])
            result["bad_magic"] = {
                "status": status.tolist(),
                "rejected_pages_intact": True,
            }
        finally:
            self.payload[0].fill_(original)
        if self.mapping.numel() > 1:
            self.poison()
            saved = int(self.ids.flat[1])
            self.mapping.flatten()[1].fill_(int(self.ids.flat[0]))
            try:
                self.operations[name]()
                status = self.status.cpu().numpy().copy()
                bad_blocks = np.unique([0, 1 // self.parts])
                if not np.all(status[bad_blocks] & 4):
                    raise AssertionError(f"{name}: duplicate target page was accepted")
                self._assert_poisoned(self.ids[bad_blocks].reshape(-1))
                result["duplicate_mapping"] = {
                    "status": status.tolist(),
                    "rejected_pages_intact": True,
                }
            finally:
                self.mapping.flatten()[1].fill_(saved)
        else:
            result["duplicate_mapping"] = {"status": "not_applicable_one_native_page"}
        return result

    def _assert_poisoned(self, ids: np.ndarray) -> None:
        for cache in self.caches:
            if not bool(torch.all(cache[:, ids.tolist()].view(torch.int16) == POISON)):
                raise AssertionError("rejected decode wrote a destination page")

    def sample(self, name: str) -> float:
        graph = self.graphs[name]
        graph.graph.replay()
        graph.end.synchronize()
        return graph.start.elapsed_time(graph.end) * 1000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cuda", type=Path, required=True)
    parser.add_argument("--candidate-cuda", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--blocks", type=_ints, help="legacy default: 1,4,16,64,256")
    parser.add_argument("--page-tokens", type=_ints, help="legacy default: 16,128")
    parser.add_argument(
        "--case-pairs",
        type=_pairs,
        help="exact N:P pairs; cannot combine with geometry lists",
    )
    parser.add_argument(
        "--read-queries",
        type=_ints,
        help="enable full restore+48 FA3 calls for each Q, e.g. 1,128; default off",
    )
    parser.add_argument("--calib-pages", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=8)
    parser.add_argument("--eval-pages", type=int, default=24)
    parser.add_argument(
        "--source-pattern",
        choices=("capture-repeat", "fallback25", "fallback100"),
        default="capture-repeat",
    )
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--input-description", required=True)
    parser.add_argument("--raw-scatter-control", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> None:
    """Run same-process ABBA rounds, preserving every sample and paired round."""
    args = _parser().parse_args()
    if min(args.rounds, args.warmup) < 1:
        raise ValueError("invalid rounds or warmup")
    cases = _case_specs(args)
    if args.build_root is None:
        args.build_root = args.out.parent / "kernel-ab-build"
    report: dict[str, Any] = {
        "schema": "fragpack.native-codec-kernel-ab.v1",
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cases": [],
        "planned_case_pairs": cases,
        "protocol": {
            "scope": (
                "GPU-only fully validated resident FPKB restore; no H2D, "
                "Python API, attention, or serving timing"
            ),
            "sources": (
                "Actual CUDA and binding included unchanged; six exported host "
                "functions uniquely renamed per module"
            ),
            "validation": (
                "Each decode_blobs retains header+length+full mapping/alias "
                "validation, record validation and decode/scatter; "
                "no unchecked stage launch"
            ),
            "same_inputs_and_storage": (
                "Both variants sequentially use identical target addresses, "
                "descriptors, maps, tables, payload, offsets and "
                "status/record-offset scratch"
            ),
            "cuda_build_flags": FLAGS,
            "timing": (
                "external=True CUDA timing events inside each graph; eager "
                "warmup, capture, validation and synchronization overhead "
                "outside reported GPU duration"
            ),
            "binding_checks": (
                "C++ tensor argument checks run during eager validation and "
                "graph capture; replay holds arguments/storage fixed"
            ),
            "ordering": (
                "ABBA then BAAB on alternating rounds; --order reverse swaps "
                "first pattern; each arm has two observations per round"
            ),
            "paired_analysis": (
                "Per-round baseline mean / candidate mean; ABBA samples and "
                "rounds are within one process, not independent process "
                "repetitions or confidence intervals"
            ),
            "cache_state": (
                "Repeated resident buffers and warm kernels; "
                "no artificial cache flushing"
            ),
            "source_pattern": (
                "capture-repeat preserves captured BF16 bits; fallback25/100 "
                "reuse scale-benchmark whole-record finite out-of-table high "
                "bytes with original low bytes. Selection is deterministic; "
                "--seed still controls the shared native-page mapping. "
                "Actual encoded fallback counts/fractions are recorded per case."
            ),
            "raw_scatter_control": (
                "Optional separate raw_scatter kernel control; not the raw-record "
                "fallback branch of decode_blobs. Forced-fallback patterns "
                "exercise that branch through fully validated decode_blobs."
            ),
        },
    }
    if args.read_queries:
        report["protocol"].update(
            {
                "scope": (
                    "GPU-only validated resident FPKB restore and separately "
                    "measured restore+48 stock FA3 layers; no H2D or serving"
                ),
                "attention": (
                    "Each full read graph contains mandatory header/map/record "
                    "validation, restore, then 48 causal FA3 v3 calls "
                    "(num_splits=0). Q/output addresses are shared by both "
                    "variants. This is immutable read replay, "
                    "not a full model or generation, including for Q=1."
                ),
                "attention_validation": (
                    "Poison all KV/outputs, verify restored bits/canaries/status, "
                    "then compare all BF16 output bits via int16 views with an "
                    "independent "
                    "FA3 reference built directly from original CPU source bits."
                ),
                "measurement_windows": (
                    "decode is restore-only; decode_fa3_qN is the complete "
                    "restore+read graph. These are separate direct measurements, "
                    "never summed medians."
                ),
            }
        )
    _write(args.out, report)
    try:
        report["source"] = _source_pair(args)
        if args.prepare_only:
            report["status"] = "prepared_cpu_only"
            return
        table, pool, report["input"] = _load_inputs(args)
        props = torch.cuda.get_device_properties(0)
        report["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gpu": props.name,
            "gpu_uuid": str(props.uuid),
            "cuda_runtime": torch.version.cuda,
        }
        modules = {
            role: _load_module(identity, args.build_root)
            for role, identity in report["source"]["modules"].items()
        }
        if args.read_queries:
            report["source"]["attention"] = _fa3_identity()
        _verify_sources(report["source"])
        host_tables = device_tables(table, torch.device("cpu"))
        digest = int.from_bytes(
            hashlib.blake2b(
                host_tables.rank_of.numpy().tobytes(), digest_size=8
            ).digest(),
            "little",
            signed=True,
        )
        tables = device_tables(table, torch.device("cuda"))
        report["status"] = "running"
        _write(args.out, report)
        for blocks, page_tokens in cases:
            row: dict[str, Any] = {
                "blocks": blocks,
                "page_tokens": page_tokens,
                "status": "preparing",
            }
            report["cases"].append(row)
            _write(args.out, report)
            case = _Case(
                args, tables, digest, pool, table, blocks, page_tokens, modules
            )
            row.update(
                {
                    "input": case.input,
                    "archive": case.archive,
                    "memory": case.memory,
                    "validation": case.validation,
                    "rejection_validation": case.rejection,
                    "graph_validation": case.graph_validation,
                    "samples": [],
                    "paired_rounds": [],
                    "summary": {},
                    "status": "running",
                }
            )
            kinds = (
                ["decode", "raw_scatter"] if args.raw_scatter_control else ["decode"]
            )
            kinds += [f"decode_fa3_q{query}" for query in args.read_queries or []]
            for round_id in range(args.rounds):
                roles = ["baseline", "candidate", "candidate", "baseline"]
                if (round_id + (args.order == "reverse")) % 2:
                    roles = ["candidate", "baseline", "baseline", "candidate"]
                for kind in kinds if round_id % 2 == 0 else list(reversed(kinds)):
                    values: dict[str, list[float]] = {role: [] for role in modules}
                    for position, role in enumerate(roles):
                        elapsed = case.sample(f"{role}_{kind}")
                        values[role].append(elapsed)
                        row["samples"].append(
                            {
                                "round": round_id,
                                "position": position,
                                "operation": kind,
                                "role": role,
                                "order": roles,
                                "gpu_graph_us": elapsed,
                            }
                        )
                    means = {
                        role: statistics.mean(times) for role, times in values.items()
                    }
                    row["paired_rounds"].append(
                        {
                            "round": round_id,
                            "operation": kind,
                            "baseline_mean_us": means["baseline"],
                            "candidate_mean_us": means["candidate"],
                            "baseline_over_candidate": means["baseline"]
                            / means["candidate"],
                        }
                    )
            for kind in kinds:
                row["summary"][kind] = {
                    role: _summary(
                        [
                            sample["gpu_graph_us"]
                            for sample in row["samples"]
                            if sample["operation"] == kind and sample["role"] == role
                        ]
                    )
                    for role in modules
                }
                row["summary"][kind]["paired_round_baseline_over_candidate"] = _summary(
                    [
                        pair["baseline_over_candidate"]
                        for pair in row["paired_rounds"]
                        if pair["operation"] == kind
                    ]
                )
            row["post_timing_validation"] = {
                name: case.validate(name, graph=True) for name in case.operations
            }
            if case.reads:
                row["post_timing_memory"] = {
                    "torch_allocated": torch.cuda.memory_allocated(),
                    "torch_peak_since_archive_preparation": (
                        torch.cuda.max_memory_allocated()
                    ),
                    "torch_reserved": torch.cuda.memory_reserved(),
                }
            row["status"] = "passed"
            _write(args.out, report)
            print(
                json.dumps(
                    {
                        "blocks": blocks,
                        "page_tokens": page_tokens,
                        "summary": row["summary"],
                    }
                ),
                flush=True,
            )
            del case
            gc.collect()
            torch.cuda.current_stream().synchronize()
            torch.cuda.empty_cache()
        _verify_sources(report["source"])
        report["sources_unchanged_after_run"] = True
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write(args.out, report)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Replay immutable KV sessions under an enforced GPU tensor-allocation budget.

Every arm receives the same logical session trace and uses whole-session LRU.
Misses transfer a real pinned-host archive in bounded chunks before running
48 attention layers; hits read resident pages. Archive creation and a separate
poison-and-restore validation trace are excluded from timed replay. This is a
cache/attention replay, not model serving, TTFT, or a scheduling benchmark.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import heapq
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import traceback
from typing import Any

import numpy as np
import torch

from benchmarks.native_codec_prefill_bench import (
    ATOL,
    RTOL,
    _hash_array,
    _hash_file,
    _overflow_quota,
    _summary,
    _tensor_bytes,
    _write,
)
from benchmarks.native_codec_prefill_bench import (
    _source_identity as previous_source_identity,
)
from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import calibrate, device_tables
from fragpack.ops.fused_attn_decode import (
    RecordTables,
    decode_attention,
    decode_units,
    workspace_elems,
)
from fragpack.ops.fused_attn_prefill import (
    prefill_attention,
    prefill_split_count,
    prefill_workspace_elems,
)
from fragpack.ops.hbm_kv import HbmKvStore, page_bytes
from fragpack.ops.native_codec import NativeCodec, NativeKVPages

ARMS = ("raw_fa3", "native_fa3", "gather_fa3", "fused_hbm")
L, H, HQ, T, D = 48, 4, 32, 128, 128
RAW_BLOCK_BYTES = L * 2 * T * H * D * 2


@dataclass(frozen=True)
class Admission:
    """One LRU decision; rejected sessions leave the cache unchanged."""

    admitted: bool
    hit: bool
    slots: tuple[int, ...]
    evicted: tuple[int, ...]


class SessionLRU:
    """Allocate disjoint physical slots to immutable whole sessions on the CPU."""

    def __init__(self, capacity_blocks: int) -> None:
        if capacity_blocks < 1:
            raise ValueError("positive block capacity required")
        self.capacity = capacity_blocks
        self.free = list(range(capacity_blocks))
        self.resident: OrderedDict[int, tuple[int, ...]] = OrderedDict()

    def access(self, session: int, blocks: int) -> Admission:
        """Touch a hit or evict least-recent sessions until a new one fits."""
        if blocks < 1:
            raise ValueError("session must contain at least one block")
        if session in self.resident:
            slots = self.resident[session]
            if len(slots) != blocks:
                raise ValueError("an immutable session changed its block count")
            self.resident.move_to_end(session)
            return Admission(True, True, slots, ())
        if blocks > self.capacity:
            return Admission(False, False, (), ())
        evicted = []
        while len(self.free) < blocks:
            victim, slots = self.resident.popitem(last=False)
            evicted.append(victim)
            for slot in slots:
                heapq.heappush(self.free, slot)
        slots = tuple(heapq.heappop(self.free) for _ in range(blocks))
        self.resident[session] = slots
        return Admission(True, False, slots, tuple(evicted))


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read explicit per-session contexts and request IDs without random traces."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty JSON list")
    names = set()
    for case in cases:
        case.setdefault("query_tokens", 1)
        case.setdefault("chunk_blocks", 4)
        case.setdefault("native_workspace", "decode")
        case.setdefault("receive_capacity", "format-max")
        contexts, trace = case["contexts"], case["trace"]
        if (
            not isinstance(case["name"], str)
            or case["name"] in names
            or not contexts
            or not trace
            or any(not isinstance(c, int) or c < 1 for c in contexts)
            or any(not isinstance(s, int) or not 0 <= s < len(contexts) for s in trace)
            or not isinstance(case["query_tokens"], int)
            or not 1 <= case["query_tokens"] <= min(contexts)
            or not isinstance(case["chunk_blocks"], int)
            or case["chunk_blocks"] < 1
            or not isinstance(case["budget_bytes"], int)
            or case["budget_bytes"] < 1
            or case["native_workspace"] not in ("all", "decode")
            or case["receive_capacity"] not in ("format-max", "observed")
        ):
            raise ValueError(f"invalid budget replay case: {case}")
        names.add(case["name"])
    return cases


def _source_identity() -> dict[str, Any]:
    identity = previous_source_identity()
    identity["files_sha256"]["benchmarks/native_codec_budget_bench.py"] = _hash_file(
        Path(__file__)
    )
    identity["digest"] = hashlib.sha256(
        json.dumps(identity["files_sha256"], sort_keys=True).encode()
    ).hexdigest()
    return identity


@dataclass
class Corpus:
    """CPU capture, codebook and independently pinned FPKB archive blocks."""

    content: np.ndarray
    table: np.ndarray
    blobs: list[torch.Tensor]
    max_overflow_per_block: int
    provenance: dict[str, Any]


def prepare_corpus(args: argparse.Namespace) -> Corpus:
    """Encode the finite capture pool once; return no GPU owners."""
    dump = np.load(args.dump, mmap_mode="r", allow_pickle=False)
    if dump.dtype != np.uint16 or dump.shape[1:] != (L, 2, T, H, D):
        raise ValueError("expected uint16 [blocks,48,2,128,4,128] capture")
    end = args.eval_start + args.eval_pages
    if args.eval_start < args.calib_pages or end > len(dump):
        raise ValueError("invalid disjoint calibration/evaluation positions")
    table = calibrate(np.asarray(dump[: args.calib_pages]))
    content = np.array(dump[args.eval_start : end])
    count = len(content)
    caches = []
    for layer in range(L):
        bits = np.ascontiguousarray(content[:, layer].transpose(1, 0, 2, 3, 4))
        caches.append(
            torch.from_numpy(bits.view(np.int16)).to("cuda").view(torch.bfloat16)
        )
    pages = NativeKVPages(caches)
    codec = NativeCodec(table, max_blocks=count, device="cuda")
    mapping = torch.arange(count, dtype=torch.int32, device="cuda")[:, None]
    status = codec.encode(pages, mapping)
    torch.cuda.current_stream().synchronize()
    if bool(torch.any(status.cpu() != 0)):
        raise RuntimeError("archive preparation encoding failed")
    offsets = codec.offsets.cpu().numpy().copy()
    lengths = codec.rec_lens.cpu().numpy().copy()
    blobs = []
    for begin, finish in zip(offsets[:-1], offsets[1:], strict=True):
        blob = torch.empty(int(finish - begin), dtype=torch.uint8, pin_memory=True)
        blob.copy_(codec.buffer[int(begin) : int(finish)], non_blocking=True)
        blobs.append(blob)
    torch.cuda.current_stream().synchronize()
    needed = [_overflow_quota(lengths[i : i + 1])[0] for i in range(count)]
    return Corpus(
        content,
        table,
        blobs,
        max(needed),
        {
            "description": args.input_description,
            "dump_sha256": _hash_file(args.dump),
            "dump_shape": list(dump.shape),
            "calibration_range": [0, args.calib_pages],
            "evaluation_pool_range": [args.eval_start, end],
            "table_sha256": _hash_array(table),
            "pool_sha256": _hash_array(content),
            "blob_bytes": [blob.numel() for blob in blobs],
            "raw_fallback_records": int((lengths == 32768 - 22528).sum()),
            "max_overflow_live_records_per_block": max(needed),
            "scope": (
                "disjoint positions from one historical capture; session contents "
                "rotate/repeat this pool, not new long-context model captures"
            ),
        },
    )


@dataclass
class Session:
    """One immutable logical session, with separate raw and FPKB host archives."""

    context: int
    indices: np.ndarray
    raw: torch.Tensor
    compressed: torch.Tensor
    offsets: np.ndarray

    @property
    def blocks(self) -> int:
        return len(self.indices)


def prepare_sessions(corpus: Corpus, contexts: list[int]) -> list[Session]:
    """Materialize independent logical identities without content deduplication."""
    sessions = []
    for identity, context in enumerate(contexts):
        count = (context + T - 1) // T
        indices = (identity * 7 + np.arange(count)) % len(corpus.content)
        raw_bits = np.ascontiguousarray(
            corpus.content[indices].transpose(0, 1, 2, 4, 3, 5)
        )
        raw = torch.empty(raw_bits.nbytes, dtype=torch.uint8, pin_memory=True)
        raw.copy_(torch.from_numpy(raw_bits.view(np.uint8).reshape(-1)))
        offsets = np.r_[0, np.cumsum([corpus.blobs[i].numel() for i in indices])]
        compressed = torch.empty(int(offsets[-1]), dtype=torch.uint8, pin_memory=True)
        for index, block in enumerate(indices):
            compressed[int(offsets[index]) : int(offsets[index + 1])].copy_(
                corpus.blobs[block]
            )
        sessions.append(Session(context, indices, raw, compressed, offsets))
    return sessions


def prepare_reference(
    corpus: Corpus, sessions: list[Session], queries: torch.Tensor
) -> list[torch.Tensor]:
    """Compute native FA3 reference directly from CPU source bits, outside budget."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    maximum = max(session.blocks for session in sessions)
    cache = torch.empty((2, maximum, T, H, D), dtype=torch.bfloat16, device="cuda")
    q = queries.to("cuda")
    output = torch.empty_like(q)
    table = torch.arange(maximum, dtype=torch.int32, device="cuda")[None]
    cu_q = torch.tensor([0, queries.shape[1]], dtype=torch.int32, device="cuda")
    lens = torch.empty(1, dtype=torch.int32, device="cuda")
    references = []
    for session in sessions:
        lens.fill_(session.context)
        for layer in range(L):
            bits = np.ascontiguousarray(
                corpus.content[session.indices, layer].transpose(1, 0, 2, 3, 4)
            )
            cache[:, : session.blocks].copy_(
                torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
            )
            flash_attn_varlen_func(
                q=q[layer],
                k=cache[0],
                v=cache[1],
                out=output[layer],
                cu_seqlens_q=cu_q,
                max_seqlen_q=queries.shape[1],
                seqused_k=lens,
                max_seqlen_k=session.context,
                softmax_scale=D**-0.5,
                causal=True,
                block_table=table[:, : session.blocks],
                fa_version=3,
                num_splits=0,
            )
        references.append(output.cpu())
    return references


class Runtime:
    """One arm's complete GPU ownership; all restores use one ordered stream."""

    def __init__(
        self,
        arm: str,
        blocks: int,
        case: dict[str, Any],
        corpus: Corpus,
        sessions: list[Session],
        queries: torch.Tensor,
    ) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        self.arm, self.blocks, self.case = arm, blocks, case
        self.corpus, self.sessions = corpus, sessions
        self.chunk = case["chunk_blocks"]
        self.max_request_blocks = max(session.blocks for session in sessions)
        self.query_tokens = case["query_tokens"]
        self.fa3 = flash_attn_varlen_func
        self.stream = torch.cuda.current_stream()
        self.done, self.restored = torch.cuda.Event(), torch.cuda.Event()
        self.pages: NativeKVPages | None = None
        self.codec: NativeCodec | None = None
        self.store: HbmKvStore | None = None
        self.arenas: list[torch.Tensor] = []
        self.gather_k: torch.Tensor | None = None
        self.gather_v: torch.Tensor | None = None
        self.tables: list[RecordTables] = []
        if arm in ("raw_fa3", "native_fa3"):
            self.pages = NativeKVPages(
                [
                    torch.zeros(
                        (2, blocks, T, H, D), dtype=torch.bfloat16, device="cuda"
                    )
                    for _ in range(L)
                ]
            )
            workspace = "raw" if arm == "raw_fa3" else case["native_workspace"]
            self.codec = NativeCodec(
                corpus.table, max_blocks=self.chunk, device="cuda", workspace=workspace
            )
        else:
            self.arenas = [
                torch.zeros((blocks, page_bytes(H)), dtype=torch.uint8, device="cuda")
                for _ in range(L)
            ]
            # Replacement import claims fresh overflow before freeing previous
            # records. Reserve all possible live blocks plus one incoming chunk.
            overflow = max(1, (blocks + self.chunk) * corpus.max_overflow_per_block)
            self.store = HbmKvStore(
                self.arenas,
                H,
                device_tables(corpus.table, torch.device("cuda")),
                max_tokens=T,
                staging_slots=1,
                overflow_records=overflow,
            )
            self.tables = [
                RecordTables(*self.store.layer_tables(layer)) for layer in range(L)
            ]
            if arm == "gather_fa3":
                self.gather_k = torch.empty(
                    self.max_request_blocks,
                    T,
                    H,
                    D,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                self.gather_v = torch.empty_like(self.gather_k)
        max_blob = (
            ap.blob_slot_bytes(L, H)
            if case.get("receive_capacity", "format-max") == "format-max"
            else max(blob.numel() for blob in corpus.blobs)
        )
        receive_bytes = self.chunk * (RAW_BLOCK_BYTES if arm == "raw_fa3" else max_blob)
        self.receive = torch.empty(receive_bytes, dtype=torch.uint8, device="cuda")
        self.offsets = torch.empty(self.chunk + 1, dtype=torch.int64, device="cuda")
        self.mapping = torch.empty((self.chunk, 1), dtype=torch.int32, device="cuda")
        self.status_host = torch.empty(self.chunk, dtype=torch.int32, pin_memory=True)
        self.offsets_host = torch.empty(
            self.chunk + 1, dtype=torch.int64, pin_memory=True
        )
        self.mapping_host = torch.empty(
            (self.chunk, 1), dtype=torch.int32, pin_memory=True
        )
        self.read_ids_host = torch.empty(
            self.max_request_blocks, dtype=torch.int32, pin_memory=True
        )
        self.read_ids = torch.arange(
            self.max_request_blocks, dtype=torch.int32, device="cuda"
        )
        self.length_host = torch.empty(1, dtype=torch.int32, pin_memory=True)
        self.length = torch.empty(1, dtype=torch.int32, device="cuda")
        self.cu_q = torch.tensor(
            [0, self.query_tokens], dtype=torch.int32, device="cuda"
        )
        self.gather_table = torch.arange(
            self.max_request_blocks, dtype=torch.int32, device="cuda"
        )[None]
        self.q = queries.to("cuda")
        self.output = torch.empty_like(self.q)
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.units = decode_units(1, H, sms)
        self.splits = {
            context: prefill_split_count(
                [self.query_tokens], [context], H, HQ // H, sms
            )
            for context in set(case["contexts"])
        }
        self.fused_workspace: tuple[torch.Tensor, ...] | None = None
        if arm == "fused_hbm":
            if self.query_tokens == 1:
                opart, ml, plan = workspace_elems(H, self.units)
                self.fused_workspace = (
                    torch.empty(opart, dtype=torch.float32, device="cuda"),
                    torch.empty(ml, dtype=torch.float32, device="cuda"),
                    torch.empty(plan, dtype=torch.int32, device="cuda"),
                )
            elif max(self.splits.values()) > 1:
                counts = prefill_workspace_elems(
                    self.query_tokens, HQ, max(self.splits.values())
                )
                self.fused_workspace = tuple(
                    torch.empty(n, dtype=torch.float32, device="cuda") for n in counts
                )
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.stream.synchronize()
        before_graph = torch.cuda.memory_allocated()
        for context in sorted(set(case["contexts"])):
            self.length.fill_(context)
            for _ in range(2):
                self.read(context)
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.read(context)
            self.graphs[context] = graph
            graph.replay()
            self.stream.synchronize()
        self.graph_allocation_delta = torch.cuda.memory_allocated() - before_graph
        self.memory = self.memory_usage()
        self.lru = SessionLRU(blocks)

    def memory_usage(self) -> dict[str, Any]:
        """Account for explicit tensors and observed graph-private allocations."""
        metadata = (
            self.offsets,
            self.mapping,
            self.read_ids,
            self.length,
            self.cu_q,
            self.gather_table,
        )
        return {
            "native_pages": self.pages.memory_usage() if self.pages else {},
            "codec": self.codec.memory_usage() if self.codec else {},
            "hbm_store": self.store.memory_usage() if self.store else {},
            "receive": _tensor_bytes(self.receive),
            "receive_capacity_policy": self.case.get("receive_capacity", "format-max"),
            "transfer_and_attention_metadata": sum(_tensor_bytes(t) for t in metadata),
            "gather_workspace": sum(
                _tensor_bytes(t)
                for t in (self.gather_k, self.gather_v)
                if t is not None
            ),
            "fused_workspace": sum(
                _tensor_bytes(t) for t in self.fused_workspace or ()
            ),
            "queries": _tensor_bytes(self.q),
            "output": _tensor_bytes(self.output),
            "graph_allocation_delta": self.graph_allocation_delta,
            "torch_allocated": torch.cuda.memory_allocated(),
            "torch_reserved": torch.cuda.memory_reserved(),
            "pinned_transfer_metadata": sum(
                _tensor_bytes(t)
                for t in (
                    self.status_host,
                    self.offsets_host,
                    self.mapping_host,
                    self.read_ids_host,
                    self.length_host,
                )
            ),
        }

    def read(self, context: int) -> None:
        """Enqueue the complete read, with one decode plan per 48-layer step."""
        count = (context + T - 1) // T
        ids = self.read_ids[:count]
        for layer in range(L):
            if self.arm == "fused_hbm":
                if self.query_tokens == 1:
                    decode_attention(
                        self.q[layer],
                        self.tables[layer],
                        ids[None],
                        self.length,
                        D**-0.5,
                        self.units,
                        self.output[layer],
                        self.fused_workspace,
                        replan=layer == 0,
                    )
                else:
                    prefill_attention(
                        self.q[layer],
                        self.tables[layer],
                        ids[None],
                        self.cu_q,
                        self.length,
                        self.query_tokens,
                        D**-0.5,
                        self.output[layer],
                        num_splits=self.splits[context],
                        workspace=self.fused_workspace,
                    )
                continue
            if self.arm == "gather_fa3":
                self.store.gather(
                    layer, ids, self.gather_k[:count], self.gather_v[:count]
                )
                key, value = self.gather_k[:count], self.gather_v[:count]
                table = self.gather_table[:, :count]
            else:
                key, value = self.pages.caches[layer][0], self.pages.caches[layer][1]
                table = ids[None]
            self.fa3(
                q=self.q[layer],
                k=key,
                v=value,
                out=self.output[layer],
                cu_seqlens_q=self.cu_q,
                max_seqlen_q=self.query_tokens,
                seqused_k=self.length,
                max_seqlen_k=context,
                softmax_scale=D**-0.5,
                causal=True,
                block_table=table,
                fa_version=3,
                num_splits=0,
            )

    def poison(self, ids: torch.Tensor) -> None:
        """Invalidate only selected destinations; leave other live sessions intact."""
        if self.pages is not None:
            for cache in self.pages.caches:
                cache.view(torch.int16).index_fill_(1, ids.long(), 0x6B6B)
        else:
            for arena in self.arenas:
                arena.index_fill_(0, ids.long(), 0x6B)
            if self.corpus.max_overflow_per_block:
                addresses = (
                    self.store.rec_addr.index_select(1, ids.long()).cpu().numpy()
                )
                base = self.store.overflow.data_ptr()
                maximum = base + _tensor_bytes(self.store.overflow)
                addresses = addresses[(addresses >= base) & (addresses < maximum)]
                slots = np.unique((addresses - base) // 32768)
                if len(slots):
                    selected = torch.from_numpy(slots).to("cuda")
                    self.store.overflow.index_fill_(0, selected, 0x6B)

    def validate_chunk(self, session: Session, begin: int, count: int) -> None:
        """Compare all restored KV bits, using CPU source values as the oracle."""
        ids = self.mapping[:count, 0]
        indices = session.indices[begin : begin + count]
        for layer in range(L):
            expected = self.corpus.content[indices, layer]
            if self.pages is not None:
                actual = self.pages.caches[layer].index_select(1, ids.long())
                bits = actual.view(torch.int16).cpu().numpy().view(np.uint16)
                if not np.array_equal(bits, expected.transpose(1, 0, 2, 3, 4)):
                    raise AssertionError(
                        f"{self.arm} restore bits differ at layer {layer}"
                    )
            else:
                # Validation-only scratch is not part of the production path.
                key = torch.empty(count, T, H, D, dtype=torch.bfloat16, device="cuda")
                value = torch.empty_like(key)
                self.store.gather(layer, ids, key, value)
                for kv, tensor in enumerate((key, value)):
                    bits = tensor.view(torch.int16).cpu().numpy().view(np.uint16)
                    if not np.array_equal(bits, expected[:, kv]):
                        raise AssertionError(
                            f"{self.arm} gather bits differ at layer {layer}"
                        )

    def restore(
        self, session: Session, slots: tuple[int, ...], *, validate: bool
    ) -> dict[str, int]:
        """Restore one session through bounded buffers, checking every chunk status."""
        transferred = metadata = status_bytes = chunks = 0
        for begin in range(0, session.blocks, self.chunk):
            count = min(self.chunk, session.blocks - begin)
            self.mapping_host.numpy()[:count, 0] = slots[begin : begin + count]
            self.mapping[:count].copy_(self.mapping_host[:count], non_blocking=True)
            metadata += count * 4
            if validate:
                self.poison(self.mapping[:count, 0])
            if self.arm == "raw_fa3":
                start, finish = (
                    begin * RAW_BLOCK_BYTES,
                    (begin + count) * RAW_BLOCK_BYTES,
                )
                payload = session.raw[start:finish]
            else:
                start, finish = (
                    int(session.offsets[begin]),
                    int(session.offsets[begin + count]),
                )
                payload = session.compressed[start:finish]
                self.offsets_host.numpy()[: count + 1] = (
                    session.offsets[begin : begin + count + 1] - start
                )
                self.offsets[: count + 1].copy_(
                    self.offsets_host[: count + 1], non_blocking=True
                )
                metadata += (count + 1) * 8
            self.receive[: payload.numel()].copy_(payload, non_blocking=True)
            if self.arm == "raw_fa3":
                status = self.codec.raw_scatter(
                    self.pages, self.mapping[:count], self.receive[: payload.numel()]
                )
            elif self.arm == "native_fa3":
                status = self.codec.decode(
                    self.pages,
                    self.mapping[:count],
                    src=self.receive[: payload.numel()],
                    offsets=self.offsets[: count + 1],
                )
            else:
                status = self.store.import_blocks(
                    self.receive[: payload.numel()],
                    self.offsets[:count],
                    self.mapping[:count, 0],
                )
            self.status_host[:count].copy_(status, non_blocking=True)
            self.restored.record(self.stream)
            self.restored.synchronize()
            if bool(torch.any(self.status_host[:count] != 0)):
                raise RuntimeError(
                    f"{self.arm} restore failed: {self.status_host[:count].tolist()}"
                )
            if validate:
                self.validate_chunk(session, begin, count)
            transferred += payload.numel()
            status_bytes += count * 4
            chunks += 1
        return {
            "payload_h2d_bytes": transferred,
            "restore_metadata_h2d_bytes": metadata,
            "status_d2h_bytes": status_bytes,
            "chunks": chunks,
        }

    def step(self, identity: int, *, validate: bool = False) -> dict[str, Any]:
        """Admit/load/read one session; complete wall includes CPU LRU decisions."""
        started = time.perf_counter_ns()
        session = self.sessions[identity]
        decision = self.lru.access(identity, session.blocks)
        if not decision.admitted:
            return {
                "session": identity,
                "admitted": False,
                "wall_us": (time.perf_counter_ns() - started) / 1000,
            }
        transfers = {
            "payload_h2d_bytes": 0,
            "restore_metadata_h2d_bytes": 0,
            "status_d2h_bytes": 0,
            "chunks": 0,
        }
        if not decision.hit:
            transfers = self.restore(session, decision.slots, validate=validate)
        ready = time.perf_counter_ns()
        self.read_ids_host.numpy()[: session.blocks] = decision.slots
        self.read_ids[: session.blocks].copy_(
            self.read_ids_host[: session.blocks], non_blocking=True
        )
        self.length_host[0] = session.context
        self.length.copy_(self.length_host, non_blocking=True)
        if validate:
            self.output.fill_(float("nan"))
        self.graphs[session.context].replay()
        self.done.record(self.stream)
        self.done.synchronize()
        finished = time.perf_counter_ns()
        return {
            "session": identity,
            "context": session.context,
            "admitted": True,
            "hit": decision.hit,
            "evicted": list(decision.evicted),
            "slots": list(decision.slots),
            "resident_sessions": len(self.lru.resident),
            "resident_blocks": self.blocks - len(self.lru.free),
            "wall_us": (finished - started) / 1000,
            "policy_restore_wall_us": (ready - started) / 1000,
            "metadata_read_wall_us": (finished - ready) / 1000,
            "read_metadata_h2d_bytes": session.blocks * 4 + 4,
            **transfers,
        }


def _cleanup_gpu() -> None:
    gc.collect()
    torch.cuda.current_stream().synchronize()
    torch.cuda.empty_cache()


def fit_runtime(
    arm: str,
    case: dict[str, Any],
    corpus: Corpus,
    sessions: list[Session],
    queries: torch.Tensor,
) -> tuple[Runtime | None, dict[str, Any]]:
    """Probe actual allocations and verify the feasible pool's next-block boundary."""
    budget = case["budget_bytes"]
    minimum = max(session.blocks for session in sessions)
    stride = RAW_BLOCK_BYTES if arm in ("raw_fa3", "native_fa3") else L * page_bytes(H)
    upper = budget // stride
    attempts = []

    def build(blocks: int) -> tuple[Runtime, int]:
        _cleanup_gpu()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        runtime = Runtime(arm, blocks, case, corpus, sessions, queries)
        largest = max(range(len(sessions)), key=lambda i: sessions[i].blocks)
        runtime.step(largest)
        runtime.memory = runtime.memory_usage()
        peak = torch.cuda.max_memory_allocated()
        attempts.append(
            {
                "blocks": blocks,
                "baseline_allocated_bytes": before,
                "setup_peak_allocated_bytes": peak,
                "probe_includes_largest_request_cold_restore_and_read": True,
                "within_budget": peak <= budget,
                "components": runtime.memory,
            }
        )
        return runtime, peak

    info: dict[str, Any] = {
        "budget_bytes": budget,
        "budget_scope": (
            "absolute torch allocated tensor peak, including KV, codec, "
            "receive, metadata, queries, outputs and graph-private buffers; "
            "allocator reserved/context outside this tensor cap"
        ),
        "minimum_blocks_for_largest_request": minimum,
        "raw_or_arena_only_upper_bound": upper,
        "attempts": attempts,
    }
    if upper < minimum:
        info.update(
            {"status": "infeasible", "reason": "resident KV alone exceeds budget"}
        )
        return None, info
    runtime, peak = build(minimum)
    if peak > budget:
        info.update(
            {
                "status": "infeasible",
                "reason": "largest request plus required buffers exceeds budget",
            }
        )
        del runtime
        _cleanup_gpu()
        return None, info
    candidate = min(upper, minimum + (budget - peak) // stride)
    if candidate == minimum:
        fit = minimum
        del runtime
        _cleanup_gpu()
    else:
        del runtime
        _cleanup_gpu()
        while True:
            runtime, peak = build(candidate)
            del runtime
            _cleanup_gpu()
            if peak <= budget:
                fit = candidate
                break
            candidate -= max(1, math.ceil((peak - budget) / stride))
            if candidate < minimum:
                raise AssertionError(
                    "capacity probe fell below proven feasible minimum"
                )
    # Allocator granularity and metadata grow in steps. Confirm that one more
    # physical block fails, rather than presenting an arithmetic floor as a run.
    while fit < upper:
        runtime, peak = build(fit + 1)
        del runtime
        _cleanup_gpu()
        if peak > budget:
            break
        fit += 1
    runtime, peak = build(fit)
    if peak > budget:
        raise RuntimeError("recreated capacity exceeded its previously verified cap")
    info.update(
        {
            "status": "admitted",
            "physical_blocks": fit,
            "physical_token_capacity": fit * T,
            "uniform_session_capacity": fit // minimum
            if len(set(case["contexts"])) == 1
            else None,
            "next_block_boundary_verified": fit == upper
            or any(a["blocks"] == fit + 1 and not a["within_budget"] for a in attempts),
            "final_setup_peak_allocated_bytes": peak,
        }
    )
    return runtime, info


def _run_trace(
    runtime: Runtime,
    trace: list[int],
    *,
    validate: bool,
    references: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    runtime.lru = SessionLRU(runtime.blocks)
    rows = []
    maximum_error = 0.0
    all_exact = True
    started = time.perf_counter_ns()
    for index, identity in enumerate(trace):
        row = runtime.step(identity, validate=validate)
        row["request_index"] = index
        rows.append(row)
        if not row["admitted"]:
            raise RuntimeError("a supposedly feasible runtime rejected a session")
        if validate:
            actual = runtime.output.cpu()
            expected = references[identity]
            torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
            exact = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            if runtime.arm != "fused_hbm" and not exact:
                raise AssertionError(
                    f"{runtime.arm} differs from source FA3 output bits"
                )
            all_exact &= exact
            maximum_error = max(
                maximum_error, float((actual.float() - expected.float()).abs().max())
            )
    elapsed = (time.perf_counter_ns() - started) / 1000
    result = {
        "samples": rows,
        "trace_wall_us": elapsed,
        "request_wall_us": _summary([row["wall_us"] for row in rows]),
        "hits": sum(row["hit"] for row in rows),
        "misses": sum(not row["hit"] for row in rows),
        "evictions": sum(len(row["evicted"]) for row in rows),
        "payload_h2d_bytes": sum(row["payload_h2d_bytes"] for row in rows),
        "metadata_h2d_bytes": sum(
            row["restore_metadata_h2d_bytes"] + row["read_metadata_h2d_bytes"]
            for row in rows
        ),
        "restore_chunks": sum(row["chunks"] for row in rows),
        "status_d2h_bytes": sum(row["status_d2h_bytes"] for row in rows),
        "peak_resident_sessions": max(row["resident_sessions"] for row in rows),
        "peak_resident_blocks": max(row["resident_blocks"] for row in rows),
        "hit_rate": sum(row["hit"] for row in rows) / len(rows),
    }
    if validate:
        result["validation"] = {
            "all_restored_chunks_poisoned_then_bitchecked": True,
            "layers_checked_per_chunk": L,
            "attention_requests_checked": len(rows),
            "all_attention_outputs_bitwise_equal": all_exact,
            "max_abs_attention_error": maximum_error,
            "atol": ATOL,
            "rtol": RTOL,
            "scope": (
                "separate validation trace; its timings are not performance evidence"
            ),
        }
    return result


def run_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    corpus: Corpus,
    row: dict[str, Any],
    report: dict[str, Any],
) -> None:
    sessions = prepare_sessions(corpus, case["contexts"])
    generator = torch.Generator().manual_seed(args.seed)
    queries = torch.randn(L, case["query_tokens"], HQ, D, generator=generator).to(
        torch.bfloat16
    )
    references = (
        [] if args.capacity_only else prepare_reference(corpus, sessions, queries)
    )
    _cleanup_gpu()
    if torch.cuda.memory_allocated() != 0:
        raise RuntimeError("archive/reference GPU owners survived preparation cleanup")
    row["setup"] = {
        "query_sha256": _hash_array(queries.view(torch.int16).numpy()),
        "archive_prep_remaining_torch_allocated_bytes": torch.cuda.memory_allocated(),
        "host_raw_history_bytes": sum(session.raw.numel() for session in sessions),
        "host_compressed_history_bytes": sum(
            session.compressed.numel() for session in sessions
        ),
        "host_source_pool_blob_bytes": sum(blob.numel() for blob in corpus.blobs),
        "sessions": [
            {
                "context": session.context,
                "blocks": session.blocks,
                "capture_pool_indices": session.indices.tolist(),
                "raw_payload_bytes": session.raw.numel(),
                "compressed_payload_bytes": session.compressed.numel(),
                "raw_sha256": _hash_array(session.raw.numpy()),
                "compressed_sha256": _hash_array(session.compressed.numpy()),
                "reference_sha256": (
                    _hash_array(references[index].view(torch.int16).numpy())
                    if references
                    else None
                ),
            }
            for index, session in enumerate(sessions)
        ],
        "input_scope": (
            "synthetic session identities with rotated/repeated captured KV; "
            "no content deduplication, no newly captured long-context "
            "distribution"
        ),
        "host_scope": (
            "both formats retain the complete same history; host archive "
            "capacity is reported, not constrained or silently used as a "
            "miss-recompute oracle"
        ),
    }
    row["arms"] = {}
    _write(args.out, report)
    for arm in args.arms:
        arm_row: dict[str, Any] = {"status": "preparing"}
        row["arms"][arm] = arm_row
        runtime, capacity = fit_runtime(arm, case, corpus, sessions, queries)
        arm_row["capacity"] = capacity
        _write(args.out, report)
        if runtime is None:
            arm_row["status"] = "infeasible"
            continue
        arm_row["components"] = runtime.memory
        if args.capacity_only:
            arm_row["status"] = "capacity_verified"
            print(
                json.dumps(
                    {
                        "case": case["name"],
                        "arm": arm,
                        "capacity_only": True,
                        "physical_blocks": runtime.blocks,
                        "current_allocated_bytes": torch.cuda.memory_allocated(),
                        "setup_peak_bytes": capacity[
                            "final_setup_peak_allocated_bytes"
                        ],
                    }
                ),
                flush=True,
            )
            del runtime
            _cleanup_gpu()
            arm_row["remaining_allocated_bytes_after_release"] = (
                torch.cuda.memory_allocated()
            )
            _write(args.out, report)
            continue
        arm_row["validation_trace"] = _run_trace(
            runtime, case["trace"], validate=True, references=references
        )
        # Validation-only reconstructed views and reference comparisons are gone
        # before resetting the peak used to enforce the timed runtime budget.
        _cleanup_gpu()
        arm_row["warmup_traces"] = [
            _run_trace(runtime, case["trace"], validate=False)
            for _ in range(args.warmup)
        ]
        arm_row["traces"] = []
        for repeat in range(args.rounds):
            torch.cuda.reset_peak_memory_stats()
            result = _run_trace(runtime, case["trace"], validate=False)
            result["round"] = repeat
            result["runtime_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["runtime_reserved_bytes"] = torch.cuda.memory_reserved()
            if result["runtime_peak_allocated_bytes"] > case["budget_bytes"]:
                raise RuntimeError(
                    f"{arm} exceeded GPU tensor budget during timed trace"
                )
            arm_row["traces"].append(result)
            _write(args.out, report)
        arm_row["summary"] = {
            "trace_wall_us": _summary(
                [trace["trace_wall_us"] for trace in arm_row["traces"]]
            ),
            "median_request_wall_us": _summary(
                [trace["request_wall_us"]["median"] for trace in arm_row["traces"]]
            ),
            "hit_rate": arm_row["traces"][0]["hit_rate"],
            "hits": arm_row["traces"][0]["hits"],
            "misses": arm_row["traces"][0]["misses"],
            "evictions": arm_row["traces"][0]["evictions"],
            "payload_h2d_bytes_per_trace": arm_row["traces"][0]["payload_h2d_bytes"],
        }
        arm_row["status"] = "passed"
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "arm": arm,
                    "physical_blocks": runtime.blocks,
                    "hit_rate": arm_row["summary"]["hit_rate"],
                    "median_trace_wall_ms": arm_row["summary"]["trace_wall_us"][
                        "median"
                    ]
                    / 1000,
                    "budget_bytes": case["budget_bytes"],
                    "setup_peak_bytes": capacity["final_setup_peak_allocated_bytes"],
                }
            ),
            flush=True,
        )
        del runtime
        _cleanup_gpu()
        arm_row["remaining_allocated_bytes_after_release"] = (
            torch.cuda.memory_allocated()
        )
        _write(args.out, report)
    row["status"] = "passed"


def main() -> None:
    """Run isolated arms on an explicitly reserved GPU and retain partial results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--calib-pages", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=8)
    parser.add_argument("--eval-pages", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--input-description", required=True)
    parser.add_argument(
        "--capacity-only",
        action="store_true",
        help="allocate/capture/probe full ownership and admission, without timed trace",
    )
    args = parser.parse_args()
    args.arms = args.arms.split(",")
    if (
        min(args.rounds, args.calib_pages, args.eval_pages) < 1
        or args.warmup < 0
        or len(set(args.arms)) != len(args.arms)
        or any(arm not in ARMS for arm in args.arms)
    ):
        raise ValueError("invalid arm selection or sample counts")
    cases = load_cases(args.cases)
    report: dict[str, Any] = {
        "schema": "fragpack.native-codec-fixed-budget.v1",
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": _source_identity(),
        "cases": [],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "vllm": importlib.metadata.version("vllm"),
        },
        "protocol": {
            "work": (
                "serial immutable-session LRU replay; each admitted access "
                "performs48 attention layers with independent synthetic per-layer "
                "Q; no QKV projection/MLP/newKV/model dependency"
            ),
            "budget": (
                "absolute torch tensor allocation peak: resident KV, all "
                "codec/receive/metadata/gather/fused buffers, Q/output, "
                "graph-private allocations; CUDA context and allocator reserved "
                "are reported separately, not treated as usable KV"
            ),
            "policy": (
                "same whole-session LRU and physical-slot allocator in every arm; "
                "a finished request leaves an inactive resident session; miss "
                "evicts least-recent inactive sessions; one request active at a "
                "time"
            ),
            "transfer": (
                "actual pinned-host H2D in bounded chunks, fresh destination "
                "metadata, public restore, per-chunk status D2H/event wait/check "
                "before dependent read"
            ),
            "validation": (
                "separate full trace; every loaded chunk poisoned then all48 "
                "layers compared bitwise with CPU capture; every output compared "
                "with independent source-bits FA3 reference"
            ),
            "history": (
                "fully available identical logical histories; raw and FPKB bytes "
                "differ and are both reported; no host capacity eviction, "
                "recompute, network or SSD"
            ),
            "attention": (
                "FA3 version3 num_splits0 causal; fused_hbm Q1 uses "
                "productiondecode with oneplan/layer0, otherQ uses current "
                "automatic split prefill"
            ),
            "timing": (
                "perrequest wall includes CPU policy, optional restore, read "
                "metadata H2D, captured48layergraph and completion; wholetrace "
                "wall also includes loop/resultcollection; validation/warmup "
                "excluded"
            ),
            "capacity": (
                "arithmetic resident-only upper bound followed by actual "
                "allocation/graph probes and nextblock boundary; infeasible arms "
                "report no fabricated latency"
            ),
            "repetitions": (
                "each timed trace resets logical LRU to empty; retained complete "
                "traces are nested process repetitions, not independent services"
            ),
            "interpretation": (
                "measured cache/attention replay latency, hits, bytes and "
                "admission; not TTFT, queueing, model serving throughput or a "
                "validated connector policy"
            ),
        },
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
        _cleanup_gpu()
        if torch.cuda.memory_allocated() != 0:
            raise RuntimeError("source archive GPU owners survived preparation cleanup")
        report["input"] = corpus.provenance
        report["archive_prep_remaining_allocated_bytes"] = torch.cuda.memory_allocated()
        report["status"] = "running"
        for case in cases:
            row: dict[str, Any] = {"case": case, "status": "preparing"}
            report["cases"].append(row)
            _write(args.out, report)
            run_case(args, case, corpus, row, report)
            _cleanup_gpu()
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

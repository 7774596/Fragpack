# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only two-slot H2D/restore scheduling over the unchanged public core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from benchmarks import native_codec_budget_bench as budget
from fragpack.compression import attn_page as ap

MODES = ("original", "serial-double", "overlap")


def pipeline_extra_bytes(
    arm: str, chunk_blocks: int, counts: list[int]
) -> dict[str, int]:
    """Count added second-slot logical tensor bytes, excluding allocator/event costs."""
    if (
        arm not in budget.ARMS
        or chunk_blocks < 1
        or not counts
        or len(set(counts)) != len(counts)
        or any(count < 1 or count > chunk_blocks for count in counts)
    ):
        raise ValueError("invalid pipeline arm, chunk capacity or graph counts")
    stride = (
        budget.RAW_BLOCK_BYTES
        if arm == "raw_fa3"
        else ap.blob_slot_bytes(budget.L, budget.H)
    )
    values = {
        "receive_payload": chunk_blocks * stride,
        "mapping": chunk_blocks * 4,
        "offsets": (chunk_blocks + 1) * 8,
        "captured_status_logical": (
            sum(counts) * 4 if arm in ("gather_fa3", "fused_hbm") else 0
        ),
    }
    return {**values, "total_logical": sum(values.values())}


def enqueue_slot(
    copy_stream: Any,
    compute_stream: Any,
    ready: Any,
    done: Any,
    copy: Callable[[], None],
    consume: Callable[[], None],
) -> None:
    """Order slot reuse/copy/consume on device; no event queries or host waits."""
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_event(done)
        copy()
        ready.record(copy_stream)
    with torch.cuda.stream(compute_stream):
        compute_stream.wait_event(ready)
        consume()
        done.record(compute_stream)


@dataclass
class Slot:
    """Own bounded inputs/events and captured statuses; native codec remains shared."""

    receive: torch.Tensor
    mapping: torch.Tensor
    offsets: torch.Tensor
    ready: torch.cuda.Event
    done: torch.cuda.Event
    graphs: dict[int, budget.RestoreGraph] = field(default_factory=dict)
    graph_allocation_delta: int = 0


class PipelineRuntime(budget.Runtime):
    """Retain original attention/policy while comparing three two-slot schedules."""

    def __init__(
        self,
        arm: str,
        blocks: int,
        case: dict,
        corpus: budget.Corpus,
        sessions: list[budget.Session],
        queries: torch.Tensor,
    ) -> None:
        if case.get("raw_transfer", "packed-scatter") != "packed-scatter":
            raise ValueError("pipeline diagnostic requires packed-scatter raw archives")
        if case.get("receive_capacity", "format-max") != "format-max":
            raise ValueError("pipeline diagnostic requires format-max receive capacity")
        # Build the inherited first slot without restore graphs; attention and
        # pinned request metadata use the original constructor unchanged.
        super().__init__(
            arm,
            blocks,
            dict(case, restore_dispatch="eager-capacity", restore_wait="request"),
            corpus,
            sessions,
            queries,
        )
        self.pipeline_mode = "original"
        self.slot_phase = 0
        self.copy_stream = torch.cuda.Stream()
        self.slots = [
            Slot(
                self.receive,
                self.mapping,
                self.offsets,
                torch.cuda.Event(),
                torch.cuda.Event(),
            ),
            Slot(
                torch.empty_like(self.receive),
                torch.empty_like(self.mapping),
                torch.empty_like(self.offsets),
                torch.cuda.Event(),
                torch.cuda.Event(),
            ),
        ]
        for slot in self.slots:
            self.capture_slot(slot)
            # First copy also waits for construction/capture initialization.
            slot.done.record(self.stream)
        self.restore_dispatch = "graph"
        self.restore_graphs = self.slots[0].graphs
        self.restore_graph_allocation_delta = self.slots[0].graph_allocation_delta
        self.select_mode("original")
        self.memory = self.memory_usage()

    def select_mode(self, mode: str) -> None:
        """Use the exact original enqueue method for the net-gain control."""
        if mode not in MODES:
            raise ValueError("unknown pipeline mode")
        self.pipeline_mode = mode
        method = (
            budget.Runtime.enqueue_request_chunk
            if mode == "original"
            else PipelineRuntime.enqueue_request_chunk
        )
        self.enqueue_request_chunk = method.__get__(self, type(self))

    def close(self) -> None:
        """Drain both streams before freeing owners, including failed enqueues."""
        try:
            self.copy_stream.synchronize()
        finally:
            self.stream.synchronize()

    def invoke_slot(self, slot: Slot, count: int, extent: int) -> torch.Tensor:
        """Enqueue one complete public restore, sharing serial decoder/store state."""
        if self.arm == "raw_fa3":
            return self.codec.raw_scatter(
                self.pages, slot.mapping[:count], slot.receive[:extent]
            )
        if self.arm == "native_fa3":
            return self.codec.decode(
                self.pages,
                slot.mapping[:count],
                src=slot.receive[:extent],
                offsets=slot.offsets[: count + 1],
            )
        return self.store.import_blocks(
            slot.receive[:extent], slot.offsets[:count], slot.mapping[:count, 0]
        )

    def capture_slot(self, slot: Slot) -> None:
        """Initialize legal inputs and retain count-specific graph/status ownership."""
        before = torch.cuda.memory_allocated()
        source = max(self.sessions, key=lambda session: session.blocks)
        for count in budget.restore_chunk_counts(self.case["contexts"], self.chunk):
            self.mapping_host[:count, 0] = torch.arange(count, dtype=torch.int32)
            slot.mapping[:count].copy_(self.mapping_host[:count], non_blocking=True)
            payload = self.request_payload(source, 0, count)
            slot.receive[: payload.numel()].copy_(payload, non_blocking=True)
            if self.arm != "raw_fa3":
                self.offsets_host.numpy()[: count + 1] = source.offsets[: count + 1]
                slot.offsets[: count + 1].copy_(
                    self.offsets_host[: count + 1], non_blocking=True
                )
            extent = payload.numel() if self.arm == "raw_fa3" else slot.receive.numel()
            for _ in range(2):
                status = self.invoke_slot(slot, count, extent)
                self.status_host[:count].copy_(status, non_blocking=True)
                self.restored.record(self.stream)
                self.restored.synchronize()
                if bool(torch.any(self.status_host[:count] != 0)):
                    raise RuntimeError("pipeline capture warmup failed")
                del status
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                status = self.invoke_slot(slot, count, extent)
            slot.graphs[count] = budget.RestoreGraph(graph, status)
            graph.replay()
            self.status_host[:count].copy_(status, non_blocking=True)
            self.restored.record(self.stream)
            self.restored.synchronize()
            if bool(torch.any(self.status_host[:count] != 0)):
                raise RuntimeError("pipeline captured restore failed")
        slot.graph_allocation_delta = torch.cuda.memory_allocated() - before

    def memory_usage(self) -> dict[str, Any]:
        """Add actual second-slot storage to the inherited first-slot accounting."""
        values = super().memory_usage()
        if hasattr(self, "slots"):
            second = self.slots[1]
            values["pipeline"] = {
                "slots": 2,
                "mode": self.pipeline_mode,
                "extra_logical_bytes": pipeline_extra_bytes(
                    self.arm, self.chunk, sorted(second.graphs)
                ),
                "second_graph_allocation_delta": second.graph_allocation_delta,
                "graph_status_bytes_are_included_in_graph_delta": True,
                "shared_codec_or_store_state": True,
                "runtime_events_and_allocator_reserved_excluded_from_tensor_cap": True,
            }
        return values

    def enqueue_request_chunk(
        self, begin: int, count: int, offset_begin: int, payload: torch.Tensor
    ) -> None:
        """Use original slot0 enqueue, or alternate equally allocated slots/streams."""
        if self.pipeline_mode == "original":
            return super().enqueue_request_chunk(begin, count, offset_begin, payload)
        slot = self.slots[(begin // self.chunk + self.slot_phase) % 2]

        def copy() -> None:
            slot.mapping[:count].copy_(
                self.request_mapping_host[begin : begin + count], non_blocking=True
            )
            if self.arm != "raw_fa3":
                slot.offsets[: count + 1].copy_(
                    self.request_offsets_host[offset_begin : offset_begin + count + 1],
                    non_blocking=True,
                )
            slot.receive[: payload.numel()].copy_(payload, non_blocking=True)

        def consume() -> None:
            captured = slot.graphs[count]
            captured.graph.replay()
            self.request_status_host[begin : begin + count].copy_(
                captured.status, non_blocking=True
            )

        copy_stream = (
            self.copy_stream if self.pipeline_mode == "overlap" else self.stream
        )
        enqueue_slot(copy_stream, self.stream, slot.ready, slot.done, copy, consume)

    def validate_pipeline(self) -> dict[str, Any]:
        """Exercise both slots, tail counts, early errors, and repeated slot use."""
        records = []
        original_mapping = self.mapping
        try:
            for phase in (0, 1):
                self.slot_phase = phase
                # The inherited tail canary writes self.mapping[count:]. Match
                # that alias to the slot actually selected by this probe.
                self.mapping = (
                    original_mapping
                    if self.pipeline_mode == "original"
                    else self.slots[phase].mapping
                )
                records.append(
                    {
                        "slot_phase": phase,
                        "dynamic_inputs": self.validate_restore_dispatch(),
                        "early_error": self.validate_request_status_retention(),
                        "reused_slot_error": self.validate_reused_slot_error(),
                    }
                )
        finally:
            self.slot_phase = 0
            self.mapping = original_mapping
        # The complete validation trace below restores every full session,
        # including slot wraparound when N > 2K; do not invent this coverage.
        return {
            "probes": records,
            "slot_wraparound_in_full_trace": self.max_request_blocks > 2 * self.chunk,
            "scope": "validation only; original mode intentionally uses slot0",
        }

    def validate_reused_slot_error(self) -> dict[str, Any]:
        """Retain an early error after i+2 reuses the slot, including tail counts."""
        source = max(self.sessions, key=lambda session: session.blocks)
        if source.blocks <= 2 * self.chunk:
            return {"status": "not_applicable", "reason": "fewer than three chunks"}
        blocks = min(source.blocks, 3 * self.chunk)
        probe = budget.Session(
            blocks * budget.T,
            source.indices[:blocks],
            source.raw[: blocks * budget.RAW_BLOCK_BYTES],
            source.compressed[: int(source.offsets[blocks])],
            source.offsets[: blocks + 1],
        )
        # Reverse IDs to avoid assuming contiguous target copies.
        slots = tuple(reversed(range(blocks)))
        layout = self.prepare_request_restore(probe, slots)
        self.read_ids[:blocks].copy_(
            self.request_mapping_host[:blocks, 0], non_blocking=True
        )
        self.poison(self.read_ids[:blocks])
        invalid = slots[0]
        before = self.snapshot_inactive_block(invalid)
        inactive = blocks if blocks < self.blocks else None
        canary = (
            self.snapshot_inactive_block(inactive) if inactive is not None else None
        )
        corrupted = None
        if self.arm == "raw_fa3":
            self.request_mapping_host[0, 0] = -1
        else:
            first = self.request_payload(probe, 0, self.chunk)
            corrupted = torch.empty(first.numel(), dtype=torch.uint8, pin_memory=True)
            corrupted.copy_(first)
            corrupted[0] = int(corrupted[0]) ^ 1
        self.output.fill_(float("nan"))
        for begin, count, offset in layout:
            payload = (
                corrupted
                if begin == 0 and corrupted is not None
                else self.request_payload(probe, begin, count)
            )
            self.enqueue_request_chunk(begin, count, offset, payload)
        try:
            self.finish_request_restore(blocks)
        except budget.RestoreStatusError:
            pass
        else:
            raise AssertionError("reused-slot error disappeared before attention gate")
        statuses = self.request_status_host[:blocks].tolist()
        if statuses[0] == 0 or any(value != 0 for value in statuses[1:]):
            raise AssertionError("first error/later success statuses were not retained")
        after = self.snapshot_inactive_block(invalid)
        if len(before) != len(after) or not all(
            torch.equal(a, b) for a, b in zip(before, after, strict=True)
        ):
            raise AssertionError("invalid reused-slot target changed")
        for begin, count, _ in layout:
            start = max(1, begin)
            if begin + count > start:
                self.validate_chunk(
                    probe,
                    start,
                    begin + count - start,
                    ids=self.read_ids[start : begin + count],
                )
        if inactive is not None:
            after = self.snapshot_inactive_block(inactive)
            if len(canary) != len(after) or not all(
                torch.equal(a, b) for a, b in zip(canary, after, strict=True)
            ):
                raise AssertionError("reused-slot restore changed inactive target")
        if not bool(torch.isnan(self.output).all()):
            raise AssertionError("failed restore changed poisoned attention output")
        index = 0 if self.pipeline_mode == "original" else self.slot_phase
        first_status = self.slots[index].graphs[layout[0][1]].status
        third_status = self.slots[index].graphs[layout[2][1]].status
        return {
            "status": "passed",
            "counts": [count for _, count, _ in layout],
            "retained_statuses": statuses,
            "same_slot_reused": True,
            "third_chunk_reuses_first_status_storage": (
                first_status.untyped_storage().data_ptr()
                == third_status.untyped_storage().data_ptr()
            ),
            "invalid_target_unchanged": True,
            "later_kv_bits_exact": True,
            "inactive_target_intact": True if inactive is not None else None,
            "output_remained_poisoned": True,
            "error_raised_before_attention": True,
        }

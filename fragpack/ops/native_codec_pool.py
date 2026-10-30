# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity codec/transfer slots with CUDA-ordered exclusive leases.

This module owns GPU working storage, not native KV, pinned host buffers,
external-cache policy or a transfer scheduler. Construct and borrow/return on
one host thread; different CUDA streams may reuse released slots through events.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from types import TracebackType
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import torch

from fragpack.compression import attn_page as ap
from fragpack.ops.native_codec import (
    NativeCodec,
    native_codec_workspace_bytes,
    validate_tables,
)

PoolWorkspace = Literal["encode", "decode", "raw"]


def native_codec_pool_bytes(
    num_layers: int,
    kv_heads: int,
    max_blocks: int,
    slots: int,
    *,
    workspace: PoolWorkspace,
) -> dict[str, int]:
    """Calculate all owned tensor bytes without tensors or CUDA initialization.

    Codec allocations are prefixed ``codec_``; ``receive_payload`` and
    ``receive_offsets`` count additional transfer storage. Encode slots alias
    existing output buffers and contribute zero additional receive storage.
    CUDA events/runtime, allocator reservation and caller-owned KV/descriptors,
    maps and host buffers are excluded. Safe to call from any host thread.
    """
    if workspace not in ("encode", "decode", "raw"):
        raise ValueError("pool workspace must be 'encode', 'decode', or 'raw'")
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
        raise ValueError("slots must be a positive integer")
    codec = native_codec_workspace_bytes(
        num_layers, kv_heads, max_blocks, workspace=workspace
    )
    payload = 0
    if workspace == "decode":
        payload = max_blocks * ap.blob_slot_bytes(num_layers, kv_heads)
    elif workspace == "raw":
        payload = max_blocks * num_layers * 2 * kv_heads * ap.RAW_RECORD_BYTES
    return {
        **{f"codec_{name}": slots * size for name, size in codec.items()},
        "receive_payload": slots * payload,
        "receive_offsets": slots * (max_blocks + 1) * 8 if workspace == "decode" else 0,
    }


@dataclass
class _Slot:
    codec: NativeCodec
    payload: torch.Tensor
    offsets: torch.Tensor | None
    completion: torch.cuda.Event
    generation: int = 0
    lease: NativeCodecLease | None = None
    stream: torch.cuda.Stream | None = None


class NativeCodecLease:
    """Exclusive access to one slot until :meth:`release` or context exit.

    Obtain leases only through :meth:`NativeCodecPool.acquire`. All GPU work
    using the slot must be ordered on the stream active at acquisition. Queue
    every consumer before release, including dependencies on other streams.
    Released leases reject access; tensor references borrowed earlier must not
    be used afterward. The pool cannot intercept such retained references.
    """

    def __init__(self, pool: NativeCodecPool, index: int, generation: int) -> None:
        self._pool = pool
        self._index = index
        self._generation = generation

    @property
    def slot_index(self) -> int:
        """Return this lease's fixed slot index, including after release."""
        return self._index

    def _slot(self) -> _Slot:
        return self._pool._checked_slot(self)  # noqa: SLF001

    @property
    def codec(self) -> NativeCodec:
        """Borrow this slot's direction-specific codec for the lease lifetime."""
        return self._slot().codec

    @property
    def payload(self) -> torch.Tensor:
        """Borrow full-capacity uint8 transfer storage; transfer valid bytes only."""
        return self._slot().payload

    @property
    def offsets(self) -> torch.Tensor | None:
        """Borrow active encode offsets, full decode capacity, or None for raw.

        Decode callers pass the first N or N+1 elements for an N-block batch.
        Encode offsets reflect the latest batch after ``codec.encode``/``plan``.
        """
        slot = self._slot()
        return slot.codec.offsets if self._pool.workspace == "encode" else slot.offsets

    def release(self) -> None:
        """Record completion on the acquiring stream without waiting on the host."""
        self._pool.release(self)

    def __enter__(self) -> NativeCodecLease:
        self._slot()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class NativeCodecPool:
    """Own a fixed number of bounded codec and GPU transfer slots.

    Construction preallocates every tensor and initializes one CUDA event per
    slot. ``encode`` aliases codec output buffers; ``decode`` and ``raw`` add
    bounded receive buffers. No acquire/release performs a tensor allocation,
    completion query or host synchronization. A Python lease is allocated per
    acquisition. Exhaustion raises immediately instead of waiting for a lease.

    Construct, acquire, access leases and release on one host thread. Release
    on the acquiring CUDA stream after all uses of the slot; any other consumer
    stream must be joined first. The next acquisition waits for that completion
    on its current stream. Acquire/release happen outside CUDA Graph capture;
    replay a graph only while its captured slot storage is exclusively leased.

    Keep the pool alive until its pending work completes. Callers separately
    retain native KV, maps and pinned host buffers until their final GPU uses,
    and establish producer dependencies before operations. Releasing a slot
    neither releases those objects nor waits for their transfers to finish.
    """

    def __init__(
        self,
        table: NDArray[np.uint8],
        *,
        max_blocks: int,
        slots: int = 1,
        workspace: PoolWorkspace,
        device: torch.device | str,
    ) -> None:
        validate_tables(table)
        layers, _, heads, _ = table.shape
        sizes = native_codec_pool_bytes(
            layers, heads, max_blocks, slots, workspace=workspace
        )
        self._owner_thread = threading.get_ident()
        self._workspace = workspace
        self._max_blocks = max_blocks
        self._slots: list[_Slot] = []
        self._next_slot = 0
        for _ in range(slots):
            codec = NativeCodec(
                table, max_blocks=max_blocks, device=device, workspace=workspace
            )
            payload = (
                codec.buffer
                if workspace == "encode"
                else torch.empty(
                    sizes["receive_payload"] // slots,
                    dtype=torch.uint8,
                    device=codec.device,
                )
            )
            offsets = (
                torch.empty(max_blocks + 1, dtype=torch.int64, device=codec.device)
                if workspace == "decode"
                else None
            )
            completion = torch.cuda.Event(enable_timing=False, blocking=False)
            # Recording here initializes the CUDA event outside the hot path
            # and orders first use after table/descriptor initialization.
            completion.record(torch.cuda.current_stream(codec.device))
            self._slots.append(_Slot(codec, payload, offsets, completion))
        self._device = self._slots[0].codec.device

    @property
    def device(self) -> torch.device:
        """Return the CUDA device owning all slots."""
        return self._device

    @property
    def workspace(self) -> PoolWorkspace:
        """Return the immutable role shared by every slot."""
        return self._workspace

    @property
    def max_blocks(self) -> int:
        """Return each slot's capacity in complete 128-token codec blocks."""
        return self._max_blocks

    @property
    def slots(self) -> int:
        """Return the fixed number of independently leased slots."""
        return len(self._slots)

    def _check_thread(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                "codec pool must be used by its constructing host thread"
            )

    def _checked_slot(self, lease: NativeCodecLease) -> _Slot:
        self._check_thread()
        if not isinstance(lease, NativeCodecLease) or lease._pool is not self:  # noqa: SLF001
            raise ValueError("lease belongs to a different codec pool")
        index = lease.slot_index
        if not 0 <= index < self.slots:
            raise RuntimeError("codec lease is stale or already released")
        slot = self._slots[index]
        if slot.lease is not lease or slot.generation != lease._generation:  # noqa: SLF001
            raise RuntimeError("codec lease is stale or already released")
        return slot

    def acquire(self) -> NativeCodecLease:
        """Lease an available slot and enqueue its prior-completion dependency.

        A released slot is eligible immediately: ``wait_event`` postpones its
        next GPU use, without asking the host whether prior work has completed.
        All GPU work on the lease must use the current stream at this call.
        """
        self._check_thread()
        for distance in range(self.slots):
            index = (self._next_slot + distance) % self.slots
            slot = self._slots[index]
            if slot.lease is not None:
                continue
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(slot.completion)
            slot.generation += 1
            lease = NativeCodecLease(self, index, slot.generation)
            slot.lease, slot.stream = lease, stream
            self._next_slot = (index + 1) % self.slots
            return lease
        raise RuntimeError("codec pool exhausted: release a lease before acquiring")

    def release(self, lease: NativeCodecLease) -> None:
        """Return an owned lease after recording all slot uses on its stream.

        Reject foreign, stale and repeated releases, or release from a different
        host thread/CUDA stream. The lease remains active after a rejected call.
        """
        slot = self._checked_slot(lease)
        stream = torch.cuda.current_stream(self.device)
        if stream != slot.stream:
            raise RuntimeError("release must use the same CUDA stream as acquire")
        slot.completion.record(stream)
        slot.lease = None
        slot.stream = None

    def memory_usage(self) -> dict[str, int]:
        """Return owned tensor bytes, matching :func:`native_codec_pool_bytes`."""
        result: dict[str, int] = {}
        for slot in self._slots:
            for name, size in slot.codec.memory_usage().items():
                key = f"codec_{name}"
                result[key] = result.get(key, 0) + size
        result["receive_payload"] = (
            0
            if self.workspace == "encode"
            else sum(
                slot.payload.numel() * slot.payload.element_size()
                for slot in self._slots
            )
        )
        result["receive_offsets"] = sum(
            slot.offsets.numel() * slot.offsets.element_size()
            for slot in self._slots
            if slot.offsets is not None
        )
        return result


__all__ = ["NativeCodecLease", "NativeCodecPool", "native_codec_pool_bytes"]

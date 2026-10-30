# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous transport of existing FragPack block records.

The caller owns engine block lifetimes. This module owns only transfer buffers;
it does not change vLLM scheduling or LMCache eviction. Device synchronization,
length readback and RPC run on the transfer thread, outside model execution.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import queue
import threading
from typing import TYPE_CHECKING, Sequence

import torch

from fragpack.integrations.lmcache.client import BlobClient
from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity
from fragpack.integrations.lmcache.protocol import canonicalize_blob, validate_blob
from fragpack.logging import init_logger

if TYPE_CHECKING:
    from fragpack.ops.hbm_kv import HbmKvStore

logger = init_logger(__name__)

# CUDA failure can make outstanding copies impossible to prove complete. Keep
# their buffers and source-ID owners alive until process teardown, even if the
# caller drops the transfer object after receiving the exception.
_quarantined_transfers: list[BlockTransfer] = []


@dataclass(frozen=True)
class StoreTicket:
    """Two independent completion points for an admitted store.

    Source engine blocks may be reused after ``source_ready`` completes.
    ``completion`` reports publication per block in input order; a False stops
    the stored prefix. An exception can follow earlier successful publications.
    The future is not cancellable because admitted copies retain their owners.
    """

    source_ready: torch.cuda.Event
    completion: Future[tuple[bool, ...]]


@dataclass(frozen=True)
class RestoreTicket:
    """Completion after import and its device status have been observed.

    The caller must retain all destinations, disjoint from attention and open
    blocks, until completion. True entries form a contiguous prefix; False
    entries must be recomputed, including any later imported blocks after a
    failed block. An exception invalidates the entire requested destination set
    for reuse. A fatal CUDA exception requires process recovery.
    """

    completion: Future[tuple[bool, ...]]


@dataclass
class _DeviceWork:
    blocks: torch.Tensor
    ready: torch.cuda.Event
    lengths: tuple[torch.Tensor, ...] = ()


@dataclass
class _Job:
    keys: tuple[BlockKey, ...]
    slots: tuple[int, ...]
    device_work: _DeviceWork
    completion: Future[tuple[bool, ...]]
    restore: bool


class _CudaBuffers:
    """Device operations separated from queue/ownership bookkeeping."""

    def __init__(self, store: HbmKvStore, num_slots: int) -> None:
        self.store = store
        self.device = store.rec_addr.device
        if self.device.type != "cuda":
            raise ValueError("block transfers require a CUDA HbmKvStore")
        self.stride = store.blob_stride
        # Submission can fail before returning its job. Keep each created
        # owner here until handoff, or until the exceptional producer drain.
        self._pending_prepare: list[torch.Tensor] = []
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("block transfers cannot be created during capture")
            self.gpu = torch.empty(
                (num_slots, self.stride), dtype=torch.uint8, device=self.device
            )
            self.host = torch.empty(
                (num_slots, self.stride), dtype=torch.uint8, pin_memory=True
            )
            self.stream = torch.cuda.Stream(device=self.device)

    def validate_blocks(self, blocks: torch.Tensor, count: int) -> None:
        if (
            blocks.dtype != torch.int32
            or blocks.device != self.device
            or blocks.ndim != 1
            or blocks.numel() != count
            or not blocks.is_contiguous()
        ):
            raise ValueError("blocks must be a contiguous device int32 vector per key")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("block transfer submission does not support capture")

    def prepare(
        self, blocks: torch.Tensor, slots: tuple[int, ...], restore: bool
    ) -> _DeviceWork:
        with torch.cuda.device(self.device):
            # Keep an independent copy of the ID tensor until every consumer
            # finishes. This does not pin the engine's physical KV block itself.
            owned = blocks.clone()
            self._pending_prepare.append(owned)
            lengths = []
            if not restore:
                for i, slot in enumerate(slots):
                    length = self.store.export_blocks(
                        owned[i : i + 1], self.gpu[slot], self.stride
                    )
                    self._pending_prepare.append(length)
                    lengths.append(length)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        work = _DeviceWork(owned, ready, tuple(lengths))
        self._pending_prepare.clear()
        return work

    def read_export(self, job: _Job) -> tuple[bytes, ...]:
        job.device_work.ready.synchronize()
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            lengths = tuple(int(t.item()) for t in job.device_work.lengths)
            for slot, length in zip(job.slots, lengths, strict=True):
                if length <= 0 or length > self.stride:
                    raise ValueError("export returned an invalid blob length")
                self.host[slot, :length].copy_(
                    self.gpu[slot, :length], non_blocking=True
                )
            self.stream.synchronize()
        return tuple(
            self.host[slot, :length].numpy().tobytes()
            for slot, length in zip(job.slots, lengths, strict=True)
        )

    def import_blobs(self, job: _Job, blobs: Sequence[bytes]) -> tuple[bool, ...]:
        job.device_work.ready.synchronize()
        if not blobs:
            return (False,) * len(job.keys)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            for slot, blob in zip(job.slots, blobs, strict=False):
                source = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
                self.host[slot, : len(blob)].copy_(source)
                self.gpu[slot, : len(blob)].copy_(
                    self.host[slot, : len(blob)], non_blocking=True
                )
            offsets = torch.tensor(
                [slot * self.stride for slot in job.slots[: len(blobs)]],
                dtype=torch.int64,
                device=self.device,
            )
            status = self.store.import_blocks(
                self.gpu.view(-1), offsets, job.device_work.blocks[: len(blobs)]
            )
            self.stream.synchronize()
            values = status.cpu().tolist()
        complete = True
        result = []
        for value in values:
            complete = complete and value == 0
            result.append(complete)
        return tuple(result) + (False,) * (len(job.keys) - len(result))

    def drain_producer(self) -> None:
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device).synchronize()
        self._pending_prepare.clear()

    def drain(self, work: _DeviceWork) -> None:
        # Producer completion matters even if no consumer copy was submitted.
        work.ready.synchronize()
        self.stream.synchronize()

    def memory_usage(self) -> dict[str, int]:
        return {"gpu_transfer": self.gpu.numel(), "pinned_transfer": self.host.numel()}


class BlockTransfer:
    """Transport complete FPKB blocks through a fixed pool and one worker.

    Args:
        store: GPU store with caller-owned arena tensors and block lifetimes.
        client: Blocking byte client with bounded network waits.
        identity: Exact model, geometry and tables for this store.
        num_slots: Maximum simultaneous block transfers. Both GPU and pinned
            host pools reserve ``num_slots * store.blob_stride`` bytes.

    Submission must run on the model's owning thread/current stream, after all
    source layers are encoded. IDs must be valid and distinct; checking device
    values here would add a hot-path readback. A full pool refuses an entire
    submission immediately. The worker may synchronize only its owned copies;
    it never calls device-wide synchronization. ``close`` drains accepted work.
    """

    def __init__(
        self,
        store: HbmKvStore,
        client: BlobClient,
        identity: CacheIdentity,
        num_slots: int = 2,
    ) -> None:
        if (
            isinstance(num_slots, bool)
            or not isinstance(num_slots, int)
            or num_slots < 1
        ):
            raise ValueError("num_slots must be a positive integer")
        if (store.num_layers, store.kv_heads, store.table_digest) != (
            identity.num_layers,
            identity.kv_heads,
            identity.table_digest,
        ) or store.blob_stride != identity.max_blob_bytes:
            raise ValueError("cache identity does not describe this GPU store")
        self._identity = identity
        self._client = client
        self._buffers = _CudaBuffers(store, num_slots)
        self._free = list(range(num_slots))
        self._lock = threading.Lock()
        self._closed = False
        self._failed = False
        self._quarantine: list[object] = []
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="fragpack-block-transfer", daemon=True
        )
        self._thread.start()

    @property
    def failed(self) -> bool:
        """Whether uncertain CUDA completion disabled buffer reuse."""
        with self._lock:
            return self._failed

    def memory_usage(self) -> dict[str, int]:
        """Return reserved pool bytes, excluding transient RPC-owned byte copies."""
        return self._buffers.memory_usage()

    def submit_store(
        self, keys: Sequence[BlockKey], blocks: torch.Tensor
    ) -> StoreTicket | None:
        """Export closed source blocks, or refuse without submitting GPU work.

        The caller must preserve each source until the ticket's source event.
        Pool refusal leaves all source ownership with the caller.
        """
        job = self._submit(keys, blocks, restore=False)
        if job is None:
            return None
        return StoreTicket(job.device_work.ready, job.completion)

    def submit_restore(
        self, keys: Sequence[BlockKey], blocks: torch.Tensor
    ) -> RestoreTicket | None:
        """Restore a contiguous prefix into retained, unread destination blocks.

        The worker stops fetching at the first miss. Pool refusal makes no
        destination change. Await completion before exposing any destination.
        """
        job = self._submit(keys, blocks, restore=True)
        return None if job is None else RestoreTicket(job.completion)

    def close(self, timeout: float | None = None) -> None:
        """Stop admission and wait for accepted jobs; no CUDA device-wide wait.

        Client RPCs must have finite timeouts. Calling this from a completion
        callback running on the transfer worker is rejected to avoid self-join.
        Fatal CUDA failures retain pool owners even after close returns.
        A finite timeout raises TimeoutError if draining takes too long; the
        live worker keeps every owner and a subsequent close can finish waiting.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("close must not run on the transfer worker")
        with self._lock:
            if not self._closed:
                self._closed = True
                self._jobs.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("transfer worker is still draining; owners retained")

    def _submit(
        self, keys: Sequence[BlockKey], blocks: torch.Tensor, *, restore: bool
    ) -> _Job | None:
        names = tuple(keys)
        if not names or len(set(names)) != len(names):
            raise ValueError("a transfer requires distinct block keys")
        if any(key.namespace != self._identity.namespace for key in names):
            raise ValueError("block key belongs to another cache identity")
        self._buffers.validate_blocks(blocks, len(names))
        with self._lock:
            if self._closed or self._failed or len(self._free) < len(names):
                return None
            slots = tuple(self._free[: len(names)])
            del self._free[: len(names)]
            try:
                work = self._buffers.prepare(blocks, slots, restore)
            except BaseException:
                # Export or event recording may fail after work was submitted.
                # This exceptional drain is necessary before reporting safe
                # source reuse or returning the pool slots.
                try:
                    self._buffers.drain_producer()
                except BaseException as drain_error:
                    self._quarantine_locked(blocks)
                    raise RuntimeError(
                        "producer completion unknown; retain source blocks and "
                        "recover the worker process"
                    ) from drain_error
                self._free.extend(slots)
                raise
            completion: Future[tuple[bool, ...]] = Future()
            completion.set_running_or_notify_cancel()
            job = _Job(names, slots, work, completion, restore)
            self._jobs.put(job)
            return job

    def _run(self) -> None:
        while (job := self._jobs.get()) is not None:
            with self._lock:
                failed = self._failed
                if failed:
                    self._quarantine.append(job)
            if failed:
                job.completion.set_exception(
                    RuntimeError("CUDA completion unknown; process recovery required")
                )
                continue
            try:
                result = self._restore(job) if job.restore else self._store(job)
            except BaseException as failure:
                error = failure
                try:
                    self._buffers.drain(job.device_work)
                except BaseException as drain_error:
                    with self._lock:
                        self._quarantine_locked(job)
                    logger.exception(
                        "CUDA completion unknown; transfer owners retained"
                    )
                    error = RuntimeError(
                        "CUDA completion unknown; do not reuse source/destination "
                        f"blocks; process recovery required: {drain_error}"
                    )
                else:
                    with self._lock:
                        self._free.extend(job.slots)
                job.completion.set_exception(error)
            else:
                with self._lock:
                    self._free.extend(job.slots)
                job.completion.set_result(result)

    def _store(self, job: _Job) -> tuple[bool, ...]:
        blobs = self._buffers.read_export(job)
        result = []
        for key, blob in zip(job.keys, blobs, strict=True):
            valid = canonicalize_blob(blob, self._identity)
            published = self._client.put(key, valid)
            result.append(published)
            if not published:
                break
        return tuple(result) + (False,) * (len(job.keys) - len(result))

    def _restore(self, job: _Job) -> tuple[bool, ...]:
        blobs = []
        for key in job.keys:
            blob = self._client.get(key)
            if blob is None:
                break
            validate_blob(blob, self._identity)
            blobs.append(blob)
        return self._buffers.import_blobs(job, blobs)

    def _quarantine_locked(self, owner: object) -> None:
        if not self._failed:
            self._failed = True
            _quarantined_transfers.append(self)
        self._quarantine.append(owner)

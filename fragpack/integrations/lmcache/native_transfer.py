# SPDX-License-Identifier: Apache-2.0
"""Bounded native-page codec transport through the CPU-L1 byte service.

Engine page ownership stays with the caller. Codec pools and all blocking
device/RPC waits belong to one background thread, outside model execution.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass
import queue
import struct
import threading
import time
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from fragpack.compression import attn_page as ap
from fragpack.integrations.lmcache.client import BlobClient
from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity
from fragpack.integrations.lmcache.protocol import (
    NATIVE_RAW_MAGIC,
    canonicalize_blob,
    validate_blob,
)
from fragpack.integrations.lmcache.transfer import RestoreTicket
from fragpack.logging import init_logger
from fragpack.ops.native_codec import NativeKVPages, validate_tables
from fragpack.ops.native_codec_ext import extension
from fragpack.ops.native_codec_pool import NativeCodecPool

logger = init_logger(__name__)
TransportMode = Literal["compressed", "raw"]
_quarantined: list[NativeBlockTransfer] = []
_quarantined_workers: list[_NativeCudaWorker] = []


def _future() -> Future:
    value: Future = Future()
    value.set_running_or_notify_cancel()
    return value


def _combine(futures: Sequence[Future[bool]]) -> Future[tuple[bool, ...]]:
    """Join publication results without executing waits on the caller thread."""
    output = _future()
    lock = threading.Lock()

    def ready(_done: Future) -> None:
        with lock:
            if output.done() or not all(item.done() for item in futures):
                return
            try:
                values = tuple(item.result() for item in futures)
            except BaseException as error:
                output.set_exception(error)
            else:
                contiguous = True
                result = []
                for value in values:
                    contiguous = contiguous and value
                    result.append(contiguous)
                output.set_result(tuple(result))

    for future in futures:
        future.add_done_callback(ready)
    return output


def _producer_event(device: torch.device) -> torch.cuda.Event:
    with torch.cuda.device(device):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        return event


def _capture_map(
    page_map: torch.Tensor, indices: Sequence[int]
) -> tuple[torch.Tensor, torch.cuda.Event]:
    """Copy metadata on the producer stream; retain it through the final DMA."""
    with torch.cuda.device(page_map.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("native transfer submission does not support capture")
        owned = page_map[list(indices)].contiguous()
        event = _producer_event(page_map.device)
        return owned, event


def pack_native_raw(raw: bytes, identity: CacheIdentity) -> bytes:
    """Frame one packed raw block without entropy coding or GPU operations.

    Input order is [layer, K/V, head, token, dimension], BF16 bytes. The output
    uses FPNR v1 token-major raw records and canonical, zeroed header padding.
    """
    if identity.representation != "native-raw":
        raise ValueError("raw transport requires native-raw representation")
    count = identity.num_layers * 2 * identity.kv_heads
    if not isinstance(raw, bytes) or len(raw) != count * ap.RAW_RECORD_BYTES:
        raise ValueError("raw block length does not match cache identity")
    head = ap.blob_head_bytes(identity.num_layers, identity.kv_heads)
    header = bytearray(head)
    struct.pack_into(
        "<4Iqq",
        header,
        0,
        NATIVE_RAW_MAGIC,
        1,
        identity.num_layers,
        identity.kv_heads,
        head + len(raw),
        identity.table_digest,
    )
    struct.pack_into(
        f"<{count}i",
        header,
        ap.BLOB_FIXED_BYTES,
        *([ap.RAW_RECORD_BYTES - ap.DENSE_BYTES] * count),
    )
    return bytes(header) + raw


def unpack_native_raw(blob: bytes, identity: CacheIdentity) -> bytes:
    """Validate FPNR all-raw token-major records and return their BF16 payload."""
    if identity.representation != "native-raw":
        raise ValueError("raw transport requires native-raw representation")
    validate_blob(blob, identity)
    count = identity.num_layers * 2 * identity.kv_heads
    lengths = struct.unpack_from(f"<{count}i", blob, ap.BLOB_FIXED_BYTES)
    if any(value != ap.RAW_RECORD_BYTES - ap.DENSE_BYTES for value in lengths):
        raise ValueError("raw transport requires all-raw FPNR records")
    return blob[ap.blob_head_bytes(identity.num_layers, identity.kv_heads) :]


def _record_counters(
    blob: bytes, identity: CacheIdentity, mode: TransportMode
) -> dict[str, int]:
    """Read a previously validated host blob's table without copying its payload."""
    lengths = np.frombuffer(
        blob,
        dtype="<i4",
        count=2 * identity.num_layers * identity.kv_heads,
        offset=ap.BLOB_FIXED_BYTES,
    ).reshape(identity.num_layers, 2, identity.kv_heads)
    dense = identity.num_layers * identity.kv_heads * ap.DENSE_BYTES
    result = {
        "k_record_bytes": int(lengths[:, 0].sum()) + dense,
        "v_record_bytes": int(lengths[:, 1].sum()) + dense,
    }
    if mode == "raw":
        result["raw_records"] = lengths.size
    else:
        result["encoded_records"] = lengths.size
        result["raw_fallback_records"] = int(
            np.count_nonzero(lengths == ap.RAW_RECORD_BYTES - ap.DENSE_BYTES)
        )
    return result


@dataclass(frozen=True)
class NativeStoreTicket:
    """Source safety and publication are independent, non-cancellable Futures.

    Only successful ``source_ready`` permits source-page release. An exceptional
    source result means GPU completion is unknown and requires process recovery.
    ``completion`` reports a contiguous publication prefix, not source safety.
    """

    source_ready: Future[None]
    completion: Future[tuple[bool, ...]]


@dataclass
class _Job:
    keys: tuple[BlockKey, ...]
    page_map: torch.Tensor
    producer: torch.cuda.Event
    source_ready: Future[None]
    outputs: tuple[Future[bool], ...]
    completion: Future[tuple[bool, ...]] | None
    queued_at: float
    restore: bool


class _NativeCudaWorker:
    """Single-thread owner of directional codec pools and one transfer stream."""

    def __init__(
        self,
        pages: NativeKVPages,
        table: NDArray[np.uint8],
        identity: CacheIdentity,
        max_blocks: int,
        mode: TransportMode,
        producer: torch.cuda.Event,
    ) -> None:
        self.pages, self.identity, self.mode = pages, identity, mode
        self.metrics: Counter = Counter()
        self.pools: dict[str, NativeCodecPool] = {}
        self.stream = None
        try:
            with torch.cuda.device(pages.device):
                self.stream = torch.cuda.Stream(device=pages.device)
                self.stream.wait_event(producer)
                with torch.cuda.stream(self.stream):
                    self._initialize(table, max_blocks)
        except BaseException:
            # Constructor failure can follow table copies or event records.
            # Drain here while the partially constructed worker still owns them.
            try:
                producer.synchronize()
                if self.stream is not None:
                    self.stream.synchronize()
            except BaseException as drain_error:
                _quarantined_workers.append(self)
                raise RuntimeError(
                    "initialization CUDA completion unknown; process recovery required"
                ) from drain_error
            raise

    def _initialize(self, table: NDArray[np.uint8], max_blocks: int) -> None:
        # JIT compilation/loading belongs to blocking construction, never the
        # first admitted request or its asynchronous completion timeout.
        extension()
        roles = ("encode", "decode") if self.mode == "compressed" else ("raw",)
        for role in roles:
            self.pools[role] = NativeCodecPool(
                table,
                max_blocks=max_blocks,
                slots=1,
                workspace=role,
                device=self.pages.device,
            )
            with self.pools[role].acquire() as lease:
                if lease.codec.digest != self.identity.table_digest:
                    raise ValueError("cache identity has a different codec table")
        capacity = max_blocks * self.identity.max_blob_bytes
        self.host = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
        self.host_offsets = torch.empty(
            max_blocks + 1,
            dtype=torch.int64,
            pin_memory=True,
        )
        self.host_status = torch.empty(
            max_blocks,
            dtype=torch.int32,
            pin_memory=True,
        )

    def wait(self, producer: torch.cuda.Event) -> None:
        self.stream.wait_event(producer)

    def drain(self, producer: torch.cuda.Event) -> None:
        producer.synchronize()
        self.stream.synchronize()

    def memory_usage(self) -> dict[str, int]:
        return {
            **{
                f"gpu_{role}": sum(pool.memory_usage().values())
                for role, pool in self.pools.items()
            },
            "pinned_transfer": self.host.numel(),
            "pinned_metadata": self.host_offsets.numel() * 8
            + self.host_status.numel() * 4,
        }

    def encode(
        self,
        page_map: torch.Tensor,
        source_finished: Callable[[], None],
    ) -> tuple[bytes, ...]:
        n = len(page_map)
        role = "encode" if self.mode == "compressed" else "raw"
        start = time.perf_counter()
        with torch.cuda.device(self.pages.device), torch.cuda.stream(self.stream):
            with self.pools[role].acquire() as lease:
                codec = lease.codec
                if self.mode == "compressed":
                    codec.encode_records(self.pages, page_map)
                else:
                    codec.raw_gather(self.pages, page_map, lease.payload)
                source_done = torch.cuda.Event()
                source_done.record(self.stream)
                if self.mode == "compressed":
                    codec.plan()
                    codec.pack()
                    self.host_offsets[: n + 1].copy_(codec.offsets, non_blocking=True)
                self.host_status[:n].copy_(codec.status, non_blocking=True)
                source_done.synchronize()
                source_finished()
                self.stream.synchronize()
                if any(self.host_status[:n].tolist()):
                    raise ValueError("native encoding/gather rejected a source page")
                if self.mode == "compressed":
                    offsets = self.host_offsets[: n + 1].tolist()
                else:
                    raw_bytes = self.identity.num_layers * 2 * self.identity.kv_heads
                    raw_bytes *= ap.RAW_RECORD_BYTES
                    offsets = [i * raw_bytes for i in range(n + 1)]
                if (
                    offsets[0] != 0
                    or any(
                        not 0 < right - left <= self.identity.max_blob_bytes
                        for left, right in zip(offsets, offsets[1:], strict=False)
                    )
                    or offsets[-1] > self.host.numel()
                ):
                    raise ValueError("native encoder returned invalid blob offsets")
                self.metrics["encode_or_gather_seconds"] += time.perf_counter() - start
                start = time.perf_counter()
                self.host[: offsets[-1]].copy_(
                    lease.payload[: offsets[-1]],
                    non_blocking=True,
                )
                self.stream.synchronize()
                self.metrics["d2h_seconds"] += time.perf_counter() - start
                self.metrics["d2h_payload_bytes"] += offsets[-1]
                start = time.perf_counter()
                blobs = tuple(
                    self.host[left:right].numpy().tobytes()
                    for left, right in zip(offsets, offsets[1:], strict=False)
                )
                if self.mode == "raw":
                    blobs = tuple(
                        pack_native_raw(blob, self.identity) for blob in blobs
                    )
                else:
                    blobs = tuple(
                        canonicalize_blob(blob, self.identity) for blob in blobs
                    )
                for blob in blobs:
                    self.metrics.update(
                        _record_counters(blob, self.identity, self.mode)
                    )
                self.metrics["cpu_pack_seconds"] += time.perf_counter() - start
                return blobs

    def restore(
        self, page_map: torch.Tensor, blobs: Sequence[bytes]
    ) -> tuple[bool, ...]:
        if not blobs:
            return ()
        start = time.perf_counter()
        payloads = (
            [unpack_native_raw(blob, self.identity) for blob in blobs]
            if self.mode == "raw"
            else blobs
        )
        offsets = [0]
        for blob in payloads:
            end = offsets[-1] + len(blob)
            self.host[offsets[-1] : end].copy_(
                torch.frombuffer(bytearray(blob), dtype=torch.uint8),
            )
            offsets.append(end)
        n = len(blobs)
        self.host_offsets[: n + 1].copy_(torch.tensor(offsets, dtype=torch.int64))
        self.metrics["cpu_unpack_seconds"] += time.perf_counter() - start
        role = "decode" if self.mode == "compressed" else "raw"
        with torch.cuda.device(self.pages.device), torch.cuda.stream(self.stream):
            with self.pools[role].acquire() as lease:
                start = time.perf_counter()
                lease.payload[: offsets[-1]].copy_(
                    self.host[: offsets[-1]],
                    non_blocking=True,
                )
                if self.mode == "compressed":
                    lease.offsets[: n + 1].copy_(
                        self.host_offsets[: n + 1],
                        non_blocking=True,
                    )
                self.stream.synchronize()
                self.metrics["h2d_seconds"] += time.perf_counter() - start
                self.metrics["h2d_payload_bytes"] += offsets[-1]
                start = time.perf_counter()
                if self.mode == "compressed":
                    lease.codec.decode(
                        self.pages,
                        page_map[:n],
                        src=lease.payload[: offsets[-1]],
                        offsets=lease.offsets[: n + 1],
                    )
                else:
                    lease.codec.raw_scatter(self.pages, page_map[:n], lease.payload)
                self.host_status[:n].copy_(lease.codec.status, non_blocking=True)
                self.stream.synchronize()
                self.metrics["decode_or_scatter_seconds"] += time.perf_counter() - start
                good, result = True, []
                for status in self.host_status[:n].tolist():
                    good = good and status == 0
                    result.append(good)
                return tuple(result)


class NativeBlockTransfer:
    """Queue native-page stores/restores with bounded ownership and one worker.

    ``num_slots`` bounds queued/running batches, each at most ``max_blocks``;
    device work uses one encode and one decode slot, or one raw slot. Submission
    runs on the engine thread/current CUDA stream. The caller retains source
    pages until source_ready succeeds, and exclusive targets until completion.
    Maps are copied, but copying a page number does not pin the engine's page.
    Initialization waits for worker-owned pools. All subsequent normal-path
    waits, metadata readback and RPC are confined to the background thread.
    """

    def __init__(
        self,
        pages: NativeKVPages,
        table: NDArray[np.uint8],
        client: BlobClient,
        identity: CacheIdentity,
        *,
        max_blocks: int = 4,
        num_slots: int = 2,
        mode: TransportMode = "compressed",
        precheck: bool = True,
    ) -> None:
        validate_tables(table)
        if mode not in ("compressed", "raw"):
            raise ValueError("mode must be compressed or raw")
        representation = "native-raw" if mode == "raw" else "fpkb"
        if identity.representation != representation:
            raise ValueError("transfer mode and identity representation mismatch")
        if any(
            type(value) is not int or value < 1 for value in (max_blocks, num_slots)
        ):
            raise ValueError("max_blocks and num_slots must be positive integers")
        if type(precheck) is not bool:
            raise ValueError("precheck must be bool")
        if (pages.num_layers, pages.kv_heads) != (
            identity.num_layers,
            identity.kv_heads,
        ):
            raise ValueError("native pages and cache identity geometry differ")
        if tuple(table.shape[:3]) != (identity.num_layers, 2, identity.kv_heads):
            raise ValueError("codec table and cache identity geometry differ")
        self.pages, self.identity = pages, identity
        self.max_blocks, self.num_slots, self.mode = max_blocks, num_slots, mode
        self.precheck, self._client = precheck, client
        self._lock = threading.RLock()
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._inflight: dict[BlockKey, Future[bool]] = {}
        self._pending = 0
        self._closed = self._failed = False
        self._held: list[object] = []
        self._stats: Counter = Counter()
        self._initialized: Future[None] = _future()
        producer = _producer_event(pages.device)
        self._thread = threading.Thread(
            target=self._run,
            args=(table.copy(), producer),
            name="fragpack-native-transfer",
            daemon=True,
        )
        self._thread.start()
        self._initialized.result()

    @property
    def failed(self) -> bool:
        """Whether unknown device completion prohibits all page/buffer reuse."""
        with self._lock:
            return self._failed

    def memory_usage(self) -> dict[str, int]:
        """Return owned pool bytes and the bounded map allocation upper bound.

        Excludes caller KV/descriptors, allocator/runtime and RPC byte copies.
        Map tensors are per admitted batch; their upper bound is charged here.
        """
        return {
            **self._memory,
            "gpu_maps_bound": (self.num_slots + 1)
            * self.max_blocks
            * (ap.PAGE_TOKENS // self.pages.page_tokens)
            * 4,
        }

    def stats(self) -> dict[str, int | float]:
        """Return completed-operation counters and host timing sums, not TTFT."""
        with self._lock:
            return dict(self._stats)

    def _validate(
        self, keys: Sequence[BlockKey], page_map: torch.Tensor
    ) -> tuple[BlockKey, ...]:
        names = tuple(keys)
        if not 1 <= len(names) <= self.max_blocks or len(set(names)) != len(names):
            raise ValueError("a batch needs distinct keys within max_blocks")
        if any(key.namespace != self.identity.namespace for key in names):
            raise ValueError("block key belongs to another cache identity")
        if (
            page_map.device != self.pages.device
            or page_map.dtype != torch.int32
            or page_map.shape != (len(names), ap.PAGE_TOKENS // self.pages.page_tokens)
            or not page_map.is_contiguous()
        ):
            raise ValueError("page_map must be contiguous int32 [N,128/page_tokens]")
        return names

    def _capture(self, page_map: torch.Tensor, indices: Sequence[int]) -> tuple:
        try:
            return _capture_map(page_map, indices)
        except BaseException:
            try:
                torch.cuda.current_stream(self.pages.device).synchronize()
            except BaseException as error:
                self._quarantine(page_map)
                raise RuntimeError(
                    "producer completion unknown; process recovery required"
                ) from error
            raise

    def submit_store(
        self,
        keys: Sequence[BlockKey],
        page_map: torch.Tensor,
    ) -> NativeStoreTicket | None:
        """Admit a bounded batch; coalesce existing in-flight keys without waits.

        None refuses the entire submission before device work. Existing-key
        followers do not read their supplied source pages. An external presence
        probe occurs before encoding and is advisory rather than a residency lease.
        """
        names = self._validate(keys, page_map)
        with self._lock:
            if self._closed or self._failed:
                return None
            indices = [i for i, key in enumerate(names) if key not in self._inflight]
            if indices and self._pending >= self.num_slots:
                return None
            source = _future()
            if indices:
                owned, producer = self._capture(page_map, indices)
                fresh = tuple(names[i] for i in indices)
                outputs = tuple(_future() for _ in fresh)
                self._inflight.update(zip(fresh, outputs, strict=True))
                self._pending += 1
                self._jobs.put(
                    _Job(
                        fresh,
                        owned,
                        producer,
                        source,
                        outputs,
                        None,
                        time.perf_counter(),
                        False,
                    )
                )
            else:
                source.set_result(None)
            combined = _combine(tuple(self._inflight[key] for key in names))
            self._stats["store_requests"] += 1
            self._stats["coalesced_blocks"] += len(names) - len(indices)
            return NativeStoreTicket(source, combined)

    def submit_restore(
        self,
        keys: Sequence[BlockKey],
        page_map: torch.Tensor,
    ) -> RestoreTicket | None:
        """Restore only a contiguous available prefix into retained native pages."""
        names = self._validate(keys, page_map)
        with self._lock:
            if self._closed or self._failed or self._pending >= self.num_slots:
                return None
            owned, producer = self._capture(page_map, range(len(names)))
            output = _future()
            self._pending += 1
            self._stats["restore_requests"] += 1
            self._jobs.put(
                _Job(
                    names,
                    owned,
                    producer,
                    _future(),
                    (),
                    output,
                    time.perf_counter(),
                    True,
                )
            )
            return RestoreTicket(output)

    def close(self, timeout: float | None = None) -> None:
        """Stop admission and drain accepted work; timeout retains live owners."""
        if threading.current_thread() is self._thread:
            raise RuntimeError("close must not run on the transfer worker")
        with self._lock:
            if not self._closed:
                self._closed = True
                self._jobs.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("native transfer is still draining; owners retained")

    def _quarantine(self, owner: object) -> None:
        with self._lock:
            if not self._failed:
                self._failed = True
                _quarantined.append(self)
            self._held.append(owner)

    def _store(self, job: _Job) -> tuple[bool, ...]:
        started = time.perf_counter()
        present = (
            self._client.probe_many(job.keys)
            if self.precheck
            else (False,) * len(job.keys)
        )
        if len(present) != len(job.keys) or any(
            type(value) is not bool for value in present
        ):
            raise ValueError("invalid probe_many result")
        self._metrics["probe_seconds"] += time.perf_counter() - started
        self._metrics["prechecked_blocks"] += len(job.keys) if self.precheck else 0
        self._metrics["skipped_blocks"] += sum(present)
        missing = [i for i, value in enumerate(present) if not value]

        def source_finished() -> None:
            if not job.source_ready.done():
                self._metrics["source_hold_seconds"] += (
                    time.perf_counter() - job.queued_at
                )
                job.source_ready.set_result(None)

        self._device.wait(job.producer)
        if not missing:
            self._device.drain(job.producer)
            source_finished()
            return tuple(present)
        with (
            torch.cuda.device(self.pages.device),
            torch.cuda.stream(self._device.stream),
        ):
            selected = job.page_map[missing].contiguous()
        blobs = self._device.encode(selected, source_finished)
        self._metrics[
            "encoded_blocks" if self.mode == "compressed" else "raw_gathered_blocks"
        ] += len(missing)
        result = list(present)
        for index, blob in zip(missing, blobs, strict=True):
            started = time.perf_counter()
            stored = self._client.put(job.keys[index], blob)
            self._metrics["put_seconds"] += time.perf_counter() - started
            self._metrics["rpc_upload_bytes"] += len(blob)
            if type(stored) is not bool:
                raise ValueError("invalid put result")
            result[index] = stored
            self._metrics["published_blocks"] += int(stored)
            if not stored:
                for later in range(index + 1, len(result)):
                    result[later] = False
                break
        return tuple(result)

    def _restore(self, job: _Job) -> tuple[bool, ...]:
        blobs = []
        for key in job.keys:
            started = time.perf_counter()
            blob = self._client.get(key)
            self._metrics["get_seconds"] += time.perf_counter() - started
            if blob is None:
                break
            self._metrics["rpc_download_bytes"] += len(blob)
            validate_blob(blob, self.identity)
            blobs.append(blob)
        self._device.wait(job.producer)
        result = self._device.restore(job.page_map, blobs)
        self._metrics["restored_blocks"] += sum(result)
        return result + (False,) * (len(job.keys) - len(result))

    def _run(self, table: NDArray[np.uint8], producer: torch.cuda.Event) -> None:
        try:
            self._device = _NativeCudaWorker(
                self.pages,
                table,
                self.identity,
                self.max_blocks,
                self.mode,
                producer,
            )
            self._memory = self._device.memory_usage()
        except BaseException as error:
            self._initialized.set_exception(error)
            return
        self._initialized.set_result(None)
        while (job := self._jobs.get()) is not None:
            self._metrics: Counter = Counter()
            self._metrics["queue_wait_seconds"] += time.perf_counter() - job.queued_at
            error = None
            result = ()
            try:
                if self.failed:
                    raise RuntimeError(
                        "CUDA completion unknown; process recovery required"
                    )
                result = self._restore(job) if job.restore else self._store(job)
                self._device.drain(job.producer)
            except BaseException as failure:
                error = failure
                try:
                    self._device.drain(job.producer)
                except BaseException as drain_error:
                    self._quarantine(job)
                    logger.exception(
                        "Native transfer CUDA completion unknown; owners retained"
                    )
                    error = RuntimeError(
                        "CUDA completion unknown; process recovery required: "
                        f"{drain_error}"
                    )
            if self.failed and error is None:
                error = RuntimeError(
                    "CUDA completion unknown; process recovery required"
                )
            if not job.source_ready.done():
                if self.failed:
                    job.source_ready.set_exception(error)
                else:
                    job.source_ready.set_result(None)
            with self._lock:
                self._stats.update(self._metrics)
                self._stats.update(self._device.metrics)
                self._device.metrics.clear()
                if not self._failed:
                    self._pending -= 1
                else:
                    self._held.append(job)
                for key, future in zip(job.keys, job.outputs, strict=False):
                    if self._inflight.get(key) is future:
                        del self._inflight[key]
            if job.restore:
                if error is None:
                    job.completion.set_result(result)
                else:
                    job.completion.set_exception(error)
            else:
                for index, future in enumerate(job.outputs):
                    if error is None:
                        future.set_result(result[index])
                    else:
                        future.set_exception(error)

# SPDX-License-Identifier: Apache-2.0
"""Strict-lossless KV compression on the LMCache-driven transfer path.

With ``--lossless-kv-codec`` the MP server stores each chunk as the
variable-length record of ``lmcache.v1.lossless_kv.KVMajorChunkCodec`` instead
of its raw bytes, so L1 and L2 hold more chunks and L2 moves fewer bytes.

A record's length is known only after encoding, and reading it back
synchronizes the GPU. Every store and retrieve of one GPU runs on one handler
thread, so encoding there would make each retrieve wait behind the host
synchronizations of earlier stores. ``RecordStoreWorker`` moves that work to
its own thread and CUDA stream: the handler only enqueues, then returns. When
each engine block holds one chunk, it packs a batch's records straight from
the blocks into one buffer of the worker's pool; otherwise it gathers the
chunks into pool buffers and the worker encodes them. The worker reads the
record lengths, reserves one L1 object per record, copies it to host and
commits it; FragPack block blobs (``fragpack_blob_transfer``) are stored by
the same worker. Retrieves copy each record into the codec's record buffers
and decode with launch metadata uploaded once per batch shape, straight into
the engine blocks where possible and otherwise into scratch scattered into
the paged KV cache, all on the context stream without host synchronization.
"""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence
import queue
import threading
import time

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.lossless_kv.chunk_codec import KVMajorChunkCodec
from lmcache.v1.lossless_kv.kv_layout import layer_major_kv_view
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.object_group_transfer import recalculate_blocks_to_skip
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.multiprocess.config import LosslessKVConfig

logger = init_logger(__name__)

# The block transfer kernel moves at most this many chunks per launch.
_KERNEL_MAX_OBJECTS = 4

# KV cache dtypes the codec stores; vLLM keeps FP8 E4M3 caches as uint8.
_CODEC_KV_DTYPES = (torch.bfloat16, torch.float8_e4m3fn, torch.uint8)


def codec_byte_layout(num_bytes: int) -> MemoryLayoutDesc:
    """Return the flat byte layout of one stored codec object.

    Args:
        num_bytes (int): Stored length of the record.

    Returns:
        MemoryLayoutDesc: A one-dimensional uint8 layout of ``num_bytes``.
    """
    return MemoryLayoutDesc(shapes=[torch.Size((num_bytes,))], dtypes=[torch.uint8])


def create_chunk_codec(
    cache_context: BaseCacheContext, config: "LosslessKVConfig"
) -> KVMajorChunkCodec:
    """Build the chunk codec matching a registered KV cache.

    Args:
        cache_context (BaseCacheContext): The registered cache context.
        config (LosslessKVConfig): Codec options of the MP server.

    Returns:
        KVMajorChunkCodec: A codec whose raw chunk is one staged LMCache chunk
        of the context's only kernel group.

    Raises:
        ValueError: If the cache is not on CUDA, has more than one object or
            kernel group, is not full attention, or holds a dtype or K/V
            layout the codec does not support.
    """
    if cache_context.device.type != "cuda":
        raise ValueError("lossless KV codec requires a CUDA KV cache")
    groups = cache_context.kv_layer_groups_manager
    if groups.num_object_groups != 1 or groups.num_kernel_groups != 1:
        raise ValueError("lossless KV codec supports one object and kernel group")
    if not groups.get_attn_desc().is_full_attention(0):
        raise ValueError("lossless KV codec supports full attention only")
    group = groups.kernel_groups[0]
    if group.dtype not in _CODEC_KV_DTYPES:
        raise ValueError(f"lossless KV codec does not support {group.dtype} KV")
    sd = group.shape_desc
    # kv_size 1 is vLLM's fused layout: each head holds K then V, so the
    # staged chunk is [layers, tokens, heads, 2 * head_dim].
    if sd.kv_size == 2:
        head_dim, fused_kv = sd.hs, False
    elif sd.kv_size == 1 and sd.hs % 2 == 0:
        head_dim, fused_kv = sd.hs // 2, True
    else:
        raise ValueError("lossless KV codec supports separate or fused K/V only")
    chunk_tokens = cache_context.get_slots_per_chunk_in_sw(0)
    codec = KVMajorChunkCodec(
        num_layers=group.num_layers,
        chunk_tokens=chunk_tokens,
        num_kv_heads=sd.nh,
        head_dim=head_dim,
        kv_dtype=group.dtype,
        device=cache_context.device,
        max_chunks=config.batch_size,
        capacity_ratio=config.capacity_ratio,
        codebook_dir=config.codebook_dir,
        fused_kv=fused_kv,
    )
    shape, dtype = cache_context.get_kernel_group_shape_dtype(
        cache_context.lmcache_tokens_per_chunk, 0
    )
    staged_bytes = shape.numel() * dtype.itemsize
    if codec.raw_bytes != staged_bytes:
        raise ValueError(
            f"codec chunk is {codec.raw_bytes} bytes, LMCache stages {staged_bytes}"
        )
    return codec


def launch_paged_kv_copy(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    start_object_idx: int,
    buffers: Sequence[torch.Tensor],
    skip_tokens_in_chunk: int,
    direction: "lmcache_native.TransferDirection",
) -> None:
    """Copy consecutive chunks between the paged KV cache and chunk buffers.

    Args:
        cache_context (BaseCacheContext): Context of a single kernel group.
        block_ids_gpu (list[torch.Tensor]): Staged GPU block IDs per kernel
            group.
        start_object_idx (int): Index of the first chunk in the request.
        buffers (Sequence[torch.Tensor]): One raw chunk buffer per chunk.
        skip_tokens_in_chunk (int): Leading tokens of the first buffer left
            untouched in the paged cache.
        direction (lmcache_native.TransferDirection): H2D scatters the
            buffers into the paged cache; D2H gathers into them.

    Note:
        Enqueues on ``cache_context.stream`` without synchronizing it.
    """
    chunk_tokens = cache_context.lmcache_tokens_per_chunk
    kernel_group_id = 0
    blocks_per_chunk = cache_context.calculate_num_blocks(chunk_tokens, 0)
    tokens_per_window = min(
        chunk_tokens,
        cache_context.kv_layer_groups_manager.get_subchunk_sw_size_tokens(0),
    )
    blocks_per_window = cache_context.calculate_num_blocks(tokens_per_window, 0)
    kv_pointers = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
    # Wider batches are split; each part skips its share of the leading tokens.
    for part in range(0, len(buffers), _KERNEL_MAX_OBJECTS):
        part_buffers = buffers[part : part + _KERNEL_MAX_OBJECTS]
        part_skip = max(0, skip_tokens_in_chunk - part * chunk_tokens)
        if part_skip >= len(part_buffers) * chunk_tokens:
            continue
        first = start_object_idx + part
        block_ids = block_ids_gpu[kernel_group_id][
            first * blocks_per_window : (first + len(part_buffers)) * blocks_per_window
        ]
        skip_blocks = recalculate_blocks_to_skip(
            blocks_per_chunk,
            blocks_per_window,
            cache_context.calculate_num_blocks(part_skip, kernel_group_id),
        )
        device_ops.multi_layer_block_kv_transfer(
            kv_pointers,
            [buf.data_ptr() for buf in part_buffers],
            block_ids,
            cache_context.device,
            direction,
            cache_context.get_shape_desc(kernel_group_id),
            cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
            cache_context.get_engine_kv_format(kernel_group_id),
            skip_blocks,
        )


@dataclass
class RecordStoreJob:
    """One batch of gathered chunks waiting to be stored as records.

    Args:
        keys (list[ObjectKey]): Object key of each chunk.
        slots (list[int]): Pool slot holding each chunk, from ``acquire``.
        ready (torch.cuda.Event): Recorded after the gather filling the slots.
        records (Callable): Called on the worker thread, with the worker's
            stream current, on the slots' buffers; returns each chunk's
            record as a uint8 device view (None to skip the chunk).
        on_done (Callable): Called on the worker thread with the committed
            objects by key and whether the batch completed without raising.
    """

    keys: list[ObjectKey]
    slots: list[int]
    ready: torch.cuda.Event
    records: Callable[[list[torch.Tensor]], list[torch.Tensor | None]]
    on_done: Callable[[dict[ObjectKey, MemoryObj], bool], None]


class StoreTally:
    """Collects the batches of one store request and reports it once.

    The handler thread calls ``add_batch`` before submitting each batch and
    ``seal`` after the last; the worker calls ``batch_done``. ``report`` runs
    exactly once, on whichever thread finishes last, with the committed key
    count and bytes (both zero if any batch or the submission failed).

    Args:
        report (Callable[[int, int], None]): Called as
            ``report(stored_count, total_bytes)``.

    Thread-safety:
        All methods may be called from any thread.
    """

    def __init__(self, report: Callable[[int, int], None]) -> None:
        self._report = report
        self._lock = threading.Lock()
        self._pending = 0
        self._sealed = False
        self._ok = True
        self._count = 0
        self._bytes = 0

    def add_batch(self) -> None:
        """Count one batch that will later call ``batch_done``."""
        with self._lock:
            self._pending += 1

    def batch_done(self, stored: dict[ObjectKey, MemoryObj], ok: bool) -> None:
        """Account one finished batch.

        Args:
            stored (dict[ObjectKey, MemoryObj]): Objects the batch committed.
            ok (bool): False if the batch raised.
        """
        with self._lock:
            self._pending -= 1
            self._ok &= ok
            self._count += len(stored)
            self._bytes += sum(obj.get_size() for obj in stored.values())
            last = self._sealed and self._pending == 0
        if last:
            self._finish()

    def seal(self, ok: bool) -> None:
        """Mark that no more batches will be added.

        Args:
            ok (bool): False if submitting the batches failed.
        """
        with self._lock:
            self._sealed = True
            self._ok &= ok
            last = self._pending == 0
        if last:
            self._finish()

    def _finish(self) -> None:
        """Report the request's committed objects."""
        if self._ok:
            self._report(self._count, self._bytes)
        else:
            self._report(0, 0)


@dataclass
class _Pool:
    """Free-slot bookkeeping of the worker's gather buffers."""

    free: list[int]
    cond: threading.Condition = field(default_factory=threading.Condition)


class RecordStoreWorker:
    """Stores variable-length records of gathered chunks on its own thread
    and stream.

    Args:
        storage_manager (StorageManager): Reserves and commits L1 objects.
        device (torch.device): CUDA device of the cache context.
        num_slots (int): Number of chunk buffers in the gather pool.
        slot_bytes (int): Bytes per buffer, a 16-byte multiple; buffers are
            consecutive in :attr:`pool`.

    Thread-safety:
        ``acquire``, ``buffer`` and ``submit`` are called by the GPU handler
        thread, ``release`` by either thread; the rest runs on the worker.
        ``acquire`` blocks while the pool is empty, so stores are throttled
        rather than dropped when encoding falls behind. A record is visible to
        lookups once its batch commits, after the handler has returned.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
        device: torch.device,
        num_slots: int,
        slot_bytes: int,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("record store pool needs at least one slot")
        if slot_bytes <= 0 or slot_bytes % 16:
            raise ValueError("record store slots must be 16-byte multiples")
        self._storage = storage_manager
        self._device = device
        self._slot_bytes = slot_bytes
        with torch.cuda.device(device):
            self._stream = torch.cuda.Stream(device=device)
            self._pool_tensor = torch.empty(
                num_slots * slot_bytes, dtype=torch.uint8, device=device
            )
        self._num_slots = num_slots
        self._pool = _Pool(free=list(range(num_slots)))
        self._jobs: queue.Queue[RecordStoreJob | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="lossless-kv-store", daemon=True
        )
        self._thread.start()

    @property
    def num_slots(self) -> int:
        """Return the number of gather buffers in the pool."""
        return self._num_slots

    @property
    def pool(self) -> torch.Tensor:
        """Return the uint8 pool; slot ``i`` starts at :meth:`offset` ``(i)``."""
        return self._pool_tensor

    def offset(self, slot: int) -> int:
        """Return the byte offset of ``slot``'s buffer in :attr:`pool`."""
        return slot * self._slot_bytes

    def acquire(self, count: int) -> list[int]:
        """Take ``count`` free pool slots, waiting for the worker if needed.

        Args:
            count (int): Number of slots, at most ``num_slots``.

        Returns:
            list[int]: Slot indices, released once their batch is stored.

        Raises:
            ValueError: If ``count`` is out of range.
        """
        if not 0 < count <= self.num_slots:
            raise ValueError(f"cannot acquire {count} of {self.num_slots} slots")
        pool = self._pool
        with pool.cond:
            pool.cond.wait_for(lambda: len(pool.free) >= count)
            slots = pool.free[:count]
            del pool.free[:count]
        return slots

    def buffer(self, slot: int) -> torch.Tensor:
        """Return the buffer of ``slot`` (uint8, ``slot_bytes``)."""
        start = self.offset(slot)
        return self._pool_tensor[start : start + self._slot_bytes]

    def release(self, slots: list[int]) -> None:
        """Return acquired slots to the pool.

        The worker releases a batch's slots after storing it; the handler
        releases slots it acquired but could not submit.
        """
        pool = self._pool
        with pool.cond:
            pool.free.extend(slots)
            pool.cond.notify_all()

    def submit(self, job: RecordStoreJob) -> None:
        """Queue a gathered batch; the worker owns its slots from now on."""
        self._jobs.put(job)

    def close(self) -> None:
        """Store every queued batch, then stop the worker thread."""
        self._jobs.put(None)
        self._thread.join()

    def _run(self) -> None:
        """Worker loop: store queued batches in submission order."""
        with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
            while True:
                job = self._jobs.get()
                if job is None:
                    return
                self._store(job)

    def _store(self, job: RecordStoreJob) -> None:
        """Encode one batch, copy each record into its own L1 object, commit."""
        st = time.perf_counter()
        stored: dict[ObjectKey, MemoryObj] = {}
        ok = True
        try:
            self._stream.wait_event(job.ready)
            records = job.records([self.buffer(slot) for slot in job.slots])
            # Records of one length share a reservation.
            by_length: dict[int, list[int]] = {}
            for index, record in enumerate(records):
                if record is not None:
                    by_length.setdefault(record.numel(), []).append(index)
            reserved: dict[ObjectKey, MemoryObj] = {}
            for length, indices in by_length.items():
                objs = self._storage.reserve_write(
                    [job.keys[i] for i in indices], codec_byte_layout(length)
                )
                for i in indices:
                    obj = objs.get(job.keys[i])
                    if obj is not None:
                        lmcache_memcpy_async_d2h(records[i], obj)
                        reserved[job.keys[i]] = obj
            if reserved:
                completion = self._storage.prepare_write_completion(reserved)
                self._stream.synchronize()
                self._storage.finish_write_by_owner(completion)
            stored = reserved
        except Exception:
            ok = False
            logger.exception("Cannot store lossless KV records")
        finally:
            self.release(job.slots)
        logger.debug(
            "Stored %d lossless KV records in %.3f seconds",
            len(stored),
            time.perf_counter() - st,
        )
        job.on_done(stored, ok)


class LosslessKVTransfer:
    """Codec store and retrieve of one registered GPU cache context.

    Args:
        cache_context (BaseCacheContext): The registered cache context.
        storage_manager (StorageManager): The MP server's storage manager.
        config (LosslessKVConfig): Codec options of the MP server.

    Raises:
        ValueError: If the codec does not support the cache (see
            ``create_chunk_codec``).

    Thread-safety:
        ``submit_store`` and ``retrieve`` run on the context's GPU handler
        thread and enqueue on ``cache_context.stream``.
    """

    def __init__(
        self,
        cache_context: BaseCacheContext,
        storage_manager: "StorageManager",
        config: "LosslessKVConfig",
    ) -> None:
        self._cache_context = cache_context
        self._storage = storage_manager
        self._codec = create_chunk_codec(cache_context, config)
        self._width = config.batch_size
        # Retrieved records are copied in on their own stream, so the copies
        # of later batches run ahead of the decodes on the context stream.
        self._fill_stream = torch.cuda.Stream(device=cache_context.device)
        self._engine_direct = self._attach_engine_cache()
        # Packing from engine blocks writes a batch's records back to back,
        # so each pool slot then holds one whole batch.
        self._engine_store = self._engine_direct and self._codec.engine_store
        pool_chunks = max(self._width, config.store_pool_chunks)
        self._store_worker = RecordStoreWorker(
            storage_manager,
            cache_context.device,
            pool_chunks // self._width if self._engine_store else pool_chunks,
            self._codec.raw_bytes * (self._width if self._engine_store else 1),
        )

    def layout_desc(self, num_tokens: int) -> MemoryLayoutDesc:
        """Return the registered object layout: a bound on stored records.

        Each stored object is sized from its own record, so this layout only
        bounds them.

        Args:
            num_tokens (int): Tokens of one LMCache object.

        Returns:
            MemoryLayoutDesc: A flat uint8 layout of the largest record.
        """
        chunks = num_tokens // self._cache_context.lmcache_tokens_per_chunk
        return codec_byte_layout(chunks * self._codec.capacity_bytes)

    def submit_store(
        self,
        block_ids_gpu: list[torch.Tensor],
        block_ids_host: list[list[int]],
        obj_keys: list[ObjectKey],
        skip_mask: Sequence[bool],
        tally: StoreTally,
    ) -> list[bool]:
        """Encode or gather chunks into the store worker's pool and queue them.

        Keys already resident in L1 are skipped up front, as ``reserve_write``
        would refuse them, so they cost no gather or encode. The remaining
        chunks go a batch at a time: when the engine's blocks each hold one
        chunk, the batch's records are packed straight from those blocks into
        one pool slot; otherwise the chunks are gathered into pool slots for
        the worker to encode. Each batch is handed to the worker with an
        event recorded after its pack or gather, so the caller's completion
        event (recorded later on the same stream) already lets the engine
        reuse the blocks.

        Args:
            block_ids_gpu (list[torch.Tensor]): Staged GPU block IDs.
            block_ids_host (list[list[int]]): The same block IDs on the host.
            obj_keys (list[ObjectKey]): Keys of the chunks, in token order.
            skip_mask (Sequence[bool]): Chunks carrying no valid KV.
            tally (StoreTally): Accounts each submitted batch.

        Returns:
            list[bool]: Per chunk, whether it was queued for storing.

        Note:
            Never synchronizes the GPU; blocks only while the worker's pool
            has no free slots.
        """
        worker = self._store_worker
        in_l1 = self._storage.l1_contains(obj_keys)
        todo = [i for i in range(len(obj_keys)) if not (skip_mask[i] or in_l1[i])]
        queued = [False] * len(obj_keys)
        width = (
            self._width if self._engine_store else min(self._width, worker.num_slots)
        )
        for window_start in range(0, len(todo), width):
            window = todo[window_start : window_start + width]
            if self._engine_store:
                job = self._pack_engine_window(
                    [block_ids_host[0][i] for i in window],
                    [obj_keys[i] for i in window],
                    tally,
                )
                tally.add_batch()
                worker.submit(job)
                for i in window:
                    queued[i] = True
                continue
            slots = worker.acquire(len(window))
            try:
                # Gather each contiguous run of chunks with one copy.
                run_start = 0
                for pos in range(1, len(window) + 1):
                    if pos == len(window) or window[pos] != window[pos - 1] + 1:
                        launch_paged_kv_copy(
                            self._cache_context,
                            block_ids_gpu,
                            window[run_start],
                            [worker.buffer(s) for s in slots[run_start:pos]],
                            0,
                            lmcache_native.TransferDirection.D2H,
                        )
                        run_start = pos
                ready = torch.cuda.Event()
                ready.record(self._cache_context.stream)
            except BaseException:
                worker.release(slots)
                raise
            tally.add_batch()
            worker.submit(
                RecordStoreJob(
                    keys=[obj_keys[i] for i in window],
                    slots=slots,
                    ready=ready,
                    records=self._encode,
                    on_done=tally.batch_done,
                )
            )
            for i in window:
                queued[i] = True
        return queued

    def retrieve(
        self,
        block_ids_gpu: list[torch.Tensor],
        block_ids_host: list[list[int]],
        memory_objs: Sequence[MemoryObj | None],
        skip_first_n_tokens: int,
    ) -> None:
        """Decode stored records into the paged KV cache.

        When the engine's blocks each hold one chunk in a layout the decoder
        addresses, compressed records decode straight into them; otherwise
        (and for batches with a raw record or a partly skipped chunk) they
        decode into codec scratch and are scattered.

        Args:
            block_ids_gpu (list[torch.Tensor]): Staged GPU block IDs.
            block_ids_host (list[list[int]]): The same block IDs on the host.
            memory_objs (Sequence[MemoryObj | None]): One stored record per
                chunk in token order; None for leading chunks not retrieved.
            skip_first_n_tokens (int): Leading tokens not written to the
                paged cache (APC-shared blocks).

        Raises:
            ValueError: If a None entry follows a record, or a record length
                is invalid for the codec.

        Note:
            Copies records in on a private stream and decodes on
            ``cache_context.stream``, which is ordered after the copies;
            never synchronizes.
        """
        chunk_tokens = self._cache_context.lmcache_tokens_per_chunk
        stream = self._cache_context.stream
        first = next(
            (i for i, obj in enumerate(memory_objs) if obj is not None),
            len(memory_objs),
        )
        # Chunks wholly inside the skipped prefix need no copy or decode.
        first = max(first, skip_first_n_tokens // chunk_tokens)
        engine_blocks = None
        if self._engine_direct and first < len(memory_objs):
            # One block per chunk; the pinned copy stays alive until it runs.
            ids = block_ids_host[0][first : len(memory_objs)]
            with torch.cuda.stream(stream):
                engine_blocks = torch.tensor(
                    ids, dtype=torch.int32, pin_memory=True
                ).to(self._cache_context.device, non_blocking=True)
        for start in range(first, len(memory_objs), self._width):
            batch = memory_objs[start : start + self._width]
            if any(obj is None for obj in batch):
                raise ValueError("lossless KV retrieve has a missing record")
            sizes = [obj.get_size() for obj in batch]
            buffers = self._codec.record_buffers(len(batch), self._fill_stream)
            with torch.cuda.stream(self._fill_stream):
                for buf, obj, size in zip(buffers, batch, sizes, strict=True):
                    lmcache_memcpy_async_h2d(obj, buf[:size])
            skip_tokens = max(0, skip_first_n_tokens - start * chunk_tokens)
            if (
                engine_blocks is not None
                and skip_tokens == 0
                and self._codec.decode_records_to_engine(
                    sizes,
                    engine_blocks[start - first : start - first + len(batch)],
                    stream,
                )
            ):
                continue
            chunks = self._codec.decode_records(sizes, stream)
            launch_paged_kv_copy(
                self._cache_context,
                block_ids_gpu,
                start,
                chunks,
                skip_tokens,
                lmcache_native.TransferDirection.H2D,
            )

    def close(self) -> None:
        """Store every queued batch, then stop the store worker.

        Queued batches hold copies of the context's KV blocks, so they are
        stored before the context goes away.
        """
        self._store_worker.close()

    def _attach_engine_cache(self) -> bool:
        """Let the codec decode into the engine blocks; return whether it can.

        Requires one engine block per chunk and per-layer fused K/V tensors
        laid back to back in one allocation.
        """
        context = self._cache_context
        chunk_tokens = context.lmcache_tokens_per_chunk
        group = context.kv_layer_groups_manager.kernel_groups[0]
        sd = group.shape_desc
        view = None
        if sd.kv_size == 1 and context.calculate_num_blocks(chunk_tokens, 0) == 1:
            view = layer_major_kv_view(
                [context.kv_caches_[i] for i in group.layer_indices],
                sd.nh,
                sd.hs // 2,
            )
        if view is None:
            logger.info("Lossless KV retrieve decodes via codec scratch")
            return False
        self._codec.attach_engine_cache(view)
        logger.info("Lossless KV retrieve decodes straight into engine blocks")
        return True

    def _pack_engine_window(
        self, blocks: list[int], keys: list[ObjectKey], tally: StoreTally
    ) -> RecordStoreJob:
        """Pack one batch from engine blocks into a pool slot; return its job."""
        worker = self._store_worker
        stream = self._cache_context.stream
        (slot,) = worker.acquire(1)
        try:
            staging = worker.buffer(slot)
            pending = self._codec.encode_engine_blocks(blocks, staging, stream)
            ready = torch.cuda.Event()
            ready.record(stream)
        except BaseException:
            worker.release([slot])
            raise

        def records(buffers: list[torch.Tensor]) -> list[torch.Tensor | None]:
            stream = torch.cuda.current_stream()
            return list(self._codec.engine_records(pending, buffers[0], stream))

        return RecordStoreJob(
            keys=keys,
            slots=[slot],
            ready=ready,
            records=records,
            on_done=tally.batch_done,
        )

    def _encode(self, chunks: list[torch.Tensor]) -> list[torch.Tensor | None]:
        """Encode chunks in place on the current stream; return the records."""
        lengths = self._codec.encode(chunks, torch.cuda.current_stream())
        return [chunk[:length] for chunk, length in zip(chunks, lengths, strict=True)]

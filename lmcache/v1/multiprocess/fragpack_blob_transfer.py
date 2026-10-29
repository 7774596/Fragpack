# SPDX-License-Identifier: Apache-2.0
"""FragPack block blobs on the LMCache-driven transfer path.

With FragPack attention the engine's paged KV cache holds compressed records,
and each block is located through record tables rather than a fixed page
layout, so LMCache cannot copy its blocks raw. The worker announces FragPack
with the ``kv_codec`` layout hint and, once its attention store exists, shares
the store's tables over CUDA IPC (``register_fragpack_store``). This module
then stores each LMCache chunk (one vLLM block) as the block's blob: its
records, still compressed, behind a small header (``fragpack.attn_page``).

Stores export the blocks into the shared ``RecordStoreWorker`` pool on the
context stream; the worker reads the blob lengths back, reserves one L1 object
per blob and commits it, off the handler thread. Retrieves copy the blobs into
a staging buffer and install them into the destination blocks, then wait for
the import status once per request: a blob that fails to install (overflow
pool full, mismatched codec tables) fails the retrieve, so the engine
recomputes instead of reading a block whose tables were not written.
"""

# Standard
from typing import TYPE_CHECKING, Callable, Sequence

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.gpu_connector.gpu_ops import lmcache_memcpy_async_h2d
from lmcache.v1.lossless_kv.fragpack import attn_page as ap
from lmcache.v1.lossless_kv.fragpack.hbm_kv import (
    BLOCK_TOKENS,
    BlobPort,
    BlobTables,
    blob_failures,
)
from lmcache.v1.lossless_kv.fragpack.registry import KV_CODEC_BF16, KV_CODEC_FP8
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.lossless_kv_transfer import (
    RecordStoreJob,
    RecordStoreWorker,
    StoreTally,
    codec_byte_layout,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.multiprocess.config import LosslessKVConfig

logger = init_logger(__name__)

FRAGPACK_KV_CODECS = (KV_CODEC_BF16, KV_CODEC_FP8)


def _blob_records(
    lengths: torch.Tensor, status: torch.Tensor
) -> Callable[[list[torch.Tensor]], list[torch.Tensor | None]]:
    """Return a job's ``records`` callback slicing exported blobs.

    Args:
        lengths (torch.Tensor): int64 device blob lengths of the batch.
        status (torch.Tensor): int32 device export status of the batch.

    Returns:
        Callable mapping the slots' buffers to their blobs; a block that was
        not closed (nonzero status) is skipped.
    """

    def records(buffers: list[torch.Tensor]) -> list[torch.Tensor | None]:
        # Runs on the worker stream after the job's ready event.
        host_lengths = lengths.cpu().tolist()
        host_status = status.cpu().tolist()
        failed = [bad for bad in host_status if bad]
        if failed:
            # The engine stores full blocks only, so this means a block was
            # freed or reopened before its store ran.
            logger.warning(
                "%d FragPack blobs of a store batch not exported: %s",
                len(failed),
                blob_failures(failed[0]),
            )
        return [
            None if bad else buf[:length]
            for buf, length, bad in zip(buffers, host_lengths, host_status, strict=True)
        ]

    return records


class FragPackBlobTransfer:
    """Block-blob store and retrieve of one registered FragPack cache.

    Args:
        cache_context (BaseCacheContext): The registered cache context; its
            kernel group addresses the FragPack pages only for bookkeeping.
        storage_manager (StorageManager): The MP server's storage manager.
        config (LosslessKVConfig): Batch and pool sizes (the codec switch
            itself is not needed).
        kv_codec (str): The ``kv_codec`` layout hint, BF16 or FP8 records.
        num_kv_heads (int): The ``num_kv_heads`` layout hint.

    Raises:
        ValueError: If the cache is not on CUDA, is not one full-attention
            object and kernel group, or an LMCache chunk is not one
            ``BLOCK_TOKENS`` block.

    Thread-safety:
        ``attach``, ``submit_store`` and ``retrieve`` run on the context's GPU
        handler thread and enqueue on ``cache_context.stream``.
    """

    def __init__(
        self,
        cache_context: BaseCacheContext,
        storage_manager: "StorageManager",
        config: "LosslessKVConfig",
        kv_codec: str,
        num_kv_heads: int,
    ) -> None:
        if kv_codec not in FRAGPACK_KV_CODECS:
            raise ValueError(f"{kv_codec!r} is not a FragPack KV codec")
        if cache_context.device.type != "cuda":
            raise ValueError("FragPack blobs require a CUDA KV cache")
        groups = cache_context.kv_layer_groups_manager
        if groups.num_object_groups != 1 or groups.num_kernel_groups != 1:
            raise ValueError("FragPack blobs support one object and kernel group")
        if not groups.get_attn_desc().is_full_attention(0):
            raise ValueError("FragPack blobs support full attention only")
        chunk_tokens = cache_context.lmcache_tokens_per_chunk
        if (
            chunk_tokens != BLOCK_TOKENS
            or cache_context.calculate_num_blocks(chunk_tokens, 0) != 1
        ):
            raise ValueError(
                f"FragPack blobs need {BLOCK_TOKENS}-token chunks of one block"
            )
        self._cache_context = cache_context
        self._storage = storage_manager
        self._num_layers = groups.kernel_groups[0].num_layers
        self._kv_heads = num_kv_heads
        self._fp8 = kv_codec == KV_CODEC_FP8
        self._slot_bytes = ap.blob_slot_bytes(self._num_layers, num_kv_heads, self._fp8)
        self._width = config.batch_size
        self._port: BlobPort | None = None
        device = cache_context.device
        self._store_worker = RecordStoreWorker(
            storage_manager,
            device,
            max(self._width, config.store_pool_chunks),
            self._slot_bytes,
        )
        with torch.cuda.device(device):
            self._retrieve_staging = torch.empty(
                self._width * self._slot_bytes, dtype=torch.uint8, device=device
            )

    @property
    def attached(self) -> bool:
        """Return whether the worker's store tables have been attached."""
        return self._port is not None

    def attach(self, tables: BlobTables) -> None:
        """Attach (or replace) the store tables shared by the worker.

        Args:
            tables (BlobTables): The worker store's tables mapped into this
                process.

        Raises:
            ValueError: If the tables' geometry does not match the
                registration.
        """
        port = BlobPort(tables)
        if (port.num_layers, port.kv_heads, port.fp8) != (
            self._num_layers,
            self._kv_heads,
            self._fp8,
        ):
            raise ValueError(
                f"FragPack store has {port.num_layers} layers x {port.kv_heads} "
                f"heads (fp8={port.fp8}); registered {self._num_layers} x "
                f"{self._kv_heads} (fp8={self._fp8})"
            )
        if port.blob_stride != self._slot_bytes:
            raise ValueError("FragPack store blob size disagrees with the hints")
        self._port = port

    def layout_desc(self, num_tokens: int) -> MemoryLayoutDesc:
        """Return the registered object layout: a bound on stored blobs.

        Args:
            num_tokens (int): Tokens of one LMCache object.

        Returns:
            MemoryLayoutDesc: A flat uint8 layout of the largest blob.
        """
        chunks = num_tokens // self._cache_context.lmcache_tokens_per_chunk
        return codec_byte_layout(chunks * self._slot_bytes)

    def submit_store(
        self,
        block_ids_host: list[list[int]],
        obj_keys: list[ObjectKey],
        skip_mask: Sequence[bool],
        tally: StoreTally,
    ) -> list[bool]:
        """Export blocks into the store worker's pool and queue them.

        Keys already resident in L1 are skipped up front. Each batch is handed
        to the worker with an event recorded after its export, so the
        caller's completion event already lets the engine reuse the blocks.

        Args:
            block_ids_host (list[list[int]]): Block ids of the only kernel
                group, one per chunk.
            obj_keys (list[ObjectKey]): Keys of the chunks, in token order.
            skip_mask (Sequence[bool]): Chunks carrying no valid KV.
            tally (StoreTally): Accounts each submitted batch.

        Returns:
            list[bool]: Per chunk, whether it was queued for storing; none
            before the worker attached its store.

        Note:
            Never synchronizes the GPU; blocks only while the worker's pool
            has no free slots.
        """
        queued = [False] * len(obj_keys)
        port = self._port
        if port is None:
            return queued
        worker = self._store_worker
        blocks = block_ids_host[0]
        in_l1 = self._storage.l1_contains(obj_keys)
        todo = [i for i in range(len(obj_keys)) if not (skip_mask[i] or in_l1[i])]
        width = min(self._width, worker.num_slots)
        for window_start in range(0, len(todo), width):
            window = todo[window_start : window_start + width]
            slots = worker.acquire(len(window))
            try:
                lengths, status = port.export_blocks(
                    [blocks[i] for i in window],
                    worker.pool,
                    [worker.offset(s) for s in slots],
                )
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
                    records=_blob_records(lengths, status),
                    on_done=tally.batch_done,
                )
            )
            for i in window:
                queued[i] = True
        return queued

    def retrieve(
        self,
        block_ids_host: list[list[int]],
        memory_objs: Sequence[MemoryObj | None],
        skip_first_n_tokens: int,
    ) -> None:
        """Install stored blobs into the destination blocks.

        Args:
            block_ids_host (list[list[int]]): Block ids of the only kernel
                group, one per chunk.
            memory_objs (Sequence[MemoryObj | None]): One stored blob per
                chunk in token order; None for leading chunks not retrieved.
            skip_first_n_tokens (int): Leading tokens not written (APC-shared
                blocks); a multiple of the block size.

        Raises:
            RuntimeError: If the worker has not attached its store, or a blob
                did not install (its blocks must be recomputed).
            ValueError: If a None entry follows a blob, a blob is larger than
                a slot, or the skip splits a block.

        Note:
            Enqueues on ``cache_context.stream``, then waits for that stream
            once to read the import status.
        """
        port = self._port
        if port is None:
            raise RuntimeError("FragPack store tables are not attached yet")
        if skip_first_n_tokens % BLOCK_TOKENS:
            raise ValueError(
                f"FragPack retrieve cannot skip {skip_first_n_tokens} tokens of a "
                f"{BLOCK_TOKENS}-token block"
            )
        blocks = block_ids_host[0]
        first = next(
            (i for i, obj in enumerate(memory_objs) if obj is not None),
            len(memory_objs),
        )
        first = max(first, skip_first_n_tokens // BLOCK_TOKENS)
        slot = self._slot_bytes
        statuses: list[torch.Tensor] = []
        for start in range(first, len(memory_objs), self._width):
            batch = memory_objs[start : start + self._width]
            offsets: list[int] = []
            for index, obj in enumerate(batch):
                if obj is None:
                    raise ValueError("FragPack retrieve has a missing blob")
                size = obj.get_size()
                if size > slot:
                    raise ValueError(f"FragPack blob of {size} B exceeds {slot} B")
                offset = index * slot
                lmcache_memcpy_async_h2d(
                    obj, self._retrieve_staging[offset : offset + size]
                )
                offsets.append(offset)
            statuses.append(
                port.import_blocks(
                    self._retrieve_staging,
                    offsets,
                    blocks[start : start + len(batch)],
                )
            )
        if not statuses:
            return
        status = torch.cat(statuses).cpu()
        failed = status.nonzero().flatten().tolist()
        if failed:
            raise RuntimeError(
                f"{len(failed)} FragPack blobs did not install: "
                f"{blob_failures(int(status[failed[0]]))}"
            )

    def close(self) -> None:
        """Store every queued batch, then stop the store worker."""
        self._store_worker.close()

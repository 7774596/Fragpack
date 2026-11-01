# SPDX-License-Identifier: Apache-2.0
"""Public-API-only, CPU L1 access for pinned LMCache 0.5.6.

Stock StorageManager lacks write cancellation and per-key finish results.
This adapter confirms publication by reading the resident object. Uncertain
failures stop this adapter; it never force-deletes keys or accesses private
owners. Failed staging stays under LMCache's existing expiration contract.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Sequence

import torch

from fragpack.integrations.lmcache.client import MAX_PREFIX_KEYS
from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity
from fragpack.integrations.lmcache.protocol import canonicalize_blob
from fragpack.logging import init_logger

logger = init_logger(__name__)


def _load_api() -> Any:
    from fragpack.integrations.lmcache.client import require_lmcache_056

    require_lmcache_056()
    from lmcache.v1.distributed import api

    return api


class LMCacheL1Store:
    """Read and publish exact-length FPKB objects through a shared manager.

    Args:
        storage_manager: Stock 0.5.6 public StorageManager.
        identity: Fixed representation/model/shard namespace for this store.
        timeout_s: Maximum wait per background prefetch operation.

    Calls serialize under a host lock and belong to server/background threads.
    All copies into LMCache reservations are synchronous CPU copies; CUDA DMA
    never targets reservations that the manager may expire. Caller shutdown
    must close this adapter before closing the shared manager.

    The full reserve/copy/finish interval must stay within LMCache's write TTL.
    Owner IDs identify managers, not reservation generations. This adapter's
    host lock cannot protect a paused writer from LMCache's expiry/eviction or
    a second plugin instance; the stock API offers no lease-extension contract.
    """

    def __init__(
        self, storage_manager: Any, identity: CacheIdentity, timeout_s: float = 30.0
    ) -> None:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("L1 timeout must be finite and positive")
        self._manager = storage_manager
        self._identity = identity
        self._api = _load_api()
        self._timeout_s = timeout_s
        self._lock = threading.RLock()
        self._failed = False
        self._closed = False
        self._pending_reads: list[tuple[Any, Any]] = []
        self._check_l2()

    def _check_l2(self) -> None:
        # Recheck on every operation: adapters can be added at runtime.
        if self._manager.l2_adapters():
            raise RuntimeError(
                "FragPack 0.5.6 plugin currently requires CPU L1 with no L2 adapters; "
                "stock variable-length L2 restore is not supported"
            )

    def _check_open(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError(
                "FragPack L1 store is closed or failed; restart required"
            )
        self._check_l2()

    def _key(self, key: BlockKey) -> Any:
        if key.namespace != self._identity.namespace:
            raise ValueError("Block key belongs to another cache identity")
        return self._api.ObjectKey(
            chunk_hash=bytes.fromhex(key.prefix_hash),
            model_name="fragpack-" + key.namespace,
            kv_rank=self._identity.rank,
            object_group_id=0,
            cache_salt=key.cache_salt,
        )

    def _layout(self, length: int) -> Any:
        return self._api.MemoryLayoutDesc(
            shapes=[torch.Size((length,))], dtypes=[torch.uint8]
        )

    def _lookup(self, key: Any) -> Any | None:
        spec = self._api.PrefetchTaskSpec(
            key_groups=[
                self._api.GroupedObjectKeys(
                    keys=[key],
                    object_group_id=0,
                    layout_desc=self._layout(self._identity.max_blob_bytes),
                )
            ]
        )
        handle = self._manager.submit_prefetch_task(spec, skip_l2=True)
        self._pending_reads.append((handle, key))
        try:
            self._manager.wait_prefetch_status(handle, self._timeout_s)
            result = self._manager.query_prefetch_status(handle)
        except Exception:
            self._failed = True
            raise
        if result is None:
            self._failed = True
            raise TimeoutError("L1 prefetch timed out; publication is not confirmed")
        self._pending_reads.pop()
        if not result.hit_cells or not result.hit_cells[0].get_indices_list():
            return None
        return result

    @staticmethod
    def _tensor(memory_obj: Any, expected_length: int | None = None) -> torch.Tensor:
        tensor = memory_obj.tensor
        if (
            tensor is None
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.uint8
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError("FragPack L1 requires a contiguous CPU uint8 object")
        if expected_length is not None and tensor.numel() != expected_length:
            raise ValueError("L1 allocation length differs from the requested bytes")
        return tensor

    def _get(self, key: Any) -> bytes | None:
        result = self._lookup(key)
        if result is None:
            return None
        # The context manager releases reads on failure, but successful exits
        # require explicit finish_read_prefetched (official 0.5.6 contract).
        with self._manager.read_prefetched_results([key], result.l1_owners) as objects:
            if objects is None:
                return None
            tensor = self._tensor(objects[0])
            if tensor.numel() > self._identity.max_blob_bytes:
                raise ValueError("Resident object exceeds the FPKB size bound")
            blob = canonicalize_blob(tensor.numpy().tobytes(), self._identity)
        self._manager.finish_read_prefetched([key], l1_owners=result.l1_owners)
        return blob

    def get(self, key: BlockKey) -> bytes | None:
        """Return validated CPU bytes, releasing all successful read locks."""
        with self._lock:
            self._check_open()
            return self._get(self._key(key))

    def put(self, key: BlockKey, blob: bytes) -> bool:
        """Publish exact bytes, or refuse a reservation without claiming a hit.

        Existing different bytes are a key conflict. Any exception after reserve
        disables the adapter because stock 0.5.6 cannot cancel by write owner.
        The exception does not imply that the remote write was rolled back.
        """
        blob = canonicalize_blob(blob, self._identity)
        with self._lock:
            self._check_open()
            obj_key = self._key(key)
            resident = self._get(obj_key)
            if resident is not None:
                if resident != blob:
                    raise ValueError("Different FPKB contents already use this key")
                return True
            try:
                objects = self._manager.reserve_write(
                    [obj_key], self._layout(len(blob))
                )
                if obj_key not in objects:
                    return False
                completion = self._manager.prepare_write_completion(objects)
                tensor = self._tensor(objects[obj_key], len(blob))
                # The mutable bytearray avoids a non-writable frombuffer view.
                tensor.copy_(torch.frombuffer(bytearray(blob), dtype=torch.uint8))
                self._manager.finish_write_by_owner(completion)
                resident = self._get(obj_key)
                if resident is None:
                    # A failed completion and immediate eviction are
                    # indistinguishable through the stock public interface.
                    self._failed = True
                    logger.error(
                        "Publication could not be confirmed; admission stopped"
                    )
                    return False
                if resident != blob:
                    raise ValueError(
                        "Published object differs from this writer's bytes"
                    )
                return True
            except Exception:
                self._failed = True
                logger.exception(
                    "L1 write outcome uncertain; admission stopped. Stock LMCache "
                    "0.5.6 has no owner-scoped cancellation; no key was force-deleted."
                )
                raise

    def lookup_prefix(self, keys: Sequence[BlockKey]) -> int:
        """Return an advisory contiguous L1 prefix and release lookup locks."""
        with self._lock:
            self._check_open()
            count = 0
            for key in keys:
                obj_key = self._key(key)
                result = self._lookup(obj_key)
                if result is None:
                    break
                self._manager.finish_read_prefetched(
                    [obj_key], l1_owners=result.l1_owners
                )
                count += 1
            return count

    def probe_many(self, keys: Sequence[BlockKey]) -> tuple[bool, ...]:
        """Return per-key L1 presence for one bounded batch without reading KV.

        Each successful lookup releases its captured read ownership before
        the next key. Missing keys do not stop the scan. These observations
        neither hold objects resident nor certify their payloads. Calls are
        serialized under the host lock and belong on background threads.
        """
        if len(keys) > MAX_PREFIX_KEYS:
            raise ValueError(f"Presence probe accepts at most {MAX_PREFIX_KEYS} keys")
        with self._lock:
            self._check_open()
            names = [self._key(key) for key in keys]
            present = []
            for obj_key in names:
                result = self._lookup(obj_key)
                if result is None:
                    present.append(False)
                    continue
                try:
                    self._manager.finish_read_prefetched(
                        [obj_key], l1_owners=result.l1_owners
                    )
                except Exception:
                    # The release may have taken effect before raising. Stop
                    # admission rather than retrying an uncertain release by
                    # key; the manager owner must perform final shutdown.
                    self._failed = True
                    logger.exception("Presence probe read release is uncertain")
                    raise
                present.append(True)
            return tuple(present)

    def report_status(self) -> dict:
        """Return capabilities and failure state without touching stored bytes."""
        with self._lock:
            return {
                "lmcache_version": "0.5.6",
                "namespace": self._identity.namespace,
                "l1_only": True,
                "l2_restore": False,
                "immediate_write_rollback": False,
                "reservation_generation_check": False,
                "publication_confirmation": "resident-byte-comparison",
                "failed": self._failed,
                "closed": self._closed,
                "pending_read_cleanup": len(self._pending_reads),
            }

    def close(self) -> None:
        """Stop admission and release completed timed-out reads, not the manager.

        Stock LMCache has no public prefetch cancellation. Unfinished handles
        remain owned by its controller until completion/manager shutdown.
        """
        with self._lock:
            self._closed = True
            pending = []
            for handle, key in self._pending_reads:
                try:
                    result = self._manager.query_prefetch_status(handle)
                except Exception:
                    logger.exception(
                        "Could not inspect pending L1 prefetch during close"
                    )
                    pending.append((handle, key))
                    continue
                if result is None:
                    pending.append((handle, key))
                elif result.hit_cells and result.hit_cells[0].get_indices_list():
                    self._manager.finish_read_prefetched(
                        [key], l1_owners=result.l1_owners
                    )
            self._pending_reads = pending
            if pending:
                logger.warning(
                    "%d prefetch handles await LMCache cleanup", len(pending)
                )

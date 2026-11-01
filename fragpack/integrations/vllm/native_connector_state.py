# SPDX-License-Identifier: Apache-2.0
"""Engine-independent state for the native-attention external codec connector.

Scheduler calls are single-threaded; only lookup RPCs run on an executor.
Worker calls are single-threaded and poll transfer futures without blocking.
Neither class imports vLLM, initializes CUDA, nor owns engine KV pages.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from fragpack.integrations.lmcache.client import BlobClient
from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity
from fragpack.logging import init_logger

logger = init_logger(__name__)


def native_layer_view(cache: torch.Tensor) -> torch.Tensor:
    """View vLLM ``[pages, heads, tokens, 256]`` as native KV planes.

    Only tensor metadata changes. CPU tensors are accepted for layout tests;
    actual GPU registration and stream ownership belong to the worker adapter.
    """
    from fragpack.ops.native_codec import validate_native_tensor

    if cache.ndim != 4 or cache.shape[-1] != 256 or cache.stride(-1) != 1:
        raise ValueError("expected standard vLLM [pages, heads, tokens, 256] KV")
    result = cache.unflatten(-1, (2, 128)).permute(3, 0, 2, 1, 4)
    validate_native_tensor(result)
    return result


@dataclass(frozen=True)
class NativeRequestPlan:
    """One request's complete logical objects and ordered physical pages.

    Keys already include the complete causal prefix. ``page_ids`` contains only
    the selected range, excluding a locally cached prefix and incomplete tail.
    The engine retains these pages until the connector reports completion.
    """

    request_id: str
    keys: tuple[BlockKey, ...]
    page_ids: tuple[int, ...]
    page_tokens: int
    store: bool

    def __post_init__(self) -> None:
        if self.page_tokens not in (16, 32, 64, 128):
            raise ValueError("native page size must divide 128")
        if len(self.page_ids) != len(self.keys) * (128 // self.page_tokens):
            raise ValueError("one physical page map is required per complete object")
        if any(type(page) is not int or page < 0 for page in self.page_ids):
            raise ValueError("physical page IDs must be nonnegative integers")
        if not self.store and len(set(self.page_ids)) != len(self.page_ids):
            raise ValueError("restore destinations must be exclusive distinct pages")

    def batch(
        self, start: int, count: int
    ) -> tuple[tuple[BlockKey, ...], tuple[tuple[int, ...], ...]]:
        """Return bounded keys and their CPU page maps without changing identity."""
        width = 128 // self.page_tokens
        stop = min(start + count, len(self.keys))
        return self.keys[start:stop], tuple(
            self.page_ids[i * width : (i + 1) * width] for i in range(start, stop)
        )


def finished_store_ids(
    plans: Sequence[NativeRequestPlan], finished_requests: set[str]
) -> tuple[str, ...]:
    """Bind each admitted store to its scheduler's actual finish notification.

    The pinned scheduler calls ``request_finished`` before adding the same ID to
    ``SchedulerOutput.finished_req_ids`` and building connector metadata. Carry
    this bounded subset with the plans so worker callback ordering cannot lose
    the finish evidence. An unexpected engine order fails explicitly.
    """
    stores = tuple(plan.request_id for plan in plans if plan.store)
    if not set(stores).issubset(finished_requests):
        raise RuntimeError("store plan lacks the scheduler's finish notification")
    return stores


@dataclass
class _Lookup:
    keys: tuple[BlockKey, ...]
    start: int
    future: Future[int]
    matched: int = 0


class NativeSchedulerState:
    """Bounded asynchronous prefix lookup and delayed-free request bookkeeping.

    ``client`` is dedicated to scheduler lookups. ``executor`` is injectable for
    deterministic CPU tests; when omitted, this class owns a single RPC thread.
    Call ``finish`` before the engine frees any store source, then honor its
    boolean delayed-free result. ``complete`` consumes worker notifications.
    """

    def __init__(
        self,
        identity: CacheIdentity,
        client: BlobClient,
        *,
        page_tokens: int,
        max_pending_requests: int,
        executor: Executor | None = None,
    ) -> None:
        if page_tokens not in (16, 32, 64, 128):
            raise ValueError("native page size must divide 128")
        if type(max_pending_requests) is not int or max_pending_requests < 1:
            raise ValueError("max_pending_requests must be positive")
        self.identity = identity
        self.client = client
        self.page_tokens = page_tokens
        self.max_pending_requests = max_pending_requests
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fragpack-native-lookup"
        )
        self._lookup_jobs: set[Future[int]] = set()
        self._lookups: dict[str, _Lookup] = {}
        self._pending: list[NativeRequestPlan] = []
        self._saving: set[str] = set()
        self._loading: set[str] = set()
        self._failed_loads: set[str] = set()
        self._worker_pending = False

    def lookup(
        self,
        request_id: str,
        tokens: Sequence[int],
        local_tokens: int,
        *,
        cache_salt: str = "",
    ) -> tuple[int | None, bool]:
        """Poll a lookup; return only an extension beyond aligned local APC.

        A complete final prompt object is left out when it would include the
        final prompt token. vLLM must compute that token to obtain logits.
        Non-object-aligned local hits remain entirely with native APC so shared
        local pages are never overwritten by a partially overlapping restore.
        """
        end = max(0, len(tokens) - 1) // 128
        if (
            request_id in self._failed_loads
            or local_tokens % 128
            or end * 128 <= local_tokens
        ):
            return 0, False
        start = local_tokens // 128
        lookup = self._lookups.get(request_id)
        if lookup is None or lookup.start != start:
            self._lookup_jobs = {f for f in self._lookup_jobs if not f.done()}
            if len(self._lookup_jobs) >= self.max_pending_requests:
                return 0, False
            keys = tuple(self.identity.block_keys(tokens, cache_salt))[:end]
            future = self._executor.submit(self.client.lookup_prefix, keys[start:])
            self._lookup_jobs.add(future)
            lookup = _Lookup(keys, start, future)
            self._lookups[request_id] = lookup
        if not lookup.future.done():
            return None, False
        try:
            count = lookup.future.result()
            if type(count) is not int or not 0 <= count <= len(lookup.keys) - start:
                raise ValueError("cache service returned an invalid prefix count")
        except Exception:
            logger.exception("Native external lookup failed; recomputing request")
            self._failed_loads.add(request_id)
            return 0, False
        lookup.matched = count
        return count * 128, count > 0

    def allocate(
        self, request_id: str, page_ids: Sequence[int], external_tokens: int
    ) -> None:
        """Bind an advisory hit to allocated, exclusive external suffix pages."""
        if external_tokens == 0:
            return
        lookup = self._lookups[request_id]
        if external_tokens != lookup.matched * 128 or request_id in self._loading:
            raise ValueError("allocated external range differs from pending lookup")
        first_page = lookup.start * 128 // self.page_tokens
        count_pages = external_tokens // self.page_tokens
        keys = lookup.keys[lookup.start : lookup.start + lookup.matched]
        plan = NativeRequestPlan(
            request_id,
            keys,
            tuple(page_ids[first_page : first_page + count_pages]),
            self.page_tokens,
            False,
        )
        self._loading.add(request_id)
        self._pending.append(plan)

    def finish(
        self,
        request_id: str,
        tokens: Sequence[int],
        page_ids: Sequence[int],
        *,
        normal_finish: bool,
        computed_tokens: int,
        cache_salt: str = "",
    ) -> bool:
        """Admit complete computed prompt objects and retain their source pages.

        Aborted requests, active restores and pool pressure decline a best-effort
        save. No page reference is taken here; True tells the engine to retain
        its existing references until the worker reports finished sending.
        """
        lookup = self._lookups.pop(request_id, None)
        if lookup is not None:
            lookup.future.cancel()
        self._failed_loads.discard(request_id)
        count = min(len(tokens), computed_tokens) // 128
        if (
            not normal_finish
            or count <= 0
            or request_id in self._loading
            or len(self._saving) >= self.max_pending_requests
        ):
            return False
        if request_id in self._saving:
            raise ValueError("request already has a delayed save")
        keys = tuple(self.identity.block_keys(tokens[: count * 128], cache_salt))
        plan = NativeRequestPlan(
            request_id,
            keys,
            tuple(page_ids[: count * 128 // self.page_tokens]),
            self.page_tokens,
            True,
        )
        self._saving.add(request_id)
        self._pending.append(plan)
        return True

    def take_plans(self) -> tuple[NativeRequestPlan, ...]:
        """Move this step's owned metadata into the engine's worker message."""
        result = tuple(self._pending)
        self._pending.clear()
        return result

    def complete(
        self,
        sent: set[str],
        received: set[str],
        failed: set[str],
        *,
        worker_pending: bool,
    ) -> None:
        """Release scheduler bookkeeping after worker ownership notifications."""
        self._saving.difference_update(sent)
        self._loading.difference_update(received)
        self._failed_loads.update(req for req in failed if req in self._lookups)
        self._worker_pending = worker_pending

    @property
    def pending(self) -> bool:
        """Whether empty model steps are required to drain transfers."""
        return bool(
            self._saving or self._loading or self._pending or self._worker_pending
        )

    def reset(self) -> bool:
        """Reset local lookup hints only when no transferred pages are owned."""
        if self.pending:
            return False
        self._lookups.clear()
        self._failed_loads.clear()
        return True

    def close(self) -> None:
        """Drain the owned lookup executor before its transport is closed."""
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=True)


@dataclass
class _Batch:
    ticket: Any
    count: int


@dataclass
class _TransferRequest:
    plan: NativeRequestPlan
    next_block: int = 0
    batches: list[_Batch] = field(default_factory=list)
    failed: bool = False
    source_reported: bool = False
    prepared_batch: tuple[tuple[BlockKey, ...], Any] | None = None


class NativeWorkerState:
    """Batch and poll request transfers without waiting on CUDA or RPC calls.

    ``make_map`` creates the device page map on the current producer stream.
    ``transfer`` owns submitted maps and futures. Store sources remain engine
    owned until all source futures succeed, and restore destinations remain
    exclusive until every submitted batch has completed, including failures.
    """

    def __init__(
        self,
        transfer: Any,
        make_map: Callable[[tuple[tuple[int, ...], ...]], Any],
        *,
        batch_blocks: int,
    ) -> None:
        if type(batch_blocks) is not int or batch_blocks < 1:
            raise ValueError("batch_blocks must be positive")
        self.transfer = transfer
        self.make_map = make_map
        self.batch_blocks = batch_blocks
        self._stores: dict[str, _TransferRequest] = {}
        self._loads: dict[str, _TransferRequest] = {}
        self._finished_notifications: set[str] = set()
        self.counters = {
            "store_requests": 0,
            "store_blocks": 0,
            "store_failures": 0,
            "restore_requests": 0,
            "restore_blocks": 0,
            "restore_failures": 0,
        }

    def accept(
        self,
        plans: Sequence[NativeRequestPlan],
        *,
        finished_requests: Sequence[str] = (),
    ) -> None:
        """Accept plans and their bounded scheduler-certified finish evidence.

        ``finished_requests`` contains only stores in these plans. It preserves
        a finish event even if a generic worker poll happened before plan
        submission; unrelated short/aborted request IDs are never retained.
        """
        admitted_stores = {plan.request_id for plan in plans if plan.store}
        if not set(finished_requests).issubset(admitted_stores):
            raise ValueError("finish evidence must belong to an admitted store plan")
        for plan in plans:
            jobs = self._stores if plan.store else self._loads
            if plan.request_id in jobs:
                raise ValueError("duplicate native transfer plan")
            jobs[plan.request_id] = _TransferRequest(plan)
        self._finished_notifications.update(finished_requests)
        self._submit_batches()

    def _submit_batches(self) -> None:
        # Background saves are best effort. Already admitted restores get first
        # access to free transfer slots; an in-flight task is never cancelled.
        for jobs in (self._loads, self._stores):
            for job in jobs.values():
                while not job.failed and job.next_block < len(job.plan.keys):
                    if job.prepared_batch is None:
                        keys, pages = job.plan.batch(job.next_block, self.batch_blocks)
                        job.prepared_batch = (keys, self.make_map(pages))
                    keys, page_map = job.prepared_batch
                    submit = (
                        self.transfer.submit_store
                        if job.plan.store
                        else self.transfer.submit_restore
                    )
                    ticket = submit(keys, page_map)
                    if ticket is None:
                        return
                    job.prepared_batch = None
                    job.batches.append(_Batch(ticket, len(keys)))
                    job.next_block += len(keys)

    def poll(self, finished_requests: set[str]) -> tuple[set[str], set[str], set[str]]:
        """Return source releases, finished receives and failed receives.

        A failed transfer backend means CUDA completion is unknown: raise and
        retain all engine pages for process recovery. Ordinary cache failures
        whose CUDA work drained may report recomputation after every in-flight
        destination write completed.
        """
        if self.transfer.failed:
            raise RuntimeError("native CUDA completion unknown; retain engine KV pages")
        self._finished_notifications.update(
            finished_requests.intersection(self._stores)
        )
        sent: set[str] = set()
        received: set[str] = set()
        failed: set[str] = set()
        for jobs in (self._loads, self._stores):
            for request_id, job in list(jobs.items()):
                for batch in job.batches:
                    future = batch.ticket.completion
                    if not future.done():
                        continue
                    try:
                        values = future.result()
                        valid = (
                            isinstance(values, tuple)
                            and len(values) == batch.count
                            and all(value is True for value in values)
                        )
                        job.failed |= not valid
                    except Exception:
                        if self.transfer.failed:
                            raise RuntimeError(
                                "native CUDA completion unknown; retain engine KV pages"
                            ) from None
                        job.failed = True
                submitted_all = job.failed or job.next_block == len(job.plan.keys)
                if not submitted_all:
                    continue
                if job.plan.store:
                    if not job.source_reported and all(
                        batch.ticket.source_ready.done() for batch in job.batches
                    ):
                        # Never turn an exceptional source-ready result into a
                        # release, even if publication has already failed.
                        for batch in job.batches:
                            batch.ticket.source_ready.result()
                        if request_id in self._finished_notifications:
                            sent.add(request_id)
                            job.source_reported = True
                    if not job.source_reported:
                        continue
                if not all(batch.ticket.completion.done() for batch in job.batches):
                    continue
                kind = "store" if job.plan.store else "restore"
                self.counters[f"{kind}_requests"] += 1
                self.counters[f"{kind}_blocks"] += len(job.plan.keys)
                self.counters[f"{kind}_failures"] += int(job.failed)
                if not job.plan.store:
                    received.add(request_id)
                    if job.failed:
                        failed.add(request_id)
                else:
                    self._finished_notifications.discard(request_id)
                del jobs[request_id]
                logger.info(
                    "FRAGPACK_NATIVE_%s request=%s blocks=%d success=%s",
                    kind.upper(),
                    request_id,
                    len(job.plan.keys),
                    not job.failed,
                )
        self._submit_batches()
        return sent, received, failed

    @property
    def pending(self) -> bool:
        """Whether GPU reads, restores or external publications remain owned."""
        return bool(self._stores or self._loads)

    @property
    def pending_map_bytes(self) -> int:
        """Bytes in int32 page maps retained across temporary pool refusal."""
        return sum(
            len(job.prepared_batch[0]) * (128 // job.plan.page_tokens) * 4
            for jobs in (self._loads, self._stores)
            for job in jobs.values()
            if job.prepared_batch is not None
        )

    @property
    def retained_finish_count(self) -> int:
        """Number of finish notifications retained for admitted stores only."""
        return len(self._finished_notifications)

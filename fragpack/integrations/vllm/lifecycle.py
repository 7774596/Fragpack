# SPDX-License-Identifier: Apache-2.0
"""Host-only tracking of writable KV blocks from vLLM scheduler outputs.

This module intentionally imports neither torch nor vLLM. The integration
supplies the block size and the scheduler's host metadata to this state machine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class OpenBlocks:
    """Mirror request block lists and identify their partially written blocks.

    A superset of writable blocks is unsafe: a freed open block can be allocated
    ahead of a new request's write position, retaining an obsolete staging slot.
    Only the block containing each request's first uncomputed token may stay open.

    Args:
        block_tokens (int): Number of token slots in one KV block.

    Async/thread-safety:
        One instance belongs to one worker and is used on its model-runner thread.
        Updates and reads must not run concurrently. No I/O or CUDA access occurs.
    """

    def __init__(self, block_tokens: int) -> None:
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        self.block_tokens = block_tokens
        self.blocks: dict[str, list[int]] = {}
        self.computed: dict[str, int] = {}
        self._batch_requests: dict[int, str] | None = None
        self._dummy_step = False

    def begin_step(self, *, dummy_run: bool) -> None:
        """Discard the previous batch identity before any runner early return.

        Args:
            dummy_run (bool): Whether this step uses synthetic request metadata.

        Returns:
            None. Persistent request block lists are unchanged.

        Async/thread-safety:
            Mutates host state on the model-runner thread; no I/O or CUDA access.
        """
        self._batch_requests = None
        self._dummy_step = dummy_run

    def bind_batch(self, req_ids: list[str], req_indices: NDArray[np.intp]) -> None:
        """Bind real requests to the runner indices carried by attention metadata.

        Args:
            req_ids (list[str]): Request ids returned by the runner's batch gather.
            req_indices (NDArray[np.intp]): Matching CPU request-table indices.

        Returns:
            None. The binding is valid only until the next begin_step call.

        Raises:
            RuntimeError: If binding synthetic requests during a dummy step.
            ValueError: If indices do not uniquely identify the supplied requests.

        Async/thread-safety:
            Reads CPU metadata and mutates host state on the model-runner thread.
            No device readback occurs.
        """
        if self._dummy_step:
            raise RuntimeError("cannot bind real requests during a dummy step")
        if req_indices.ndim != 1 or len(req_ids) != len(req_indices):
            raise ValueError("request ids and runner indices must have equal lengths")
        binding = {
            int(index): req_id
            for req_id, index in zip(req_ids, req_indices, strict=True)
        }
        if len(binding) != len(req_ids) or any(index < 0 for index in binding):
            raise ValueError("runner request indices must be unique and nonnegative")
        self._batch_requests = binding

    def prefill_blocks(
        self,
        req_indices: NDArray[np.intp] | None,
        page_counts: NDArray[np.int64],
    ) -> list[list[int]] | None:
        """Resolve attention rows to their live host-side physical block lists.

        Args:
            req_indices (NDArray[np.intp] | None): CPU runner indices for these
                attention rows, including any reorder or microbatch slice.
            page_counts (NDArray[np.int64]): Number of context pages per row.

        Returns:
            list[list[int]] | None: Independent lists trimmed to page_counts, in
            attention-row order. None means this is explicitly a dummy step.

        Raises:
            RuntimeError: If a real step has no batch binding, an unknown request,
                missing row indices, or fewer allocated blocks than required.
            ValueError: If row counts differ or a requested page count is negative.

        Async/thread-safety:
            Reads host state on the model-runner thread; no I/O or CUDA access.
            Real batches fail closed instead of reusing stale or guessed mappings.
        """
        if self._dummy_step:
            return None
        if self._batch_requests is None:
            raise RuntimeError("prefill requires a real runner batch binding")
        if req_indices is None:
            raise RuntimeError("prefill requires CPU runner request indices")
        if (
            req_indices.ndim != 1
            or page_counts.ndim != 1
            or len(req_indices) != len(page_counts)
        ):
            raise ValueError("prefill request indices and page counts must match")
        rows: list[list[int]] = []
        for index, count in zip(req_indices, page_counts, strict=True):
            count = int(count)
            if count < 0:
                raise ValueError("prefill page counts must be nonnegative")
            req_id = self._batch_requests.get(int(index))
            blocks = self.blocks.get(req_id) if req_id is not None else None
            if blocks is None:
                raise RuntimeError(
                    f"prefill runner request index {int(index)} is unknown"
                )
            if len(blocks) < count:
                raise RuntimeError(
                    f"prefill request {req_id!r} needs {count} blocks; "
                    f"only {len(blocks)} are tracked"
                )
            rows.append(blocks[:count])
        return rows

    def update(self, so: Any) -> None:
        """Apply host scheduler metadata in the V2 model runner's update order.

        Args:
            so (Any): A SchedulerOutput-compatible object with finished and
                preempted request ids, new/cached requests, and block-table updates.

        Returns:
            None.

        Async/thread-safety:
            Mutates worker-local state on the model-runner thread; no I/O.
        """
        for req_id in so.finished_req_ids | (so.preempted_req_ids or set()):
            self.blocks.pop(req_id, None)
            self.computed.pop(req_id, None)
        for new in so.scheduled_new_reqs:
            self.blocks[new.req_id] = list(new.block_ids[0])
            self.computed[new.req_id] = new.num_computed_tokens
        updates = so.block_table_updates or {}
        for req_id, block_ids in updates.items():
            if req_id in self.blocks:
                self.blocks[req_id] = list(block_ids[0])
        cached = so.scheduled_cached_reqs
        for req_id, nct, new_ids in zip(
            cached.req_ids,
            cached.num_computed_tokens,
            cached.new_block_ids,
            strict=True,
        ):
            if req_id not in self.blocks:
                continue
            self.computed[req_id] = nct
            if new_ids is not None and req_id not in updates:
                if req_id in cached.resumed_req_ids:
                    self.blocks[req_id] = list(new_ids[0])
                else:
                    self.blocks[req_id].extend(new_ids[0])

    def live_blocks(self) -> NDArray[np.int32]:
        """Return the partially written block ids of tracked requests.

        Returns:
            NDArray[np.int32]: One id per request with a non-aligned computed
            length and an allocated block at that position; otherwise no entry.

        Async/thread-safety:
            Reads host state on the model-runner thread; no I/O or CUDA access.
        """
        open_ids = []
        for req_id, blocks in self.blocks.items():
            index, offset = divmod(self.computed[req_id], self.block_tokens)
            if offset and index < len(blocks):
                open_ids.append(blocks[index])
        return np.asarray(open_ids, dtype=np.int32)

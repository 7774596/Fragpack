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

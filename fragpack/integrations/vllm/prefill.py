# SPDX-License-Identifier: Apache-2.0
"""CPU planning of shared decoded pages for bounded prefill workspaces.

The planner uses physical block identity, not token equality. It only aliases
pages within one attention pass, where the workspace is read-only. It imports
neither torch nor vLLM and never examines tensor contents or GPU state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _finish_plan(
    first: int,
    rows: list[list[int]],
    source_rows: list[int],
    source_columns: list[int],
) -> PrefillPlan:
    """Materialize one pass; padded columns are ignored by sequence lengths."""
    table = np.zeros((len(rows), max(map(len, rows))), dtype=np.int32)
    for i, slots in enumerate(rows):
        table[i, : len(slots)] = slots
    return PrefillPlan(
        first=first,
        stop=first + len(rows),
        source_rows=np.asarray(source_rows, dtype=np.int64),
        source_columns=np.asarray(source_columns, dtype=np.int64),
        block_table=table,
    )


@dataclass(frozen=True)
class PrefillPlan:
    """One contiguous group of requests sharing decoded workspace pages.

    Attributes:
        first (int): First request row in the supplied block lists.
        stop (int): Exclusive end of the request rows.
        source_rows (NDArray[np.int64]): Request row containing each unique block.
        source_columns (NDArray[np.int64]): Column of each unique block in that row.
        block_table (NDArray[np.int32]): Per-request workspace indices. Trailing
            columns are padding and must be masked by the request's token length.

    Async/thread-safety:
        Host metadata owned by one model step. Consumers must not mutate arrays.
    """

    first: int
    stop: int
    source_rows: NDArray[np.int64]
    source_columns: NDArray[np.int64]
    block_table: NDArray[np.int32]


def plan_prefill_groups(
    block_rows: Sequence[Sequence[int]], workspace_pages: int
) -> list[PrefillPlan]:
    """Group consecutive requests by their union of physical KV blocks.

    Args:
        block_rows (Sequence[Sequence[int]]): Physical block IDs for each prefill
            request, already trimmed to its context length. Extra allocated or
            padded blocks must not be included. Each request must be nonempty.
        workspace_pages (int): Maximum number of distinct blocks in one pass.

    Returns:
        list[PrefillPlan]: Ordered passes covering every request exactly once.
        Each source block occurs once per pass; logical block order is preserved
        through the workspace block table. An empty batch returns an empty list.

    Raises:
        ValueError: For a nonpositive budget, empty request, negative block ID,
            or one request whose distinct blocks exceed the workspace budget.

    Async/thread-safety:
        Pure CPU planning; no I/O, engine imports, device operations, or shared
        mutable state. Plans are rebuilt each step, so block reuse is safe.
    """
    if workspace_pages <= 0:
        raise ValueError("workspace_pages must be positive")
    plans: list[PrefillPlan] = []
    slots: dict[int, int] = {}
    rows: list[list[int]] = []
    source_rows: list[int] = []
    source_columns: list[int] = []
    first = 0
    for row, blocks in enumerate(block_rows):
        if not len(blocks):
            raise ValueError("prefill requests must contain at least one block")
        unique = set(blocks)
        if min(unique) < 0:
            raise ValueError("physical block IDs must be nonnegative")
        if len(unique) > workspace_pages:
            raise ValueError("one prefill request exceeds the workspace page budget")
        if len(slots) + len(unique.difference(slots)) > workspace_pages:
            plans.append(_finish_plan(first, rows, source_rows, source_columns))
            slots, rows, source_rows, source_columns = {}, [], [], []
            first = row
        mapped: list[int] = []
        for column, block in enumerate(blocks):
            if block not in slots:
                slots[block] = len(slots)
                source_rows.append(row)
                source_columns.append(column)
            mapped.append(slots[block])
        rows.append(mapped)
    if rows:
        plans.append(_finish_plan(first, rows, source_rows, source_columns))
    return plans

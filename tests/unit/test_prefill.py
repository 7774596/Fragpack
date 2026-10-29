# SPDX-License-Identifier: Apache-2.0
"""Shared prefill pages preserve request order, KV contents, and workspace bounds."""

import numpy as np
import pytest

from fragpack.integrations.vllm.prefill import PrefillPlan, plan_prefill_groups


def _sources(plan: PrefillPlan, rows: list[list[int]]) -> list[int]:
    """Resolve gather coordinates against the original physical block lists."""
    return [
        rows[r][c] for r, c in zip(plan.source_rows, plan.source_columns, strict=True)
    ]


def _check_reconstruction(rows: list[list[int]], capacity: int) -> None:
    """Verify coverage and exact logical blocks through every workspace table."""
    plans = plan_prefill_groups(rows, capacity)
    covered: list[int] = []
    for plan in plans:
        sources = _sources(plan, rows)
        assert len(sources) == len(set(sources)) <= capacity
        assert len(sources) == len(set(sum(rows[plan.first : plan.stop], [])))
        for local, row in enumerate(range(plan.first, plan.stop)):
            restored = [sources[s] for s in plan.block_table[local, : len(rows[row])]]
            assert restored == rows[row]
            covered.append(row)
    assert covered == list(range(len(rows)))


def test_shared_prefix_fits_one_pass_by_unique_page_count() -> None:
    """Nine logical pages use five decoded pages without splitting requests."""
    rows = [[10, 11, 12], [10, 11, 13], [10, 11, 14]]
    plans = plan_prefill_groups(rows, workspace_pages=5)
    assert len(plans) == 1
    assert len(plans[0].source_rows) == 5
    _check_reconstruction(rows, 5)


def test_disjoint_requests_split_at_workspace_capacity() -> None:
    """Unique blocks cannot alias merely because requests have equal lengths."""
    rows = [[9, 1, 7], [2, 8, 3], [4]]
    plans = plan_prefill_groups(rows, workspace_pages=4)
    assert [(p.first, p.stop) for p in plans] == [(0, 1), (1, 3)]
    _check_reconstruction(rows, 4)


def test_each_pass_regathers_its_own_shared_pages() -> None:
    """Workspace reuse must not point into pages overwritten by another pass."""
    rows = [[0, 1, 2], [0, 1, 3], [0, 1, 4]]
    plans = plan_prefill_groups(rows, workspace_pages=4)
    assert [(p.first, p.stop) for p in plans] == [(0, 2), (2, 3)]
    assert _sources(plans[1], rows) == [0, 1, 4]
    _check_reconstruction(rows, 4)


def test_ragged_contexts_preserve_attention_inputs_and_causality() -> None:
    """Shared storage preserves K/V and causal outputs, including partial tails.

    This is a CPU numeric reference for the page mapping, not a GPU/FA test.
    """
    rows = [[8, 2, 5], [8, 2], [8, 7]]
    context_lengths = [10, 7, 5]
    query_lengths = [3, 2, 2]
    block_tokens, dim = 4, 6
    rng = np.random.default_rng(42)
    kv = rng.normal(size=(2, 9, block_tokens, dim))
    plan = plan_prefill_groups(rows, workspace_pages=4)[0]
    sources = _sources(plan, rows)
    workspace = kv[:, sources]
    for row, (blocks, length, query_length) in enumerate(
        zip(rows, context_lengths, query_lengths, strict=True)
    ):
        mapped = plan.block_table[row, : len(blocks)]
        actual = workspace[:, mapped].reshape(2, -1, dim)[:, :length]
        expected = kv[:, blocks].reshape(2, -1, dim)[:, :length]
        np.testing.assert_array_equal(actual, expected)
        query = rng.normal(size=(query_length, dim))
        outputs = []
        for keys, values in (actual, expected):
            scores = query @ keys.T / np.sqrt(dim)
            allowed = np.arange(length)[None, :] <= (
                length - query_length + np.arange(query_length)[:, None]
            )
            scores = np.where(allowed, scores, -np.inf)
            weights = np.exp(scores - scores.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            outputs.append(weights @ values)
        np.testing.assert_array_equal(outputs[0], outputs[1])


def test_random_shared_and_ragged_batches_reconstruct_exactly() -> None:
    """Exercise overlapping identities, arbitrary ordering, and repeated passes."""
    rng = np.random.default_rng(9)
    for _ in range(100):
        rows = [
            rng.choice(24, size=int(rng.integers(1, 9)), replace=False).tolist()
            for _ in range(int(rng.integers(1, 17)))
        ]
        _check_reconstruction(rows, capacity=9)


@pytest.mark.parametrize(
    "rows,capacity,message",
    [
        ([[0]], 0, "must be positive"),
        ([[]], 4, "at least one block"),
        ([[1, -1]], 4, "nonnegative"),
        ([[1, 2, 3]], 2, "exceeds the workspace"),
    ],
)
def test_invalid_input_cannot_create_an_out_of_bounds_plan(
    rows: list[list[int]], capacity: int, message: str
) -> None:
    """Reject bad metadata before it can generate an unsafe GPU page table."""
    with pytest.raises(ValueError, match=message):
        plan_prefill_groups(rows, capacity)


def test_empty_batch_has_no_workspace_passes() -> None:
    """A decode-only batch needs no prefill plan."""
    assert plan_prefill_groups([], workspace_pages=4) == []

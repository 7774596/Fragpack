# SPDX-License-Identifier: Apache-2.0
"""OpenBlocks lists exactly the partially written block of each request."""

from types import SimpleNamespace

import numpy as np
import pytest

from fragpack.integrations.vllm.lifecycle import OpenBlocks


def _output(
    new: tuple = (),
    cached: tuple = (),
    finished: frozenset = frozenset(),
    resumed: frozenset = frozenset(),
    preempted: frozenset = frozenset(),
    updates: dict[str, list[int]] | None = None,
) -> SimpleNamespace:
    """Build a minimal scheduler output.

    Args:
        new: ``(req_id, block_ids, num_computed_tokens)`` per new request.
        cached: ``(req_id, num_computed_tokens, new_block_ids or None)``.
        finished: Finished request ids.
        resumed: Resumed request ids among ``cached``.
        preempted: Request ids whose old block lists must be discarded.
        updates: Authoritative replacement block lists for tracked requests.

    Returns:
        Object with the fields ``OpenBlocks.update`` reads.
    """
    return SimpleNamespace(
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, block_ids=(list(b),), num_computed_tokens=c)
            for r, b, c in new
        ],
        block_table_updates=(
            {req_id: (ids,) for req_id, ids in updates.items()} if updates else None
        ),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, _, _ in cached],
            num_computed_tokens=[c for _, c, _ in cached],
            new_block_ids=[None if b is None else (list(b),) for _, _, b in cached],
            resumed_req_ids=set(resumed),
        ),
    )


def test_lists_only_partially_written_blocks() -> None:
    """Track each writer's partial block across token and allocation updates."""
    blocks = OpenBlocks(128)
    blocks.update(_output(new=(("a", [10, 11, 12], 0), ("b", [20], 0))))
    assert blocks.live_blocks().tolist() == []
    blocks.update(_output(cached=(("a", 200, None), ("b", 128, None))))
    assert blocks.live_blocks().tolist() == [11]
    blocks.update(_output(cached=(("a", 300, None), ("b", 129, [21]))))
    assert sorted(blocks.live_blocks().tolist()) == [12, 21]


def test_reallocated_open_block_is_not_live_ahead_of_its_writer() -> None:
    """Do not retain a finished writer's block ahead of a new writer."""
    blocks = OpenBlocks(128)
    blocks.update(_output(new=(("a", [10, 11], 0),)))
    blocks.update(_output(cached=(("a", 200, None),)))
    assert blocks.live_blocks().tolist() == [11]
    # ``a`` finishes with block 11 open; ``b`` gets 11 as its second block
    # and has written part of its first.
    blocks.update(_output(new=(("b", [30, 11], 0),), finished=frozenset({"a"})))
    blocks.update(_output(cached=(("b", 50, None),)))
    assert blocks.live_blocks().tolist() == [30]


def test_resumed_request_replaces_its_blocks() -> None:
    """Resumption uses the new allocation instead of appending to the old one."""
    blocks = OpenBlocks(128)
    blocks.update(_output(new=(("a", [10, 11], 0),)))
    blocks.update(_output(cached=(("a", 150, None),)))
    blocks.update(_output(cached=(("a", 130, [40, 41]),), resumed=frozenset({"a"})))
    assert blocks.live_blocks().tolist() == [41]


def test_preemption_discards_old_blocks_until_request_is_reintroduced() -> None:
    """Preempted writers do not keep staging alive through stale cached updates."""
    blocks = OpenBlocks(128)
    blocks.update(_output(new=(("a", [10, 11], 150),)))
    assert blocks.live_blocks().tolist() == [11]
    blocks.update(_output(preempted=frozenset({"a"}), cached=(("a", 160, None),)))
    assert blocks.live_blocks().tolist() == []
    blocks.update(_output(new=(("a", [40, 41], 130),)))
    assert blocks.live_blocks().tolist() == [41]


def test_block_table_update_is_authoritative_over_incremental_blocks() -> None:
    """A full replacement must not also append the cached incremental list."""
    blocks = OpenBlocks(128)
    blocks.update(_output(new=(("a", [10, 11], 150),)))
    blocks.update(_output(cached=(("a", 150, [90]),), updates={"a": [40, 41]}))
    assert blocks.live_blocks().tolist() == [41]
    # If the incremental list had been appended as well, the next update would
    # incorrectly report block 90, despite its absence from the full table.
    blocks.update(_output(cached=(("a", 260, None),)))
    assert blocks.live_blocks().tolist() == []


def test_preemption_and_readmission_in_one_step_use_new_allocation() -> None:
    """Removing the old allocation must precede reintroducing the same id."""
    blocks = OpenBlocks(128)
    blocks.update(_output(new=(("a", [10, 11], 150),)))
    blocks.update(_output(preempted=frozenset({"a"}), new=(("a", [40, 41], 130),)))
    assert blocks.live_blocks().tolist() == [41]


def test_prefill_rows_follow_runner_indices_after_reorder_and_slicing() -> None:
    """Shared pages and spare allocations retain their correct row identities."""
    blocks = OpenBlocks(128)
    blocks.begin_step(dummy_run=False)
    blocks.update(
        _output(
            new=(("a", [10, 11, 12], 128), ("b", [10, 20, 21], 128), ("c", [30], 0))
        )
    )
    blocks.bind_batch(["c", "b", "a"], np.array([4, 0, 7], dtype=np.intp))
    # Attention may receive reordered rows or a microbatch slice. Resolve by
    # actual runner index, not insertion order or the position within this slice.
    rows = blocks.prefill_blocks(np.array([7, 0]), np.array([2, 3]))
    assert rows == [[10, 11], [10, 20, 21]]
    assert blocks.prefill_blocks(np.array([0]), np.array([2])) == [[10, 20]]
    rows[0][0] = 999
    assert blocks.prefill_blocks(np.array([7]), np.array([1])) == [[10]]


def test_request_index_reuse_resolves_the_new_owner() -> None:
    """A recycled runner slot must not retain the finished request's KV blocks."""
    blocks = OpenBlocks(128)
    blocks.begin_step(dummy_run=False)
    blocks.update(_output(new=(("a", [10, 11], 128),)))
    blocks.bind_batch(["a"], np.array([2]))
    assert blocks.prefill_blocks(np.array([2]), np.array([2])) == [[10, 11]]
    blocks.begin_step(dummy_run=False)
    blocks.update(_output(new=(("b", [40, 41], 128),), finished=frozenset({"a"})))
    blocks.bind_batch(["b"], np.array([2]))
    assert blocks.prefill_blocks(np.array([2]), np.array([2])) == [[40, 41]]


def test_dummy_and_unbound_real_steps_do_not_reuse_previous_batch() -> None:
    """Dummy and zero-token steps clear identities while preserving live writers."""
    blocks = OpenBlocks(128)
    blocks.begin_step(dummy_run=False)
    blocks.update(_output(new=(("a", [10, 11], 150),)))
    blocks.bind_batch(["a"], np.array([2]))
    blocks.begin_step(dummy_run=True)
    assert blocks.prefill_blocks(np.array([2]), np.array([2])) is None
    assert blocks.live_blocks().tolist() == [11]
    blocks.begin_step(dummy_run=False)
    with pytest.raises(RuntimeError, match="batch binding"):
        blocks.prefill_blocks(np.array([2]), np.array([2]))


def test_real_prefill_fails_closed_without_identity_or_sufficient_blocks() -> None:
    """Missing metadata cannot silently produce an incomplete or stale KV view."""
    blocks = OpenBlocks(128)
    with pytest.raises(RuntimeError, match="batch binding"):
        blocks.prefill_blocks(np.array([2]), np.array([1]))
    blocks.begin_step(dummy_run=False)
    blocks.update(_output(new=(("a", [10, 11], 150),)))
    blocks.bind_batch(["a"], np.array([2]))
    with pytest.raises(RuntimeError, match="CPU runner request indices"):
        blocks.prefill_blocks(None, np.array([1]))
    with pytest.raises(RuntimeError, match="index 3 is unknown"):
        blocks.prefill_blocks(np.array([3]), np.array([1]))
    with pytest.raises(RuntimeError, match="needs 3 blocks; only 2"):
        blocks.prefill_blocks(np.array([2]), np.array([3]))
    blocks.update(_output(finished=frozenset({"a"})))
    with pytest.raises(RuntimeError, match="index 2 is unknown"):
        blocks.prefill_blocks(np.array([2]), np.array([1]))


def test_prefill_rejects_ambiguous_runner_bindings() -> None:
    """Two requests cannot claim one runner index or bind during a dummy step."""
    blocks = OpenBlocks(128)
    blocks.begin_step(dummy_run=False)
    with pytest.raises(ValueError, match="equal lengths"):
        blocks.bind_batch(["a"], np.array([1, 2]))
    with pytest.raises(ValueError, match="unique and nonnegative"):
        blocks.bind_batch(["a", "b"], np.array([2, 2]))
    blocks.begin_step(dummy_run=True)
    with pytest.raises(RuntimeError, match="dummy step"):
        blocks.bind_batch(["a"], np.array([2]))

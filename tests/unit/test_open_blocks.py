# SPDX-License-Identifier: Apache-2.0
"""OpenBlocks lists exactly the partially written block of each request."""

from types import SimpleNamespace

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

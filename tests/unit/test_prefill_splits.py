# SPDX-License-Identifier: Apache-2.0
"""Host split plans respect bounded partial storage without device metadata."""

import random

import pytest

from fragpack.ops.fused_attn_prefill import (
    prefill_split_count,
    prefill_workspace_capacity,
    prefill_workspace_elems,
)


def test_short_suffix_exposes_more_work_without_splitting_a_full_wave() -> None:
    """Split long histories with few queries, but keep full query waves direct."""
    short = prefill_split_count([16], [8192], 4, 8, 132)
    assert short > 1
    assert prefill_split_count([512], [8192], 4, 8, 132) == 1
    assert prefill_split_count([16] * 32, [8192] * 32, 4, 8, 132) == 1
    assert prefill_split_count([16], [128], 4, 8, 132) == 1


def test_automatic_plans_fit_one_reusable_workspace_for_ragged_batches() -> None:
    """Actual packed query/head/split state always fits the startup allocation."""
    rng = random.Random(314)
    for _ in range(500):
        sms, heads, group = (
            rng.choice([16, 80, 132]),
            rng.choice([1, 4, 8]),
            rng.choice([1, 2, 4, 8]),
        )
        lengths = [rng.randrange(0, 513) for _ in range(rng.randrange(0, 33))]
        contexts = [q + rng.randrange(0, 32769) for q in lengths]
        splits = prefill_split_count(lengths, contexts, heads, group, sms)
        required = prefill_workspace_elems(sum(lengths), heads * group, splits)
        capacity = prefill_workspace_capacity(sms)
        assert all(need <= cap for need, cap in zip(required, capacity, strict=True))
        if splits > 1:
            assert max((ctx + 127) // 128 for ctx in contexts) >= splits


def test_empty_requests_do_not_change_a_real_requests_split_budget() -> None:
    """Padding neither steals useful CTAs nor invents a longer active context."""
    expected = prefill_split_count([17, 3], [8192, 129], 4, 8, 132)
    assert (
        prefill_split_count([0, 17, 0, 3], [0, 8192, 131072, 129], 4, 8, 132)
        == expected
    )
    assert prefill_split_count([], [], 4, 8, 132) == 1
    assert prefill_workspace_elems(2048, 32, 1) == (0, 0)


@pytest.mark.parametrize(
    "queries,contexts",
    [([2], [1]), ([-1], [8]), ([1, 2], [128])],
)
def test_bad_host_lengths_are_rejected_before_launch(
    queries: list[int], contexts: list[int]
) -> None:
    """Inconsistent request metadata must not drive an unsafe split schedule."""
    with pytest.raises(ValueError):
        prefill_split_count(queries, contexts, 4, 8, 132)


def test_forced_split_workspace_can_exceed_the_automatic_capacity() -> None:
    """Forced splits allocate exact buffers instead of assuming the auto bound."""
    forced = prefill_workspace_elems(512, 32, 32)
    automatic = prefill_workspace_capacity(132)
    assert forced[0] > automatic[0] and forced[1] > automatic[1]
    with pytest.raises(ValueError, match="splits"):
        prefill_workspace_elems(16, 32, 33)

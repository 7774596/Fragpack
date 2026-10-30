# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for paired ordering, storage identity and failure retention."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from benchmarks import native_codec_dispatch_ab_bench as paired


@pytest.mark.parametrize("reverse", [False, True])
def test_latin_cycles_balance_every_position_and_preserve_all_variants(
    reverse: bool,
) -> None:
    orders = paired.variant_orders(12, reverse)
    for order in orders:
        assert set(order) == set(paired.VARIANTS)
    for position in range(4):
        assert Counter(order[position] for order in orders) == dict.fromkeys(
            paired.VARIANTS, 3
        )
    expected = tuple(paired.VARIANTS)
    assert orders[0] == (expected[::-1] if reverse else expected)


@pytest.mark.parametrize("rounds", [0, 3, 5])
def test_incomplete_position_cycles_are_rejected(rounds: int) -> None:
    with pytest.raises(ValueError, match="multiple"):
        paired.variant_orders(rounds, False)


def test_paired_contrasts_use_same_round_wall_boundaries() -> None:
    first = dict(zip(paired.VARIANTS, [100, 50, 80, 40], strict=True))
    second = dict(zip(paired.VARIANTS, [120, 80, 60, 40], strict=True))
    first_result = paired.paired_ratios(
        {name: {"trace_wall_us": wall} for name, wall in first.items()}
    )
    second_result = paired.paired_ratios(
        {name: {"trace_wall_us": wall} for name, wall in second.items()}
    )
    assert first_result["dispatch_with_chunk_wait"] == 2
    assert second_result["dispatch_with_chunk_wait"] == 1.5
    assert first_result["combined"] == 2.5
    assert second_result["combined"] == 3


def test_storage_identity_ignores_active_view_shape_but_detects_reallocation() -> None:
    owner = torch.zeros(16, dtype=torch.int32)
    assert paired.storage_signature(owner[:1]) == paired.storage_signature(owner[3:9])
    assert paired.storage_signature(owner) != paired.storage_signature(owner.clone())


def test_budget_failure_retains_the_measured_trace_without_refitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {"trace_wall_us": 137.0, "samples": [{"session": 0}]}
    calls = []
    monkeypatch.setattr(
        paired, "run_trace", lambda *a, **k: calls.append("trace") or original
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 101)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 256)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 90)
    runtime = SimpleNamespace(restore_dispatch="eager-capacity", restore_wait="chunk")
    with pytest.raises(paired.TraceInvariantError, match="not refit") as error:
        paired.measure_trace(
            runtime, {"trace": [0], "budget_bytes": 100}, {}, 90, 4, None
        )
    assert error.value.trace["trace_wall_us"] == 137.0
    assert error.value.trace["runtime_peak_allocated_bytes"] == 101
    assert error.value.trace["restore_dispatch"] == "eager-capacity"
    assert calls == ["trace"]

# SPDX-License-Identifier: Apache-2.0
"""CPU-only cache-policy contracts for the fixed-budget replay."""

import json
from pathlib import Path

import pytest

from benchmarks.native_codec_budget_bench import SessionLRU, load_cases


def test_hit_changes_eviction_order_without_changing_physical_slots() -> None:
    cache = SessionLRU(6)
    first = cache.access(10, 2)
    second = cache.access(20, 2)
    cache.access(30, 2)
    touched = cache.access(10, 2)
    assert touched.hit and touched.slots == first.slots
    replacement = cache.access(40, 2)
    assert replacement.evicted == (20,)
    assert replacement.slots == second.slots
    assert set(cache.resident) == {10, 30, 40}


def test_larger_request_evicts_multiple_whole_sessions_and_never_aliases() -> None:
    cache = SessionLRU(8)
    cache.access(0, 2)
    cache.access(1, 3)
    kept = cache.access(2, 3)
    incoming = cache.access(3, 5)
    assert incoming.evicted == (0, 1)
    assert set(incoming.slots).isdisjoint(kept.slots)
    assert len(incoming.slots) == 5
    assert set(incoming.slots) | set(kept.slots) == set(range(8))
    assert not cache.free


def test_oversized_admission_does_not_evict_existing_sessions() -> None:
    cache = SessionLRU(4)
    existing = cache.access(0, 3)
    before = list(cache.resident.items()), list(cache.free)
    rejected = cache.access(1, 5)
    assert not rejected.admitted
    assert rejected.slots == rejected.evicted == ()
    assert (list(cache.resident.items()), list(cache.free)) == before
    assert cache.access(0, 3).slots == existing.slots


def test_immutable_session_length_change_is_rejected() -> None:
    cache = SessionLRU(8)
    cache.access(0, 2)
    with pytest.raises(ValueError, match="immutable"):
        cache.access(0, 3)


def test_trace_has_reproducible_hits_with_fragmented_slot_reuse() -> None:
    trace = [0, 1, 0, 2, 1, 2, 3, 0]
    lengths = [2, 3, 2, 1]
    outputs = []
    for _ in range(2):
        cache = SessionLRU(6)
        records = []
        for session in trace:
            decision = cache.access(session, lengths[session])
            records.append(decision)
            live = [slot for slots in cache.resident.values() for slot in slots]
            assert len(live) == len(set(live))
            assert set(live).isdisjoint(cache.free)
            assert set(live) | set(cache.free) == set(range(6))
        outputs.append(records)
    assert outputs[0] == outputs[1]
    assert [item.hit for item in outputs[0]] == [
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"trace": [0, 2]},
        {"query_tokens": 1025},
        {"budget_bytes": 0},
        {"chunk_blocks": 0},
        {"native_workspace": "encode"},
        {"receive_capacity": "average"},
    ],
)
def test_case_loader_rejects_unusable_replay_contract(
    tmp_path: Path, override: dict
) -> None:
    case = {
        "name": "bad",
        "contexts": [1024, 2048],
        "trace": [0, 1, 0],
        "budget_bytes": 512000000,
    }
    case.update(override)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


def test_ragged_contexts_and_explicit_trace_are_preserved(tmp_path: Path) -> None:
    case = {
        "name": "mixed",
        "contexts": [513, 4096],
        "trace": [1, 0, 1],
        "budget_bytes": 768000000,
        "query_tokens": 16,
        "chunk_blocks": 4,
        "native_workspace": "all",
        "receive_capacity": "observed",
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case]), encoding="utf-8")
    assert load_cases(path) == [case]

# SPDX-License-Identifier: Apache-2.0
"""CPU-only cache-policy contracts for the fixed-budget replay."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.native_codec_budget_bench import (
    RestoreStatusError,
    Runtime,
    Session,
    SessionLRU,
    load_cases,
    request_restore_layout,
    restore_chunk_counts,
    validate_request_slots,
)


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
        {"restore_dispatch": "unchecked"},
        {"restore_wait": "never"},
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
        "restore_dispatch": "eager",
        "restore_wait": "chunk",
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case]), encoding="utf-8")
    assert load_cases(path) == [case]


@pytest.mark.parametrize(
    "contexts,chunk,counts",
    [
        ([513, 1024], 4, [1, 4]),
        ([129, 257], 16, [2, 3]),
        ([1, 128], 256, [1]),
        ([8192, 8193], 64, [1, 64]),
        ([384, 640], 4, [1, 3, 4]),
    ],
)
def test_restore_graph_counts_include_tails_without_oversized_capture(
    contexts: list[int], chunk: int, counts: list[int]
) -> None:
    assert restore_chunk_counts(contexts, chunk) == counts
    assert max(counts) <= max((context + 127) // 128 for context in contexts)


@pytest.mark.parametrize("dispatch", [None, "eager", "eager-capacity", "graph"])
@pytest.mark.parametrize("wait", [None, "chunk", "request"])
def test_restore_dispatch_preserves_archive_and_trace_configuration(
    tmp_path: Path, dispatch: str | None, wait: str | None
) -> None:
    case = {
        "name": "dispatch",
        "contexts": [513, 1024],
        "trace": [0, 1, 0],
        "budget_bytes": 768000000,
        "chunk_blocks": 4,
    }
    if dispatch is not None:
        case["restore_dispatch"] = dispatch
    if wait is not None:
        case["restore_wait"] = wait
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case]), encoding="utf-8")
    loaded = load_cases(path)[0]
    assert loaded["restore_dispatch"] == (dispatch or "eager")
    assert loaded["restore_wait"] == (wait or "chunk")
    assert loaded["contexts"] == case["contexts"]
    assert loaded["trace"] == case["trace"]
    assert loaded["budget_bytes"] == case["budget_bytes"]
    assert restore_chunk_counts(loaded["contexts"], loaded["chunk_blocks"]) == [1, 4]


@pytest.mark.parametrize("blocks,chunk", [(1, 16), (9, 4), (8, 4), (17, 1)])
def test_request_metadata_segments_cover_blocks_without_aliasing(
    blocks: int, chunk: int
) -> None:
    layout = request_restore_layout(blocks, chunk)
    block_ranges = [
        i for begin, count, _ in layout for i in range(begin, begin + count)
    ]
    offset_ranges = [
        i for _, count, offset in layout for i in range(offset, offset + count + 1)
    ]
    assert block_ranges == list(range(blocks))
    assert offset_ranges == list(range(blocks + len(layout)))
    assert all(1 <= count <= chunk for _, count, _ in layout)


@pytest.mark.parametrize(
    "slots,blocks,capacity",
    [((0, 1, 2, 3, 0), 5, 8), ((0, 1, -1), 3, 8), ((0, 8), 2, 8), ((0,), 2, 8)],
)
def test_request_slot_guard_rejects_cross_chunk_alias_and_invalid_ids(
    slots: tuple[int, ...], blocks: int, capacity: int
) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        validate_request_slots(slots, blocks, capacity)


def test_prepared_request_keeps_all_chunk_offsets_and_maps_alive() -> None:
    runtime = Runtime.__new__(Runtime)
    runtime.max_request_blocks, runtime.chunk, runtime.blocks = 9, 4, 20
    runtime.arm = "native_fa3"
    runtime.request_mapping_host = torch.empty((9, 1), dtype=torch.int32)
    runtime.request_offsets_host = torch.empty(12, dtype=torch.int64)
    runtime.request_layouts = {}
    offsets = np.r_[0, np.cumsum(np.arange(11, 20))]
    session = Session(9 * 128, np.arange(9), torch.empty(0), torch.empty(0), offsets)
    slots = (9, 8, 7, 6, 5, 4, 3, 2, 1)
    layout = runtime.prepare_request_restore(session, slots)
    for begin, count, offset in layout:
        assert runtime.request_mapping_host[begin : begin + count, 0].tolist() == list(
            slots[begin : begin + count]
        )
        assert (
            runtime.request_offsets_host[offset : offset + count + 1].tolist()
            == (offsets[begin : begin + count + 1] - offsets[begin]).tolist()
        )
    before = runtime.request_mapping_host.clone(), runtime.request_offsets_host.clone()
    with pytest.raises(ValueError):
        runtime.prepare_request_restore(session, slots[:-1] + (slots[0],))
    assert torch.equal(before[0], runtime.request_mapping_host)
    assert torch.equal(before[1], runtime.request_offsets_host)


@pytest.mark.parametrize("first_status", [0, -7])
def test_request_completion_checks_early_status_after_one_wait(
    first_status: int,
) -> None:
    runtime = Runtime.__new__(Runtime)
    runtime.request_status_host = torch.tensor(
        [first_status, 0, 0, 0], dtype=torch.int32
    )
    runtime.arm, runtime.stream = "native_fa3", object()
    calls = []
    runtime.restored = SimpleNamespace(
        record=lambda stream: calls.append(("record", stream)),
        synchronize=lambda: calls.append(("wait", None)),
    )
    if first_status:
        with pytest.raises(RestoreStatusError, match="-7"):
            runtime.finish_request_restore(4)
    else:
        runtime.finish_request_restore(4)
    assert calls == [("record", runtime.stream), ("wait", None)]


def test_failed_request_status_prevents_attention_dispatch() -> None:
    runtime = Runtime.__new__(Runtime)
    runtime.arm, runtime.stream = "native_fa3", object()
    runtime.blocks, runtime.lru = 4, SessionLRU(4)
    runtime.sessions = [
        Session(512, np.arange(4), torch.empty(0), torch.empty(0), np.arange(5))
    ]
    runtime.request_status_host = torch.tensor([-7, 0, 0, 0], dtype=torch.int32)
    runtime.restored = runtime.done = SimpleNamespace(
        record=lambda _: None, synchronize=lambda: None
    )
    runtime.read_ids_host = torch.empty(4, dtype=torch.int32)
    runtime.read_ids = torch.empty_like(runtime.read_ids_host)
    runtime.length_host = torch.empty(1, dtype=torch.int32)
    runtime.length = torch.empty_like(runtime.length_host)
    calls = []
    runtime.graphs = {512: SimpleNamespace(replay=lambda: calls.append("attention"))}

    def completed_restore(session, slots, *, validate):
        runtime.finish_request_restore(session.blocks)
        return {}

    runtime.restore = completed_restore
    with pytest.raises(RestoreStatusError):
        runtime.step(0)
    assert calls == []

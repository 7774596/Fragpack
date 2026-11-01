# SPDX-License-Identifier: Apache-2.0
"""Native connector lookup, engine ownership and completion contracts on CPU."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from fragpack.integrations.lmcache.identity import CacheIdentity
from fragpack.integrations.vllm.native_connector_state import (
    NativeRequestPlan,
    NativeSchedulerState,
    NativeWorkerState,
    finished_store_ids,
    native_layer_view,
)


class ControlledExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, function, *args):
        future = Future()
        self.tasks.append((function, args, future))
        return future


def scheduler(*, page_tokens=128, limit=4):
    executor = ControlledExecutor()
    state = NativeSchedulerState(
        CacheIdentity("model", 7, 2, 1),
        SimpleNamespace(lookup_prefix=lambda keys: len(keys)),
        page_tokens=page_tokens,
        max_pending_requests=limit,
        executor=executor,
    )
    return state, executor


def plan(*, count=3, store=True, request_id="r"):
    identity = CacheIdentity("model", 7, 2, 1)
    return NativeRequestPlan(
        request_id,
        tuple(identity.block_keys(list(range(128 * count)))),
        tuple(range(10, 10 + count)),
        128,
        store,
    )


class FakeTransfer:
    def __init__(self, slots=2):
        self.slots = slots
        self.jobs = []
        self.failed = False

    def submit(self, keys, page_map, store):
        if sum(not job[3].completion.done() for job in self.jobs) >= self.slots:
            return None
        source_ready, completion = Future(), Future()
        source_ready.set_running_or_notify_cancel()
        completion.set_running_or_notify_cancel()
        ticket = SimpleNamespace(source_ready=source_ready, completion=completion)
        self.jobs.append((keys, page_map, store, ticket))
        return ticket

    def submit_store(self, keys, page_map):
        return self.submit(keys, page_map, True)

    def submit_restore(self, keys, page_map):
        return self.submit(keys, page_map, False)

    def done(self, index, *, success=True):
        keys, _, store, ticket = self.jobs[index]
        if store and not ticket.source_ready.done():
            ticket.source_ready.set_result(None)
        ticket.completion.set_result((success,) * len(keys))


def worker(*, slots=2, batch=2):
    transfer = FakeTransfer(slots)
    state = NativeWorkerState(transfer, lambda value: value, batch_blocks=batch)
    return state, transfer


@pytest.mark.parametrize("physical_order", ["NHD", "HND"])
def test_native_view_preserves_bits_and_writes_original_storage(physical_order):
    raw = torch.arange(3 * 2 * 16 * 256).to(torch.bfloat16)
    if physical_order == "NHD":
        original = raw.view(3, 16, 2, 256).transpose(1, 2)
    else:
        original = raw.view(3, 2, 16, 256)
    view = native_layer_view(original)
    assert view.shape == (2, 3, 16, 2, 128)
    assert view.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert torch.equal(view[0, 2, 7, 1], original[2, 1, 7, :128])
    assert torch.equal(view[1, 2, 7, 1], original[2, 1, 7, 128:])
    view[1, 0, 0, 0, 3] = 19
    assert original[0, 0, 0, 131] == 19


def test_lookup_is_nonblocking_and_restore_excludes_shared_local_prefix():
    state, executor = scheduler(page_tokens=16)
    tokens = list(range(513))
    assert state.lookup("r", tokens, 128) == (None, False)
    function, arguments, future = executor.tasks[0]
    assert len(arguments[0]) == 3
    future.set_result(2)
    assert state.lookup("r", tokens, 128) == (256, True)
    assert len(executor.tasks) == 1
    state.allocate("r", list(range(32)), 256)
    (load,) = state.take_plans()
    assert load.page_ids == tuple(range(8, 24))
    assert load.keys == tuple(state.identity.block_keys(tokens))[1:3]
    assert not load.store and state.pending


def test_lookup_leaves_final_prompt_token_and_non_aligned_local_apc_alone():
    state, executor = scheduler()
    assert state.lookup("short", [1] * 128, 0) == (0, False)
    assert state.lookup("local", [1] * 512, 64) == (0, False)
    assert executor.tasks == []
    assert state.lookup("full", [1] * 256, 0) == (None, False)
    assert len(executor.tasks[0][1][0]) == 1


def test_lookup_queue_is_bounded_and_failed_receive_recomputes():
    state, executor = scheduler(limit=1)
    assert state.lookup("r", [1] * 257, 0) == (None, False)
    assert state.lookup("other", [1] * 257, 0) == (0, False)
    executor.tasks[0][2].set_result(2)
    assert state.lookup("r", [1] * 257, 0) == (256, True)
    state.allocate("r", [5, 6], 256)
    state.take_plans()
    state.complete(set(), {"r"}, {"r"}, worker_pending=False)
    assert state.lookup("r", [1] * 257, 0) == (0, False)
    assert len(executor.tasks) == 1


@pytest.mark.parametrize("bad", [True, -1, 3, None])
def test_invalid_lookup_response_never_reports_external_tokens(bad):
    state, executor = scheduler()
    state.lookup("r", [1] * 257, 0)
    executor.tasks[0][2].set_result(bad)
    assert state.lookup("r", [1] * 257, 0) == (0, False)


def test_finish_saves_only_complete_computed_prompt_and_bounds_delayed_sources():
    state, _ = scheduler(limit=1)
    assert state.finish(
        "r",
        list(range(385)),
        [7, 3, 5, 9],
        normal_finish=True,
        computed_tokens=300,
    )
    (store,) = state.take_plans()
    assert store.page_ids == (7, 3)
    assert len(store.keys) == 2
    assert not state.finish(
        "other", [1] * 256, [1, 2], normal_finish=True, computed_tokens=256
    )
    assert not state.reset()
    state.complete({"r"}, set(), set(), worker_pending=False)
    assert state.reset()
    assert state.finish(
        "third", [1] * 256, [1, 2], normal_finish=True, computed_tokens=256
    )


def test_aborting_remote_load_does_not_admit_an_uninitialized_store():
    state, executor = scheduler()
    state.lookup("r", [1] * 257, 0)
    executor.tasks[0][2].set_result(2)
    state.lookup("r", [1] * 257, 0)
    state.allocate("r", [3, 8], 256)
    assert not state.finish(
        "r", [1] * 257, [3, 8], normal_finish=False, computed_tokens=256
    )
    assert state.pending
    assert all(not item.store for item in state.take_plans())
    state.complete(set(), {"r"}, {"r"}, worker_pending=False)
    assert not state.pending


def test_source_release_precedes_publication_but_keeps_empty_step_progress():
    state, transfer = worker(slots=1)
    state.accept([plan(count=2)])
    ticket = transfer.jobs[0][3]
    ticket.source_ready.set_result(None)
    assert state.poll({"r"}) == ({"r"}, set(), set())
    assert state.pending
    assert state.poll(set()) == (set(), set(), set())
    transfer.done(0)
    assert state.poll(set()) == (set(), set(), set())
    assert not state.pending
    assert state.counters["store_requests"] == 1


def test_batching_retains_source_until_last_batch_read_and_drains_without_tokens():
    state, transfer = worker(slots=1, batch=2)
    item = plan(count=5)
    state.accept([item])
    assert len(transfer.jobs) == 1
    transfer.done(0)
    assert state.poll({"r"}) == (set(), set(), set())
    assert transfer.jobs[1][0] == item.keys[2:4]
    transfer.done(1)
    assert state.poll(set()) == (set(), set(), set())
    assert transfer.jobs[2][1] == ((14,),)
    transfer.jobs[2][3].source_ready.set_result(None)
    assert state.poll(set()) == ({"r"}, set(), set())
    transfer.done(2)
    state.poll(set())
    assert not state.pending


def test_pool_refusal_reuses_prepared_map_across_empty_steps():
    transfer = FakeTransfer(slots=1)
    maps = []

    def make_map(rows):
        maps.append(rows)
        return rows

    state = NativeWorkerState(transfer, make_map, batch_blocks=1)
    state.accept([plan(count=3)])
    assert len(maps) == 2  # One submitted, one retained on refusal.
    for _ in range(5):
        state.poll({"r"})
    assert len(maps) == 2
    assert state.pending_map_bytes == 4
    transfer.done(0)
    state.poll(set())
    assert len(maps) == 3
    assert transfer.jobs[1][1] is maps[1]


def test_finish_before_plan_is_preserved_without_retaining_unadmitted_requests():
    state, transfer = worker()
    unrelated = {f"short-or-aborted-{i}" for i in range(10000)}
    state.poll(unrelated | {"r"})
    assert state.retained_finish_count == 0
    item = plan(count=1)
    # The scheduler attaches this certified subset to the plan in the same
    # SchedulerOutput. Generic poll ordering cannot lose it.
    certified = finished_store_ids([item], unrelated | {"r"})
    state.accept([item], finished_requests=certified)
    assert state.retained_finish_count == 1
    transfer.done(0)
    assert state.poll(unrelated) == ({"r"}, set(), set())
    assert state.retained_finish_count == 0
    assert not state.pending


def test_store_plan_without_scheduler_finish_evidence_fails_explicitly():
    with pytest.raises(RuntimeError, match="finish notification"):
        finished_store_ids([plan()], {"unrelated"})
    state, _ = worker()
    with pytest.raises(ValueError, match="admitted store plan"):
        state.accept([plan()], finished_requests=["unrelated"])


def test_failed_restore_waits_for_other_inflight_destination_writes():
    state, transfer = worker(slots=2, batch=1)
    state.accept([plan(count=3, store=False)])
    transfer.done(0, success=False)
    assert state.poll(set()) == (set(), set(), set())
    assert len(transfer.jobs) == 2
    transfer.done(1)
    assert state.poll(set()) == (set(), {"r"}, {"r"})
    assert not state.pending


def test_publication_exception_after_source_ready_is_a_safe_cache_failure():
    state, transfer = worker(slots=1)
    state.accept([plan(count=2)])
    ticket = transfer.jobs[0][3]
    ticket.source_ready.set_result(None)
    ticket.completion.set_exception(TimeoutError("uncertain publication"))
    assert state.poll({"r"}) == ({"r"}, set(), set())
    assert not state.pending
    assert state.counters["store_failures"] == 1


def test_unknown_cuda_completion_never_releases_engine_sources():
    state, transfer = worker()
    state.accept([plan(count=2)])
    transfer.failed = True
    with pytest.raises(RuntimeError, match="retain engine KV pages"):
        state.poll({"r"})
    assert state.pending


def test_restore_destinations_must_be_distinct_and_failed_source_future_is_fatal():
    item = plan(count=2, store=False)
    with pytest.raises(ValueError, match="exclusive distinct"):
        NativeRequestPlan("r", item.keys, (4, 4), 128, False)
    state, transfer = worker()
    state.accept([plan(count=2)])
    transfer.jobs[0][3].source_ready.set_exception(RuntimeError("GPU unknown"))
    with pytest.raises(RuntimeError, match="GPU unknown"):
        state.poll({"r"})
    assert state.pending

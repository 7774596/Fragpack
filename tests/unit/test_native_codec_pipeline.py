# SPDX-License-Identifier: Apache-2.0
"""CPU event-generation model for bounded copy/restore ownership."""

from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmarks import native_codec_pipeline as pipeline
from fragpack.compression import attn_page


def test_direct_raw_archive_rejected_before_runtime_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_init(*args: object, **kwargs: object) -> None:
        pytest.fail("incompatible archive reached GPU runtime allocation")

    monkeypatch.setattr(pipeline.budget.Runtime, "__init__", unexpected_init)
    with pytest.raises(ValueError, match="packed-scatter raw archives"):
        pipeline.PipelineRuntime(
            "raw_fa3", 8, {"raw_transfer": "native-direct"}, None, [], None
        )


class Queue:
    def __init__(self) -> None:
        self.pending: list[Callable[[], bool]] = []

    def submit(self, action: Callable[[], None]) -> None:
        def run() -> bool:
            action()
            return True

        self.pending.append(run)

    def wait_event(self, event: "Event") -> None:
        # CUDA waits bind the most recently recorded generation at enqueue time.
        generation = event.generation
        self.pending.append(lambda: generation in event.completed)


class Event:
    def __init__(self) -> None:
        self.generation = 0
        self.completed = {0}

    def record(self, stream: Queue) -> None:
        self.generation += 1
        generation = self.generation
        stream.submit(lambda: self.completed.add(generation))


class Scheduler:
    def __init__(self) -> None:
        self.current: Queue | None = None

    @contextmanager
    def stream(self, stream: Queue):
        previous, self.current = self.current, stream
        try:
            yield stream
        finally:
            self.current = previous

    def submit(self, action: Callable[[], None]) -> None:
        assert self.current is not None
        self.current.submit(action)

    def drain(self, queues: list[Queue]) -> None:
        # Drain the preferred queue whenever possible, exposing missing fences.
        queues = list(dict.fromkeys(queues))
        for _ in range(1000):
            if not any(queue.pending for queue in queues):
                return
            for queue in queues:
                if queue.pending and queue.pending[0]():
                    queue.pending.pop(0)
                    break
            else:
                pytest.fail("cyclic or unsatisfied event-generation dependency")
        pytest.fail("scheduler did not finish")


@pytest.mark.parametrize("shared_status", [True, False])
@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("copy_first", [True, False])
def test_reuse_tail_and_early_error_survive_async_queues(
    monkeypatch: pytest.MonkeyPatch, serial: bool, copy_first: bool, shared_status: bool
) -> None:
    scheduler = Scheduler()
    monkeypatch.setattr(pipeline.torch.cuda, "stream", scheduler.stream)
    compute = Queue()
    copy = compute if serial else Queue()
    ready, done = [Event(), Event()], [Event(), Event()]
    buffers: list[tuple[int, tuple[int, ...]] | None] = [None, None]
    payloads = [
        (i, tuple(range(4 * i, 4 * i + n))) for i, n in enumerate([4] * 4 + [1])
    ]
    restored, statuses = [], [None] * sum(len(p[1]) for p in payloads)
    gpu_status = {}  # Native: shared codec; HBM: one retained tensor per slot/count.
    log: list[tuple[str, int]] = []

    def enqueue(index: int) -> None:
        slot, payload = index % 2, payloads[index]
        count = len(payload[1])
        status = gpu_status.setdefault(
            None if shared_status else (slot, count), [None] * 4
        )

        def copy_work() -> None:
            buffers[slot] = payload
            log.append(("copy", index))

        def restore_work() -> None:
            assert buffers[slot] == payload, "slot overwritten or copy not ready"
            restored.append(buffers[slot])
            status[:count] = [int(block == 0) for block in payload[1]]
            log.append(("restore", index))

        def status_work() -> None:
            statuses[4 * index : 4 * index + count] = status[:count]
            log.append(("status", index))

        def consume() -> None:
            scheduler.submit(restore_work)
            scheduler.submit(status_work)

        pipeline.enqueue_slot(
            copy,
            compute,
            ready[slot],
            done[slot],
            lambda: scheduler.submit(copy_work),
            consume,
        )

    for index in range(len(payloads)):
        enqueue(index)
    assert restored == []  # Helpers enqueue; they cannot execute device work here.
    scheduler.drain([copy, compute] if copy_first else [compute, copy])
    assert restored == payloads  # Also the same result for serial and overlap.
    assert statuses == [1] + [0] * 16
    for index in range(2, len(payloads)):
        assert log.index(("copy", index)) > log.index(("status", index - 2))
    assert [event.generation for event in done] == [3, 2]


@pytest.mark.parametrize("chunk,counts", [(1, [1]), (4, [1, 4])])
def test_extra_bytes_follow_slot_ownership_and_format_bound(
    chunk: int, counts: list[int]
) -> None:
    raw = pipeline.pipeline_extra_bytes("raw_fa3", chunk, counts)
    native = pipeline.pipeline_extra_bytes("native_fa3", chunk, counts)
    raw_shape = torch.empty(
        (chunk, 48, 2, 4, 128, 128), dtype=torch.bfloat16, device="meta"
    )
    mapping = torch.empty((chunk, 1), dtype=torch.int32, device="meta")
    offsets = torch.empty(chunk + 1, dtype=torch.int64, device="meta")
    assert raw["receive_payload"] == raw_shape.nbytes
    assert native["receive_payload"] == chunk * attn_page.blob_slot_bytes(48, 4)
    assert native["receive_payload"] > raw["receive_payload"]
    for values in (raw, native):
        assert values["mapping"] == mapping.nbytes
        assert values["offsets"] == offsets.nbytes
        assert values["captured_status_logical"] == 0  # Shared original codec.
        assert values["total_logical"] == sum(
            v for k, v in values.items() if k != "total_logical"
        )
    for arm in ("gather_fa3", "fused_hbm"):
        extra = pipeline.pipeline_extra_bytes(arm, chunk, counts)
        status = torch.empty(sum(counts), dtype=torch.int32, device="meta")
        assert extra["captured_status_logical"] == status.nbytes
        assert extra["total_logical"] == native["total_logical"] + status.nbytes


@pytest.mark.parametrize(
    "arm,chunk,counts",
    [
        ("unknown", 4, [4]),
        ("native_fa3", 0, [1]),
        ("raw_fa3", 4, [0]),
        ("gather_fa3", 4, [5]),
        ("fused_hbm", 4, [-1]),
        ("native_fa3", 4, []),
        ("gather_fa3", 4, [1, 1]),
    ],
)
def test_extra_bytes_reject_invalid_ownership_sizes(
    arm: str, chunk: int, counts: list[int]
) -> None:
    with pytest.raises(ValueError):
        pipeline.pipeline_extra_bytes(arm, chunk, counts)


def test_original_mode_calls_base_enqueue_without_pipeline_events(monkeypatch) -> None:
    runtime = object.__new__(pipeline.PipelineRuntime)
    calls = []
    monkeypatch.setattr(
        pipeline.budget.Runtime,
        "enqueue_request_chunk",
        lambda self, *args: calls.append(args),
    )
    monkeypatch.setattr(
        pipeline, "enqueue_slot", lambda *args: pytest.fail("original used slot events")
    )
    runtime.select_mode("overlap")
    runtime.select_mode("original")
    runtime.enqueue_request_chunk(4, 4, 5, None)
    assert calls == [(4, 4, 5, None)]


@pytest.mark.parametrize("copy_fails", [False, True])
def test_close_always_attempts_both_stream_drains(copy_fails: bool) -> None:
    runtime = object.__new__(pipeline.PipelineRuntime)
    drained = []

    def drain_copy() -> None:
        drained.append("copy")
        if copy_fails:
            raise RuntimeError("copy drain failed")

    runtime.copy_stream = SimpleNamespace(synchronize=drain_copy)
    runtime.stream = SimpleNamespace(synchronize=lambda: drained.append("compute"))
    if copy_fails:
        with pytest.raises(RuntimeError, match="copy drain failed"):
            runtime.close()
    else:
        runtime.close()
    assert drained == ["copy", "compute"]

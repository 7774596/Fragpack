# SPDX-License-Identifier: Apache-2.0
"""CPU ownership and accounting checks for CUDA-ordered fixed codec slots."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

import numpy as np
import pytest
import torch

from fragpack.ops.native_codec_pool import (
    NativeCodecLease,
    NativeCodecPool,
    native_codec_pool_bytes,
)


@dataclass(eq=False)
class _Stream:
    waits: list[tuple[_Event, int]] = field(default_factory=list)

    def wait_event(self, event: _Event) -> None:
        self.waits.append((event, event.generation))

    def synchronize(self) -> None:
        raise AssertionError("pool must not synchronize streams")

    def query(self) -> None:
        raise AssertionError("pool must not query stream completion")


class _Event:
    def __init__(self, **kwargs: Any) -> None:
        self.generation = 0
        self.records: list[_Stream] = []

    def record(self, stream: _Stream) -> None:
        self.generation += 1
        self.records.append(stream)

    def synchronize(self) -> None:
        raise AssertionError("pool must not synchronize events")

    def query(self) -> None:
        raise AssertionError("pool must not query event completion")


@pytest.fixture
def host_pool(monkeypatch: pytest.MonkeyPatch) -> list[_Stream]:
    """Use real tensors/codec allocation but model only CUDA event ordering."""
    original_empty, original_to = torch.empty, torch.Tensor.to
    current = [_Stream()]

    def host_empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    def host_to(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if args and isinstance(args[0], (str, torch.device)):
            if torch.device(args[0]).type == "cuda":
                args = (torch.device("cpu"), *args[1:])
        if "device" in kwargs and torch.device(kwargs["device"]).type == "cuda":
            kwargs["device"] = torch.device("cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", host_empty)
    monkeypatch.setattr(torch.Tensor, "to", host_to)
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current[0])
    return current


def _pool(workspace: str = "decode", slots: int = 2) -> NativeCodecPool:
    table = np.broadcast_to(np.arange(22, dtype=np.uint8), (2, 2, 1, 22)).copy()
    return NativeCodecPool(
        table, max_blocks=3, slots=slots, workspace=workspace, device="cuda:0"
    )


@pytest.mark.parametrize("workspace", ["encode", "decode", "raw"])
def test_pool_budget_matches_owned_allocations_without_counting_encode_aliases(
    workspace: str, host_pool: list[_Stream]
) -> None:
    pool = _pool(workspace)
    assert pool.memory_usage() == native_codec_pool_bytes(
        2, 1, 3, 2, workspace=workspace
    )
    first, second = pool.acquire(), pool.acquire()
    assert first.payload.data_ptr() != second.payload.data_ptr()
    assert first.codec is not second.codec
    if workspace == "encode":
        assert first.payload is first.codec.buffer
        assert first.offsets.data_ptr() == first.codec.offsets.data_ptr()
        assert pool.memory_usage()["receive_payload"] == 0
        assert pool.memory_usage()["receive_offsets"] == 0
    elif workspace == "decode":
        assert first.offsets.shape == (pool.max_blocks + 1,)
        assert first.offsets.data_ptr() != second.offsets.data_ptr()
    else:
        assert first.offsets is second.offsets is None
    first.release()
    second.release()


def test_exclusive_leases_wait_for_prior_release_before_cross_stream_reuse(
    host_pool: list[_Stream], monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool()
    initial = host_pool[0]
    stream_a, stream_b = _Stream(), _Stream()
    host_pool[0] = stream_a
    first, second = pool.acquire(), pool.acquire()
    payload_pointer = first.payload.data_ptr()
    event, generation = stream_a.waits[0]
    assert event.records == [initial] and generation == 1
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.acquire()
    first.release()
    assert event.records == [initial, stream_a]
    host_pool[0] = stream_b

    def reject_allocation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("lease operations must not allocate storage or events")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    monkeypatch.setattr(torch.cuda, "Event", reject_allocation)
    repeated = pool.acquire()
    assert repeated.slot_index == first.slot_index
    assert repeated.payload.data_ptr() == payload_pointer
    assert stream_b.waits == [(event, 2)]
    assert second.codec is not repeated.codec
    with pytest.raises(RuntimeError, match="stale"):
        _ = first.payload
    with pytest.raises(RuntimeError, match="stale"):
        first.release()
    # Even a forged lease for this slot/generation cannot borrow the live slot.
    forged = NativeCodecLease(pool, repeated.slot_index, 2)
    with pytest.raises(RuntimeError, match="stale"):
        pool.release(forged)
    repeated.release()
    host_pool[0] = stream_a
    second.release()


def test_wrong_pool_or_stream_release_leaves_lease_active(
    host_pool: list[_Stream],
) -> None:
    pool, other = _pool(), _pool()
    acquired_stream = host_pool[0]
    lease = pool.acquire()
    with pytest.raises(ValueError, match="different"):
        other.release(lease)
    host_pool[0] = _Stream()
    with pytest.raises(RuntimeError, match="same CUDA stream"):
        lease.release()
    assert lease.codec.workspace == "decode"
    host_pool[0] = acquired_stream
    lease.release()
    with pytest.raises(RuntimeError, match="stale"):
        _ = lease.codec


def test_context_exception_records_completion_and_releases_slot(
    host_pool: list[_Stream],
) -> None:
    pool = _pool(slots=1)
    with pytest.raises(ValueError, match="caller failure"):
        with pool.acquire() as lease:
            pointer = lease.payload.data_ptr()
            raise ValueError("caller failure")
    with pool.acquire() as next_lease:
        assert next_lease.payload.data_ptr() == pointer
    assert [generation for _, generation in host_pool[0].waits] == [1, 2]


def test_pool_rejects_other_host_threads_without_touching_cuda(
    host_pool: list[_Stream],
) -> None:
    pool = _pool()
    lease = pool.acquire()
    errors: list[Exception] = []

    def wrong_thread() -> None:
        for operation in (pool.acquire, lease.release, lambda: lease.payload):
            try:
                operation()
            except Exception as error:
                errors.append(error)

    worker = threading.Thread(target=wrong_thread)
    worker.start()
    worker.join()
    assert len(errors) == 3
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert all("constructing host thread" in str(error) for error in errors)
    assert len(host_pool[0].waits) == 1
    lease.release()


@pytest.mark.parametrize("slots,workspace", [(0, "raw"), (True, "raw"), (1, "all")])
def test_pool_budget_rejects_invalid_contract_without_cuda(
    slots: int, workspace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_allocation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("planning must not allocate or initialize CUDA")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    monkeypatch.setattr(torch.cuda, "current_device", reject_allocation)
    with pytest.raises(ValueError):
        native_codec_pool_bytes(48, 4, 16, slots, workspace=workspace)


def test_pool_budget_scales_with_fixed_slots_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_allocation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("planning must not allocate or initialize CUDA")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    monkeypatch.setattr(torch.cuda, "current_device", reject_allocation)
    single = native_codec_pool_bytes(48, 4, 16, 1, workspace="decode")
    double = native_codec_pool_bytes(48, 4, 16, 2, workspace="decode")
    assert double == {name: 2 * size for name, size in single.items()}
    assert single["codec_record_scratch"] == single["codec_blob_buffer"] == 0
    assert single["receive_payload"] > single["codec_record_offsets"] > 0

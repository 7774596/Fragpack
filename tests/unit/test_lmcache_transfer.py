# SPDX-License-Identifier: Apache-2.0
"""Queue ownership and failure contracts without CUDA initialization."""

from __future__ import annotations

from contextlib import nullcontext
import gc
import struct
import threading
from types import SimpleNamespace
import weakref

import pytest
import torch

from fragpack.compression import attn_page as ap
from fragpack.integrations.lmcache import transfer
from fragpack.integrations.lmcache.identity import CacheIdentity
from fragpack.integrations.lmcache.protocol import canonicalize_blob


def _blob(identity: CacheIdentity) -> bytes:
    data = bytearray([0xA5] * identity.max_blob_bytes)
    struct.pack_into(
        "<4Iqq", data, 0, ap.BLOB_MAGIC, 1, 1, 1, len(data), identity.table_digest
    )
    struct.pack_into("<2i", data, 64, *([ap.RAW_RECORD_BYTES - ap.DENSE_BYTES] * 2))
    return bytes(data)


class _Event:
    def synchronize(self) -> None:
        pass


class _Buffers:
    def __init__(self, store, num_slots):
        self.store = store
        self.num_slots = num_slots
        self.prepares = 0
        self.drains = 0
        self.imports = []
        self.prepare_error = False
        self.drain_error = False
        self.import_result = None

    def validate_blocks(self, blocks, count):
        if blocks.shape != (count,):
            raise ValueError("block count mismatch")

    def prepare(self, blocks, slots, restore):
        self.prepares += 1
        if self.prepare_error:
            raise RuntimeError("event recording failed")
        return SimpleNamespace(blocks=blocks.clone(), ready=_Event())

    def read_export(self, job):
        return (_blob(self.store.identity),) * len(job.keys)

    def import_blobs(self, job, blobs):
        self.imports.append((job.device_work.blocks.tolist(), list(blobs)))
        if self.import_result is not None:
            return self.import_result
        return (True,) * len(blobs) + (False,) * (len(job.keys) - len(blobs))

    def drain_producer(self):
        self.drains += 1
        if self.drain_error:
            raise RuntimeError("producer still running")

    def drain(self, work):
        self.drains += 1
        if self.drain_error:
            raise RuntimeError("copy completion unknown")

    def memory_usage(self):
        size = self.num_slots * self.store.blob_stride
        return {"gpu_transfer": size, "pinned_transfer": size}


class _Client:
    def __init__(self):
        self.blobs = {}
        self.gets = []
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.error = False
        self.refuse = False

    def put(self, key, blob):
        self.entered.set()
        assert self.gate.wait(5), "test client was not released"
        if self.error:
            raise RuntimeError("network failed")
        if self.refuse:
            return False
        self.blobs[key] = blob
        return True

    def get(self, key):
        self.gets.append(key)
        return self.blobs.get(key)


@pytest.fixture
def setup(monkeypatch):
    identity = CacheIdentity("test-model", 41, 1, 1)
    store = SimpleNamespace(
        identity=identity,
        num_layers=1,
        kv_heads=1,
        table_digest=41,
        blob_stride=identity.max_blob_bytes,
    )
    buffers = []

    def factory(store, slots):
        result = _Buffers(store, slots)
        buffers.append(result)
        return result

    monkeypatch.setattr(transfer, "_CudaBuffers", factory)
    workers = []

    def create(slots=2):
        client = _Client()
        worker = transfer.BlockTransfer(store, client, identity, slots)
        workers.append((worker, client))
        return worker, client, buffers[-1]

    yield identity, identity.block_keys(list(range(384))), create
    for worker, client in workers:
        client.gate.set()
        worker.close()


def test_pool_retains_buffers_until_publication(setup):
    identity, keys, create = setup
    worker, client, buffers = create(1)
    client.gate.clear()
    ticket = worker.submit_store(keys[:1], torch.tensor([0]))
    assert ticket is not None and not ticket.completion.cancel()
    assert client.entered.wait(5)
    ticket.source_ready.synchronize()
    assert not ticket.completion.done()
    assert worker.submit_store(keys[1:2], torch.tensor([1])) is None
    assert buffers.prepares == 1
    client.gate.set()
    assert ticket.completion.result(5) == (True,)
    assert client.blobs[keys[0]] == canonicalize_blob(_blob(identity), identity)
    retry = worker.submit_store(keys[1:2], torch.tensor([1]))
    assert retry is not None and retry.completion.result(5) == (True,)


def test_restore_stops_at_first_missing_block(setup):
    identity, keys, create = setup
    worker, client, buffers = create(3)
    client.blobs[keys[0]] = _blob(identity)
    client.blobs[keys[2]] = _blob(identity)
    ticket = worker.submit_restore(keys, torch.tensor([3, 4, 5]))
    assert ticket is not None
    assert ticket.completion.result(5) == (True, False, False)
    assert client.gets == keys[:2]
    assert len(buffers.imports[0][1]) == 1


def test_network_failure_drains_then_releases_pool(setup):
    _, keys, create = setup
    worker, client, buffers = create(1)
    client.error = True
    ticket = worker.submit_store(keys[:1], torch.tensor([0]))
    with pytest.raises(RuntimeError, match="network failed"):
        ticket.completion.result(5)
    assert buffers.drains == 1 and not worker.failed
    client.error = False
    retry = worker.submit_store(keys[:1], torch.tensor([0]))
    assert retry is not None and retry.completion.result(5) == (True,)


def test_unknown_cuda_completion_disables_pool_reuse(setup):
    _, keys, create = setup
    worker, client, buffers = create(1)
    client.error = buffers.drain_error = True
    ticket = worker.submit_store(keys[:1], torch.tensor([0]))
    with pytest.raises(RuntimeError, match="process recovery"):
        ticket.completion.result(5)
    assert worker.failed
    assert worker.submit_store(keys[:1], torch.tensor([0])) is None


def test_recording_failure_drains_producer_before_reuse(setup):
    _, keys, create = setup
    worker, _, buffers = create(1)
    buffers.prepare_error = True
    with pytest.raises(RuntimeError, match="event recording"):
        worker.submit_store(keys[:1], torch.tensor([0]))
    assert buffers.drains == 1
    buffers.prepare_error = False
    retry = worker.submit_store(keys[:1], torch.tensor([0]))
    assert retry is not None and retry.completion.result(5) == (True,)


def test_recording_and_drain_failure_retains_owners(setup):
    _, keys, create = setup
    worker, _, buffers = create(1)
    buffers.prepare_error = buffers.drain_error = True
    with pytest.raises(RuntimeError, match="retain source blocks"):
        worker.submit_store(keys[:1], torch.tensor([0]))
    assert worker.failed
    assert worker.submit_restore(keys[:1], torch.tensor([0])) is None


def test_corrupt_restore_is_not_imported_and_pool_can_retry(setup):
    identity, keys, create = setup
    worker, client, buffers = create(1)
    client.blobs[keys[0]] = b"truncated"
    ticket = worker.submit_restore(keys[:1], torch.tensor([0]))
    with pytest.raises(ValueError):
        ticket.completion.result(5)
    assert not buffers.imports
    client.blobs[keys[0]] = _blob(identity)
    retry = worker.submit_restore(keys[:1], torch.tensor([0]))
    assert retry is not None and retry.completion.result(5) == (True,)


def test_refusal_and_close_never_publish_later_prefix(setup):
    _, keys, create = setup
    worker, client, _ = create(2)
    client.refuse = True
    ticket = worker.submit_store(keys[:2], torch.tensor([0, 1]))
    assert ticket.completion.result(5) == (False, False)
    assert not client.blobs
    worker.close()
    assert worker.submit_restore(keys[:1], torch.tensor([0])) is None


def test_foreign_namespace_rejected_before_gpu_work(setup):
    _, _, create = setup
    worker, _, buffers = create(1)
    foreign = CacheIdentity("other-model", 41, 1, 1).block_keys(list(range(128)))
    with pytest.raises(ValueError, match="another cache identity"):
        worker.submit_store(foreign, torch.tensor([0]))
    assert buffers.prepares == 0


def test_close_timeout_preserves_pending_job_and_can_finish_later(setup):
    _, keys, create = setup
    worker, client, _ = create(1)
    client.gate.clear()
    ticket = worker.submit_store(keys[:1], torch.tensor([0]))
    assert client.entered.wait(5)
    with pytest.raises(TimeoutError, match="owners retained"):
        worker.close(timeout=0)
    assert not ticket.completion.done()
    assert worker.submit_store(keys[:1], torch.tensor([0])) is None
    client.gate.set()
    worker.close(timeout=5)
    assert ticket.completion.result(0) == (True,)


def test_partial_prepare_retains_created_tensors_until_successful_drain(monkeypatch):
    # Isolate the CUDA adapter to observe actual tensor lifetimes on CPU.
    buffers_type = transfer._CudaBuffers  # noqa: SLF001
    buffers = object.__new__(buffers_type)
    owned_refs = []
    length_refs = []

    class ExportStore:
        def export_blocks(self, blocks, out, stride):
            owned_refs.append(weakref.ref(blocks._base))  # noqa: SLF001
            if length_refs:
                raise RuntimeError("second export failed")
            lengths = torch.tensor([4096], dtype=torch.int64)
            length_refs.append(weakref.ref(lengths))
            return lengths

    class Producer:
        fails = True

        def synchronize(self):
            if self.fails:
                raise RuntimeError("completion unknown")

    producer = Producer()
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: producer)
    buffers.device = torch.device("cpu")
    buffers.store = ExportStore()
    buffers.stride = 1
    buffers.gpu = torch.empty((2, 1), dtype=torch.uint8)
    buffers._pending_prepare = []  # noqa: SLF001
    with pytest.raises(RuntimeError, match="second export failed"):
        buffers.prepare(torch.tensor([0, 1]), (0, 1), restore=False)
    gc.collect()
    assert all(ref() is not None for ref in owned_refs + length_refs)
    with pytest.raises(RuntimeError, match="completion unknown"):
        buffers.drain_producer()
    gc.collect()
    assert all(ref() is not None for ref in owned_refs + length_refs)
    producer.fails = False
    buffers.drain_producer()
    gc.collect()
    assert all(ref() is None for ref in owned_refs + length_refs)

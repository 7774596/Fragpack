# SPDX-License-Identifier: Apache-2.0
"""CPU fault injection for the plugin's public LMCache ownership contract."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import struct
import threading
from types import SimpleNamespace

import pytest
import torch

from fragpack.compression import attn_page as ap
from fragpack.integrations.lmcache.client import RemoteBlobClient, require_lmcache_056
from fragpack.integrations.lmcache.identity import CacheIdentity
from fragpack.integrations.lmcache.protocol import decode_message, encode_message
from fragpack.integrations.lmcache.server import FragPackServerModule
from fragpack.integrations.lmcache.storage import LMCacheL1Store


@dataclass(frozen=True)
class ObjectKey:
    chunk_hash: bytes
    model_name: str
    kv_rank: int
    object_group_id: int
    cache_salt: str


class MemoryObject:
    def __init__(self, length):
        self.tensor = torch.zeros(length, dtype=torch.uint8)


class FakeManager:
    """Model staging/resident separation and captured-owner read releases."""

    def __init__(self):
        self.resident = {}
        self.staging = {}
        self.reads = {}
        self.adapters = []
        self.mode = "ok"
        self.reservation_lengths = []
        self.next_handle = 0
        self.results = {}

    def l2_adapters(self):
        return self.adapters

    def reserve_write(self, keys, layout):
        length = layout.shapes[0][0]
        self.reservation_lengths.append(length)
        if self.mode == "refuse":
            return {}
        result = {key: MemoryObject(length) for key in keys}
        self.staging.update(result)
        return result

    def prepare_write_completion(self, objects):
        if self.mode == "capture_error":
            raise RuntimeError("injected owner capture failure")
        return [(7, list(objects))]

    def finish_write_by_owner(self, completion):
        if self.mode == "finish_error":
            raise RuntimeError("injected finish failure")
        for owner, keys in completion:
            assert owner == 7
            for key in keys:
                obj = self.staging.pop(key)
                if self.mode != "expire":
                    self.resident.setdefault(key, obj)

    def submit_prefetch_task(self, spec, skip_l2):
        assert skip_l2 is True
        key = spec.key_groups[0].keys[0]
        hit = key in self.resident
        if hit:
            self.reads[key] = self.reads.get(key, 0) + 1
        handle = self.next_handle
        self.next_handle += 1
        self.results[handle] = SimpleNamespace(
            hit_cells=[SimpleNamespace(get_indices_list=lambda: [0] if hit else [])],
            l1_owners={key: 7} if hit else {},
        )
        return handle

    def wait_prefetch_status(self, handle, timeout):
        assert timeout > 0
        if self.mode == "wait_error":
            raise RuntimeError("injected wait failure")
        return self.mode != "timeout"

    def query_prefetch_status(self, handle):
        if self.mode == "query_error":
            raise RuntimeError("injected query failure")
        if self.mode == "timeout":
            return None
        return self.results.pop(handle, None)

    @contextmanager
    def read_prefetched_results(self, keys, owners):
        try:
            yield [self.resident[key] for key in keys]
        except Exception:
            self.finish_read_prefetched(keys, l1_owners=owners)
            raise

    def finish_read_prefetched(self, keys, l1_owners):
        assert l1_owners == {key: 7 for key in keys}
        for key in keys:
            assert self.reads[key] > 0
            self.reads[key] -= 1


def raw_blob(identity, byte=42):
    blob = bytearray(identity.max_blob_bytes)
    struct.pack_into(
        "<4Iqq", blob, 0, ap.BLOB_MAGIC, 1, 1, 1, len(blob), identity.table_digest
    )
    struct.pack_into("<2i", blob, ap.BLOB_FIXED_BYTES, *([10240] * 2))
    blob[ap.blob_head_bytes(1, 1) :] = bytes([byte]) * (2 * ap.RAW_RECORD_BYTES)
    return bytes(blob)


@pytest.fixture
def setup(monkeypatch):
    api = SimpleNamespace(
        ObjectKey=ObjectKey,
        MemoryLayoutDesc=SimpleNamespace,
        GroupedObjectKeys=SimpleNamespace,
        PrefetchTaskSpec=SimpleNamespace,
    )
    monkeypatch.setattr("fragpack.integrations.lmcache.storage._load_api", lambda: api)
    identity = CacheIdentity("test-model-revision", 123, 1, 1)
    keys = identity.block_keys(list(range(128 * 3)))
    manager = FakeManager()
    return identity, keys, manager, LMCacheL1Store(manager, identity)


def test_exact_bytes_publication_read_locks_and_duplicate_suppression(setup):
    identity, keys, manager, store = setup
    blob = raw_blob(identity)
    assert store.get(keys[0]) is None
    assert store.put(keys[0], blob)
    assert store.put(keys[0], blob)
    assert manager.reservation_lengths == [len(blob)]
    assert store.get(keys[0]) == blob
    assert store.put(keys[2], blob)
    assert store.lookup_prefix(keys) == 1
    assert all(value == 0 for value in manager.reads.values())


def test_compressed_extent_and_undefined_padding_do_not_change_identity(setup):
    identity, keys, manager, store = setup
    head = ap.blob_head_bytes(1, 1)
    declared = head + 2 * (ap.DENSE_BYTES + ap.HEADER_BYTES)
    size = (declared + ap.BLOB_ALIGN - 1) // ap.BLOB_ALIGN * ap.BLOB_ALIGN
    blob = bytearray(size)
    struct.pack_into("<4Iqq", blob, 0, ap.BLOB_MAGIC, 1, 1, 1, declared, 123)
    struct.pack_into("<2i", blob, ap.BLOB_FIXED_BYTES, *([ap.HEADER_BYTES] * 2))
    assert store.put(keys[0], bytes(blob))
    blob[32:64] = b"\xff" * 32
    blob[declared:] = b"\xff" * (size - declared)
    assert store.put(keys[0], bytes(blob))
    assert manager.reservation_lengths == [size]
    assert size < identity.max_blob_bytes


@pytest.mark.parametrize("mode", ["capture_error", "finish_error"])
def test_uncertain_write_never_publishes_or_deletes_other_owners(setup, mode):
    identity, keys, manager, store = setup
    assert store.put(keys[1], raw_blob(identity))
    resident_before = dict(manager.resident)
    manager.mode = mode
    with pytest.raises(RuntimeError, match="injected"):
        store.put(keys[0], raw_blob(identity))
    assert manager.resident == resident_before
    assert len(manager.staging) == 1
    assert store.report_status()["failed"]
    with pytest.raises(RuntimeError, match="restart"):
        store.put(keys[2], raw_blob(identity))


@pytest.mark.parametrize("mode", ["expire", "refuse"])
def test_void_finish_and_refused_allocation_do_not_claim_publication(setup, mode):
    identity, keys, manager, store = setup
    manager.mode = mode
    assert store.put(keys[0], raw_blob(identity)) is False
    if mode == "expire":
        assert store.report_status()["failed"]
    else:
        assert store.lookup_prefix(keys) == 0


def test_invalid_blob_and_namespace_never_reserve(setup):
    identity, keys, manager, store = setup
    with pytest.raises(ValueError):
        store.put(keys[0], b"not-a-blob")
    other = CacheIdentity("other-model", 123, 1, 1)
    with pytest.raises(ValueError, match="identity"):
        store.put(other.block_keys(list(range(128)))[0], raw_blob(identity))
    assert not manager.reservation_lengths


def test_corrupt_read_releases_lock_and_duplicate_conflict_keeps_original(setup):
    identity, keys, manager, store = setup
    assert store.put(keys[0], raw_blob(identity))
    with pytest.raises(ValueError, match="Different"):
        store.put(keys[0], raw_blob(identity, byte=8))
    next(iter(manager.resident.values())).tensor[0] = 0
    with pytest.raises(ValueError):
        store.get(keys[0])
    assert all(value == 0 for value in manager.reads.values())


def test_l2_is_rejected_even_if_added_after_construction(setup):
    identity, keys, manager, store = setup
    manager.adapters.append(object())
    with pytest.raises(RuntimeError, match="no L2"):
        store.put(keys[0], raw_blob(identity))
    assert not manager.reservation_lengths


def test_prefetch_timeout_fails_closed_and_close_releases_late_lock(setup):
    identity, keys, manager, store = setup
    assert store.put(keys[0], raw_blob(identity))
    manager.mode = "timeout"
    with pytest.raises(TimeoutError):
        store.get(keys[0])
    assert store.report_status()["failed"]
    assert any(manager.reads.values())
    manager.mode = "ok"
    store.close()
    assert not any(manager.reads.values())


@pytest.mark.parametrize("mode", ["wait_error", "query_error"])
def test_prefetch_exceptions_retain_handles_for_shutdown(setup, mode):
    identity, keys, manager, store = setup
    assert store.put(keys[0], raw_blob(identity))
    manager.mode = mode
    with pytest.raises(RuntimeError, match="injected"):
        store.get(keys[0])
    assert store.report_status()["failed"]
    assert store.report_status()["pending_read_cleanup"] == 1
    manager.mode = "query_error"
    store.close()
    assert store.report_status()["pending_read_cleanup"] == 1
    assert any(manager.reads.values())
    manager.mode = "ok"
    store.close()
    assert not any(manager.reads.values())
    assert store.report_status()["pending_read_cleanup"] == 0


def test_module_rpc_roundtrip_and_contiguous_lookup(setup):
    identity, keys, _, store = setup
    module = FragPackServerModule(object(), store, identity)
    response = module.handle(
        encode_message({"op": "put", "key": keys[0].to_dict()}, raw_blob(identity))
    )
    header, payload = decode_message(response, identity.max_blob_bytes)
    assert header["stored"] is True and not payload
    response = module.handle(encode_message({"op": "get", "key": keys[0].to_dict()}))
    header, payload = decode_message(response, identity.max_blob_bytes)
    assert header["found"] is True and payload == raw_blob(identity)
    response = module.handle(
        encode_message({"op": "lookup", "keys": [key.to_dict() for key in keys]})
    )
    assert decode_message(response, identity.max_blob_bytes)[0]["prefix_blocks"] == 1
    with pytest.raises(ValueError):
        module.handle(encode_message({"op": "unknown"}))
    with pytest.raises(ValueError):
        module.handle(encode_message({"op": "get", "key": keys[0].to_dict()}, b"x"))


def test_lmcache_version_is_pinned(monkeypatch):
    monkeypatch.setattr(
        "fragpack.integrations.lmcache.client.metadata.version", lambda _: "0.5.5"
    )
    with pytest.raises(RuntimeError, match="0.5.6"):
        require_lmcache_056()


@pytest.fixture
def rpc_client_factory(monkeypatch):
    monkeypatch.setattr(
        "fragpack.integrations.lmcache.client.require_lmcache_056", lambda: None
    )
    monkeypatch.setattr(
        "fragpack.integrations.lmcache.client._load_request_type",
        lambda: SimpleNamespace,
    )
    identity = CacheIdentity("rpc-test", 123, 1, 1)
    key = identity.block_keys(list(range(128)))[0]

    def create(transport):
        return RemoteBlobClient(transport, identity, timeout_s=0.1), identity, key

    return create


@pytest.mark.parametrize("fail_during_send", [False, True])
def test_uncertain_rpc_failure_disables_retries(rpc_client_factory, fail_during_send):
    class Transport:
        def __init__(self):
            self.requests = []

        def server_module_call(self, request):
            self.requests.append(request)
            if fail_during_send:
                raise RuntimeError("send outcome uncertain")
            return self

        def result(self, timeout):
            raise TimeoutError("wait timed out but request remains pending")

    transport = Transport()
    client, identity, key = rpc_client_factory(transport)
    with pytest.raises((RuntimeError, TimeoutError), match="uncertain|timed out"):
        client.put(key, raw_blob(identity))
    assert client.failed
    for operation in (
        lambda: client.put(key, raw_blob(identity)),
        lambda: client.get(key),
        lambda: client.lookup_prefix([key]),
        lambda: client.lookup_prefix([]),
    ):
        with pytest.raises(RuntimeError, match="close its messaging client"):
            operation()
    assert len(transport.requests) == 1


def test_completed_rpc_rejection_does_not_disable_client(rpc_client_factory):
    class Transport:
        def __init__(self):
            self.calls = 0

        def server_module_call(self, request):
            self.calls += 1
            return self

        def result(self, timeout):
            return SimpleNamespace(
                success=self.calls > 1,
                error="application refused object",
                payload=encode_message({"found": False}),
            )

    transport = Transport()
    client, _, key = rpc_client_factory(transport)
    with pytest.raises(RuntimeError, match="server rejected"):
        client.get(key)
    assert not client.failed
    assert client.get(key) is None
    assert transport.calls == 2


def test_rpc_concurrency_never_accumulates_pending_requests(rpc_client_factory):
    class Transport:
        def __init__(self):
            self.lock = threading.Lock()
            self.entered = threading.Event()
            self.overlap = threading.Event()
            self.release = threading.Event()
            self.pending = 0
            self.max_pending = 0
            self.calls = 0

        def server_module_call(self, request):
            with self.lock:
                self.calls += 1
                self.pending += 1
                self.max_pending = max(self.max_pending, self.pending)
                if self.pending > 1:
                    self.overlap.set()
            return self

        def result(self, timeout):
            self.entered.set()
            assert self.release.wait(3)
            with self.lock:
                self.pending -= 1
            return SimpleNamespace(
                success=True, error="", payload=encode_message({"found": False})
            )

    transport = Transport()
    client, _, key = rpc_client_factory(transport)
    barrier = threading.Barrier(4)

    def get():
        barrier.wait(timeout=3)
        return client.get(key)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(get) for _ in range(4)]
        try:
            assert transport.entered.wait(3)
            assert not client.failed
            assert not transport.overlap.wait(0.05)
        finally:
            transport.release.set()
        assert [future.result(timeout=3) for future in futures] == [None] * 4
    assert transport.calls == 4
    assert transport.max_pending == 1
    assert not client.failed

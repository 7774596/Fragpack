# SPDX-License-Identifier: Apache-2.0
"""Native transport ownership and dedup contracts without a CUDA context."""

from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fragpack.compression import attn_page as ap
from fragpack.integrations.lmcache import native_transfer as nt
from fragpack.integrations.lmcache.identity import CacheIdentity
from fragpack.integrations.lmcache.protocol import NATIVE_RAW_MAGIC, validate_blob


def _fixture_blob(raw, identity):
    # CPU ownership tests use constant-valued records: token and kernel order
    # coincide. Explicitly select the fixture's wire magic after raw framing.
    blob = bytearray(
        nt.pack_native_raw(
            raw,
            replace(identity, representation="native-raw"),
        )
    )
    if identity.representation == "fpkb":
        struct.pack_into("<I", blob, 0, ap.BLOB_MAGIC)
    return bytes(blob)


def test_record_counters_distinguish_raw_control_from_compressed_fallback():
    identity = CacheIdentity("record-stats", 17, 1, 1)
    raw = _fixture_blob(bytes(2 * ap.RAW_RECORD_BYTES), identity)
    control = nt._record_counters(raw, identity, "raw")  # noqa: SLF001
    assert control == {
        "raw_records": 2,
        "k_record_bytes": ap.RAW_RECORD_BYTES,
        "v_record_bytes": ap.RAW_RECORD_BYTES,
    }
    head = ap.blob_head_bytes(1, 1)
    coded_bytes = ap.DENSE_BYTES + ap.HEADER_BYTES
    declared = head + ap.RAW_RECORD_BYTES + coded_bytes
    extent = -(-declared // ap.BLOB_ALIGN) * ap.BLOB_ALIGN
    mixed = bytearray(raw[:extent])
    struct.pack_into("<q", mixed, 16, declared)
    struct.pack_into("<i", mixed, ap.BLOB_FIXED_BYTES + 4, ap.HEADER_BYTES)
    validate_blob(bytes(mixed), identity)
    counters = nt._record_counters(bytes(mixed), identity, "compressed")  # noqa: SLF001
    assert counters == {
        "encoded_records": 2,
        "raw_fallback_records": 1,
        "k_record_bytes": ap.RAW_RECORD_BYTES,
        "v_record_bytes": coded_bytes,
    }


class _Event:
    def synchronize(self):
        pass


class _Device:
    def __init__(self, pages, table, identity, max_blocks, mode, producer):
        self.identity = identity
        self.stream = object()
        self.metrics = Counter()
        self.encoded, self.restored = [], []
        self.fail_drain = False
        self.fail_encode = False

    def wait(self, producer):
        pass

    def drain(self, producer):
        if self.fail_drain:
            raise RuntimeError("device completion unknown")

    def encode(self, page_map, source_finished):
        self.encoded.append(page_map.clone())
        if self.fail_encode:
            raise RuntimeError("encoding failed")
        source_finished()
        return tuple(
            _fixture_blob(bytes([int(row[0]) % 256]) * 65536, self.identity)
            for row in page_map
        )

    def restore(self, page_map, blobs):
        self.restored.append((page_map.clone(), tuple(blobs)))
        return (True,) * len(blobs)

    def memory_usage(self):
        return {"gpu_encode": 128, "gpu_decode": 128, "pinned_transfer": 256}


class _Client:
    def __init__(self):
        self.blobs = {}
        self.probes, self.puts, self.gets = [], [], []
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.error = False
        self.probe_error = False
        self.refuse = False

    def probe_many(self, keys):
        self.probes.append(tuple(keys))
        if self.probe_error:
            raise TimeoutError("probe uncertain")
        return tuple(key in self.blobs for key in keys)

    def put(self, key, blob):
        self.entered.set()
        assert self.gate.wait(5), "test publication was not released"
        if self.error:
            raise RuntimeError("publication uncertain")
        self.puts.append((key, blob))
        if self.refuse:
            return False
        self.blobs[key] = blob
        return True

    def get(self, key):
        self.gets.append(key)
        return self.blobs.get(key)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(nt, "_NativeCudaWorker", _Device)
    monkeypatch.setattr(nt, "_producer_event", lambda device: _Event())
    monkeypatch.setattr(
        nt,
        "_capture_map",
        lambda page_map, rows: (page_map[list(rows)].clone(), _Event()),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: nullcontext())
    identity = CacheIdentity("native-unit", 17, 1, 1)
    pages = SimpleNamespace(
        device=torch.device("cpu"), num_layers=1, kv_heads=1, page_tokens=128
    )
    table = np.tile(np.arange(ap.TABLE_ENTRIES, dtype=np.uint8), (1, 2, 1, 1))
    workers = []

    def create(*, slots=2, precheck=True, mode="compressed"):
        client = _Client()
        transfer = nt.NativeBlockTransfer(
            pages,
            table,
            client,
            identity,
            max_blocks=3,
            num_slots=slots,
            precheck=precheck,
            mode=mode,
        )
        workers.append((transfer, client))
        return transfer, client, transfer._device  # noqa: SLF001

    yield identity, identity.block_keys(list(range(384))), create
    for transfer, client in workers:
        client.gate.set()
        transfer.close(5)


def _map(*ids):
    return torch.tensor(ids, dtype=torch.int32).reshape(-1, 1)


def test_raw_envelope_roundtrip_and_geometry_validation():
    identity = CacheIdentity("raw", -123, 3, 2, representation="native-raw")
    raw = bytes(range(256)) * (3 * 2 * 2 * ap.RAW_RECORD_BYTES // 256)
    blob = nt.pack_native_raw(raw, identity)
    validate_blob(blob, identity)
    assert len(blob) == identity.max_blob_bytes
    assert nt.unpack_native_raw(blob, identity) == raw
    with pytest.raises(ValueError, match="length"):
        nt.pack_native_raw(raw[:-1], identity)
    with pytest.raises(ValueError, match="identity mismatch"):
        nt.unpack_native_raw(blob, replace(identity, table_digest=-124))


def test_raw_restore_rejects_compressed_record_even_with_valid_envelope():
    identity = CacheIdentity("raw", 4, 1, 1, representation="native-raw")
    head = ap.blob_head_bytes(1, 1)
    record_length = ap.DENSE_BYTES + ap.HEADER_BYTES
    logical = head + 2 * record_length
    blob = bytearray((logical + ap.BLOB_ALIGN - 1) // ap.BLOB_ALIGN * ap.BLOB_ALIGN)
    struct.pack_into("<4Iqq", blob, 0, NATIVE_RAW_MAGIC, 1, 1, 1, logical, 4)
    struct.pack_into("<2i", blob, 64, ap.HEADER_BYTES, ap.HEADER_BYTES)
    with pytest.raises(ValueError, match="all-raw"):
        nt.unpack_native_raw(bytes(blob), identity)


@pytest.mark.parametrize(
    "mode,representation",
    [("raw", "fpkb"), ("compressed", "native-raw")],
)
def test_mode_identity_mismatch_rejected_before_gpu_initialization(
    mode, representation
):
    identity = CacheIdentity("mode", 3, 1, 1, representation=representation)
    table = np.tile(np.arange(ap.TABLE_ENTRIES, dtype=np.uint8), (1, 2, 1, 1))
    with pytest.raises(ValueError, match="mode and identity"):
        nt.NativeBlockTransfer(SimpleNamespace(), table, object(), identity, mode=mode)


def test_wrong_wire_magic_rejected_before_restore_writes(setup):
    identity, keys, create = setup
    transfer, client, device = create()
    raw_identity = replace(identity, representation="native-raw")
    client.blobs[keys[0]] = nt.pack_native_raw(bytes(65536), raw_identity)
    ticket = transfer.submit_restore(keys[:1], _map(8))
    with pytest.raises(ValueError, match="identity mismatch"):
        ticket.completion.result(5)
    assert not device.restored


def test_source_safety_precedes_publication_and_queue_reuse(setup):
    _, keys, create = setup
    transfer, client, device = create(slots=1)
    client.gate.clear()
    ticket = transfer.submit_store(keys[:1], _map(3))
    assert client.entered.wait(5)
    assert ticket.source_ready.result(5) is None
    assert not ticket.completion.done()
    assert not ticket.source_ready.cancel() and not ticket.completion.cancel()
    assert transfer.submit_store(keys[1:2], _map(4)) is None
    client.gate.set()
    assert ticket.completion.result(5) == (True,)
    assert len(device.encoded) == 1


def test_precheck_skips_both_sides_of_a_missing_key(setup):
    identity, keys, create = setup
    transfer, client, device = create()
    client.blobs[keys[0]] = client.blobs[keys[2]] = _fixture_blob(
        bytes(65536), identity
    )
    ticket = transfer.submit_store(keys, _map(11, 12, 13))
    assert ticket.completion.result(5) == (True, True, True)
    assert device.encoded[0].tolist() == [[12]]
    assert [key for key, _ in client.puts] == [keys[1]]
    assert transfer.stats()["skipped_blocks"] == 2
    repeat = transfer.submit_store(keys, _map(31, 32, 33))
    assert repeat.completion.result(5) == (True, True, True)
    assert len(device.encoded) == 1


def test_inflight_followers_do_not_read_or_pin_their_source(setup):
    _, keys, create = setup
    transfer, client, device = create(slots=1)
    client.gate.clear()
    owner = transfer.submit_store(keys[:1], _map(5))
    assert client.entered.wait(5)
    follower = transfer.submit_store(keys[:1], _map(91))
    assert follower.source_ready.result(1) is None
    assert not follower.completion.done()
    client.gate.set()
    assert owner.completion.result(5) == follower.completion.result(5) == (True,)
    assert [item.tolist() for item in device.encoded] == [[[5]]]
    assert transfer.stats()["coalesced_blocks"] == 1


def test_mixed_inflight_and_new_keys_keep_input_result_order(setup):
    _, keys, create = setup
    transfer, client, device = create(slots=2)
    client.gate.clear()
    first = transfer.submit_store(keys[:1], _map(5))
    assert client.entered.wait(5)
    second = transfer.submit_store(keys[:2], _map(91, 6))
    client.gate.set()
    assert first.completion.result(5) == (True,)
    assert second.completion.result(5) == (True, True)
    assert [item.tolist() for item in device.encoded] == [[[5]], [[6]]]


def test_precheck_ablation_keeps_repeated_encoding(setup):
    _, keys, create = setup
    transfer, client, device = create(precheck=False)
    for _ in range(2):
        assert transfer.submit_store(keys[:1], _map(5)).completion.result(5) == (True,)
    assert len(device.encoded) == len(client.puts) == 2
    assert not client.probes


def test_external_eviction_allows_new_encoding(setup):
    _, keys, create = setup
    transfer, client, device = create()
    assert transfer.submit_store(keys[:1], _map(5)).completion.result(5) == (True,)
    del client.blobs[keys[0]]
    assert transfer.submit_store(keys[:1], _map(5)).completion.result(5) == (True,)
    assert len(device.encoded) == 2


def test_restore_stops_at_first_missing_object_and_retains_external_copy(setup):
    identity, keys, create = setup
    transfer, client, device = create()
    client.blobs[keys[0]] = client.blobs[keys[2]] = _fixture_blob(
        bytes(65536), identity
    )
    result = transfer.submit_restore(keys, _map(4, 5, 6)).completion.result(5)
    assert result == (True, False, False)
    assert client.gets == keys[:2]
    assert len(device.restored[0][1]) == 1
    assert keys[0] in client.blobs and keys[2] in client.blobs


def test_corrupt_restore_never_reaches_gpu(setup):
    _, keys, create = setup
    transfer, client, device = create()
    client.blobs[keys[0]] = b"bad"
    with pytest.raises(ValueError):
        transfer.submit_restore(keys[:1], _map(1)).completion.result(5)
    assert not device.restored and not transfer.failed


def test_probe_failure_is_not_converted_to_a_miss(setup):
    _, keys, create = setup
    transfer, client, device = create()
    client.probe_error = True
    ticket = transfer.submit_store(keys[:1], _map(7))
    with pytest.raises(TimeoutError, match="probe uncertain"):
        ticket.completion.result(5)
    assert ticket.source_ready.result(5) is None
    assert not device.encoded and not client.puts


def test_uncertain_publication_does_not_retry(setup):
    _, keys, create = setup
    transfer, client, device = create()
    client.error = True
    ticket = transfer.submit_store(keys[:1], _map(7))
    with pytest.raises(RuntimeError, match="publication uncertain"):
        ticket.completion.result(5)
    assert ticket.source_ready.result(5) is None
    assert len(device.encoded) == 1
    assert not transfer.failed


def test_refused_publication_stops_following_keys(setup):
    _, keys, create = setup
    transfer, client, _ = create()
    client.refuse = True
    assert (
        transfer.submit_store(keys, _map(1, 2, 3)).completion.result(5) == (False,) * 3
    )
    assert [key for key, _ in client.puts] == keys[:1]


def test_unknown_device_completion_quarantines_all_owners(setup):
    _, keys, create = setup
    transfer, _, device = create()
    device.fail_encode = device.fail_drain = True
    ticket = transfer.submit_store(keys[:1], _map(7))
    with pytest.raises(RuntimeError, match="process recovery"):
        ticket.completion.result(5)
    with pytest.raises(RuntimeError, match="process recovery"):
        ticket.source_ready.result(5)
    assert transfer.failed and transfer in nt._quarantined  # noqa: SLF001
    assert transfer.submit_restore(keys[:1], _map(9)) is None


def test_close_timeout_retains_live_work_and_stops_admission(setup):
    _, keys, create = setup
    transfer, client, _ = create()
    client.gate.clear()
    ticket = transfer.submit_store(keys[:1], _map(7))
    assert client.entered.wait(5)
    with pytest.raises(TimeoutError, match="owners retained"):
        transfer.close(0.001)
    assert transfer.submit_store(keys[1:2], _map(9)) is None
    client.gate.set()
    assert ticket.completion.result(5) == (True,)
    transfer.close(5)


def test_namespace_and_page_map_validation_precedes_admission(setup):
    _, keys, create = setup
    transfer, _, device = create()
    with pytest.raises(ValueError, match="distinct keys"):
        transfer.submit_store([keys[0], keys[0]], _map(1, 2))
    with pytest.raises(ValueError, match="page_map"):
        transfer.submit_restore(keys[:1], torch.tensor([[1]], dtype=torch.int64))
    alien = CacheIdentity("other", 17, 1, 1).block_keys(list(range(128)))
    with pytest.raises(ValueError, match="another cache identity"):
        transfer.submit_store(alien, _map(1))
    assert not device.encoded and not device.restored


@pytest.mark.parametrize("drain_fails", [False, True])
def test_partial_initialization_drains_or_retains_worker(monkeypatch, drain_fails):
    events = []

    class Stream:
        def wait_event(self, producer):
            events.append("wait")

        def synchronize(self):
            events.append("drain")
            if drain_fails:
                raise RuntimeError("initialization still running")

    def initialize(worker, table, max_blocks):
        events.append("allocate")
        worker.pools["partial"] = object()
        raise ValueError("initialization failed")

    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: Stream())
    monkeypatch.setattr(nt._NativeCudaWorker, "_initialize", initialize)  # noqa: SLF001
    pages = SimpleNamespace(device=torch.device("cuda:0"))
    identity = CacheIdentity("native-unit", 17, 1, 1)
    expected = "process recovery" if drain_fails else "initialization failed"
    with pytest.raises((RuntimeError, ValueError), match=expected):
        nt._NativeCudaWorker(  # noqa: SLF001
            pages, np.empty((0,), dtype=np.uint8), identity, 1, "raw", _Event()
        )
    assert events == ["wait", "allocate", "drain"]
    if drain_fails:
        assert nt._quarantined_workers[-1].pages is pages  # noqa: SLF001


def test_jobs_queued_before_fatal_failure_never_report_source_safe(setup):
    _, keys, create = setup
    transfer, _, device = create(slots=2)
    entered, release = threading.Event(), threading.Event()

    def blocked_encode(page_map, source_finished):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("encoding could still be running")

    device.encode = blocked_encode
    device.fail_drain = True
    first = transfer.submit_store(keys[:1], _map(1))
    assert entered.wait(5)
    second = transfer.submit_store(keys[1:2], _map(2))
    release.set()
    for ticket in (first, second):
        with pytest.raises(RuntimeError, match="process recovery"):
            ticket.completion.result(5)
        with pytest.raises(RuntimeError, match="process recovery"):
            ticket.source_ready.result(5)

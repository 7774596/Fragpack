# SPDX-License-Identifier: Apache-2.0
"""Native GPU-to-owned-host-to-GPU transport, independent of LMCache install."""

import hashlib
import threading

import pytest
from test_native_codec import _case, _check_owners
import torch

from fragpack.compression.attn_tables import device_tables
from fragpack.integrations.lmcache.identity import CacheIdentity
from fragpack.integrations.lmcache.native_transfer import NativeBlockTransfer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


class _Client:
    def __init__(self):
        self.blobs = {}
        self.puts = 0
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()

    def probe_many(self, keys):
        return tuple(key in self.blobs for key in keys)

    def put(self, key, blob):
        self.entered.set()
        assert self.gate.wait(30), "test publication was not released"
        assert key not in self.blobs or self.blobs[key] == blob
        self.blobs[key] = blob
        self.puts += 1
        return True

    def get(self, key):
        return self.blobs.get(key)


def _identity(case, mode):
    rank = device_tables(case.table, torch.device("cpu")).rank_of
    digest = int.from_bytes(
        hashlib.blake2b(rank.numpy().tobytes(), digest_size=8).digest(),
        "little",
        signed=True,
    )
    return CacheIdentity(
        f"native-transfer-gpu-{mode}",
        digest,
        case.source.num_layers,
        case.source.kv_heads,
        representation="native-raw" if mode == "raw" else "fpkb",
    )


@pytest.mark.parametrize("mode", ["compressed", "raw"])
@pytest.mark.parametrize("page_tokens", [16, 128])
def test_native_transfer_roundtrip_skip_and_external_eviction(mode, page_tokens):
    case = _case(tokens=page_tokens, blocks=2, layers=2, heads=1)
    identity = _identity(case, mode)
    keys = identity.block_keys(list(range(256)))
    client = _Client()
    source = NativeBlockTransfer(
        case.source,
        case.table,
        client,
        identity,
        max_blocks=2,
        num_slots=2,
        mode=mode,
    )
    destination = NativeBlockTransfer(
        case.target,
        case.table,
        client,
        identity,
        max_blocks=2,
        num_slots=2,
        mode=mode,
    )
    try:
        before = [owner.cpu().clone() for owner in case.source_owners]
        store = source.submit_store(keys, case.source_map)
        assert store.source_ready.result(60) is None
        assert store.completion.result(60) == (True, True)
        assert destination.submit_restore(keys, case.target_map).completion.result(
            60
        ) == (
            True,
            True,
        )
        _check_owners(case.target_owners, case.expected)
        _check_owners(case.source_owners, before)
        first_bytes = source.stats()["d2h_payload_bytes"]
        assert source.submit_store(keys, case.source_map).completion.result(60) == (
            True,
            True,
        )
        assert client.puts == 2 and source.stats()["d2h_payload_bytes"] == first_bytes
        assert len(client.blobs) == 2
        del client.blobs[keys[0]]
        assert source.submit_store(keys, case.source_map).completion.result(60) == (
            True,
            True,
        )
        assert client.puts == 3
        assert source.stats()["d2h_payload_bytes"] > first_bytes
        assert destination.stats()["h2d_payload_bytes"] == first_bytes
    finally:
        client.gate.set()
        source.close(60)
        destination.close(60)


def test_source_pages_can_change_after_read_before_remote_publication():
    case = _case(tokens=32, blocks=1, layers=2, heads=1)
    identity = _identity(case, "compressed")
    keys = identity.block_keys(list(range(128)))
    client = _Client()
    source = NativeBlockTransfer(
        case.source, case.table, client, identity, max_blocks=1
    )
    destination = NativeBlockTransfer(
        case.target, case.table, client, identity, max_blocks=1
    )
    try:
        client.gate.clear()
        ticket = source.submit_store(keys, case.source_map)
        assert ticket.source_ready.result(60) is None
        assert client.entered.wait(30) and not ticket.completion.done()
        for owner in case.source_owners:
            owner.zero_()
        client.gate.set()
        assert ticket.completion.result(60) == (True,)
        assert destination.submit_restore(keys, case.target_map).completion.result(
            60
        ) == (True,)
        _check_owners(case.target_owners, case.expected)
    finally:
        client.gate.set()
        source.close(60)
        destination.close(60)


def test_aliasing_restore_targets_are_rejected_without_writes():
    case = _case(tokens=16, blocks=2, layers=2, heads=1)
    identity = _identity(case, "compressed")
    keys = identity.block_keys(list(range(256)))
    client = _Client()
    source = NativeBlockTransfer(
        case.source, case.table, client, identity, max_blocks=2
    )
    destination = NativeBlockTransfer(
        case.target, case.table, client, identity, max_blocks=2
    )
    try:
        assert source.submit_store(keys, case.source_map).completion.result(60) == (
            True,
            True,
        )
        before = [owner.cpu().clone() for owner in case.target_owners]
        mapping = case.target_map.clone()
        mapping[1].copy_(mapping[0])
        assert destination.submit_restore(keys, mapping).completion.result(60) == (
            False,
            False,
        )
        _check_owners(case.target_owners, before)
    finally:
        source.close(60)
        destination.close(60)

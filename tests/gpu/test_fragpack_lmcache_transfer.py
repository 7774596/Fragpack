# SPDX-License-Identifier: Apache-2.0
"""Real CUDA export/import through bounded transport and an in-memory client."""

from __future__ import annotations

from collections.abc import Callable
import threading

import numpy as np
import pytest
import torch

from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import device_tables
from fragpack.integrations.lmcache.client import BlobClient
from fragpack.integrations.lmcache.identity import CacheIdentity
from fragpack.integrations.lmcache.protocol import validate_blob
from fragpack.integrations.lmcache.transfer import BlockTransfer
from fragpack.ops.hbm_kv import HbmKvStore, page_bytes

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


class _Client:
    def __init__(self):
        self.blobs = {}
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def put(self, key, blob):
        self.entered.set()
        assert self.gate.wait(20)
        self.blobs[key] = blob
        return True

    def get(self, key):
        return self.blobs.get(key)

    def lookup_prefix(self, keys):
        count = 0
        for key in keys:
            if key not in self.blobs:
                break
            count += 1
        return count


class _Harness:
    def __init__(self, *, overflow_records=16):
        self.table = np.tile(np.arange(ap.TABLE_ENTRIES, dtype=np.uint8), (2, 2, 1, 1))
        self.caches = [
            torch.empty((4, page_bytes(1)), dtype=torch.uint8, device="cuda")
            for _ in range(2)
        ]
        self.store = HbmKvStore(
            self.caches,
            1,
            device_tables(self.table, torch.device("cuda")),
            max_tokens=256,
            staging_slots=2,
            overflow_records=overflow_records,
        )

    def fill(self):
        """Block 0 compresses; block 1 requires exact raw fallback records."""
        rng = np.random.default_rng(448)
        self.bits = rng.integers(0, 65536, (2, 2, 256, 1, 128), dtype=np.uint16)
        self.bits[:, :, :128] &= 0x00FF
        slots = torch.arange(256, dtype=torch.int64, device="cuda")
        self.store.prepare(slots, 256, np.array([0, 1], dtype=np.int32))
        for layer in range(2):
            k, v = [
                torch.from_numpy(self.bits[layer, kv].view(np.int16).copy())
                .cuda()
                .view(torch.bfloat16)
                for kv in range(2)
            ]
            self.store.write(layer, k, v, slots)
            self.store.encode(layer, 2)
        self.store.commit()


def _verify_restored(source, target, destinations):
    for layer in range(2):
        k = torch.empty((2, 128, 1, 128), dtype=torch.bfloat16, device="cuda")
        v = torch.empty_like(k)
        target.store.gather(layer, destinations, k, v)
        for kv, actual in enumerate((k, v)):
            observed = actual.view(torch.int16).cpu().numpy().view(np.uint16)
            np.testing.assert_array_equal(
                observed.reshape(256, 1, 128), source.bits[layer, kv]
            )


def make_remote_roundtrip() -> tuple[CacheIdentity, Callable[[BlobClient], dict]]:
    """Allocate a small CUDA case; return its identity and live-client check.

    The integration smoke launches its stock CPU server using this identity,
    then calls the returned function with the real RemoteBlobClient. Closure
    ownership keeps source and destination arenas alive throughout the check.
    """
    source, target = _Harness(), _Harness()
    source.fill()
    identity = CacheIdentity("gpu-rpc-smoke", source.store.table_digest, 2, 1)
    keys = identity.block_keys(list(range(256)))

    def run(client: BlobClient) -> dict:
        writer = BlockTransfer(source.store, client, identity, num_slots=2)
        reader = BlockTransfer(target.store, client, identity, num_slots=2)
        try:
            ticket = writer.submit_store(
                keys, torch.tensor([0, 1], dtype=torch.int32, device="cuda")
            )
            assert ticket is not None and ticket.completion.result(30) == (True, True)
            assert client.lookup_prefix(keys) == 2
            blobs = [client.get(key) for key in keys]
            assert all(blob is not None for blob in blobs)
            lengths = [len(blob) for blob in blobs]
            assert lengths[0] < lengths[1] == identity.max_blob_bytes
            destinations = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
            ticket = reader.submit_restore(keys, destinations)
            assert ticket is not None and ticket.completion.result(30) == (True, True)
            _verify_restored(source, target, destinations)
            return {"prefix_blocks": 2, "blob_bytes": lengths, "bit_exact": True}
        finally:
            writer.close(timeout=30)
            reader.close(timeout=30)

    return identity, run


def test_transfer_roundtrip_mixed_records_and_bounded_admission():
    source = _Harness()
    target = _Harness()
    source.fill()
    identity = CacheIdentity("gpu-test-model", source.store.table_digest, 2, 1)
    keys = identity.block_keys(list(range(256)))
    client = _Client()
    writer = BlockTransfer(source.store, client, identity, num_slots=2)
    reader = BlockTransfer(target.store, client, identity, num_slots=2)
    try:
        client.gate.clear()
        write = writer.submit_store(
            keys, torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        )
        assert write is not None
        write.source_ready.synchronize()
        assert client.entered.wait(20)
        assert not write.completion.done()
        assert (
            writer.submit_store(
                keys[:1], torch.tensor([0], dtype=torch.int32, device="cuda")
            )
            is None
        )
        client.gate.set()
        assert write.completion.result(20) == (True, True)
        for blob in client.blobs.values():
            validate_blob(blob, identity)
        assert len(client.blobs[keys[0]]) < len(client.blobs[keys[1]])
        assert len(client.blobs[keys[1]]) == identity.max_blob_bytes

        destinations = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
        read = reader.submit_restore(keys, destinations)
        assert read is not None and read.completion.result(20) == (True, True)
        _verify_restored(source, target, destinations)
        assert writer.memory_usage() == {
            "gpu_transfer": 2 * identity.max_blob_bytes,
            "pinned_transfer": 2 * identity.max_blob_bytes,
        }
    finally:
        client.gate.set()
        writer.close()
        reader.close()


def test_restore_reports_native_import_overflow_as_failure():
    source = _Harness()
    target = _Harness(overflow_records=1)
    source.fill()
    identity = CacheIdentity("gpu-test-model", source.store.table_digest, 2, 1)
    key = identity.block_keys(list(range(256)))[1]
    client = _Client()
    writer = BlockTransfer(source.store, client, identity, num_slots=1)
    reader = BlockTransfer(target.store, client, identity, num_slots=1)
    try:
        write = writer.submit_store(
            [key], torch.tensor([1], dtype=torch.int32, device="cuda")
        )
        assert write is not None and write.completion.result(20) == (True,)
        read = reader.submit_restore(
            [key], torch.tensor([2], dtype=torch.int32, device="cuda")
        )
        assert read is not None and read.completion.result(20) == (False,)
    finally:
        writer.close()
        reader.close()

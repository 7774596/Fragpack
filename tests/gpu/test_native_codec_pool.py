# SPDX-License-Identifier: Apache-2.0
"""Actual two-slot reuse across streams, with no additional tensor storage."""

import numpy as np
import pytest
from test_native_codec import POISON, _case, _check_owners
import torch

from fragpack.ops.native_codec import NativeCodec
from fragpack.ops.native_codec_pool import NativeCodecPool, native_codec_pool_bytes

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.mark.parametrize("workspace", ["encode", "decode", "raw"])
def test_two_slot_pool_orders_consumers_before_cross_stream_overwrite(
    workspace: str,
) -> None:
    """Later leases cannot overwrite a delayed earlier transfer/restore consumer."""
    case = _case(tokens=16, blocks=2, layers=2, heads=1)
    encoder = NativeCodec(case.table, max_blocks=2, device="cuda", workspace="encode")
    assert not bool(encoder.encode(case.source, case.source_map).any())
    encoded_offsets = encoder.offsets.cpu().pin_memory()
    encoded_bytes = int(encoded_offsets[-1])
    encoded_payload = encoder.buffer[:encoded_bytes].cpu().pin_memory()
    raw_payload = torch.from_numpy(
        case.bits.copy().view(np.uint8).reshape(-1)
    ).pin_memory()
    reference_payload = raw_payload if workspace == "raw" else encoded_payload
    payload_bytes = reference_payload.numel()
    pool = NativeCodecPool(
        case.table, max_blocks=2, slots=2, workspace=workspace, device="cuda"
    )
    assert pool.memory_usage() == native_codec_pool_bytes(
        2, 1, 2, 2, workspace=workspace
    )

    # Initialize both slots, compile the actual operation and establish active
    # batch metadata before measuring allocation stability or reuse ordering.
    for _ in range(2):
        with pool.acquire() as lease:
            if workspace == "encode":
                lease.codec.encode(case.source, case.source_map)
                assert lease.payload is lease.codec.buffer
                assert lease.offsets.data_ptr() == lease.codec.offsets.data_ptr()
            else:
                lease.payload[:payload_bytes].copy_(
                    reference_payload, non_blocking=True
                )
                if workspace == "decode":
                    lease.offsets.copy_(encoded_offsets, non_blocking=True)
                    lease.codec.decode(
                        case.target,
                        case.target_map,
                        src=lease.payload[:payload_bytes],
                        offsets=lease.offsets,
                    )
                else:
                    lease.codec.raw_scatter(case.target, case.target_map, lease.payload)
            assert not bool(lease.codec.status.any())
    for owner in case.target_owners:
        owner.fill_(POISON)
    snapshots = (
        [torch.empty(payload_bytes, dtype=torch.uint8, device="cuda") for _ in range(2)]
        if workspace == "encode"
        else [
            [torch.empty_like(owner) for owner in case.target_owners] for _ in range(2)
        ]
    )
    status = torch.empty((2, 2), dtype=torch.int32, device="cuda")
    offset_snapshots = torch.empty((2, 3), dtype=torch.int64, device="cuda")
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    produced = torch.cuda.Event()
    produced.record()
    torch.cuda.current_stream().synchronize()
    allocated = torch.cuda.memory_allocated()

    with torch.cuda.stream(stream_a):
        leases = [pool.acquire(), pool.acquire()]
        assert [lease.slot_index for lease in leases] == [0, 1]
        if workspace == "encode":
            for lease in leases:
                lease.codec.encode(case.source, case.source_map)
        produced.record(stream_a)
        # Give stream B a chance to corrupt these buffers if pool dependencies
        # are absent. The delay belongs only to this ordering correctness test.
        torch.cuda._sleep(5_000_000)  # noqa: SLF001
        for index, lease in enumerate(leases):
            if workspace == "encode":
                snapshots[index].copy_(lease.payload[:payload_bytes])
                offset_snapshots[index].copy_(lease.offsets)
            else:
                if workspace == "decode":
                    lease.codec.decode(
                        case.target,
                        case.target_map,
                        src=lease.payload[:payload_bytes],
                        offsets=lease.offsets,
                    )
                else:
                    lease.codec.raw_scatter(case.target, case.target_map, lease.payload)
                for snapshot, owner in zip(
                    snapshots[index], case.target_owners, strict=True
                ):
                    snapshot.copy_(owner)
            status[index].copy_(lease.codec.status)
            lease.release()

    with torch.cuda.stream(stream_b):
        stream_b.wait_event(produced)
        repeated = [pool.acquire(), pool.acquire()]
        assert [lease.slot_index for lease in repeated] == [0, 1]
        for lease in repeated:
            lease.payload.zero_()
            if lease.offsets is not None:
                lease.offsets.zero_()
            lease.release()
    assert torch.cuda.memory_allocated() == allocated
    torch.cuda.current_stream().wait_stream(stream_b)
    assert not bool(status.any())
    if workspace == "encode":
        for snapshot in snapshots:
            assert torch.equal(snapshot.cpu(), encoded_payload)
        assert torch.equal(offset_snapshots.cpu(), encoded_offsets.expand(2, -1))
    else:
        for snapshot in snapshots:
            _check_owners(snapshot, case.expected)
        _check_owners(case.target_owners, case.expected)

# SPDX-License-Identifier: Apache-2.0
"""Coalesced native restore: per-head alignment and fresh graph inputs."""

import numpy as np
import pytest
from test_native_codec import POISON, _case, _check_owners, _expected
import torch

from fragpack.ops.native_codec import NativeCodec, NativeKVPages

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _mixed_alignment_pages(
    count: int, tokens: int, heads: int
) -> tuple[NativeKVPages, list[torch.Tensor]]:
    """Mix aligned heads, odd head strides, and two-byte-offset layers."""
    shape = (2, count, tokens, heads, 128)
    owners, caches = [], []
    for layer in range(3):
        head_stride = 128 if layer == 0 else 129
        token_stride = (heads * head_stride + 7) // 8 * 8
        page_stride = tokens * token_stride + 8
        kv_stride = count * page_stride
        strides = (kv_stride, page_stride, token_stride, head_stride, 1)
        offset = int(layer == 2)
        extent = 1 + sum(
            (size - 1) * stride for size, stride in zip(shape, strides, strict=True)
        )
        owner = torch.full(
            (extent + offset + 8,), POISON, dtype=torch.int16, device="cuda"
        )
        caches.append(
            owner.as_strided(shape, strides, storage_offset=offset).view(torch.bfloat16)
        )
        owners.append(owner)
    return NativeKVPages(caches), owners


@pytest.mark.parametrize("tokens", [16, 128])
def test_native_restore_graph_reads_fresh_records_with_mixed_head_alignment(
    tokens: int,
) -> None:
    """Each replay restores updated bytes, including raw records, into guarded pages."""
    case = _case(tokens=tokens, strided=False, blocks=2, layers=3, heads=3)
    target, owners = _mixed_alignment_pages(
        case.target.caches[0].shape[1], tokens, heads=3
    )
    codec = NativeCodec(case.table, max_blocks=2, device="cuda")
    assert not bool(codec.encode(case.source, case.source_map).any())
    size = int(codec.offsets[-1])
    received = codec.buffer[:size].clone()
    offsets = codec.offsets.clone()
    assert not bool(
        codec.decode(target, case.target_map, src=received, offsets=offsets).any()
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        codec.decode(target, case.target_map, src=received, offsets=offsets)

    source_map = case.source_map.cpu().numpy()
    target_map = case.target_map.cpu().numpy()
    for low_mask in (0x5A, 0xC3, 0):
        bits = case.bits ^ np.uint16(low_mask)
        populated = _expected(case.source_owners, case.source, source_map, bits)
        for owner, data in zip(case.source_owners, populated, strict=True):
            owner.copy_(data)
        assert not bool(codec.encode(case.source, case.source_map).any())
        # Low-byte changes preserve all record lengths but change decoded bits.
        assert torch.equal(codec.offsets, offsets)
        received.copy_(codec.buffer[:size])
        for owner in owners:
            owner.fill_(POISON)
        expected = _expected(owners, target, target_map, bits)
        graph.replay()
        assert not bool(codec.status.any())
        _check_owners(owners, expected)

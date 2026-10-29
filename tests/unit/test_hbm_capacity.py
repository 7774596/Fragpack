# SPDX-License-Identifier: Apache-2.0
"""CPU checks of allocation accounting, including padded engine pages."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from fragpack.compression.attn_tables import device_tables
from fragpack.ops.hbm_kv import HbmKvStore, page_bytes


def test_memory_usage_counts_page_stride_and_separates_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Account real padded pages and auxiliary storage without CUDA execution.

    Args:
        monkeypatch: Pytest fixture replacing only the pinned polling allocation.

    Returns:
        None; assertions reject payload-only arena accounting and conflation
        of the raw overflow pool with the engine's page allocation.

    Async/thread-safety:
        All tensors reside on CPU. No kernels or CUDA events are submitted.
    """
    zeros = torch.zeros

    def host_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        # Pinning requires a driver even though memory_usage only reads shapes.
        # Keep every other allocation argument and the real constructor intact.
        kwargs.pop("pin_memory", None)
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", host_zeros)
    layers, blocks, heads = 2, 3, 1
    payload = page_bytes(heads)
    strides = (payload + 16, payload + 48)
    owners = [torch.empty((blocks, stride), dtype=torch.uint8) for stride in strides]
    arenas = [owner[:, :payload] for owner in owners]
    table = np.broadcast_to(
        np.arange(22, dtype=np.uint8), (layers, 2, heads, 22)
    ).copy()
    tables = device_tables(table, torch.device("cpu"))

    def build(overflow_records: int) -> HbmKvStore:
        return HbmKvStore(
            arenas,
            heads,
            tables,
            max_tokens=128,
            staging_slots=2,
            overflow_records=overflow_records,
        )

    small, larger = build(1).memory_usage(), build(2).memory_usage()
    assert small["arena"] == sum(owner.numel() for owner in owners)
    assert small["arena"] > layers * blocks * payload
    assert small["staging"] > 0 and small["scratch"] > 0
    assert small["metadata"] > 0 and small["codebooks"] > 0
    assert larger["overflow"] == 2 * small["overflow"]
    assert {k: v for k, v in small.items() if k != "overflow"} == {
        k: v for k, v in larger.items() if k != "overflow"
    }

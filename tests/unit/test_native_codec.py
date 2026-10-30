# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation of independent native-page codec boundaries."""

import numpy as np
import pytest
import torch

from fragpack.ops.native_codec import (
    NativeCodec,
    NativeKVPages,
    validate_native_tensor,
    validate_tables,
)


@pytest.mark.parametrize("page_tokens", [16, 32, 64, 128])
def test_native_tensor_geometry_accepts_permuted_physical_axes(
    page_tokens: int,
) -> None:
    """Logical KV/page/token/head axes need not follow physical storage order."""
    owner = torch.empty((7, 2, 3, page_tokens, 128), dtype=torch.bfloat16)
    tensor = owner.permute(1, 0, 3, 2, 4)
    layout = validate_native_tensor(tensor)
    assert (layout.num_pages, layout.page_tokens, layout.kv_heads) == (
        7,
        page_tokens,
        3,
    )
    assert layout.strides == tensor.stride()
    assert layout.span_bytes == owner.numel() * owner.element_size()


def test_native_tensor_span_includes_holes_but_not_storage_offset() -> None:
    shape = (2, 5, 32, 3, 128)
    strides = (70000, 13000, 400, 130, 1)
    extent = 1 + sum(
        (size - 1) * stride for size, stride in zip(shape, strides, strict=True)
    )
    owner = torch.empty(extent + 20, dtype=torch.bfloat16)
    tensor = owner.as_strided(shape, strides, storage_offset=1)
    layout = validate_native_tensor(tensor)
    assert tensor.data_ptr() % 4 == 2
    assert layout.strides == strides
    assert layout.span_bytes == extent * 2
    assert layout.span_bytes > tensor.numel() * tensor.element_size()


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((2, 3, 32, 2, 128), torch.float16),
        ((2, 3, 32, 128), torch.bfloat16),
        ((3, 3, 32, 2, 128), torch.bfloat16),
        ((2, 0, 32, 2, 128), torch.bfloat16),
        ((2, 3, 24, 2, 128), torch.bfloat16),
        ((2, 3, 32, 0, 128), torch.bfloat16),
        ((2, 3, 32, 17, 128), torch.bfloat16),
        ((2, 3, 32, 2, 64), torch.bfloat16),
    ],
)
def test_native_tensor_rejects_unsupported_geometry(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    with pytest.raises(ValueError):
        validate_native_tensor(torch.empty(shape, dtype=dtype))


def test_native_tensor_rejects_overlapping_positive_strides() -> None:
    owner = torch.empty((2, 3, 32, 2, 128), dtype=torch.bfloat16)
    strides = list(owner.stride())
    strides[1] = strides[2]  # Adjacent physical pages alias token rows.
    with pytest.raises(ValueError, match="overlap"):
        validate_native_tensor(owner.as_strided(owner.shape, strides))


def test_native_tensor_rejects_broadcast_and_noncontiguous_head_dimensions() -> None:
    broadcast = torch.empty((1, 3, 32, 2, 128), dtype=torch.bfloat16).expand(
        2, -1, -1, -1, -1
    )
    scattered = torch.empty((2, 3, 32, 2, 256), dtype=torch.bfloat16)[..., ::2]
    for tensor in (broadcast, scattered):
        with pytest.raises(ValueError, match="strides|contiguous"):
            validate_native_tensor(tensor)


def test_native_tables_accept_distinct_per_layer_kv_head_calibration() -> None:
    table = np.empty((3, 2, 4, 22), dtype=np.uint8)
    for index in np.ndindex(table.shape[:3]):
        table[index] = np.random.default_rng(sum(index) + 17).permutation(256)[:22]
    # Table storage itself may be noncontiguous; construction normalizes it.
    validate_tables(table[:, :, ::2])
    duplicate = table.copy()
    duplicate[2, 1, 3, -1] = duplicate[2, 1, 3, 0]
    with pytest.raises(ValueError, match="distinct"):
        validate_tables(duplicate)


@pytest.mark.parametrize(
    "table",
    [
        np.zeros((1, 2, 1, 22), dtype=np.int16),
        np.zeros((0, 2, 1, 22), dtype=np.uint8),
        np.zeros((1, 1, 1, 22), dtype=np.uint8),
        np.zeros((1, 2, 0, 22), dtype=np.uint8),
        np.zeros((1, 2, 17, 22), dtype=np.uint8),
        np.zeros((1, 2, 1, 21), dtype=np.uint8),
    ],
)
def test_native_tables_reject_invalid_geometry_or_type(table: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_tables(table)


def test_constructor_rejects_invalid_device_or_capacity_without_cuda() -> None:
    table = np.arange(22, dtype=np.uint8).reshape(1, 1, 1, 22).repeat(2, axis=1)
    with pytest.raises(ValueError, match="positive"):
        NativeCodec(table, max_blocks=0, device="cuda")
    with pytest.raises(ValueError, match="CUDA"):
        NativeCodec(table, max_blocks=1, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        NativeKVPages([torch.empty((2, 3, 32, 1, 128), dtype=torch.bfloat16)])
    with pytest.raises(ValueError, match="at least one"):
        NativeKVPages([])

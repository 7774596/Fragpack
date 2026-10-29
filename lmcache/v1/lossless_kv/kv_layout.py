# SPDX-License-Identifier: Apache-2.0

"""Logical and physical views of the codec's cross-layer KV scratch cache.

The GPU codec kernels address a block-outermost ``BLNHC`` buffer whose
physical order is ``[blocks, layers, tokens, heads, 2, head_dim]`` (vLLM's
cross-layer layout). Records keep the plane-major order ``[layers, 2, tokens,
heads, head_dim]`` so each (layer, K/V) plane is one contiguous codec record;
the kernels therefore take the strided logical view ``[blocks, layers, 2,
tokens, heads, head_dim]``, and copies that address raw memory use the
contiguous physical view.

The kernels also address vLLM's layer-major cache, where each layer is its
own ``[blocks, tokens, heads, 2 * head_dim]`` tensor and the layers are
consecutive in one allocation: its physical order is ``[layers, blocks,
tokens, heads, 2, head_dim]`` and its logical view has the same shape as a
block-major one.
"""

# Future
from __future__ import annotations

# Third Party
import torch

# Physical [B, L, N, H, 2, D] -> logical [B, L, 2, N, H, D].
_PHYSICAL_TO_LOGICAL = (0, 1, 4, 2, 3, 5)
# Logical [B, L, 2, N, H, D] -> physical [B, L, N, H, 2, D].
_LOGICAL_TO_PHYSICAL = (0, 1, 3, 4, 2, 5)
# Logical [B, L, 2, N, H, D] -> layer-major physical [L, B, N, H, 2, D].
_LOGICAL_TO_LAYER_MAJOR = (1, 0, 3, 4, 2, 5)
# Layer-major physical [L, B, N, H, 2, D] -> logical [B, L, 2, N, H, D].
_PHYSICAL_TO_LOGICAL_FROM_LAYER_MAJOR = (1, 0, 4, 2, 3, 5)


def physical_kv_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """Return the contiguous ``[blocks, layers, tokens, heads, 2, head_dim]`` view.

    Args:
        kv_cache: Logical ``[blocks, layers, 2, tokens, heads, head_dim]`` view.

    Returns:
        The same memory permuted into its contiguous physical order.

    Raises:
        ValueError: If ``kv_cache`` is not a logical view of a contiguous
            block-outermost ``BLNHC`` buffer.
    """
    if kv_cache.dim() != 6 or kv_cache.shape[2] != 2:
        raise ValueError(f"unsupported cross-layer KV shape {tuple(kv_cache.shape)}")
    physical = kv_cache.permute(_LOGICAL_TO_PHYSICAL)
    if not physical.is_contiguous():
        raise ValueError("cross-layer KV cache must be a contiguous BLNHC buffer")
    return physical


def kv_is_layer_major(kv_cache: torch.Tensor) -> bool:
    """Return whether a logical KV view is over a layer-major buffer.

    Args:
        kv_cache: Logical ``[blocks, layers, 2, tokens, heads, head_dim]`` view.

    Returns:
        True for a contiguous ``[layers, blocks, tokens, heads, 2, head_dim]``
        buffer, False for a contiguous block-major ``BLNHC`` buffer.

    Raises:
        ValueError: If ``kv_cache`` is a logical view of neither.
    """
    if kv_cache.dim() != 6 or kv_cache.shape[2] != 2:
        raise ValueError(f"unsupported cross-layer KV shape {tuple(kv_cache.shape)}")
    if kv_cache.permute(_LOGICAL_TO_PHYSICAL).is_contiguous():
        return False
    if kv_cache.permute(_LOGICAL_TO_LAYER_MAJOR).is_contiguous():
        return True
    raise ValueError("KV cache must be a contiguous block- or layer-major buffer")


def kernel_kv_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """Return the contiguous physical view the codec kernels address.

    Args:
        kv_cache: Logical ``[blocks, layers, 2, tokens, heads, head_dim]`` view.

    Returns:
        ``[blocks, layers, tokens, heads, 2, head_dim]`` for a block-major
        buffer, ``[layers, blocks, tokens, heads, 2, head_dim]`` for a
        layer-major one (see :func:`kv_is_layer_major`).
    """
    if kv_is_layer_major(kv_cache):
        return kv_cache.permute(_LOGICAL_TO_LAYER_MAJOR)
    return kv_cache.permute(_LOGICAL_TO_PHYSICAL)


def layer_major_kv_view(
    layers: list[torch.Tensor], heads: int, head_dim: int
) -> torch.Tensor | None:
    """View vLLM's per-layer fused K/V tensors as one logical KV cache.

    Args:
        layers: Per-layer ``[blocks, tokens, heads, 2 * head_dim]``
            tensors in layer order (vLLM's fused K/V layout).
        heads: KV heads per layer.
        head_dim: Scalars per K or V head.

    Returns:
        A logical ``[blocks, layers, 2, tokens, heads, head_dim]`` view over
        the layers' shared allocation, or None unless every layer is
        contiguous and they follow each other back to back in one storage.
    """
    first = layers[0]
    if first.dim() != 4 or tuple(first.shape[2:]) != (heads, 2 * head_dim):
        return None
    layer_elems = first.numel()
    for index, layer in enumerate(layers):
        if (
            not layer.is_contiguous()
            or layer.shape != first.shape
            or layer.dtype != first.dtype
            or layer.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
            or layer.storage_offset() != first.storage_offset() + index * layer_elems
        ):
            return None
    num_blocks, tokens = int(first.shape[0]), int(first.shape[1])
    physical = first.as_strided(
        (len(layers), num_blocks, tokens, heads, 2, head_dim),
        (
            layer_elems,
            tokens * heads * 2 * head_dim,
            heads * 2 * head_dim,
            2 * head_dim,
            head_dim,
            1,
        ),
    )
    return physical.permute(_PHYSICAL_TO_LOGICAL_FROM_LAYER_MAJOR)


def allocate_cross_layer_kv_cache(
    num_blocks: int,
    num_layers: int,
    block_tokens: int,
    heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Allocate a logical cross-layer KV view over fresh ``BLNHC`` memory.

    Args:
        num_blocks: Number of KV blocks.
        num_layers: Number of attention layers.
        block_tokens: Tokens per block.
        heads: KV heads per layer.
        head_dim: Scalars per head.
        dtype: KV element type (``torch.bfloat16`` or ``torch.uint8`` for FP8).
        device: Allocation device.

    Returns:
        A logical ``[blocks, layers, 2, tokens, heads, head_dim]`` view over a
        freshly allocated physical ``[blocks, layers, tokens, heads, 2,
        head_dim]`` buffer.
    """
    physical = torch.empty(
        num_blocks,
        num_layers,
        block_tokens,
        heads,
        2,
        head_dim,
        dtype=dtype,
        device=device,
    )
    return physical.permute(_PHYSICAL_TO_LOGICAL)


__all__ = [
    "allocate_cross_layer_kv_cache",
    "kernel_kv_view",
    "kv_is_layer_major",
    "layer_major_kv_view",
    "physical_kv_view",
]

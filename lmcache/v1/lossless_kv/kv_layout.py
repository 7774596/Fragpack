# SPDX-License-Identifier: Apache-2.0

"""Logical and physical views of the codec's cross-layer KV scratch cache.

The GPU codec kernels address a block-outermost ``BLNHC`` buffer whose
physical order is ``[blocks, layers, tokens, heads, 2, head_dim]`` (vLLM's
cross-layer layout). Records keep the plane-major order ``[layers, 2, tokens,
heads, head_dim]`` so each (layer, K/V) plane is one contiguous codec record;
the kernels therefore take the strided logical view ``[blocks, layers, 2,
tokens, heads, head_dim]``, and copies that address raw memory use the
contiguous physical view.
"""

# Future
from __future__ import annotations

# Third Party
import torch

# Physical [B, L, N, H, 2, D] -> logical [B, L, 2, N, H, D].
_PHYSICAL_TO_LOGICAL = (0, 1, 4, 2, 3, 5)
# Logical [B, L, 2, N, H, D] -> physical [B, L, N, H, 2, D].
_LOGICAL_TO_PHYSICAL = (0, 1, 3, 4, 2, 5)


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


__all__ = ["allocate_cross_layer_kv_cache", "physical_kv_view"]

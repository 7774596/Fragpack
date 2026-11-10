# SPDX-License-Identifier: Apache-2.0
"""FlashInfer paged decode as vLLM 0.30's FLASHINFER backend runs it on Hopper.

The wrapper uses the NHD layout and tensor cores (vLLM's setting), is planned
once per batch shape, and reads the same raw paged K/V tensors that FA3 reads.
"""

# Standard
from typing import Callable

# Third Party
import torch

_WORKSPACE: torch.Tensor | None = None


def flashinfer_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    lens: list[int],
    page_size: int,
    scale: float,
    out: torch.Tensor,
    use_tensor_cores: bool = True,
) -> Callable[[], torch.Tensor]:
    """Plan FlashInfer batch decode and return a zero-argument runner.

    Args:
        q (torch.Tensor): ``[B, Hq, D]`` BF16 queries, one token per request.
        k_cache (torch.Tensor): ``[pages, page_size, Hkv, D]`` paged K (BF16 or
            FP8 E4M3).
        v_cache (torch.Tensor): paged V, same shape and dtype as ``k_cache``.
        block_table (torch.Tensor): ``[B, width]`` int32 page ids per request.
        lens (list[int]): context length of each request.
        page_size (int): tokens per page.
        scale (float): softmax scale.
        out (torch.Tensor): ``[B, Hq, D]`` output buffer.
        use_tensor_cores (bool): FlashInfer's tensor-core decode path.

    Returns:
        Callable[[], torch.Tensor]: enqueues one decode on the current stream.
        Planning (a host synchronization) happens here, not in the runner.
    """
    # Third Party
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = torch.empty(256 << 20, dtype=torch.uint8, device=q.device)
    pages = [(n + page_size - 1) // page_size for n in lens]
    indptr = torch.tensor([0] + pages, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    indices = torch.cat([block_table[i, :p] for i, p in enumerate(pages)]).int()
    last = torch.tensor(
        [n - (p - 1) * page_size for n, p in zip(lens, pages, strict=True)],
        dtype=torch.int32,
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        _WORKSPACE, "NHD", use_tensor_cores=use_tensor_cores
    )
    wrapper.plan(
        indptr.to(q.device),
        indices,
        last.to(q.device),
        q.shape[1],
        k_cache.shape[2],
        q.shape[2],
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
        sm_scale=scale,
    )
    kv = (k_cache, v_cache)
    return lambda: wrapper.run(q, kv, out=out)

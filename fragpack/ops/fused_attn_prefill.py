# SPDX-License-Identifier: Apache-2.0
"""Causal prefill attention consuming FragPack records without a BF16 workspace.

The native kernel reconstructs tiles in CTA-local memory and reuses them across
queries. This module owns no KV tensors, transfer policy, or inference-engine
state. Open/raw records and compressed history share the same address tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fragpack.ops.fragpack_ext import extension
from fragpack.ops.fused_attn_decode import RecordTables


def prefill_split_count(
    query_lengths: Sequence[int],
    context_lengths: Sequence[int],
    kv_heads: int,
    group: int,
    num_sms: int,
) -> int:
    """Choose a bounded split count using host metadata, without device access.

    Args:
        query_lengths: Query token counts in request order, including empty rows.
        context_lengths: Context lengths including queries, in the same order.
        kv_heads: Number of KV heads per request on this device.
        group: Query heads per KV head, one of 1, 2, 4, or 8.
        num_sms: Device SM count, obtained during initialization.

    Returns:
        A count in [1, 32]. Splitting fills at most one wave of active CTAs
        without exceeding the page count of the longest active context.
        Batches already occupying one wave keep the direct one-split path.

    Raises:
        ValueError: For unsupported geometry or inconsistent host lengths.

    Async/thread-safety:
        Pure host calculation. Inputs must already be available on the host;
        callers must not introduce a device readback to obtain them.
    """
    if kv_heads < 1 or num_sms < 1 or group not in (1, 2, 4, 8):
        raise ValueError("positive KV heads/SM count and GQA group 1/2/4/8 required")
    if len(query_lengths) != len(context_lengths):
        raise ValueError("query/context lengths must describe the same requests")
    tile_queries = 64 // group
    tiles = 0
    max_pages = 0
    for queries, context in zip(query_lengths, context_lengths, strict=True):
        if queries < 0 or context < queries:
            raise ValueError("context lengths must be at least nonnegative queries")
        if queries:
            tiles += (int(queries) + tile_queries - 1) // tile_queries
            max_pages = max(max_pages, (int(context) + 127) // 128)
    if tiles == 0:
        return 1
    return max(1, min(32, num_sms // (tiles * kv_heads), max_pages))


def prefill_workspace_elems(
    total_queries: int, query_heads: int, num_splits: int
) -> tuple[int, int]:
    """Return required FP32 element counts for a split attention call.

    Args:
        total_queries: Packed query token count, excluding empty request rows.
        query_heads: Query heads on this device, including GQA expansion.
        num_splits: Explicit split count in [1, 32].

    Returns:
        Counts for partial output ``[Q, Hq, S, 128]`` and softmax state
        ``[Q, Hq, S, 2]`` (base-2 maximum, exponential sum). Both are zero
        for one split, which writes the final output without a combine kernel.

    Raises:
        ValueError: For invalid sizes or a split count outside [1, 32].

    Async/thread-safety:
        Pure host calculation; allocates no tensors.
    """
    if total_queries < 0 or query_heads < 1 or not 1 <= num_splits <= 32:
        raise ValueError(
            "nonnegative queries, positive heads and 1..32 splits required"
        )
    rows = total_queries * query_heads * num_splits if num_splits > 1 else 0
    return rows * 128, rows * 2


def prefill_workspace_capacity(num_sms: int) -> tuple[int, int]:
    """Bound reusable FP32 buffers for every automatic split plan on a device.

    Args:
        num_sms: Positive number of device SMs.

    Returns:
        Partial-output and softmax-state element capacities. The automatic
        planner keeps active CTAs within one wave of 64-column tiles, so these
        buffers suffice independently of batch/context size. Forced split
        counts must instead use :func:`prefill_workspace_elems`.

    Raises:
        ValueError: If num_sms is not positive.

    Async/thread-safety:
        Pure host calculation. A caller may allocate these buffers once and
        reuse them on an ordered stream; concurrent calls need separate owners.
    """
    if num_sms < 1:
        raise ValueError("num_sms must be positive")
    rows = num_sms * 64
    return rows * 128, rows * 2


def prefill_attention(
    q: torch.Tensor,
    tables: RecordTables,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen_q: int,
    scale: float,
    out: torch.Tensor,
    *,
    num_splits: int = 1,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Run fused, right-aligned causal attention over packed prefill queries.

    Args:
        q (torch.Tensor): CUDA BF16 queries ``[total_queries, H * G, 128]``.
            Heads/dimensions are contiguous; token rows may be strided by an
            even number of BF16 elements. The data pointer is 4-byte aligned.
        tables (RecordTables): This layer's record addresses, lengths, and tables.
        block_table (torch.Tensor): CUDA int32 ``[requests, max_pages]`` physical
            block IDs. Rows may be strided; no workspace remapping is required.
        cu_seqlens_q (torch.Tensor): Contiguous CUDA int32 query offsets, starting
            at zero and ending at total_queries. Empty request rows are allowed.
        seq_lens (torch.Tensor): Contiguous CUDA int32 context lengths including
            each request's query tokens; lengths must be at least query lengths.
        max_seqlen_q (int): Host upper bound on query tokens in any request.
        scale (float): Attention score multiplier, normally ``128**-0.5``.
        out (torch.Tensor): CUDA BF16 output with q's shape; token rows may be
            strided. It must not alias q, any KV record, or metadata.
        num_splits (int): Number of page partitions per query tile, from 1 to 32.
            One split bypasses the partial buffers and combine kernel.
        workspace: Caller-owned contiguous CUDA FP32 tensors sized by
            :func:`prefill_workspace_elems` or the automatic planner's capacity.
            Required when num_splits > 1; never contains reconstructed KV.
            Partial-output and softmax buffers must not alias each other or any
            input/output. Their old contents are ignored on every invocation.

    Returns:
        None; writes all valid output rows without allocating a decoded KV buffer.

    Raises:
        ValueError: If splitting is requested without a two-tensor workspace,
            or if the split count is outside [1, 32].
        RuntimeError: For unsupported dtype, shape, device, or GQA group. G must
            be 1, 2, 4, or 8. Value-level metadata is the caller's responsibility;
            the hot path never copies it back to the CPU for validation.

    Async/thread-safety:
        Enqueues on the current CUDA stream without synchronization. The caller
        retains all tensor owners and keeps records immutable until completion;
        previous writes/imports must be ordered before this call. Separate calls
        may share read-only records but must own disjoint output and partial
        tensors. Split writes, combination and later reuse are ordered on the
        current stream. The first call JIT-compiles the extension and must
        precede graph capture. There is no hot-path allocation or device readback.
    """
    if not 1 <= num_splits <= 32:
        raise ValueError("num_splits must be in [1, 32]")
    if num_splits > 1 and (workspace is None or len(workspace) != 2):
        raise ValueError("split prefill requires a two-tensor FP32 workspace")
    partial, softmax = workspace if num_splits > 1 else (None, None)
    extension().fused_prefill(
        q,
        tables.rec_addr,
        tables.rec_len,
        tables.plut,
        tables.slut,
        block_table,
        cu_seqlens_q,
        seq_lens,
        out,
        partial,
        softmax,
        num_splits,
        max_seqlen_q,
        scale,
    )


__all__ = [
    "prefill_attention",
    "prefill_split_count",
    "prefill_workspace_capacity",
    "prefill_workspace_elems",
]

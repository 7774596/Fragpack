# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode attention over FragPack KV records.

One CTA (256 threads) handles one chunk of pages of one (sequence, KV head)
pair for all ``GQA`` query heads, with an online softmax; a small combine
kernel merges each sequence's chunks. The MMAs are transposed,
``S^T = K Q^T`` and ``O^T += V^T P^T``, so K / V^T is the 16-row A operand and
the 8 GQA heads fill the n8 side without padding. Thread 0 bulk-copies (TMA)
each K and V record into its own shared slot, and every thread decodes its 64
kernel-order scalars straight into A fragments, overlapping the next record's
copy.

Records come from :mod:`.attn_page` and are located through per-record
tables: ``rec_addr[P, 2, H]`` absolute device addresses and
``rec_len[P, 2, H]`` record bytes minus ``DENSE_BYTES`` (raw records, whose
length exceeds ``VAR_CAP``, included). The kernel cuts every sequence into
chunks of the same page count, the smallest for which the batch fits
``units`` CTAs per KV head: a sequence gets CTAs in proportion to its length,
so a batch mixing short and long contexts does not wait on the long one's few
CTAs. A one-block plan kernel picks the chunk size and maps units to
(sequence, page) on device once per step, and every layer of the step reuses
the plan. A launch depends only on the batch size and is CUDA-graph safe.

FP8 E4M3 records (``RecordTables.fp8``) decode to exact FP16 fragments and run
FP16 MMAs (the query is rounded to FP16); the K scale folds into the softmax
scale and the V scale into the output. Their symbols are restored with the
per-channel exponent offsets of the tables (``RecordTables.offset``).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lossless_kv.fragpack.fragpack_ext import extension

logger = init_logger(__name__)

GQA = 8


@dataclass(frozen=True)
class RecordTables:
    """One layer's record tables for :func:`decode_attention`.

    Attributes:
        rec_addr: int64 ``[P, 2, H]`` record device addresses.
        rec_len: int32 ``[P, 2, H]`` record bytes minus ``DENSE_BYTES``.
        plut: int32 ``[2, H, 2]`` primary ``prmt`` tables.
        slut: uint8 ``[2, H, 16]`` extended tables.
        offset: uint8 ``[2, H, 128]`` FP8 exponent offsets, pre-shifted
            (``attn_tables.DeviceTables.offset``); zeros for BF16.
        fp8: The records are FP8 E4M3 records.
    """

    rec_addr: torch.Tensor
    rec_len: torch.Tensor
    plut: torch.Tensor
    slut: torch.Tensor
    offset: torch.Tensor
    fp8: bool = False


def decode_units(batch: int, kv_heads: int, num_sms: int) -> int:
    """Return the CTAs per KV head of a decode launch.

    Two CTAs fit per SM; the units of all KV heads fill one wave, and every
    sequence gets at least one. Depending on the batch size only keeps the
    launch shape fixed under CUDA graphs.

    Args:
        batch: Number of sequences (padded batch size under CUDA graphs).
        kv_heads: KV heads per sequence.
        num_sms: Streaming multiprocessor count.

    Returns:
        Unit count, at least ``batch``.

    Async/thread-safety:
        Pure function.
    """
    return max(batch, (2 * num_sms) // max(1, kv_heads))


def workspace_elems(kv_heads: int, units: int) -> tuple[int, int, int]:
    """Return the element counts of the ``(opart, ml, plan)`` workspace.

    Args:
        kv_heads: KV heads per sequence.
        units: Units per KV head.

    Returns:
        Float elements of ``opart`` and ``ml``, int32 elements of ``plan``.

    Async/thread-safety:
        Pure function.
    """
    rows = kv_heads * units * GQA
    return rows * 128, rows * 2, 1 + 4 * units


def decode_attention(
    q: torch.Tensor,
    tables: RecordTables,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    units: int,
    out: torch.Tensor,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    replan: bool,
    out_scale: float = 1.0,
) -> None:
    """Run split-KV decode attention for one query token per sequence.

    The kernel attends ``GQA`` query heads per KV head. A model with a
    smaller group ``G`` (a divisor of ``GQA``) runs on zero-padded queries,
    and only its ``G`` rows per KV head are copied to ``out``.

    Args:
        q: BF16 ``[B, H * G, 128]`` queries (rows may be strided).
        tables: Record tables of this layer.
        block_table: int32 ``[B, max_pages]`` page ids (rows may be strided).
        seq_lens: int32 ``[B]`` context lengths; 0 yields a zero output row.
        scale: Softmax scale (for FP8 records, times the K scale).
        units: CTAs per KV head, at least ``B``, e.g. from
            :func:`decode_units`.
        out: BF16 ``[B, H * G, 128]`` output (rows may be strided).
        workspace: ``(opart, ml, plan)`` float, float, int32 tensors sized
            by :func:`workspace_elems`.
        replan: Recompute the work plan from ``seq_lens`` and ``units``;
            later calls with the same batch (the other layers of a step) may
            pass False to reuse it.
        out_scale: Factor applied to the output (the V scale of FP8
            records).

    Returns:
        None; writes ``out``.

    Raises:
        ValueError: If the query group does not divide ``GQA``.

    Async/thread-safety:
        Enqueues on the current CUDA stream; not synchronized. Allocations
        for padded groups come from the caching allocator and are CUDA-graph
        safe.
    """
    kv_heads = tables.rec_addr.shape[2]
    group = q.shape[1] // kv_heads
    if group * kv_heads != q.shape[1] or GQA % group:
        raise ValueError(f"query heads {q.shape[1]} need a group dividing {GQA}")
    if group == GQA:
        _launch(
            q,
            tables,
            block_table,
            seq_lens,
            scale,
            units,
            out,
            workspace,
            replan,
            out_scale,
        )
        return
    rows = q.unflatten(1, (kv_heads, group))
    qp = q.new_zeros(q.shape[0], kv_heads, GQA, q.shape[2])
    qp[:, :, :group] = rows
    op = torch.empty_like(qp)
    _launch(
        qp.flatten(1, 2),
        tables,
        block_table,
        seq_lens,
        scale,
        units,
        op.flatten(1, 2),
        workspace,
        replan,
        out_scale,
    )
    out.unflatten(1, (kv_heads, group)).copy_(op[:, :, :group])


def _launch(
    q: torch.Tensor,
    tables: RecordTables,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    units: int,
    out: torch.Tensor,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    replan: bool,
    out_scale: float,
) -> None:
    """Launch the kernel on ``GQA`` query rows per KV head.

    Args:
        q: BF16 ``[B, H * GQA, 128]`` queries.
        tables: Record tables of this layer.
        block_table: int32 ``[B, max_pages]`` page ids.
        seq_lens: int32 ``[B]`` context lengths.
        scale: Softmax scale.
        units: CTAs per KV head.
        out: BF16 ``[B, H * GQA, 128]`` output.
        workspace: ``(opart, ml, plan)`` workspace tensors.
        replan: Recompute the work plan.
        out_scale: Factor applied to the output.

    Returns:
        None; writes ``out``.

    Async/thread-safety:
        Enqueues on the current CUDA stream; not synchronized.
    """
    extension().fused_decode(
        q,
        tables.rec_addr,
        tables.rec_len,
        tables.plut,
        tables.slut,
        tables.offset,
        block_table,
        seq_lens,
        workspace[0],
        workspace[1],
        workspace[2],
        out,
        units,
        replan,
        scale,
        out_scale,
        tables.fp8,
    )


__all__ = [
    "GQA",
    "RecordTables",
    "decode_attention",
    "decode_units",
    "workspace_elems",
]

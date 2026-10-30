# SPDX-License-Identifier: Apache-2.0
"""Fused prefill over compressed, overflow, and unfinished raw KV records.

The numeric reference reads the original BF16 tensors, never gather output or
the fused kernel's intermediates. NaNs beyond an unfinished block's valid tail
make an unmasked V read visible even when its attention probability is zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import calibrate, device_tables
from fragpack.ops.fused_attn_decode import RecordTables
from fragpack.ops.fused_attn_prefill import (
    prefill_attention,
    prefill_workspace_elems,
)
from fragpack.ops.hbm_kv import HbmKvStore, page_bytes

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

HEADS, BLOCKS, TAIL = 2, 5, 37
TOKENS, DIM = ap.PAGE_TOKENS, ap.HEAD_DIM
SCALE = DIM**-0.5


@dataclass
class StoreFixture:
    """Keep the store, arena owner, and independent source KV alive together."""

    store: HbmKvStore
    arena: torch.Tensor
    source: torch.Tensor  # [2, blocks, tokens, heads, dim]


def _make_store(*, encode: bool, last_value_delta: float = 0.0) -> StoreFixture:
    """Build four full pages and one open tail through the public store API."""
    generator = torch.Generator().manual_seed(392)
    source = torch.randn(2, BLOCKS, TOKENS, HEADS, DIM, generator=generator) * 0.5
    bits = source.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    table = calibrate(bits.transpose(1, 0, 2, 3, 4)[:, None])
    # Bounded but widely distributed exponents force one K record into the raw
    # path without producing huge logits or relying on private record metadata.
    exponents = torch.arange(TOKENS * DIM).reshape(TOKENS, DIM) % 59 - 60
    signs = torch.where(torch.arange(DIM) % 2 == 0, 1.0, -1.0)
    source[0, 1, :, 0] = torch.ldexp(signs.expand(TOKENS, DIM), exponents)
    source[1, 2, -1] += last_value_delta
    source = source.to(dtype=torch.bfloat16, device="cuda")
    arena = torch.empty(BLOCKS, page_bytes(HEADS), dtype=torch.uint8, device="cuda")
    total = (BLOCKS - 1) * TOKENS + TAIL
    store = HbmKvStore(
        [arena],
        HEADS,
        device_tables(table, torch.device("cuda")),
        max_tokens=total,
        staging_slots=1,
        overflow_records=BLOCKS * 2 * HEADS,
    )
    slots = torch.arange(total, dtype=torch.int64, device="cuda")
    store.prepare(slots, total, np.zeros(0, np.int32))
    store.write(
        0,
        source[0].reshape(-1, HEADS, DIM)[:total],
        source[1].reshape(-1, HEADS, DIM)[:total],
        slots,
    )
    # The open record already owns staging. Poison only its unwritten suffix;
    # no query may consume these token positions. This also tests 0 * NaN in PV.
    padding = torch.arange(total, BLOCKS * TOKENS, dtype=torch.int64, device="cuda")
    poison = torch.full(
        (padding.numel(), HEADS, DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    store.write(0, poison, poison, padding)
    if encode:
        store.encode(0, BLOCKS - 1)
        store.commit()
        stats = store.stats()
        assert 0 < stats.fallbacks < stats.encoded
    return StoreFixture(store, arena, source)


@pytest.fixture(scope="module", params=["mixed", "raw"])
def records(request: pytest.FixtureRequest) -> StoreFixture:
    """Expose closed coded/raw records or an entirely unencoded prefill step."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FragPack targets SM90")
    return _make_store(encode=request.param == "mixed")


def _metadata(
    rows: list[list[int]], queries: list[int], contexts: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deliberately strided block-table rows and ragged query offsets."""
    width = max(map(len, rows))
    owner = torch.full((len(rows), width + 3), -1, dtype=torch.int32, device="cuda")
    table = owner[:, :width]
    for row, blocks in enumerate(rows):
        table[row, : len(blocks)] = torch.tensor(
            blocks, dtype=torch.int32, device="cuda"
        )
    offsets = torch.tensor([0, *np.cumsum(queries)], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(contexts, dtype=torch.int32, device="cuda")
    return table, offsets, lengths


def _reference(
    q: torch.Tensor,
    source: torch.Tensor,
    rows: list[list[int]],
    queries: list[int],
    contexts: list[int],
) -> torch.Tensor:
    """Compute independent FP32 causal attention with right-aligned queries."""
    group = q.shape[1] // HEADS
    output = torch.empty_like(q, dtype=torch.float32)
    offset = 0
    for blocks, nq, context in zip(rows, queries, contexts, strict=True):
        if nq == 0:
            continue
        key, value = source[:, blocks].reshape(2, -1, HEADS, DIM)[:, :context].float()
        key = key.repeat_interleave(group, dim=1)
        value = value.repeat_interleave(group, dim=1)
        query = q[offset : offset + nq].float()
        scores = torch.einsum("qhd,khd->hqk", query, key) * SCALE
        positions = context - nq + torch.arange(nq, device=q.device)
        mask = torch.arange(context, device=q.device)[None, :] > positions[:, None]
        scores.masked_fill_(mask[None], -float("inf"))
        output[offset : offset + nq] = torch.einsum(
            "hqk,khd->qhd", scores.softmax(-1), value
        )
        offset += nq
    return output


def _workspace(
    q: torch.Tensor, num_splits: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Allocate caller-owned partials with deliberately stale finite contents."""
    if num_splits == 1:
        return None
    counts = prefill_workspace_elems(q.shape[0], q.shape[1], num_splits)
    workspace = tuple(
        torch.empty(n, dtype=torch.float32, device=q.device) for n in counts
    )
    _poison_workspace(workspace)
    return workspace


def _poison_workspace(workspace: tuple[torch.Tensor, torch.Tensor] | None) -> None:
    """Make any unwritten empty split dominate the final softmax incorrectly."""
    if workspace is not None:
        workspace[0].fill_(37.0)
        softmax = workspace[1].view(-1, 2)
        softmax[:, 0].fill_(80.0)
        softmax[:, 1].fill_(1.0)


@pytest.mark.parametrize("group", [1, 2, 4, 8])
@pytest.mark.parametrize("num_splits", [1, 2, 8])
def test_prefill_matches_causal_reference_with_ragged_strided_inputs(
    records: StoreFixture, group: int, num_splits: int
) -> None:
    """Cover ragged queries, an empty row, shared pages, and an open KV tail."""
    # The empty row has no query/output span and its block table remains -1;
    # an implementation that reads its cache metadata can access invalid memory.
    rows = [[0, 1, 2, 3, 4], [], [2, 0], [3, 0, 1, 4], [1, 2]]
    queries, contexts = [35, 0, 1, 17, 3], [549, 0, 255, 389, 129]
    table, offsets, lengths = _metadata(rows, queries, contexts)
    owner = torch.randn(
        sum(queries), HEADS * group + 2, DIM, dtype=torch.bfloat16, device="cuda"
    )
    q = owner[:, : HEADS * group]
    output_owner = torch.full_like(owner, float("nan"))
    out = output_owner[:, : HEADS * group]
    tables = RecordTables(*records.store.layer_tables(0))
    workspace = _workspace(q, num_splits)
    prefill_attention(
        q,
        tables,
        table,
        offsets,
        lengths,
        max(queries),
        SCALE,
        out,
        num_splits=num_splits,
        workspace=workspace,
    )
    expected = _reference(q, records.source, rows, queries, contexts)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.isnan(output_owner[:, HEADS * group :]).all()
    if num_splits > 1:
        unsplit = torch.empty_like(q)
        prefill_attention(
            q, tables, table, offsets, lengths, max(queries), SCALE, unsplit
        )
        torch.testing.assert_close(unsplit.float(), expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(out, unsplit, atol=2e-2, rtol=2e-2)


def test_unaligned_bf16_query_view_is_rejected(records: StoreFixture) -> None:
    """Reject an odd BF16 storage offset before native packed-word reads."""
    table, offsets, lengths = _metadata([[0]], [3], [TOKENS])
    owner = torch.empty(3 * HEADS * DIM + 1, dtype=torch.bfloat16, device="cuda")
    q = owner[1:].view(3, HEADS, DIM)
    out = torch.empty_like(q)
    with pytest.raises(RuntimeError, match="q data must be 4-byte aligned"):
        prefill_attention(
            q,
            RecordTables(*records.store.layer_tables(0)),
            table,
            offsets,
            lengths,
            3,
            SCALE,
            out,
        )


@pytest.mark.parametrize("num_splits", [1, 4])
def test_cold_prefill_matches_reference(records: StoreFixture, num_splits: int) -> None:
    """No prefix is available: the first query may attend only the first token."""
    rows, queries = [[0, 1, 2, 3, 4]], [549]
    table, offsets, lengths = _metadata(rows, queries, queries)
    q = torch.randn(549, HEADS * 8, DIM, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(q)
    workspace = _workspace(q, num_splits)
    prefill_attention(
        q,
        RecordTables(*records.store.layer_tables(0)),
        table,
        offsets,
        lengths,
        549,
        SCALE,
        out,
        num_splits=num_splits,
        workspace=workspace,
    )
    expected = _reference(q, records.source, rows, queries, queries)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_splits", [1, 8])
def test_cuda_graph_replay_writes_fresh_outputs(
    records: StoreFixture, num_splits: int
) -> None:
    """Changing graph metadata cannot expose stale output or empty-split partials."""
    rows, queries, contexts = [[0, 1, 2, 3, 4], [1, 2]], [5, 19], [549, 129]
    table, offsets, lengths = _metadata(rows, queries, contexts)
    q = torch.randn(sum(queries), HEADS * 4, DIM, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(q)
    tables = RecordTables(*records.store.layer_tables(0))
    workspace = _workspace(q, num_splits)
    # A fixed host bound covers all subsequent ragged layouts at the same total Q.
    max_queries = sum(queries)

    def run() -> None:
        prefill_attention(
            q,
            tables,
            table,
            offsets,
            lengths,
            max_queries,
            SCALE,
            out,
            num_splits=num_splits,
            workspace=workspace,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for query_lengths, context_lengths in (
        (queries, contexts),
        ([17, 7], [19, 7]),
        ([0, 24], [0, 129]),
    ):
        # Shrinking context makes previously active splits empty. Repartitioning
        # Q changes which request owns each global output/partial row, and the
        # final replay removes a request entirely without changing allocation.
        offsets.copy_(
            torch.tensor(
                [0, *np.cumsum(query_lengths)], dtype=torch.int32, device=q.device
            )
        )
        lengths.copy_(torch.tensor(context_lengths, dtype=torch.int32, device=q.device))
        out.fill_(float("nan"))
        _poison_workspace(workspace)
        graph.replay()
        expected = _reference(q, records.source, rows, query_lengths, context_lengths)
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_splits", [1, 4])
def test_future_value_cannot_affect_earlier_queries(num_splits: int) -> None:
    """Changing the last KV token affects its query but no earlier causal row."""
    before = _make_store(encode=True)
    after = _make_store(encode=True, last_value_delta=17.0)
    rows, queries, contexts = [[0, 2]], [33], [256]
    table, offsets, lengths = _metadata(rows, queries, contexts)
    # Uniform logits make the changed last token's contribution deterministic.
    q = torch.zeros(33, HEADS * 2, DIM, dtype=torch.bfloat16, device="cuda")
    outputs = []
    for fixture in (before, after):
        out = torch.empty_like(q)
        workspace = _workspace(q, num_splits)
        prefill_attention(
            q,
            RecordTables(*fixture.store.layer_tables(0)),
            table,
            offsets,
            lengths,
            33,
            SCALE,
            out,
            num_splits=num_splits,
            workspace=workspace,
        )
        expected = _reference(q, fixture.source, rows, queries, contexts)
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
        outputs.append(out)
    torch.testing.assert_close(outputs[0][:-1], outputs[1][:-1], atol=0, rtol=0)
    assert float((outputs[0][-1].float() - outputs[1][-1].float()).abs().max()) > 0.01

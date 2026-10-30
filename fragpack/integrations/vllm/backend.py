# SPDX-License-Identifier: Apache-2.0
"""vLLM attention backend that keeps the KV cache FragPack-compressed in HBM.

Each paged KV block holds, per layer, one page of
:func:`~fragpack.ops.hbm_kv.page_bytes` that its ``2 H`` FragPack records share (23.25
KiB per record on average instead of 32 KiB raw), so the same KV memory holds
1.38x as many blocks. :class:`~fragpack.ops.hbm_kv.HbmKvStore`
manages the records over the engine's cache; decode attention decompresses
records inside the attention kernel
(:func:`~fragpack.ops.fused_attn_decode.decode_attention`), and prefill decodes
the context pages into a BF16 workspace and runs FlashAttention over it by
default. ``FRAGPACK_PREFILL=fused`` instead consumes records directly inside
prefill attention and allocates no BF16 prefill workspace.
``FRAGPACK_WRITE=direct`` with the default gather reader skips scratch for
complete new blocks, using the original K/V for gather and subsequent encode.

Per step: :meth:`FragPackMetadataBuilder.build` calls ``HbmKvStore.prepare``
(eagerly, also before a FULL CUDA-graph replay); per layer the impl writes the
new tokens, attends, and encodes the blocks closed this step; the last layer
commits. The store needs the ids of every block a running request may still
write, which vLLM does not pass to attention backends, so the model runner's
``execute_model`` is wrapped by the version-pinned runtime adapter in
:mod:`fragpack.integrations.vllm.compat` to mirror per-request block lists
from the scheduler output (:class:`OpenBlocks`). This adapter is not a stable
public vLLM lifecycle API.

Requirements: V2 model runner, BF16 KV, head size 128, ``H * G`` query heads
with ``G`` dividing 8,
block size 128, SM90, no speculative decoding, a TP size that divides the KV
heads. Calibrated tables are read from ``FRAGPACK_TABLES`` (see
:mod:`fragpack.compression.attn_tables`); each TP rank uses its own heads' tables.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import ClassVar

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

from fragpack.compression.attn_tables import load_tables
from fragpack.integrations.vllm.capacity import FragPackCapacity, publish_capacity
from fragpack.integrations.vllm.compat import install_tracker, require_supported_vllm
from fragpack.integrations.vllm.lifecycle import OpenBlocks
from fragpack.integrations.vllm.prefill import PrefillPlan, plan_prefill_groups
from fragpack.logging import init_logger
from fragpack.ops.fused_attn_decode import (
    GQA,
    RecordTables,
    decode_attention,
    decode_units,
    workspace_elems,
)
from fragpack.ops.fused_attn_prefill import (
    prefill_attention,
    prefill_split_count,
    prefill_workspace_capacity,
)
from fragpack.ops.hbm_kv import (
    BLOCK_TOKENS,
    HEAD_DIM,
    HbmKvStore,
    page_bytes,
    register_store,
)

logger = init_logger(__name__)

_POLL_EVERY = 16
_LOG_EVERY = 4096
# Raw (incompressible) records; FRAGPACK_OVERFLOW. Real KV almost never
# falls back (~1 in 10^6 records), but a degenerate prompt (one token
# repeated) falls back ~96%: 8192 holds such a 2048-token prompt (~5.9k
# records) cached alongside real traffic, at 256 MiB.
_OVERFLOW_RECORDS = 8192


_TRACKER = OpenBlocks(BLOCK_TOKENS)


@dataclass
class PrefillGroup:
    """Prefill rows attended in one workspace pass.

    Attributes:
        src_blocks: int32 ``[P]`` distinct blocks decoded into workspace pages.
            Requests in this pass share pages for common physical blocks.
        block_table: int32 ``[R, max_pages]`` workspace page ids per row.
        cu_seqlens_q: int32 ``[R + 1]`` query offsets relative to ``tok0``.
        seqused_k: int32 ``[R]`` context lengths.
        tok0: First token of the group in the batch.
        tok1: One past the last token.
        max_seqlen_q: Longest query of the group.
        max_seqlen_k: Longest context of the group.
    """

    src_blocks: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    tok0: int
    tok1: int
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass
class FusedPrefill:
    """One prefill sub-batch reading physical records without a KV workspace.

    Attributes:
        block_table (torch.Tensor): Physical block IDs, in request order.
        cu_seqlens_q (torch.Tensor): Query offsets relative to tok0.
        seq_lens (torch.Tensor): Total context lengths, including new queries.
        tok0 (int): First prefill token in the mixed batch.
        tok1 (int): Exclusive end of the prefill token range.
        max_seqlen_q (int): Largest query length in this sub-batch.
        num_splits (int): KV partitions per query tile, selected on the host.
        workspace: Shared FP32 attention partials; None for one-split calls.

    Async/thread-safety:
        GPU metadata belongs to this model step and is consumed on its stream.
    """

    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    tok0: int
    tok1: int
    max_seqlen_q: int
    num_splits: int
    workspace: tuple[torch.Tensor, torch.Tensor] | None


@dataclass
class FragPackMetadata:
    """Per-step metadata shared by every layer.

    Attributes:
        store: The KV store.
        layer_tables: Record tables per layer index.
        layer_index: Layer name to index.
        last_layer: Index of the layer that commits the step.
        slot_mapping: int64 slot ids of the batch tokens.
        num_tokens: Tokens with a slot (rows of ``key`` to write).
        num_decodes: Leading single-token rows.
        decode_block_table: int32 ``[num_decodes, max_pages]``.
        decode_seq_lens: int32 ``[num_decodes]``.
        units: Decode CTAs per KV head.
        decode_workspace: ``(opart, ml, plan)`` of the decode kernel; the
            first layer of a step writes the plan, the others reuse it.
        max_close: Grid bound of the per-layer encode.
        prefills: Workspace-based prefill groups; empty in fused mode.
        fused_prefill: Direct-record prefill metadata, or None when not used.
        wk: BF16 prefill K workspace ``[W, 128, H, 128]``; None in fused mode.
        wv: BF16 prefill V workspace; None in fused mode.
    """

    store: HbmKvStore
    layer_tables: list[RecordTables]
    layer_index: dict[str, int]
    last_layer: int
    slot_mapping: torch.Tensor
    num_tokens: int
    num_decodes: int
    decode_block_table: torch.Tensor
    decode_seq_lens: torch.Tensor
    units: int
    decode_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    max_close: int
    prefills: list[PrefillGroup]
    fused_prefill: FusedPrefill | None
    wk: torch.Tensor | None
    wv: torch.Tensor | None


class FragPackMetadataBuilder(AttentionMetadataBuilder[FragPackMetadata]):
    """Builds :class:`FragPackMetadata` and owns the store and workspaces.

    Args:
        kv_cache_spec: KV spec of the layer group.
        layer_names: Attention layers of the group.
        vllm_config: Engine config.
        device: CUDA device.

    Async/thread-safety:
        Used from the model runner thread only.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        require_supported_vllm(use_v2_model_runner=vllm_config.use_v2_model_runner)
        self.prefill_mode = os.environ.get("FRAGPACK_PREFILL", "gather")
        if self.prefill_mode not in ("gather", "fused"):
            raise ValueError("FRAGPACK_PREFILL must be 'gather' or 'fused'")
        self.write_mode = os.environ.get("FRAGPACK_WRITE", "staged")
        if self.write_mode not in ("staged", "direct"):
            raise ValueError("FRAGPACK_WRITE must be 'staged' or 'direct'")
        if self.write_mode == "direct" and self.prefill_mode != "gather":
            raise ValueError("FRAGPACK_WRITE=direct requires FRAGPACK_PREFILL=gather")
        if vllm_config.speculative_config is not None:
            raise ValueError("FragPack attention does not support speculative decoding")
        if vllm_config.cache_config.block_size != BLOCK_TOKENS:
            raise ValueError(f"FragPack attention needs --block-size {BLOCK_TOKENS}")
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        install_tracker(_TRACKER, use_v2_model_runner=vllm_config.use_v2_model_runner)

        sched = vllm_config.scheduler_config
        capture = vllm_config.compilation_config.max_cudagraph_capture_size or 0
        self.max_tokens = max(sched.max_num_batched_tokens, capture)
        self.kv_heads = kv_cache_spec.num_kv_heads
        query_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.group = query_heads // self.kv_heads
        order = sorted(layer_names, key=extract_layer_index)
        self.layer_names = order
        self.layer_index = {name: i for i, name in enumerate(order)}

        props = torch.cuda.get_device_properties(device)
        self.num_sms = props.multi_processor_count
        self.max_reqs = sched.max_num_seqs
        opart, ml, plan = workspace_elems(
            self.kv_heads,
            decode_units(max(self.max_reqs, capture), self.kv_heads, self.num_sms),
        )
        f32 = dict(dtype=torch.float32, device=device)
        self.decode_workspace = (
            torch.empty(opart, **f32),
            torch.empty(ml, **f32),
            torch.empty(plan, dtype=torch.int32, device=device),
        )
        self.workspace_pages = -(
            -vllm_config.model_config.max_model_len // BLOCK_TOKENS
        )
        ws = (self.workspace_pages, BLOCK_TOKENS, self.kv_heads, HEAD_DIM)
        self.wk = self.wv = None
        self.prefill_partial_workspace = None
        if self.prefill_mode == "gather":
            self.wk = torch.empty(ws, dtype=torch.bfloat16, device=device)
            self.wv = torch.empty(ws, dtype=torch.bfloat16, device=device)
        else:
            # The host split planner caps active partial rows by one wave of
            # 64-column CTAs. This fixed allocation is shared across layers,
            # and every split/combine completes on the ordered worker stream.
            partial, softmax = prefill_workspace_capacity(self.num_sms)
            self.prefill_partial_workspace = (
                torch.empty(partial, **f32),
                torch.empty(softmax, **f32),
            )
        logger.info(
            "FragPack prefill path: %s, write path: %s",
            self.prefill_mode,
            self.write_mode,
        )

        path = os.environ.get("FRAGPACK_TABLES")
        if not path:
            raise ValueError(
                "FRAGPACK_TABLES must point at calibrated FragPack tables (.npz)"
            )
        # vLLM gives TP rank r the KV heads [r * H, (r + 1) * H).
        rank = get_tensor_model_parallel_rank()
        heads = slice(rank * self.kv_heads, (rank + 1) * self.kv_heads)
        self.tables = load_tables(path, device, heads)
        if self.tables.plut.shape[2] != self.kv_heads:
            raise ValueError(
                f"FragPack tables {path} do not cover KV heads "
                f"{heads.start}..{heads.stop}"
            )
        self.store: HbmKvStore | None = None
        self.store_key: tuple[int, int] | None = None
        self.layer_tables: list[RecordTables] = []
        self.steps = 0

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FragPackMetadata:
        """Prepare the store for this step and build the metadata.

        Args:
            common_prefix_len: Unused (no cascade attention).
            common_attn_metadata: Batch metadata.
            fast_build: Unused.

        Returns:
            Metadata for every layer of the step.

        Raises:
            StoreError: If the store dropped KV data since the last poll.

        Async/thread-safety:
            Enqueues the store's prepare on the current stream.
        """
        return self._build(common_attn_metadata, capture=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FragPackMetadata:
        """Build capture metadata without touching store state.

        Args:
            common_attn_metadata: Dummy batch metadata.

        Returns:
            Metadata whose launch shapes match replays of this batch shape.
        """
        return self._build(common_attn_metadata, capture=True)

    def _build(self, cam: CommonAttentionMetadata, capture: bool) -> FragPackMetadata:
        store = self._store()
        num_reqs = cam.num_reqs
        nd, npf, _, _ = split_decodes_and_prefills(cam, decode_threshold=1)
        qsl = cam.query_start_loc_cpu.numpy().astype(np.int64)
        seq = cam.seq_lens_cpu_upper_bound.numpy().astype(np.int64)[:num_reqs]
        qlen = qsl[1 : num_reqs + 1] - qsl[:num_reqs]
        num_tokens = min(int(qsl[num_reqs]), cam.slot_mapping.shape[0])

        # Blocks receiving their last token: one per crossed 128 boundary.
        closes = int(
            np.sum(seq // BLOCK_TOKENS - np.maximum(seq - qlen, 0) // BLOCK_TOKENS)
        )
        max_close = min(max(closes, nd), self.max_tokens)
        if not capture:
            store.prepare(cam.slot_mapping, num_tokens, _TRACKER.live_blocks())
            self.steps += 1
            if self.steps % _POLL_EVERY == 0:
                stats = store.poll()
                if stats is not None and self.steps % _LOG_EVERY == 0:
                    logger.info("FragPack store: %s", stats)

        prefills: list[PrefillGroup] = []
        fused_prefill = None
        if npf and not capture:
            if self.prefill_mode == "fused":
                tok0 = int(qsl[nd])
                splits = prefill_split_count(
                    qlen[nd:num_reqs],
                    seq[nd:num_reqs],
                    self.kv_heads,
                    self.group,
                    self.num_sms,
                )
                fused_prefill = FusedPrefill(
                    block_table=cam.block_table_tensor[nd:num_reqs],
                    cu_seqlens_q=cam.query_start_loc[nd : num_reqs + 1] - tok0,
                    seq_lens=cam.seq_lens[nd:num_reqs],
                    tok0=tok0,
                    tok1=int(qsl[num_reqs]),
                    max_seqlen_q=int(qlen[nd:num_reqs].max()),
                    num_splits=splits,
                    workspace=self.prefill_partial_workspace if splits > 1 else None,
                )
            else:
                prefills = self._prefill_groups(cam, nd, num_reqs, qsl, seq)
        return FragPackMetadata(
            store=store,
            layer_tables=self.layer_tables,
            layer_index=self.layer_index,
            last_layer=len(self.layer_names) - 1,
            slot_mapping=cam.slot_mapping,
            num_tokens=num_tokens,
            num_decodes=nd,
            decode_block_table=cam.block_table_tensor[:nd],
            decode_seq_lens=cam.seq_lens[:nd],
            units=decode_units(max(nd, 1), self.kv_heads, self.num_sms),
            decode_workspace=self.decode_workspace,
            max_close=max_close,
            prefills=prefills,
            fused_prefill=fused_prefill,
            wk=self.wk,
            wv=self.wv,
        )

    def _store(self) -> HbmKvStore:
        layers = get_layers_from_vllm_config(
            self.vllm_config, Attention, self.layer_names
        )
        caches = [layers[name].kv_cache for name in self.layer_names]
        if caches[0].numel() == 0:
            raise RuntimeError(
                "FragPack attention built metadata before the KV cache was bound"
            )
        key = (caches[0].data_ptr(), caches[0].shape[0])
        if self.store is None or key != self.store_key:
            # Every running request holds at most one block open across steps;
            # the store hands a closing block's slot to the request's new tail
            # when a chunk needs both within the step.
            self.store = HbmKvStore(
                caches,
                self.kv_heads,
                self.tables,
                self.max_tokens,
                staging_slots=self.max_reqs,
                overflow_records=int(
                    os.environ.get("FRAGPACK_OVERFLOW", _OVERFLOW_RECORDS)
                ),
                direct_new_blocks=self.write_mode == "direct",
            )
            self.store_key = key
            register_store(self.store)
            self.layer_tables = [
                RecordTables(*self.store.layer_tables(i)) for i in range(len(caches))
            ]
            # Capture sizes once at allocation, including auxiliary memory.
            # The connector forwards this host-only snapshot through vLLM's
            # public statistics channel; no telemetry RPC runs in attention.
            memory = self.store.memory_usage()
            memory["prefill_workspace"] = sum(
                t.numel() * t.element_size()
                for t in (self.wk, self.wv)
                if t is not None
            )
            memory["decode_workspace"] = sum(
                t.numel() * t.element_size() for t in self.decode_workspace
            )
            memory["prefill_partial_workspace"] = sum(
                t.numel() * t.element_size()
                for t in self.prefill_partial_workspace or ()
            )
            publish_capacity(
                FragPackCapacity(
                    blocks=self.store.num_blocks,
                    block_tokens=BLOCK_TOKENS,
                    layers=self.store.num_layers,
                    kv_heads=self.kv_heads,
                    page_bytes=self.store.arena_stride[0],
                    raw_page_bytes=2 * self.kv_heads * BLOCK_TOKENS * HEAD_DIM * 2,
                    memory=memory,
                )
            )
            logger.info(
                "FragPack store: %d layers x %d blocks, %d B/block-layer (raw %d), "
                "%d staging slots, %d overflow records, %.0f MiB staging+scratch"
                "+overflow, %.0f MiB prefill workspace",
                len(caches),
                self.store.num_blocks,
                page_bytes(self.kv_heads),
                2 * self.kv_heads * BLOCK_TOKENS * HEAD_DIM * 2,
                self.store.staging.shape[0],
                self.store.overflow.shape[0],
                (
                    self.store.staging.numel()
                    + self.store.scratch.numel()
                    + self.store.overflow.numel()
                )
                / 2**20,
                memory["prefill_workspace"] / 2**20,
            )
        return self.store

    def _prefill_groups(
        self,
        cam: CommonAttentionMetadata,
        first: int,
        num_reqs: int,
        qsl: np.ndarray,
        seq: np.ndarray,
    ) -> list[PrefillGroup]:
        pages = -(-seq[first:num_reqs] // BLOCK_TOKENS)
        indices = None if cam.req_idx is None else cam.req_idx[first:num_reqs]
        blocks = _TRACKER.prefill_blocks(indices, pages)
        if blocks is None:
            # Dummy/profile runs have no real request ownership. Unique synthetic
            # identities retain the original non-sharing plan; source coordinates
            # still read the runner's GPU block table, never these synthetic IDs.
            offsets = np.concatenate(([0], np.cumsum(pages)))
            blocks = [
                list(range(a, b))
                for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            ]
        return [
            self._prefill_group(cam, first, plan, qsl, seq)
            for plan in plan_prefill_groups(blocks, self.workspace_pages)
        ]

    def _prefill_group(
        self,
        cam: CommonAttentionMetadata,
        first: int,
        plan: PrefillPlan,
        qsl: np.ndarray,
        seq: np.ndarray,
    ) -> PrefillGroup:
        r0, r1 = first + plan.first, first + plan.stop
        n, width = plan.block_table.shape
        total = len(plan.source_rows)
        host = np.concatenate(
            (
                plan.source_rows + first,
                plan.source_columns,
                plan.block_table.ravel(),
                qsl[r0 : r1 + 1] - qsl[r0],
                seq[r0:r1],
            )
        )
        dev = torch.from_numpy(host).pin_memory().to(self.device, non_blocking=True)
        a, b, c, d = (
            total,
            2 * total,
            2 * total + n * width,
            2 * total + n * width + n + 1,
        )
        src = cam.block_table_tensor[dev[:a], dev[a:b]].to(torch.int32)
        return PrefillGroup(
            src_blocks=src,
            block_table=dev[b:c].view(n, width).to(torch.int32),
            cu_seqlens_q=dev[c:d].to(torch.int32),
            seqused_k=dev[d:].to(torch.int32),
            tok0=int(qsl[r0]),
            tok1=int(qsl[r1]),
            max_seqlen_q=int((qsl[r0 + 1 : r1 + 1] - qsl[r0:r1]).max()),
            max_seqlen_k=int(seq[r0:r1].max()),
        )


class FragPackAttentionImpl(AttentionImpl[FragPackMetadata]):
    """Decoder self-attention over the FragPack HBM store.

    Args:
        num_heads: Query heads.
        head_size: Head size (128).
        scale: Softmax scale.
        num_kv_heads: KV heads; ``num_heads / num_kv_heads`` must divide 8.
        alibi_slopes: Must be None.
        sliding_window: Must be None.
        kv_cache_dtype: ``auto`` or ``bfloat16``.
        logits_soft_cap: Must be None or 0.
        attn_type: Must be decoder.
        kv_sharing_target_layer_name: Must be None.
        **kwargs: Ignored.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs: object,
    ) -> None:
        num_kv_heads = num_kv_heads or num_heads
        if (
            num_heads % num_kv_heads
            or GQA % (num_heads // num_kv_heads)
            or head_size != HEAD_DIM
        ):
            raise ValueError(
                f"FragPack attention needs a GQA group dividing {GQA} and head "
                f"size {HEAD_DIM}"
            )
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap:
            raise ValueError("FragPack attention supports plain causal attention only")
        if (
            attn_type != AttentionType.DECODER
            or kv_sharing_target_layer_name is not None
        ):
            raise ValueError(
                "FragPack attention supports unshared decoder self-attention only"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.fa_version = get_flash_attn_version()

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FragPackMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write this layer's new KV, attend, and encode closed blocks.

        Args:
            layer: The attention layer.
            query: BF16 ``[T, Hq, 128]``.
            key: BF16 ``[T, H, 128]``.
            value: BF16 ``[T, H, 128]``.
            kv_cache: Unused (the store addresses the cache directly).
            attn_metadata: Step metadata, None in profiling runs.
            output: BF16 ``[T, Hq, 128]`` output buffer.
            output_scale: Unsupported.
            output_block_scale: Unsupported.

        Returns:
            ``output``.

        Async/thread-safety:
            Enqueues kernels on the current stream; CUDA-graph safe for
            decode-only batches.
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "FragPack attention does not fuse output quantization"
            )
        md = attn_metadata
        if md is None:
            return output.zero_()
        li = md.layer_index[layer.layer_name]
        store = md.store
        n = md.num_tokens
        layer_key, layer_value = key[:n], value[:n]
        store.write(li, layer_key, layer_value, md.slot_mapping[:n])
        # Every gather group indexes the complete batch's original K/V. Keep
        # these tensor views alive through encoding on the same worker stream.
        direct_key = layer_key if store.direct_new_blocks else None
        direct_value = layer_value if store.direct_new_blocks else None
        nd = md.num_decodes
        if nd:
            decode_attention(
                query[:nd],
                md.layer_tables[li],
                md.decode_block_table,
                md.decode_seq_lens,
                self.scale,
                md.units,
                output[:nd],
                md.decode_workspace,
                replan=li == 0,
            )
        for g in md.prefills:
            assert md.wk is not None and md.wv is not None
            store.gather(
                li, g.src_blocks, md.wk, md.wv, key=direct_key, value=direct_value
            )
            flash_attn_varlen_func(
                q=query[g.tok0 : g.tok1],
                k=md.wk,
                v=md.wv,
                out=output[g.tok0 : g.tok1],
                cu_seqlens_q=g.cu_seqlens_q,
                max_seqlen_q=g.max_seqlen_q,
                seqused_k=g.seqused_k,
                max_seqlen_k=g.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=g.block_table,
                fa_version=self.fa_version,
            )
        if md.fused_prefill is not None:
            fp = md.fused_prefill
            prefill_attention(
                query[fp.tok0 : fp.tok1],
                md.layer_tables[li],
                fp.block_table,
                fp.cu_seqlens_q,
                fp.seq_lens,
                fp.max_seqlen_q,
                self.scale,
                output[fp.tok0 : fp.tok1],
                num_splits=fp.num_splits,
                workspace=fp.workspace,
            )
        store.encode(li, md.max_close, key=direct_key, value=direct_value)
        if li == md.last_layer:
            store.commit()
        return output


class FragPackAttentionBackend(AttentionBackend):
    """FragPack-compressed HBM KV cache backend (``--attention-backend CUSTOM``)."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        """Return the only block size the kernels handle (128)."""
        return [BLOCK_TOKENS]

    @staticmethod
    def get_name() -> str:
        """Return the registry name."""
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[FragPackAttentionImpl]:
        """Return the impl class."""
        return FragPackAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FragPackMetadataBuilder]:
        """Return the metadata builder class."""
        return FragPackMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """Return the supported head sizes."""
        return [HEAD_DIM]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Size a block as one FragPack record page per layer.

        Args:
            spec: Dense KV spec.

        Returns:
            Spec with one head slot of ``page_bytes(H) / 128`` bytes per token.
        """
        if spec.state_content_bytes is not None:
            return spec
        return replace(
            spec,
            num_head_slots=1,
            state_content_bytes=page_bytes(spec.num_kv_heads) // BLOCK_TOKENS,
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        """Return the layer-major layout (each layer's blocks contiguous)."""
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def supports_compute_capability(cls, capability: object) -> bool:
        """Return True on SM90 (TMA bulk copies, FA3 prefill)."""
        return capability.major == 9


__all__ = [
    "FragPackAttentionBackend",
    "FragPackAttentionImpl",
    "FragPackMetadata",
    "FragPackMetadataBuilder",
    "PrefillGroup",
]

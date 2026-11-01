# SPDX-License-Identifier: Apache-2.0
"""Public vLLM connector for native BF16 KV and external lossless compression.

This module is loaded through ``kv_connector_module_path``. It neither selects
CUSTOM attention nor modifies vLLM's model runner. Runtime support is limited to
the pinned vLLM version, TP=1, eager full-attention text models and CPU LMCache L1.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    KVConnectorWorkerMetadata,
)

from fragpack.integrations.lmcache.client import RemoteBlobClient
from fragpack.integrations.lmcache.identity import CacheIdentity
from fragpack.integrations.vllm.compat import require_supported_vllm
from fragpack.integrations.vllm.native_connector_state import (
    NativeRequestPlan,
    NativeSchedulerState,
    NativeWorkerState,
    finished_store_ids,
    native_layer_view,
)
from fragpack.logging import init_logger
from fragpack.ops.native_codec import NativeKVPages

logger = init_logger(__name__)


@dataclass
class NativeConnectorMetadata(KVConnectorMetadata):
    """Serializable scheduler-owned keys and retained page maps for one step."""

    plans: tuple[NativeRequestPlan, ...] = ()
    finished_stores: tuple[str, ...] = ()


@dataclass
class NativeConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    """Cumulative worker counters and pending-publication liveness for TP=1."""

    pending: bool
    counters: dict[str, int]

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        """Reject accidental cross-rank use; TP aggregation is not implemented."""
        raise RuntimeError("FragPack native connector supports exactly one worker")


class FragPackNativeConnector(KVConnectorBase_V1):
    """Finish-time external caching with unmodified native attention.

    Extra configuration requires ``endpoint``, ``tables`` and the exact server
    ``identity`` dictionary. Optional ``mode`` is compressed/raw; ``batch_blocks``
    and ``transfer_slots`` bound scratch and transport work. ``precheck`` enables
    existing-object checks before encoding; ``max_pending_requests`` bounds
    delayed source ownership and background scheduler lookups.

    Scheduler RPCs use a dedicated background client. Worker codec/RPC work uses
    NativeBlockTransfer's background worker and producer events. Source and
    destination engine ownership are released only through vLLM completion hooks.
    """

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any):
        require_supported_vllm(use_v2_model_runner=vllm_config.use_v2_model_runner)
        super().__init__(vllm_config, role, kv_cache_config)
        from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        allowed = {
            "endpoint",
            "tables",
            "identity",
            "mode",
            "batch_blocks",
            "transfer_slots",
            "max_pending_requests",
            "timeout_s",
            "precheck",
        }
        if set(extra) - allowed:
            raise ValueError(
                f"unknown native codec options: {sorted(set(extra) - allowed)}"
            )
        model = vllm_config.model_config
        if vllm_config.parallel_config.world_size != 1:
            raise ValueError("native codec serving currently requires TP=PP=1")
        if not model.enforce_eager:
            raise ValueError("native codec serving currently requires enforce_eager")
        if model.is_encoder_decoder or model.is_multimodal_model:
            raise ValueError("native codec serving currently requires a text decoder")
        if vllm_config.lora_config is not None:
            raise ValueError("native codec namespace does not yet cover LoRA adapters")
        if vllm_config.speculative_config is not None:
            raise ValueError("native codec serving does not yet support speculation")
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1 or not isinstance(
            groups[0].kv_cache_spec, FullAttentionSpec
        ):
            raise ValueError("native codec requires one full-attention KV cache group")
        spec = groups[0].kv_cache_spec
        if spec.head_size != 128 or spec.dtype != torch.bfloat16:
            raise ValueError("native codec requires BF16 KV with head dimension 128")
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("native codec serving requires prefix caching")
        self._page_tokens = spec.block_size
        if self._page_tokens not in (16, 32, 64, 128):
            raise ValueError("native codec requires 16/32/64/128-token physical pages")
        if self._kv_transfer_config.kv_load_failure_policy != "recompute":
            raise ValueError("native codec requires kv_load_failure_policy='recompute'")
        self.identity = CacheIdentity(**extra["identity"])
        if self.identity.world_size != 1 or self.identity.rank != 0:
            raise ValueError("native codec identity must describe one TP rank")
        if (
            self.identity.num_layers != len(groups[0].layer_names)
            or self.identity.kv_heads != spec.num_kv_heads
        ):
            raise ValueError("native codec identity differs from engine KV geometry")
        self._tables_path = str(extra["tables"])
        self._mode = str(extra.get("mode", "compressed"))
        if self._mode not in ("compressed", "raw"):
            raise ValueError("mode must be compressed or raw")
        self._batch_blocks = self._positive(extra, "batch_blocks", 4)
        self._transfer_slots = self._positive(extra, "transfer_slots", 2)
        self._max_pending_requests = self._positive(extra, "max_pending_requests", 16)
        self._precheck = extra.get("precheck", True)
        if type(self._precheck) is not bool:
            raise ValueError("precheck must be boolean")
        self._messaging = RequestClientFactory.create(str(extra["endpoint"]))
        self._client = RemoteBlobClient(
            self._messaging, self.identity, timeout_s=extra.get("timeout_s", 30.0)
        )
        self._scheduler: NativeSchedulerState | None = None
        self._worker: NativeWorkerState | None = None
        self._transfer: Any | None = None
        self._pages: NativeKVPages | None = None
        self._metadata_consumed = False
        self._last_counters: dict[str, int] = {}
        self._layer_names = tuple(
            sorted(groups[0].layer_names, key=extract_layer_index)
        )
        if [extract_layer_index(name) for name in self._layer_names] != list(
            range(self.identity.num_layers)
        ):
            raise ValueError(
                "codec tables require exactly one attention per model layer"
            )
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = NativeSchedulerState(
                self.identity,
                self._client,
                page_tokens=self._page_tokens,
                max_pending_requests=self._max_pending_requests,
            )

    @staticmethod
    def _positive(options: dict[str, Any], name: str, default: int) -> int:
        value = options.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @property
    def requires_kv_delivery(self) -> bool:
        """A lost best-effort save may become a later cache miss."""
        return False

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: Any) -> str:
        """Require disjoint layer spans without changing attention computation."""
        return "LBNHC"

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register native views and allocate bounded codec/transfer workspaces."""
        from fragpack.integrations.lmcache.native_transfer import NativeBlockTransfer

        if self._role != KVConnectorRole.WORKER:
            raise RuntimeError("only the worker registers native GPU KV")
        if set(kv_caches) != set(self._layer_names):
            raise ValueError(
                "registered layer names differ from the full-attention group"
            )
        # The numerical model layer order matches the calibrated table axis;
        # lexical sorting would put layer 10 before layer 2.
        self._pages = NativeKVPages(
            [native_layer_view(kv_caches[name]) for name in self._layer_names]
        )
        with np.load(self._tables_path) as archive:
            table = np.ascontiguousarray(archive["table"])
        self._transfer = NativeBlockTransfer(
            self._pages,
            table,
            self._client,
            self.identity,
            max_blocks=self._batch_blocks,
            num_slots=self._transfer_slots,
            mode=self._mode,
            precheck=self._precheck,
        )
        device = self._pages.device
        self._worker = NativeWorkerState(
            self._transfer,
            lambda rows: torch.tensor(rows, dtype=torch.int32, device=device),
            batch_blocks=self._batch_blocks,
        )
        logger.info(
            "FRAGPACK_NATIVE_REGISTER mode=%s layers=%d page_tokens=%d namespace=%s",
            self._mode,
            self._pages.num_layers,
            self._pages.page_tokens,
            self.identity.namespace,
        )

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Bind one scheduler message; each plan is submitted at most once."""
        if not isinstance(connector_metadata, NativeConnectorMetadata):
            raise TypeError("expected native codec metadata")
        super().bind_connector_metadata(connector_metadata)
        self._metadata_consumed = False

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Submit all-layer loads/saves after the current forward's producer work."""
        if self._worker is None or self._metadata_consumed:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, NativeConnectorMetadata)
        self._worker.accept(metadata.plans, finished_requests=metadata.finished_stores)
        self._metadata_consumed = True

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No layer fence: asynchronous receives are excluded from forward."""

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any
    ) -> None:
        """Per-layer callbacks do not encode or transfer KV."""

    def wait_for_save(self) -> None:
        """Source ownership is released through completion, not a forward wait."""

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        """Poll drained receives and safe source releases, including empty steps."""
        if self._worker is None:
            return KVConnectorTransferResults()
        sent, received, failed = self._worker.poll(finished_req_ids)
        return KVConnectorTransferResults(sent, received, failed)

    def build_connector_worker_meta(self) -> NativeConnectorWorkerMetadata | None:
        """Keep the scheduler alive while released-source publications drain."""
        if self._worker is None:
            return None
        return NativeConnectorWorkerMetadata(
            self._worker.pending, dict(self._worker.counters)
        )

    @staticmethod
    def _eligible(request: Any) -> bool:
        return not (
            getattr(request, "lora_request", None)
            or getattr(request, "mm_features", None)
            or request.prompt_token_ids is None
        )

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Poll a background lookup and preserve the engine's local APC prefix."""
        assert self._scheduler is not None
        if not self._eligible(request):
            return 0, False
        return self._scheduler.lookup(
            request.request_id,
            request.prompt_token_ids,
            num_computed_tokens,
            cache_salt=request.cache_salt or "",
        )

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        """Bind only the newly allocated external suffix to a load plan."""
        assert self._scheduler is not None
        if num_external_tokens:
            groups = blocks.get_block_ids()
            if len(groups) != 1:
                raise ValueError("native codec does not support hybrid cache groups")
            self._scheduler.allocate(request.request_id, groups[0], num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> NativeConnectorMetadata:
        """Move pending load and finish-time save plans to the worker."""
        assert self._scheduler is not None
        plans = self._scheduler.take_plans()
        return NativeConnectorMetadata(
            plans, finished_store_ids(plans, scheduler_output.finished_req_ids)
        )

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Retain complete computed prompt sources after normal completion."""
        assert self._scheduler is not None
        normal = request.status.name in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")
        normal = normal and self._eligible(request)
        delayed = self._scheduler.finish(
            request.request_id,
            request.prompt_token_ids or (),
            block_ids,
            normal_finish=normal,
            computed_tokens=max(
                0,
                request.num_computed_tokens
                - getattr(request, "num_in_flight_tokens", 0),
            ),
            cache_salt=request.cache_salt or "",
        )
        return delayed, None

    def update_connector_output(self, connector_output: Any) -> None:
        """Consume worker completion and cumulative counters without GPU access."""
        assert self._scheduler is not None
        metadata = connector_output.kv_connector_worker_meta
        pending = (
            isinstance(metadata, NativeConnectorWorkerMetadata) and metadata.pending
        )
        self._scheduler.complete(
            set(connector_output.finished_sending or ()),
            set(connector_output.finished_recving or ()),
            set(connector_output.failed_recving or ()),
            worker_pending=pending,
        )
        if (
            isinstance(metadata, NativeConnectorWorkerMetadata)
            and metadata.counters != self._last_counters
        ):
            self._last_counters = metadata.counters
            logger.info("FRAGPACK_NATIVE_COUNTERS %s", json.dumps(metadata.counters))

    def has_pending_push_work(self) -> bool:
        """Progress admitted stores even when no request is generating tokens."""
        return self._scheduler is not None and self._scheduler.pending

    def has_pending_block_frees(self) -> bool:
        """Allow vLLM to await transfer progress before unnecessary preemption."""
        return self.has_pending_push_work()

    def reset_cache(self) -> bool:
        """Reset local connector hints; externally stored immutable objects remain."""
        return self._scheduler is None or self._scheduler.reset()

    def snapshot(self) -> dict[str, Any]:
        """Return CPU diagnostics for an explicit worker collective RPC.

        This is not a synchronization or drain operation. Inspect ``pending``
        before resetting GPU APC or ending a timed full-lifecycle experiment.
        """
        result: dict[str, Any] = {
            "mode": self._mode,
            "namespace": self.identity.namespace,
        }
        if self._worker is not None:
            result["pending"] = self._worker.pending
            result["counters"] = dict(self._worker.counters)
            result["pending_map_bytes"] = self._worker.pending_map_bytes
            result["retained_finish_count"] = self._worker.retained_finish_count
        elif self._scheduler is not None:
            result["pending"] = self._scheduler.pending
            result["counters"] = dict(self._last_counters)
        if self._transfer is not None:
            result["memory"] = self._transfer.memory_usage()
            result["transfer"] = self._transfer.stats()
        return result

    def shutdown(self) -> None:
        """Drain submitted operations before closing their RPC transport.

        An engine shutdown may abandon unsubmitted best-effort saves. Submitted
        CUDA consumers still drain before the engine can destroy their pages.
        """
        if self._scheduler is not None:
            self._scheduler.close()
        if self._transfer is not None:
            self._transfer.close()
        self._messaging.close()

# SPDX-License-Identifier: Apache-2.0
"""Measure a stock vLLM model's profiled KV budget before separate read replay.

This offline probe loads the actual checkpoint, executes vLLM's startup forward
profile, allocates native KV, and generates a short response. It does not run
FragPack or measure codec/serving performance. Run it under an exclusive-GPU
supervisor; all CUDA work belongs to this one process and is drained at shutdown.
The worker accounting fields are intentionally pinned to vLLM 0.30.0.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any


def profile_accounting(worker: Any) -> dict[str, Any]:
    """Read and reconcile completed startup profiling, without launching work.

    The returned budget includes vLLM's configured utilization reserve and measured
    persistent/transient non-KV costs. Model loading bytes are reported separately;
    they must not be added to ``total_consumed_bytes`` a second time.
    """
    budget = int(worker.available_kv_cache_memory_bytes)
    requested = int(worker.requested_memory)
    consumed = int(worker.total_consumed)
    activation = int(worker.peak_activation_memory)
    if min(budget, requested) <= 0 or min(consumed, activation) < 0:
        raise ValueError("invalid completed model memory profile")
    # int() can round the fractional requested-utilization byte once.
    if abs(budget - (requested - consumed - activation)) > 1:
        raise ValueError("KV budget does not reconcile with the worker profile")
    cache = worker.model_runner.kv_cache_config
    allocated = sum(int(tensor.size) for tensor in cache.kv_cache_tensors)
    if allocated <= 0 or allocated > budget:
        raise ValueError("native cache allocation exceeds the profiled KV budget")
    return {
        "available_kv_cache_memory_bytes": budget,
        "requested_memory_bytes": requested,
        "total_consumed_bytes": consumed,
        "peak_activation_memory_bytes": activation,
        "cudagraph_memory_estimate_bytes": int(worker.cudagraph_memory_estimate),
        "model_loading_memory_bytes": int(worker.model_runner.model_memory_usage),
        "allocated_native_kv_bytes": allocated,
        "num_gpu_blocks": int(cache.num_blocks),
        "initial_memory": {
            name: int(getattr(worker.init_snapshot, name))
            for name in (
                "free_memory",
                "total_memory",
                "torch_allocated",
                "torch_memory",
                "non_torch_memory",
            )
        },
        "kv_cache_groups": [
            {
                "num_layers": len(group.layer_names),
                "block_size": int(group.kv_cache_spec.block_size),
                "page_size_bytes": int(group.kv_cache_spec.page_size_bytes),
                "dtype": str(group.kv_cache_spec.dtype),
                "spec_class": type(group.kv_cache_spec).__name__,
            }
            for group in cache.kv_cache_groups
        ],
    }


def worker_snapshot(worker: Any) -> dict[str, Any]:
    """Collect scalar accounting on the worker after all stream work finishes.

    Synchronization is deliberate initialization/diagnostic work outside timed
    replay. The callable retains no model or CUDA tensor outside the worker.
    """
    import torch

    torch.cuda.synchronize()
    result = profile_accounting(worker)
    runner = worker.model_runner
    backends = sorted(
        {
            ".".join(group.backend.full_cls_name())
            for groups in runner.attn_groups
            for group in groups
        }
    )
    if not backends or any("fragpack" in name.lower() for name in backends):
        raise ValueError("budget probe requires an initialized stock attention backend")
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in runner.kv_caches
    }
    free, total = torch.cuda.mem_get_info()
    result.update(
        {
            "attention_backends": backends,
            "model_runner_class": f"{type(runner).__module__}.{type(runner).__name__}",
            "model_class": (
                f"{type(runner.model).__module__}.{type(runner.model).__name__}"
            ),
            "unique_kv_tensor_storage_bytes": sum(storages.values()),
            "gpu_name": torch.cuda.get_device_name(),
            "current_device_free_bytes": free,
            "current_device_total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "actual_dtype": str(worker.model_config.dtype),
            "actual_quantization": worker.model_config.quantization,
            "actual_kv_dtype": worker.cache_config.cache_dtype,
            "actual_gpu_memory_utilization": worker.cache_config.gpu_memory_utilization,
        }
    )
    return result


def model_provenance(model: Path) -> dict[str, Any]:
    """Record local configuration/index hashes and shard sizes without loading KV."""
    config_file = model / "config.json"
    config_bytes = config_file.read_bytes()
    config = json.loads(config_bytes)
    index_file = model / "model.safetensors.index.json"
    index_bytes = index_file.read_bytes()
    index = json.loads(index_bytes)
    shards = sorted(set(index["weight_map"].values()))
    return {
        "checkpoint_name": model.name,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "weight_files_sha256_verified": False,
        "shards": [
            {"name": name, "bytes": (model / name).stat().st_size} for name in shards
        ],
        "geometry": {
            name: config[name]
            for name in (
                "num_hidden_layers",
                "num_key_value_heads",
                "num_attention_heads",
                "head_dim",
                "hidden_size",
                "max_position_embeddings",
            )
        },
        "checkpoint_quantization": config.get("quantization_config"),
    }


def main() -> None:
    """Run one actual model profile, retain JSON evidence, and stop its engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--smoke-prompt-tokens", type=int, default=2048)
    parser.add_argument("--smoke-output-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; retain the earlier probe evidence")
    if (
        min(
            args.max_model_len,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            args.smoke_prompt_tokens,
            args.smoke_output_tokens,
        )
        <= 0
        or args.smoke_prompt_tokens + args.smoke_output_tokens > args.max_model_len
        or not 0 < args.gpu_memory_utilization < 1
    ):
        parser.error("invalid model profiling or smoke configuration")
    versions = {
        package: importlib.metadata.version(package)
        for package in ("vllm", "torch", "numpy", "triton")
    }
    if not versions["vllm"].startswith("0.30.0"):
        raise RuntimeError("worker accounting requires the verified vLLM 0.30.0 API")
    # One offline process owns all GPU memory; no server, network RPC, plugin,
    # unsafe callable serialization, or custom attention is required.
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    from vllm import LLM, SamplingParams

    kwargs = {
        "model": str(args.model),
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": 128,
        "enforce_eager": True,
        "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 3},
        "async_scheduling": True,
        "enable_prefix_caching": True,
        "generation_config": "vllm",
        "seed": args.seed,
    }
    report: dict[str, Any] = {
        "status": "started",
        "scope": "Actual stock-vLLM model load/profile and smoke; not replay timing",
        "started_unix": time.time(),
        "versions": versions,
        "model": model_provenance(args.model),
        "configuration": {
            key: value for key, value in kwargs.items() if key != "model"
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "replay_weights_resident": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    llm = None
    try:
        llm = LLM(**kwargs)
        report["after_startup"] = llm.collective_rpc(worker_snapshot, timeout=60)
        outputs = llm.generate(
            [{"prompt_token_ids": [42] * args.smoke_prompt_tokens}],
            SamplingParams(
                temperature=0,
                max_tokens=args.smoke_output_tokens,
                min_tokens=args.smoke_output_tokens,
                ignore_eos=True,
                seed=args.seed,
            ),
            use_tqdm=False,
        )
        counts = [
            len(result.token_ids) for request in outputs for result in request.outputs
        ]
        if counts != [args.smoke_output_tokens]:
            raise RuntimeError(f"smoke output length mismatch: {counts}")
        report["smoke"] = {
            "prompt_tokens": args.smoke_prompt_tokens,
            "output_token_counts": counts,
            "output_token_ids": [
                list(result.token_ids)
                for request in outputs
                for result in request.outputs
            ],
        }
        report["after_smoke"] = llm.collective_rpc(worker_snapshot, timeout=60)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown(timeout=30)
                report["engine_shutdown_returned"] = True
        except BaseException as exc:
            report["status"] = "failed"
            report["shutdown_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            report["finished_unix"] = time.time()
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            del llm
            gc.collect()


if __name__ == "__main__":
    main()

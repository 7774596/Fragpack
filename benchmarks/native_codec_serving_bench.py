# SPDX-License-Identifier: Apache-2.0
"""Run a fixed token-ID trace through a real model and native codec connector.

This is offline vLLM 0.30.0 full-model execution, not HTTP serving or attention
replay. TTFT and TPOT use vLLM's actual request metrics; complete trace timing
includes codec work, real L1 RPC/transfer, and publication drain. Model startup,
initial memory diagnostics and final shutdown are outside that timing window.
The parent starts a fresh finite CPU L1 server and selects an exclusive GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any

from fragpack.logging import init_logger

logger = init_logger(__name__)


def persist(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace the current run report, preserving valid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: Any) -> str:
    """Hash a canonical JSON value without reading model or GPU state."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_trace(
    trace: Any, *, vocab_size: int, max_model_len: int, max_num_seqs: int
) -> dict[str, Any]:
    """Validate fixed prompts and sequential batches without synthesizing data.

    ``requests`` maps stable session names to token IDs and output counts;
    ``steps`` selects those names in a phase-labelled order. Repeated names
    reuse the same immutable input. Explicit GPU-cache resets are retained as
    functional interventions, never silently inserted into a pressure trace.
    """
    if not isinstance(trace, dict) or type(trace.get("version")) is not int:
        raise ValueError("trace requires an integer version")
    if trace["version"] != 1:
        raise ValueError("unsupported trace version")
    requests, steps = trace.get("requests"), trace.get("steps")
    if not isinstance(requests, dict) or not requests:
        raise ValueError("trace requires named requests")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trace requires nonempty steps")
    normalized = {}
    for name, request in requests.items():
        if not isinstance(name, str) or not name or not isinstance(request, dict):
            raise ValueError("request names and records are malformed")
        tokens = request.get("prompt_token_ids")
        output = request.get("output_tokens", 32)
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(
                type(token) is not int or not 0 <= token < vocab_size
                for token in tokens
            )
            or type(output) is not int
            or output <= 0
            or len(tokens) + output > max_model_len
        ):
            raise ValueError(f"invalid tokens or length for request {name}")
        normalized[name] = {"prompt_token_ids": tokens, "output_tokens": output}
    normalized_steps = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("each trace step must be an object")
        names = step.get("request_ids")
        phase = step.get("phase")
        reset = step.get("reset_gpu_prefix_cache", False)
        if (
            not isinstance(names, list)
            or not 1 <= len(names) <= max_num_seqs
            or any(not isinstance(name, str) or name not in requests for name in names)
            or not isinstance(phase, str)
            or not phase
            or type(reset) is not bool
        ):
            raise ValueError("invalid step phase, request list, batch size or reset")
        normalized_steps.append(
            {"phase": phase, "request_ids": names, "reset_gpu_prefix_cache": reset}
        )
    return {"version": 1, "requests": normalized, "steps": normalized_steps}


def request_record(output: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Extract real per-request latency without mixing timestamp clock domains.

    vLLM first-token latency is a frontend duration. First/last-token timestamps
    share the engine's monotonic clock, so their difference measures decode.
    No completion wall time or TTFT is inferred from a multi-request batch.
    """
    if output.prompt_token_ids != request["prompt_token_ids"]:
        raise ValueError("model returned different prompt token IDs")
    if not output.finished or len(output.outputs) != 1:
        raise ValueError("expected one finished completion per request")
    completion = output.outputs[0]
    tokens = list(completion.token_ids)
    if len(tokens) != request["output_tokens"]:
        raise ValueError("generated token count differs from the fixed trace")
    if completion.finish_reason != "length":
        raise ValueError("request did not finish at the fixed output length")
    metrics = output.metrics
    if metrics is None:
        raise ValueError("request metrics missing; enable vLLM statistics")
    raw_metrics = (
        asdict(metrics)
        if is_dataclass(metrics)
        else {
            name: getattr(metrics, name)
            for name in (
                "num_generation_tokens",
                "num_preemptions",
                "arrival_time",
                "queued_ts",
                "scheduled_ts",
                "first_token_ts",
                "last_token_ts",
                "first_token_latency",
                "is_corrupted",
            )
        }
    )
    first, last, ttft = (
        float(raw_metrics[name])
        for name in ("first_token_ts", "last_token_ts", "first_token_latency")
    )
    if (
        not all(math.isfinite(value) for value in (first, last, ttft))
        or first <= 0
        or last < first
        or ttft <= 0
        or raw_metrics["num_generation_tokens"] != len(tokens)
        or raw_metrics["is_corrupted"]
    ):
        raise ValueError("invalid or incomplete request timing/correctness metrics")
    cached = output.num_cached_tokens
    if type(cached) is not int or not 0 <= cached <= len(output.prompt_token_ids):
        raise ValueError("invalid cached-token count")
    return {
        "engine_request_id": output.request_id,
        "prompt_tokens": len(output.prompt_token_ids),
        "prompt_sha256": digest(output.prompt_token_ids),
        "output_tokens": len(tokens),
        "output_token_ids": tokens,
        "output_text": completion.text,
        "output_sha256": digest(tokens),
        "finish_reason": completion.finish_reason,
        "cached_tokens_total": cached,
        "cached_tokens_scope": "GPU APC plus external cache; see connector snapshots",
        "ttft_ms": ttft * 1000,
        "tpot_ms": (last - first) / (len(tokens) - 1) * 1000
        if len(tokens) > 1
        else None,
        "engine_decode_ms": (last - first) * 1000,
        "engine_request_metrics": raw_metrics,
    }


def connector_snapshot(worker: Any) -> dict[str, Any]:
    """Read FragPack worker diagnostics through vLLM's public collective RPC."""
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    connector = get_kv_transfer_group()
    result = connector.snapshot()
    if not isinstance(result, dict) or type(result.get("pending")) is not bool:
        raise ValueError("native connector snapshot lacks explicit pending state")
    return result


def worker_memory_snapshot(worker: Any) -> dict[str, Any]:
    """Collect pinned-version memory/accounting outside the timed request trace."""
    import torch

    torch.cuda.synchronize()
    runner = worker.model_runner
    cache = runner.kv_cache_config
    backends = sorted(
        {
            ".".join(group.backend.full_cls_name())
            for groups in runner.attn_groups
            for group in groups
        }
    )
    if not backends or any("fragpack" in backend.lower() for backend in backends):
        raise ValueError("native codec experiment requires stock attention")
    free, total = torch.cuda.mem_get_info()
    return {
        "attention_backends": backends,
        "model_runner_class": f"{type(runner).__module__}.{type(runner).__name__}",
        "gpu_name": torch.cuda.get_device_name(),
        "device_total_bytes": total,
        "device_free_bytes": free,
        "model_loading_memory_bytes": int(runner.model_memory_usage),
        "configured_native_kv_budget_bytes": worker.cache_config.kv_cache_memory_bytes,
        "profiled_available_kv_bytes": getattr(
            worker, "available_kv_cache_memory_bytes", None
        ),
        "profiled_consumed_bytes": getattr(worker, "total_consumed", None),
        "profiled_activation_bytes": getattr(worker, "peak_activation_memory", None),
        "requested_memory_bytes": int(worker.requested_memory),
        "allocated_native_kv_bytes": sum(int(t.size) for t in cache.kv_cache_tensors),
        "num_gpu_blocks": int(cache.num_blocks),
        "kv_cache_groups": [
            {
                "layers": len(group.layer_names),
                "block_tokens": int(group.kv_cache_spec.block_size),
                "page_bytes": int(group.kv_cache_spec.page_size_bytes),
                "dtype": str(group.kv_cache_spec.dtype),
            }
            for group in cache.kv_cache_groups
        ],
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "connector": connector_snapshot(worker),
    }


def drain(llm: Any, timeout_s: float) -> list[dict[str, Any]]:
    """Step until publication drains, including idle scheduler/worker handoffs.

    In-process LLM.generate may return with finished-request store metadata
    still in the scheduler. Always step before observing worker pending state,
    then require two consecutive idle observations to deliver final completion
    notifications back to the scheduler. No future request is needed to flush.
    """
    deadline = time.monotonic() + timeout_s
    consecutive_idle = 0
    while time.monotonic() < deadline:
        extra_outputs = llm.llm_engine.step()
        if extra_outputs:
            raise RuntimeError("unexpected request outputs while draining stores")
        snapshots = llm.collective_rpc(connector_snapshot, timeout=timeout_s)
        if not snapshots:
            raise RuntimeError("worker snapshot list is empty")
        if any(snapshot["pending"] for snapshot in snapshots):
            consecutive_idle = 0
        else:
            consecutive_idle += 1
            if consecutive_idle == 2:
                return snapshots
        # Polling is measured as drain overhead. It is not a device synchronize
        # and does not prevent the bounded transfer worker from making progress.
        time.sleep(0.001)
    raise TimeoutError("native connector publications did not drain before deadline")


def metric_snapshot(llm: Any) -> list[dict[str, Any]]:
    """Retain public cumulative metrics without inventing per-request data."""
    return [asdict(metric) for metric in llm.get_metrics()]


def phase_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate observed requests; missing TPOT for one-token outputs stays absent."""
    phases: dict[str, Any] = {}
    for step in steps:
        phase = phases.setdefault(
            step["phase"],
            {
                "steps": 0,
                "requests": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "cached_tokens_total": 0,
                "generate_and_drain_wall_s": 0.0,
                "drain_wall_s": 0.0,
                "gpu_cache_resets": 0,
                "ttft_ms": [],
                "tpot_ms": [],
            },
        )
        phase["steps"] += 1
        phase["gpu_cache_resets"] += int(step["reset_gpu_prefix_cache"])
        phase["generate_and_drain_wall_s"] += step["generate_and_drain_wall_s"]
        phase["drain_wall_s"] += step["drain_wall_s"]
        for request in step["requests"]:
            phase["requests"] += 1
            for field in ("prompt_tokens", "output_tokens", "cached_tokens_total"):
                phase[field] += request[field]
            phase["ttft_ms"].append(request["ttft_ms"])
            if request["tpot_ms"] is not None:
                phase["tpot_ms"].append(request["tpot_ms"])
        phase["connector_after"] = step["connector_after"]
    for phase in phases.values():
        phase["token_hit_rate"] = phase["cached_tokens_total"] / phase["prompt_tokens"]
        phase["output_tokens_per_s_including_publication"] = (
            phase["output_tokens"] / phase["generate_and_drain_wall_s"]
        )
    return phases


def main() -> None:
    """Run one arm in one fresh model process against a caller-owned L1 server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--connector-config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "compressed"), required=True)
    parser.add_argument("--label", default="performance")
    parser.add_argument("--l1-memory-bytes", type=int, required=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--drain-timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    sidecar = args.out.with_suffix(".steps.jsonl")
    if args.out.exists() or sidecar.exists():
        parser.error("output or step sidecar exists; preserve prior evidence")
    if (
        min(
            args.l1_memory_bytes,
            args.max_model_len,
            args.max_num_seqs,
            args.max_num_batched_tokens,
        )
        <= 0
        or not 0 < args.gpu_memory_utilization < 1
        or not math.isfinite(args.drain_timeout)
        or args.drain_timeout <= 0
        or (args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes <= 0)
    ):
        parser.error("invalid memory, scheduling or timing configuration")
    model_bytes = (args.model / "config.json").read_bytes()
    model_config = json.loads(model_bytes)
    trace_bytes = args.trace.read_bytes()
    trace = validate_trace(
        json.loads(trace_bytes),
        vocab_size=model_config["vocab_size"],
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    connector_config = json.loads(args.connector_config.read_text(encoding="utf-8"))
    extra = connector_config["kv_connector_extra_config"]
    if (
        connector_config.get("kv_connector") != "FragPackNativeConnector"
        or connector_config.get("kv_connector_module_path")
        != "fragpack.integrations.vllm.native_connector"
        or extra.get("mode") != args.mode
    ):
        parser.error("connector class, module or actual codec mode differs from arm")
    if args.validate_only:
        print(
            json.dumps({"valid": True, "steps": len(trace["steps"]), "mode": args.mode})
        )
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or len(visible.split(",")) != 1:
        parser.error("parent must select one exclusively allocated GPU")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "torch", "lmcache", "numpy", "triton")
    }
    if not versions["vllm"].startswith("0.30.0") or versions["lmcache"] != "0.5.6":
        raise RuntimeError("driver requires pinned vLLM 0.30.0 and LMCache 0.5.6")
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    from vllm import LLM, SamplingParams

    kwargs: dict[str, Any] = {
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
        "disable_hybrid_kv_cache_manager": True,
        "disable_log_stats": False,
        "generation_config": "vllm",
        "kv_transfer_config": connector_config,
        "seed": args.seed,
    }
    if args.kv_cache_memory_bytes is not None:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    report: dict[str, Any] = {
        "status": "started",
        "scope": (
            "offline full-model execution with real CPU L1; "
            "no HTTP, SSD or quality benchmark"
        ),
        "timing_boundary": (
            "generate plus explicit publication drain per step; "
            "initial load/profile and final shutdown excluded"
        ),
        "label": args.label,
        "mode": args.mode,
        "started_unix": time.time(),
        "versions": versions,
        "cuda_visible_devices": visible,
        "model": {
            "name": args.model.name,
            "config_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "configuration": model_config,
            "weight_files_sha256_verified": False,
        },
        "engine_configuration": {
            key: value for key, value in kwargs.items() if key != "model"
        },
        "l1_memory_bytes": args.l1_memory_bytes,
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "trace_steps": trace["steps"],
        "requests": {
            name: {
                "prompt_tokens": len(request["prompt_token_ids"]),
                "prompt_sha256": digest(request["prompt_token_ids"]),
                "output_tokens": request["output_tokens"],
            }
            for name, request in trace["requests"].items()
        },
        "steps": [],
    }
    persist(args.out, report)
    llm = None
    try:
        llm = LLM(**kwargs)
        report["startup_memory"] = llm.collective_rpc(
            worker_memory_snapshot, timeout=60
        )
        report["metrics_before"] = metric_snapshot(llm)
        connector_before = llm.collective_rpc(connector_snapshot, timeout=60)
        persist(args.out, report)
        trace_started = time.perf_counter()
        with sidecar.open("x", encoding="utf-8") as observations:
            for index, step in enumerate(trace["steps"]):
                reset_started = time.perf_counter()
                if step["reset_gpu_prefix_cache"]:
                    if not llm.reset_prefix_cache(reset_connector=False):
                        raise RuntimeError("explicit functional GPU cache reset failed")
                reset_wall_s = time.perf_counter() - reset_started
                requests = [trace["requests"][name] for name in step["request_ids"]]
                sampling = [
                    SamplingParams(
                        temperature=0,
                        max_tokens=request["output_tokens"],
                        min_tokens=request["output_tokens"],
                        ignore_eos=True,
                        seed=args.seed,
                    )
                    for request in requests
                ]
                started = time.perf_counter()
                outputs = llm.generate(
                    [
                        {"prompt_token_ids": request["prompt_token_ids"]}
                        for request in requests
                    ],
                    sampling,
                    use_tqdm=False,
                )
                generated = time.perf_counter()
                connector_after = drain(llm, args.drain_timeout)
                finished = time.perf_counter()
                if len(outputs) != len(requests):
                    raise RuntimeError("engine returned the wrong number of requests")
                records = [
                    {"session": name, **request_record(output, request)}
                    for name, output, request in zip(
                        step["request_ids"], outputs, requests, strict=True
                    )
                ]
                observed = {
                    "index": index,
                    "phase": step["phase"],
                    "request_ids": step["request_ids"],
                    "reset_gpu_prefix_cache": step["reset_gpu_prefix_cache"],
                    "reset_wall_s": reset_wall_s,
                    "generate_wall_s": generated - started,
                    "drain_wall_s": finished - generated,
                    "generate_and_drain_wall_s": finished - started,
                    "requests": records,
                    "connector_before": connector_before,
                    "connector_after": connector_after,
                }
                report["steps"].append(observed)
                connector_before = connector_after
                observations.write(json.dumps(observed) + "\n")
                observations.flush()
                logger.info(
                    "native step=%d phase=%s requests=%d wall=%.3fs drain=%.3fs",
                    index,
                    step["phase"],
                    len(records),
                    observed["generate_and_drain_wall_s"],
                    observed["drain_wall_s"],
                )
        report["trace_wall_s"] = time.perf_counter() - trace_started
        report["timed_request_and_publication_wall_s"] = sum(
            step["generate_and_drain_wall_s"] for step in report["steps"]
        )
        report["metrics_after"] = metric_snapshot(llm)
        report["final_memory"] = llm.collective_rpc(worker_memory_snapshot, timeout=60)
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["phases"] = phase_summary(report["steps"])
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown(timeout=args.drain_timeout)
                report["engine_shutdown_returned"] = True
        except BaseException as exc:
            report.update(
                status="failed", shutdown_error=f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            report["finished_unix"] = time.time()
            report["peak_process_rss_bytes"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
            persist(args.out, report)
            del llm
            gc.collect()


if __name__ == "__main__":
    main()

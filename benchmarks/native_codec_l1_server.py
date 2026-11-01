# SPDX-License-Identifier: Apache-2.0
"""Run a finite, real LMCache 0.5.6 CPU L1 for native-codec experiments.

CUDA must be hidden from this process. The parent owns its stdin: EOF or a
``stop`` line drains and closes the server; a ``status`` line prints a snapshot.
This server uses the official module factory and StorageManager, not a fake
cache. It intentionally has no L2 adapters or persistent data directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
from typing import Any


def persist(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace this process's JSON status artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """Serve bounded CPU memory until parent EOF, retaining final status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--trigger-watermark", type=float, default=0.8)
    parser.add_argument("--eviction-ratio", type=float, default=0.2)
    parser.add_argument("--rpc-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("set CUDA_VISIBLE_DEVICES='' for this CPU-only server")
    if args.ready_file.exists():
        parser.error("ready/status file exists; preserve the earlier run")
    if (
        args.memory_bytes <= 0
        or args.memory_bytes % 4096
        or not 0 < args.trigger_watermark <= 1
        or not 0 < args.eviction_ratio <= 1
        or args.rpc_timeout <= 0
        or not 0 <= args.port <= 65535
    ):
        parser.error("invalid aligned memory budget, policy, timeout or port")

    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.ext_server_module import (
        ExtServerModuleSpec,
        build_server_module_router,
        load_server_module_components,
    )
    from lmcache.v1.multiprocess.transport.server_factory import create_request_server
    import torch

    from fragpack.integrations.lmcache.client import require_lmcache_056
    from fragpack.integrations.lmcache.identity import CacheIdentity

    require_lmcache_056()
    identity = CacheIdentity(**json.loads(args.identity.read_text(encoding="utf-8")))
    if args.memory_bytes < identity.max_blob_bytes:
        parser.error("L1 budget cannot hold one all-raw block object")
    port = args.port
    if port == 0:
        with socket.socket() as sock:
            sock.bind((args.host, 0))
            port = sock.getsockname()[1]
    report: dict[str, Any] = {
        "status": "starting",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "endpoint": f"tcp://{args.host}:{port}",
        "identity": asdict(identity),
        "namespace": identity.namespace,
        "memory_bytes": args.memory_bytes,
        "eviction": {
            "policy": "LRU",
            "trigger_watermark": args.trigger_watermark,
            "eviction_ratio": args.eviction_ratio,
        },
        "lmcache": "0.5.6",
        "storage": "CPU L1 only; no SHM, L2 or persistent store",
    }
    persist(args.ready_file, report)
    context = server = None
    modules = []

    def stop_on_signal(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"server received signal {signum}")

    signal.signal(signal.SIGTERM, stop_on_signal)

    def snapshot() -> dict[str, Any]:
        assert context is not None
        used, total = context.storage_manager.get_l1_usage()
        return {
            "used_bytes": int(used),
            "total_bytes": int(total),
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "manager": context.storage_manager.report_status(),
            "modules": [module.report_status() for module in modules],
            "gpu_initialized": torch.cuda.is_initialized(),
        }

    try:
        mp_config = MPServerConfig(host=args.host, port=port, max_cpu_workers=2)
        context = MPCacheServerContext(
            StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=args.memory_bytes, use_lazy=False, shm_name=""
                    )
                ),
                eviction_config=EvictionConfig(
                    eviction_policy="LRU",
                    trigger_watermark=args.trigger_watermark,
                    eviction_ratio=args.eviction_ratio,
                ),
            ),
            chunk_size=128,
        )
        components = load_server_module_components(
            [
                ExtServerModuleSpec(
                    "fragpack.integrations.lmcache.server",
                    config={
                        "identity": asdict(identity),
                        "timeout_s": args.rpc_timeout,
                    },
                )
            ],
            server_context=context,
            mp_config=mp_config,
            coordinator_config=CoordinatorConfig(),
            built_modules=[],
        )
        modules = list(components.modules)
        router = build_server_module_router(context, modules)
        if router is None:
            raise RuntimeError("FragPack server module router was not registered")
        server = create_request_server([router], mp_config)
        server.start()
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU L1 server unexpectedly initialized CUDA")
        report.update(status="ready", initial=snapshot())
        persist(args.ready_file, report)
        print("FRAGPACK_L1_READY " + json.dumps(report), flush=True)
        for command in sys.stdin:
            command = command.strip()
            if command == "stop":
                break
            if command == "status":
                report["latest"] = snapshot()
                persist(args.ready_file, report)
                print("FRAGPACK_L1_STATUS " + json.dumps(report["latest"]), flush=True)
            elif command:
                raise ValueError("stdin accepts only status, stop, or EOF")
        report["status"] = "closed"
    except KeyboardInterrupt as exc:
        report.update(status="interrupted", error=str(exc))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        already_raising = sys.exc_info()[0] is not None
        cleanup_errors = []
        if context is not None:
            try:
                report["final"] = snapshot()
            except BaseException as exc:
                cleanup_errors.append(f"final snapshot: {type(exc).__name__}: {exc}")
        # One failing diagnostic or close must not prevent the other owners
        # from draining. Only objects created by this process are closed here.
        for owner in [server, *modules, context]:
            if owner is not None:
                try:
                    owner.close()
                except BaseException as exc:
                    cleanup_errors.append(f"close: {type(exc).__name__}: {exc}")
        report["shutdown_complete"] = not cleanup_errors
        if cleanup_errors:
            report.update(status="failed", cleanup_errors=cleanup_errors)
        report["finished_unix"] = time.time()
        persist(args.ready_file, report)
        if cleanup_errors and not already_raising:
            raise RuntimeError(
                "CPU L1 cleanup did not complete; inspect status artifact"
            )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Run one or more lossless KV evaluation arms end to end.

Every arm starts its own services with the same GPU KV budget, block size
and scheduler limits, sends one untimed warmup request, runs one client
(the AgentX harness, the trace replay, or any harness-compatible client),
and keeps only small artifacts: vLLM metric snapshots, LMCache's final
``/status``, service logs and the client's own output. The arm's scratch
store is deleted as soon as its services stop.

Arms:
    native        vLLM alone (GPU prefix cache only), FLASH_ATTN.
    lmcache       LMCache raw: L1 (+ bounded fs_native L2).
    tier          LMCache with the lossless codec in both tiers.
    pool          FragPack attention (CUSTOM backend) holding compressed KV
                  in vLLM's GPU pool; LMCache stores its records as blobs.
    pool-native   FragPack attention without LMCache.

Run from the repository root::

    python -m benchmarks.lossless_kv.arms --arms lmcache tier ...
"""

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import argparse
import asyncio
import json
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

# Third Party
import httpx

# First Party
from benchmarks.lossless_kv.services import (
    LMCACHE_HTTP_PORT,
    LMCacheConfig,
    Services,
    VLLMConfig,
    check_runtime_environment,
    pid_running,
)

ARMS = ("native", "lmcache", "tier", "pool", "pool-native")
# FragPack allocates staging, overflow records and a decode workspace after
# vLLM sized its KV pool; CUSTOM arms give vLLM this much less KV so every
# arm uses the same GPU memory.
FRAGPACK_RESERVED_BYTES = 858 << 20
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# A dist-info shim that makes vLLM discover the FragPack plugin entry point
# when LMCache runs from a source checkout instead of an installed wheel.
PLUGIN_METADATA = HERE / "plugin_metadata"


def parse_size(text: str) -> int:
    """Parse a byte size such as ``400gib``, ``2tib`` or ``512mib``.

    Args:
        text: Integer with an optional binary unit suffix.

    Returns:
        Size in bytes.

    Raises:
        argparse.ArgumentTypeError: For an unknown format.
    """
    match = re.fullmatch(r"(\d+)\s*([kmgt]i?b)?", text.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(f"bad size {text!r}")
    unit = (match[2] or "b")[0]
    return int(match[1]) << {"b": 0, "k": 10, "m": 20, "g": 30, "t": 40}[unit]


def store_disk_usage(store_dir: Path) -> dict[str, int]:
    """Return the allocated bytes under each top-level entry of a store.

    Args:
        store_dir: Arm scratch directory.

    Returns:
        Entry name to allocated bytes, plus ``"total"``.
    """
    usage: dict[str, int] = {}
    if store_dir.is_dir():
        for entry in sorted(store_dir.iterdir()):
            paths = [entry] if entry.is_file() else entry.rglob("*")
            usage[entry.name] = sum(
                p.lstat().st_blocks * 512 for p in paths if p.is_file()
            )
    usage["total"] = sum(usage.values())
    return usage


class ArmTerminated(Exception):
    """Raised when SIGTERM/SIGINT arrives while no client is running."""


class StopSignals:
    """Turn SIGTERM/SIGINT into an orderly stop of the current arm.

    During the client run the signal is forwarded to the client, which
    writes its output and exits; the arm then tears down as usual and no
    further arm starts. Outside the client the first signal raises
    :class:`ArmTerminated` so the teardown in ``run_arm`` still runs.

    Thread-safety:
        Install and use from the main thread only.
    """

    def __init__(self) -> None:
        """Create a handler with no client attached."""
        self.requested = False
        self.child: subprocess.Popen[bytes] | None = None

    def install(self) -> None:
        """Route SIGTERM and SIGINT to :meth:`handle`."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.handle)

    def handle(self, signum: int, frame: Any) -> None:
        """Forward a stop signal to the client, or unwind into teardown.

        Args:
            signum: Received signal.
            frame: Interrupted frame (unused).

        Raises:
            ArmTerminated: On the first signal while no client runs.
        """
        del frame
        first = not self.requested
        self.requested = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)
            return
        if first:
            raise ArmTerminated(f"signal {signum}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runner arguments.

    Args:
        argv: Arguments without the program name.

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--artifact", type=Path, required=True, help="kept outputs")
    p.add_argument("--scratch", type=Path, required=True, help="stores, deleted")
    p.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES (PCI order)")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--vllm-port", type=int, default=8047)
    p.add_argument("--gpu-kv-size", type=parse_size, default="54gib")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default="auto")
    p.add_argument("--vllm-extra-args", default="", help="appended verbatim")
    p.add_argument("--l1-size", type=parse_size, default="400gib")
    p.add_argument(
        "--l2-size", type=parse_size, default="2tib", help="0 runs LMCache L1-only"
    )
    p.add_argument("--l2-workers", type=int, default=16)
    p.add_argument("--l2-serde", default=None, help="LMCache L2 serde (lmcache arm)")
    p.add_argument("--eviction-watermark", type=float, default=0.9)
    p.add_argument("--eviction-ratio", type=float, default=0.05)
    p.add_argument("--codebooks", type=Path, default=None, help="tier arm codebooks")
    p.add_argument("--codec-batch", type=int, default=None)
    p.add_argument(
        "--fragpack-tables", type=Path, default=None, help="pool arms (.npz)"
    )
    p.add_argument(
        "--lmcache-pythonpath",
        type=Path,
        nargs="*",
        default=[REPO_ROOT],
        help="PYTHONPATH entries for LMCache and its vLLM (checkout, extra deps)",
    )
    client = p.add_mutually_exclusive_group(required=True)
    client.add_argument(
        "--agentx-bin",
        help="command (shell-split) of the AgentX harness or a compatible "
        "client such as 'python slotgate.py', taking "
        "'profile --url --endpoint --model --output-artifact-dir ...'",
    )
    client.add_argument("--trace", type=Path, help="Mooncake-format trace replay")
    p.add_argument("--agentx-args", default="", help="extra harness arguments")
    p.add_argument("--max-inflight", type=int, default=8, help="trace replay")
    p.add_argument("--timeout", type=float, default=3600.0, help="trace replay")
    return p.parse_args(argv)


def _vllm_config(arm: str, args: argparse.Namespace) -> VLLMConfig:
    """Return the vLLM configuration of one arm."""
    custom = arm.startswith("pool")
    extra = ["--attention-backend", "CUSTOM" if custom else "FLASH_ATTN"]
    if args.kv_cache_dtype == "fp8":
        extra += ["--kv-cache-dtype", "fp8"]
    extra += shlex.split(args.vllm_extra_args)
    pythonpath: tuple[Path, ...] = ()
    if custom:
        pythonpath = (PLUGIN_METADATA,)
        if arm == "pool-native":
            # No LMCache server, but the plugin still lives in LMCache.
            pythonpath += tuple(args.lmcache_pythonpath)
    return VLLMConfig(
        model=args.model,
        gpus=args.gpus,
        port=args.vllm_port,
        tensor_parallel_size=args.tensor_parallel_size,
        kv_cache_bytes=args.gpu_kv_size - (FRAGPACK_RESERVED_BYTES if custom else 0),
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        extra_args=tuple(extra),
        extra_pythonpath=pythonpath,
        # Read by the FragPack backend inside the vLLM workers.
        env=(
            (("LMCACHE_FRAGPACK_TABLES", str(args.fragpack_tables)),)
            if custom and args.fragpack_tables is not None
            else ()
        ),
    )


def _lmcache_config(arm: str, args: argparse.Namespace) -> LMCacheConfig | None:
    """Return the LMCache configuration of one arm, None without LMCache."""
    if arm in ("native", "pool-native"):
        return None
    return LMCacheConfig(
        l1_bytes=args.l1_size,
        l2_bytes=args.l2_size,
        l2_workers=args.l2_workers,
        eviction_watermark=args.eviction_watermark,
        eviction_ratio=args.eviction_ratio,
        l2_serde=args.l2_serde if arm == "lmcache" else None,
        codec=arm == "tier",
        codebooks=args.codebooks if arm == "tier" else None,
        codec_batch=args.codec_batch if arm == "tier" else None,
        chunk_tokens=args.block_size,
        pythonpath=tuple(args.lmcache_pythonpath),
    )


def _client_command(
    args: argparse.Namespace, vllm: VLLMConfig, out_dir: Path
) -> list[str]:
    """Return the client argv of one arm."""
    if args.agentx_bin:
        return [
            *shlex.split(args.agentx_bin),
            "profile",
            "--url",
            vllm.url,
            "--endpoint",
            "/v1/chat/completions",
            "--model",
            vllm.served_name,
            "--output-artifact-dir",
            str(out_dir / "agentx"),
            "--ui-type",
            "simple",
            *shlex.split(args.agentx_args),
        ]
    return [
        sys.executable,
        "-m",
        "benchmarks.lossless_kv.trace_replay",
        "--trace",
        str(args.trace),
        "--tokenizer",
        args.model,
        "--url",
        vllm.url,
        "--model",
        vllm.served_name,
        "--output-dir",
        str(out_dir),
        "--max-inflight",
        str(args.max_inflight),
        "--timeout",
        str(args.timeout),
    ]


async def _text(url: str) -> str:
    """Fetch a text endpoint; an empty string on failure."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError:
        return ""


async def _warmup(vllm: VLLMConfig) -> float:
    """Send one untimed request outside the workload's token space.

    Returns:
        Warmup latency in seconds.
    """
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{vllm.url}/v1/completions",
            json={
                "model": vllm.served_name,
                "prompt": [7] * 2048,
                "max_tokens": 4,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
    return time.perf_counter() - started


async def run_arm(arm: str, args: argparse.Namespace, stop: StopSignals) -> int:
    """Run one arm and persist its artifacts.

    Services are stopped and the scratch store deleted however the arm
    ends, including a stop signal.

    Args:
        arm: Arm name.
        args: Parsed runner arguments.
        stop: Installed stop-signal handler; the client is attached to it.

    Returns:
        The client's exit code.
    """
    vllm = _vllm_config(arm, args)
    lmcache = _lmcache_config(arm, args)
    if arm.startswith("pool") and args.fragpack_tables is None:
        raise ValueError(f"{arm} needs --fragpack-tables")
    out_dir = args.artifact / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    services = Services(args.scratch / arm, vllm, lmcache)
    returncode = -1
    try:
        await services.start()
        warmup_s = await _warmup(vllm)
        (out_dir / "vllm_metrics_before.prom").write_text(
            await _text(f"{vllm.url}/metrics")
        )
        with (out_dir / "client.log").open("wb") as log:
            client = subprocess.Popen(
                _client_command(args, vllm, out_dir),
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            # The watchdog finds the client here (AgentX renames its
            # processes, so its argv cannot be matched).
            (out_dir / "client.pid").write_text(str(client.pid))
            try:
                stop.child = client
                returncode = client.wait()
            except ArmTerminated:
                client.terminate()
                client.wait()
                raise
            finally:
                stop.child = None
        (out_dir / "vllm_metrics_after.prom").write_text(
            await _text(f"{vllm.url}/metrics")
        )
        if lmcache is not None:
            (out_dir / "lmcache_status_after.json").write_text(
                await _text(f"http://127.0.0.1:{LMCACHE_HTTP_PORT}/status")
            )
        (args.artifact / f"{arm}.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "client_returncode": returncode,
                    "warmup_seconds": warmup_s,
                    "vllm_command": services.vllm_command(),
                    "lmcache_command": (
                        services.lmcache_command() if lmcache is not None else None
                    ),
                },
                indent=2,
            )
        )
        return returncode
    finally:
        for log in services.log_dir.glob("*.log"):
            shutil.copyfile(log, out_dir / log.name)
        services.stop()
        pids = (
            [int(item["pid"]) for item in json.loads(services.pid_file.read_text())]
            if services.pid_file.is_file()
            else []
        )
        deadline = time.monotonic() + 60.0
        while any(map(pid_running, pids)) and time.monotonic() < deadline:
            time.sleep(0.5)
        if any(map(pid_running, pids)):
            print(f"{arm}: services still alive; kept {services.scratch}", flush=True)
        else:
            (out_dir / "store_disk_usage.json").write_text(
                json.dumps(store_disk_usage(services.scratch))
            )
            shutil.rmtree(services.scratch, ignore_errors=True)


async def main_async(args: argparse.Namespace) -> int:
    """Run the selected arms one after another.

    Args:
        args: Parsed runner arguments.

    Returns:
        0 when every client exited 0, else 1.
    """
    args.artifact.mkdir(parents=True, exist_ok=True)
    stop = StopSignals()
    stop.install()
    failed = False
    for arm in args.arms:
        returncode = await run_arm(arm, args, stop)
        print(f"{arm}: rc={returncode}", flush=True)
        failed |= returncode != 0
        if stop.requested:
            print(f"{arm}: stopped by signal; skipping remaining arms", flush=True)
            break
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    check_runtime_environment()
    try:
        sys.exit(asyncio.run(main_async(args)))
    except ArmTerminated:
        sys.exit(128 + signal.SIGTERM)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Service lifecycle for the lossless KV evaluation arms.

Starts an LMCache MP server (L1 plus an optional bounded ``fs_native`` L2,
optionally with the lossless codec) and a vLLM server whose KV connector is
LMCache's MP connector, or vLLM alone. Every service leads its own process
group; the groups are recorded next to ``pids.json`` so a watchdog or a
parent shell can stop them even after the runner was killed.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import asyncio
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

# Third Party
import httpx

LMCACHE_MP_HOST = "tcp://localhost"
LMCACHE_MP_PORT = 5555
LMCACHE_HTTP_PORT = 8080
# A benchmark-local alias of LMCache's MP connector; vLLM's built-in
# registry entry for "LMCacheMPConnector" would load vLLM's bundled copy.
CONNECTOR_NAME = "LosslessKVBenchMPConnector"
CONNECTOR_MODULE = "benchmarks.lossless_kv.connector"
LMCACHE_CLI_MAIN = "from lmcache.cli.main import main; main()"
REPO_ROOT = Path(__file__).resolve().parents[2]
BYTES_PER_GIB = 1 << 30
MIN_VLLM_VERSION = (0, 30)


def _proc_table() -> dict[int, tuple[int, int, str]]:
    """Map every PID to ``(ppid, pgid, state)`` read from ``/proc``.

    Returns:
        PID -> (parent PID, process group ID, state letter); processes that
        exit while the table is read are skipped.
    """
    table: dict[int, tuple[int, int, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name may contain spaces or parentheses.
        fields = stat.rsplit(")", 1)[1].split()
        table[int(entry.name)] = (int(fields[1]), int(fields[2]), fields[0])
    return table


def descendant_process_groups(roots: list[int]) -> list[int]:
    """Return the process groups of ``roots`` and all their descendants.

    Args:
        roots: PIDs whose process trees are scanned.

    Returns:
        Sorted distinct PGIDs, excluding the caller's own group.
    """
    table = _proc_table()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _pgid, _state) in table.items():
        children.setdefault(ppid, []).append(pid)
    groups: set[int] = set()
    stack = [pid for pid in roots if pid in table]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        groups.add(table[pid][1])
        stack.extend(children.get(pid, []))
    groups.discard(os.getpgrp())
    return sorted(groups)


def read_process_groups(path: str | Path) -> list[int]:
    """Read a PGID file (one PGID per line).

    Args:
        path: PGID file path; a missing file yields an empty list.

    Returns:
        Recorded PGIDs in file order.
    """
    pgid_path = Path(path)
    if not pgid_path.is_file():
        return []
    return [int(line) for line in pgid_path.read_text().split() if line.isdigit()]


def pgid_file_for(pid_file: str | Path) -> Path:
    """Return the PGID file that sits next to a ``pids.json``.

    Args:
        pid_file: ``pids.json`` path.

    Returns:
        ``pgids`` path in the same directory.
    """
    return Path(pid_file).with_name("pgids")


def record_process_groups(path: str | Path, roots: list[int]) -> list[int]:
    """Merge the process groups under ``roots`` into a PGID file.

    Recorded groups are kept: a group whose leader exited may still hold
    live members that are no longer descendants of any root.

    Args:
        path: PGID file path, rewritten in place.
        roots: Service PIDs whose process trees are scanned.

    Returns:
        All recorded PGIDs after the merge.
    """
    groups = set(read_process_groups(path))
    groups.update(descendant_process_groups(roots))
    ordered = sorted(groups)
    Path(path).write_text("".join(f"{g}\n" for g in ordered))
    return ordered


def group_running(pgid: int) -> bool:
    """Return whether group ``pgid`` has a live, non-zombie member.

    Args:
        pgid: Process group ID.

    Returns:
        True if any member is alive and not a zombie.
    """
    return any(g == pgid and st != "Z" for _p, g, st in _proc_table().values())


def pid_running(pid: int) -> bool:
    """Return whether ``pid`` is alive and not a zombie.

    Args:
        pid: Process ID.

    Returns:
        True for a live process.
    """
    state = _proc_table().get(pid)
    return state is not None and state[2] != "Z"


def _signal(pid: int, sig: int) -> bool:
    """Signal a service's process group, or the PID alone if it leads none."""
    for kill in (os.killpg, os.kill):
        try:
            kill(pid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            continue
    return False


def stop_process_groups(path: str | Path, grace_s: float = 15.0) -> list[int]:
    """SIGTERM every group in a PGID file, then SIGKILL the survivors.

    Args:
        path: PGID file path.
        grace_s: Seconds between SIGTERM and SIGKILL.

    Returns:
        PGIDs that needed SIGKILL.

    Thread-safety:
        Sends process signals; call from one owner.
    """
    groups = [g for g in read_process_groups(path) if g != os.getpgrp()]
    alive = []
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
            alive.append(group)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + grace_s
    while alive and time.monotonic() < deadline:
        time.sleep(0.2)
        alive = [g for g in alive if group_running(g)]
    for group in alive:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return alive


def stop_from_pid_file(path: str | Path) -> None:
    """Stop the services listed in ``pids.json`` and their recorded groups.

    Args:
        path: ``pids.json`` path; a missing file is a no-op.

    Thread-safety:
        Sends process signals; call from one owner.
    """
    pid_path = Path(path)
    if not pid_path.is_file():
        return
    pgid_path = pgid_file_for(pid_path)
    pids = [int(item["pid"]) for item in json.loads(pid_path.read_text())]
    # Catch groups started after the last record (engine workers).
    record_process_groups(pgid_path, pids)
    for pid in reversed(pids):
        _signal(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    for pid in reversed(pids):
        while time.monotonic() < deadline and (pid_running(pid) or group_running(pid)):
            time.sleep(0.2)
        if pid_running(pid) or group_running(pid):
            _signal(pid, signal.SIGKILL)
    stop_process_groups(pgid_path)


def check_runtime_environment() -> None:
    """Fail fast when services would start from a mismatched environment.

    Checks that vLLM >= ``MIN_VLLM_VERSION`` is installed for this
    interpreter and that the ``vllm`` on ``PATH`` belongs to it.

    Raises:
        RuntimeError: If a check fails.

    Thread-safety:
        Reads interpreter metadata only.
    """
    try:
        installed = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"vLLM is not installed for {sys.executable}") from exc
    release = tuple(int(part) for part in re.findall(r"\d+", installed)[:2])
    if release < MIN_VLLM_VERSION:
        raise RuntimeError(f"vLLM {installed} is older than 0.30")
    vllm_bin = shutil.which("vllm")
    expected = Path(sys.executable).parent / "vllm"
    if vllm_bin is None or Path(vllm_bin).resolve() != expected.resolve():
        raise RuntimeError(f"vllm on PATH ({vllm_bin}) is not {expected}")


def _is_port_open(port: int) -> bool:
    """Return whether something listens on local TCP ``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _l1_total_bytes(status: Any) -> int | None:
    """Return the committed L1 size from an LMCache ``/status`` payload."""
    try:
        total = status["storage_manager"]["l1_manager"]["memory_total_bytes"]
    except (KeyError, TypeError):
        return None
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return None
    return int(total)


def _l2_ready(status: Any, num_adapters: int) -> bool:
    """Return whether ``/status`` reports ``num_adapters`` healthy L2 adapters."""
    storage = status.get("storage_manager") if isinstance(status, dict) else None
    if not isinstance(storage, dict) or storage.get("is_healthy") is not True:
        return False
    adapters = storage.get("l2_adapters")
    return (
        isinstance(adapters, list)
        and len(adapters) == num_adapters
        and all(isinstance(a, dict) and a.get("is_healthy") is True for a in adapters)
    )


@dataclass(frozen=True)
class LMCacheConfig:
    """LMCache MP server deployment of one arm.

    Attributes:
        l1_bytes: Pinned DRAM L1 capacity (rounded down to whole GiB).
        l2_bytes: Bounded ``fs_native`` NVMe L2 capacity; 0 runs L1 only.
        l2_workers: ``fs_native`` IO threads.
        eviction_watermark: L1 usage fraction that triggers eviction.
        eviction_ratio: Fraction of L1 keys evicted per trigger.
        l2_serde: LMCache L2 serde plugin (e.g. ``fp8``), or None.
        codec: Store strict-lossless compressed records in both tiers.
        codebooks: Calibrated codebook directory for the codec, or None.
        codec_batch: Chunks per codec launch, or None for the default.
        chunk_tokens: LMCache chunk size; matches vLLM's block size.
        pythonpath: Entries put first on the services' ``PYTHONPATH``.
    """

    l1_bytes: int
    l2_bytes: int = 0
    l2_workers: int = 16
    eviction_watermark: float = 0.9
    eviction_ratio: float = 0.05
    l2_serde: str | None = None
    codec: bool = False
    codebooks: Path | None = None
    codec_batch: int | None = None
    chunk_tokens: int = 128
    pythonpath: tuple[Path, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate option combinations.

        Raises:
            ValueError: On an inconsistent configuration.
        """
        if self.l2_serde and self.codec:
            raise ValueError("an L2 serde cannot transform codec records")
        if not self.codec and (self.codebooks or self.codec_batch):
            raise ValueError("codebooks and codec batch need the codec")


@dataclass(frozen=True)
class VLLMConfig:
    """vLLM server of one arm.

    Attributes:
        model: Model path.
        gpus: ``CUDA_VISIBLE_DEVICES`` value (PCI bus order).
        port: HTTP port.
        tensor_parallel_size: TP ranks.
        kv_cache_bytes: ``--kv-cache-memory-bytes`` per GPU.
        block_size: KV block size in tokens.
        max_model_len: Context limit.
        max_num_seqs: Scheduler sequence limit.
        extra_args: Extra ``vllm serve`` arguments, appended verbatim.
        extra_pythonpath: Extra ``PYTHONPATH`` entries (plugin metadata).
        env: Extra environment entries for vLLM and its workers.
    """

    model: str
    gpus: str
    port: int = 8047
    tensor_parallel_size: int = 1
    kv_cache_bytes: int | None = None
    block_size: int = 128
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    extra_args: tuple[str, ...] = ()
    extra_pythonpath: tuple[Path, ...] = ()
    env: tuple[tuple[str, str], ...] = ()

    @property
    def url(self) -> str:
        """Return the server base URL."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def served_name(self) -> str:
        """Return the served model name (the model directory name)."""
        return Path(self.model).name


class Services:
    """Start and stop the services of one arm.

    Args:
        scratch: Per-arm scratch directory (logs, pids, L2 store).
        vllm: vLLM configuration.
        lmcache: LMCache configuration, or None for vLLM alone.
        startup_timeout: Seconds each service gets to become ready.

    Thread-safety:
        Owned by one asyncio task.
    """

    def __init__(
        self,
        scratch: Path,
        vllm: VLLMConfig,
        lmcache: LMCacheConfig | None,
        startup_timeout: float = 900.0,
    ) -> None:
        """Store the configuration; nothing starts until :meth:`start`."""
        self.scratch = scratch
        self.vllm = vllm
        self.lmcache = lmcache
        self.startup_timeout = startup_timeout
        self.log_dir = scratch / "logs"
        self.pid_file = scratch / "pids.json"
        self._procs: list[subprocess.Popen[bytes]] = []

    async def start(self) -> None:
        """Start LMCache (if configured) and vLLM; return once both serve.

        Raises:
            RuntimeError: If a port is busy, a service exits, or a service
                is not ready before the startup timeout.
        """
        self.scratch.mkdir(parents=True, exist_ok=True)
        ports = [self.vllm.port]
        if self.lmcache is not None:
            ports += [LMCACHE_MP_PORT, LMCACHE_HTTP_PORT]
        busy = [p for p in ports if _is_port_open(p)]
        if busy:
            raise RuntimeError(f"service port(s) already in use: {busy}")
        if self.lmcache is not None:
            proc = self._spawn(self.lmcache_command(), "lmcache_mp_server.log")
            base = f"http://127.0.0.1:{LMCACHE_HTTP_PORT}"
            await self._wait_http(f"{base}/healthcheck", proc)
            await self._wait_lmcache_tiers(proc)
        proc = self._spawn(self.vllm_command(), "vllm.log", hash_seed=True)
        await self._wait_http(f"{self.vllm.url}/health", proc)

    def stop(self) -> None:
        """Stop every service started so far.

        Thread-safety:
            Sends process signals; call from the owner only.
        """
        stop_from_pid_file(self.pid_file)

    def lmcache_command(self) -> list[str]:
        """Return the LMCache MP server argv.

        Returns:
            argv; creates the L2 directory when an L2 is configured.
        """
        cfg = self.lmcache
        assert cfg is not None
        cmd = [
            sys.executable,
            "-c",
            LMCACHE_CLI_MAIN,
            "server",
            "--host",
            "localhost",
            "--port",
            str(LMCACHE_MP_PORT),
            "--chunk-size",
            str(cfg.chunk_tokens),
            "--max-workers",
            "4",
            "--l1-size-gb",
            str(cfg.l1_bytes // BYTES_PER_GIB),
            # Pin the whole L1 before serving; lazy growth registers host
            # memory while transfers run.
            "--no-l1-use-lazy",
            "--eviction-policy",
            "LRU",
            "--eviction-trigger-watermark",
            str(cfg.eviction_watermark),
            "--eviction-ratio",
            str(cfg.eviction_ratio),
            "--http-port",
            str(LMCACHE_HTTP_PORT),
            "--disable-metrics",
        ]
        if cfg.codec:
            cmd.append("--lossless-kv-codec")
            if cfg.codebooks is not None:
                cmd += ["--lossless-kv-codebooks", str(cfg.codebooks)]
            if cfg.codec_batch is not None:
                cmd += ["--lossless-kv-batch", str(cfg.codec_batch)]
        if cfg.l2_bytes:
            store = self.scratch / "lmcache_l2"
            store.mkdir(parents=True, exist_ok=True)
            adapter: dict[str, Any] = {
                "type": "fs_native",
                "base_path": str(store),
                "use_odirect": True,
                "num_workers": cfg.l2_workers,
                "max_capacity_gb": cfg.l2_bytes / BYTES_PER_GIB,
                # Evict late and little so nearly all of the bound is usable.
                "eviction": {
                    "eviction_policy": "LRU",
                    "trigger_watermark": 0.95,
                    "eviction_ratio": 0.05,
                },
            }
            if cfg.l2_serde:
                adapter["serde"] = {"type": cfg.l2_serde}
            cmd += ["--l2-adapter", json.dumps(adapter)]
        return cmd

    def kv_transfer_config(self) -> dict[str, Any] | None:
        """Return vLLM's ``--kv-transfer-config``, or None without LMCache."""
        if self.lmcache is None:
            return None
        return {
            "kv_connector": CONNECTOR_NAME,
            "kv_connector_module_path": CONNECTOR_MODULE,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": LMCACHE_MP_HOST,
                "lmcache.mp.port": LMCACHE_MP_PORT,
            },
        }

    def vllm_command(self) -> list[str]:
        """Return the ``vllm serve`` argv.

        Prefix caching is on, the KV budget is fixed in bytes and prompt
        token details report per-request cached tokens.

        Returns:
            argv.
        """
        cfg = self.vllm
        cmd = [
            "vllm",
            "serve",
            cfg.model,
            "--served-model-name",
            cfg.served_name,
            "--port",
            str(cfg.port),
            "--enable-prefix-caching",
            "--generation-config",
            "vllm",
            "--block-size",
            str(cfg.block_size),
            "--async-scheduling",
            "--stream-interval",
            "1",
            "--enable-prompt-tokens-details",
            "--trust-remote-code",
        ]
        if cfg.kv_cache_bytes is not None:
            cmd += ["--kv-cache-memory-bytes", str(cfg.kv_cache_bytes)]
        if cfg.max_model_len:
            cmd += ["--max-model-len", str(cfg.max_model_len)]
        if cfg.max_num_seqs:
            cmd += ["--max-num-seqs", str(cfg.max_num_seqs)]
        if cfg.tensor_parallel_size > 1:
            cmd += ["--tensor-parallel-size", str(cfg.tensor_parallel_size)]
        kv_config = self.kv_transfer_config()
        if kv_config is not None:
            cmd += ["--kv-transfer-config", json.dumps(kv_config)]
        return cmd + list(cfg.extra_args)

    def _spawn(
        self, cmd: list[str], log_name: str, hash_seed: bool = False
    ) -> subprocess.Popen[bytes]:
        """Start one service in its own session and record its groups."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = self.vllm.gpus
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        entries = [str(REPO_ROOT)]
        if self.lmcache is not None:
            entries = [*map(str, self.lmcache.pythonpath), *entries]
        entries += [str(p) for p in self.vllm.extra_pythonpath]
        if env.get("PYTHONPATH"):
            entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(entries)
        if hash_seed:
            # LMCache chunk keys hash token IDs; fix the seed across runs.
            env["PYTHONHASHSEED"] = "42"
            env.update(self.vllm.env)
        with (self.log_dir / log_name).open("wb") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self._procs.append(proc)
        self.pid_file.write_text(
            json.dumps([{"pid": p.pid, "cmd": p.args} for p in self._procs], indent=2)
        )
        record_process_groups(pgid_file_for(self.pid_file), [proc.pid])
        return proc

    async def _wait_http(self, url: str, proc: subprocess.Popen[bytes]) -> None:
        """Poll ``url`` until it answers 200 or ``proc`` exits."""
        deadline = time.monotonic() + self.startup_timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"{url}: service exited ({proc.returncode})")
                try:
                    if (await client.get(url, timeout=5.0)).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2.0)
        raise RuntimeError(f"{url} not healthy after {self.startup_timeout:.0f}s")

    async def _wait_lmcache_tiers(self, proc: subprocess.Popen[bytes]) -> None:
        """Wait until LMCache reports its full L1 and a healthy L2 adapter.

        The health endpoint answers before a non-lazy L1 is fully pinned;
        load must not start while ``cudaHostRegister`` still runs.
        """
        cfg = self.lmcache
        assert cfg is not None
        expected = cfg.l1_bytes // BYTES_PER_GIB * BYTES_PER_GIB
        adapters = 1 if cfg.l2_bytes else 0
        url = f"http://127.0.0.1:{LMCACHE_HTTP_PORT}/status"
        deadline = time.monotonic() + self.startup_timeout
        total: int | None = None
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"LMCache exited ({proc.returncode})")
                try:
                    response = await client.get(url, timeout=5.0)
                    status = response.json() if response.status_code == 200 else None
                except (httpx.HTTPError, ValueError):
                    status = None
                if status is not None:
                    total = _l1_total_bytes(status)
                    if total is not None and total >= expected:
                        if _l2_ready(status, adapters):
                            return
                await asyncio.sleep(1.0)
        raise RuntimeError(f"LMCache tiers not ready: L1 {total} of {expected} bytes")

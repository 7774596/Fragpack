# SPDX-License-Identifier: Apache-2.0
"""Watch a running arm and stop it on the first anomaly.

Runs next to ``benchmarks.lossless_kv.arms``. Silent while the arm is
healthy; rewrites ``watchdog.status`` every tick and exits with one line
when the client finishes (exit 0) or an anomaly is found (exit 1). On an
anomaly it sends SIGTERM to the client so the runner collects artifacts and
tears down as usual; service groups still alive after ``--reap-grace-s``
are then killed from the ``pgids`` record.

Checks per tick:
- failed requests in ``requests.jsonl`` (trace replay only);
- new ``Traceback``/``ERROR`` lines in the service logs;
- a stall: no prompt tokens for ``--stall-minutes`` while requests wait, or
  no prompt and no generation tokens while requests run;
- a service process exiting while the client still runs;
- free disk below the thresholds.
"""

from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import signal
import sys
import time
import urllib.request

# First Party
from benchmarks.lossless_kv.services import (
    group_running,
    pgid_file_for,
    pid_running,
    read_process_groups,
)

_METRIC = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})? ([0-9.eE+-]+)$")
_LOG_ALARM = re.compile(r"Traceback|\bERROR\b")
_CLIENT_PID_FILE = "client.pid"


def _vllm_counters(url: str) -> dict[str, float] | None:
    """Return summed vLLM counters and gauges, or None when unreachable."""
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
            text = response.read().decode()
    except OSError:
        return None
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _METRIC.match(line)
        if match:
            totals[match[1]] = totals.get(match[1], 0.0) + float(match[2])
    return totals


def _client_pid(output_dir: Path) -> int | None:
    """Return the live client PID recorded by the runner, if any."""
    try:
        pid = int((output_dir / _CLIENT_PID_FILE).read_text())
    except (FileNotFoundError, ValueError):
        return None
    return pid if pid_running(pid) else None


class _Tail:
    """Incrementally read new complete lines from a growing file."""

    def __init__(self, path: Path) -> None:
        """Start at the beginning of ``path``."""
        self.path = path
        self.offset = 0
        self.partial = b""

    def new_lines(self) -> list[str]:
        """Return complete lines appended since the previous call."""
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        return [line.decode(errors="replace") for line in lines]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse watchdog arguments.

    Args:
        argv: Arguments without the program name.

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True, help="<artifact>/<arm>")
    p.add_argument("--service-dir", type=Path, required=True, help="<scratch>/<arm>")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8047")
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--stall-minutes", type=float, default=20.0)
    p.add_argument("--data-path", default=".", help="filesystem of the stores")
    p.add_argument("--min-free-data-gib", type=float, default=500.0)
    p.add_argument("--min-free-root-gib", type=float, default=2.0)
    p.add_argument("--startup-timeout", type=float, default=1800.0)
    p.add_argument("--reap-grace-s", type=float, default=180.0)
    p.add_argument("--no-stop", action="store_true", help="report only")
    return p.parse_args(argv)


def watch(args: argparse.Namespace) -> tuple[bool, str]:
    """Run the watch loop.

    Args:
        args: Parsed watchdog arguments.

    Returns:
        ``(ok, message)``; ``ok`` is False for an anomaly.
    """
    deadline = time.monotonic() + args.startup_timeout
    pid = _client_pid(args.output_dir)
    while pid is None:
        if time.monotonic() > deadline:
            return False, "client never started"
        time.sleep(args.interval / 4)
        pid = _client_pid(args.output_dir)
    requests = _Tail(args.output_dir / "requests.jsonl")
    logs: dict[Path, _Tail] = {}
    done = failures = stall = 0
    stall_limit = max(1, round(args.stall_minutes * 60 / args.interval))
    prev = _vllm_counters(args.vllm_url)
    started = time.monotonic()
    while True:
        time.sleep(args.interval)
        for line in requests.new_lines():
            if line:
                failed = json.loads(line).get("error") is not None
                failures += failed
                done += not failed
        status = (
            f"pid={pid} elapsed={time.monotonic() - started:.0f}s "
            f"done={done} failed={failures} stall_ticks={stall}"
        )
        (args.output_dir / "watchdog.status").write_text(status + "\n")
        if not pid_running(pid):
            return True, f"client exited; {status}"
        if failures:
            return False, f"{failures} failed request(s); {status}"
        for log in (args.service_dir / "logs").glob("*.log"):
            for line in logs.setdefault(log, _Tail(log)).new_lines():
                if _LOG_ALARM.search(line):
                    return False, f"{log.name}: {line.strip()[:300]}"
        pid_file = args.service_dir / "pids.json"
        if pid_file.is_file():
            for item in json.loads(pid_file.read_text()):
                service = int(item["pid"])
                if not pid_running(service) and not group_running(service):
                    return False, f"service {item['cmd'][:2]} exited; {status}"
        free_data = shutil.disk_usage(args.data_path).free / 2**30
        free_root = shutil.disk_usage("/").free / 2**30
        if free_data < args.min_free_data_gib or free_root < args.min_free_root_gib:
            return False, f"low disk: data={free_data:.0f}G root={free_root:.0f}G"
        cur = _vllm_counters(args.vllm_url)
        if cur is not None and prev is not None:
            prompt = cur.get("vllm:prompt_tokens_total", 0) - prev.get(
                "vllm:prompt_tokens_total", 0
            )
            gen = cur.get("vllm:generation_tokens_total", 0) - prev.get(
                "vllm:generation_tokens_total", 0
            )
            live = cur.get("vllm:num_requests_running", 0)
            waiting = cur.get("vllm:num_requests_waiting", 0)
            stalled = prompt == 0 and (waiting > 0 or (live > 0 and gen == 0))
            stall = stall + 1 if stalled else 0
            if stall >= stall_limit:
                return False, (
                    f"stall: no prompt tokens for {stall} ticks "
                    f"(running={live:.0f} waiting={waiting:.0f}); {status}"
                )
        prev = cur


def _reap_after_teardown(args: argparse.Namespace) -> None:
    """Kill service groups the runner's own teardown left behind."""
    # Read now: the runner deletes the scratch dir after its teardown.
    groups = read_process_groups(pgid_file_for(args.service_dir / "pids.json"))
    deadline = time.monotonic() + args.reap_grace_s
    while groups and time.monotonic() < deadline:
        groups = [g for g in groups if group_running(g)]
        time.sleep(2.0)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if groups:
        print(f"watchdog: SIGKILL leftover service groups {groups}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    ok, message = watch(args)
    print(f"watchdog: {'done' if ok else 'ANOMALY'}: {message}", flush=True)
    if not ok and not args.no_stop:
        pid = _client_pid(args.output_dir)
        if pid is not None:
            os.kill(pid, signal.SIGTERM)
            print(f"watchdog: sent SIGTERM to client pid {pid}", flush=True)
        _reap_after_teardown(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

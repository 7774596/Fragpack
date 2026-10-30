# SPDX-License-Identifier: Apache-2.0
"""Opt-in GPU -> real stock LMCache CPU L1 -> GPU functional smoke.

Run only on an explicitly authorized, idle GPU, with LMCache 0.5.6 installed.
The CPU server reuses lmcache_plugin_smoke.py and runs in a child with CUDA
hidden. This exercises compressed record transfer, not a serving workload.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import runpy
import subprocess
import sys
import threading


def main() -> None:
    """Join the real GPU transfer and server-module RPC validation paths."""
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

    from fragpack.integrations.lmcache.client import RemoteBlobClient

    tests = Path(__file__).resolve().parents[1]
    case = runpy.run_path(str(tests / "gpu/test_fragpack_lmcache_transfer.py"))
    identity, run = case["make_remote_roundtrip"]()
    process = subprocess.Popen(
        [
            sys.executable,
            str(tests / "integration/lmcache_plugin_smoke.py"),
            "--serve-identity",
            json.dumps(asdict(identity)),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    ready = queue.Queue()

    def read_server() -> None:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("FRAGPACK_SERVER_READY "):
                ready.put(json.loads(line.split(" ", 1)[1])["endpoint"])
        ready.put(None)

    reader = threading.Thread(target=read_server, daemon=True)
    reader.start()
    request_client = None
    try:
        endpoint = ready.get(timeout=60)
        if endpoint is None:
            raise RuntimeError("LMCache server exited before becoming ready")
        request_client = RequestClientFactory.create(endpoint)
        result = run(RemoteBlobClient(request_client, identity))
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "lmcache": "0.5.6",
                    "path": "GPU -> real ZMQ -> stock CPU L1 -> GPU",
                    **result,
                }
            ),
            flush=True,
        )
    finally:
        if request_client is not None:
            request_client.close()
        process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            raise RuntimeError("LMCache server did not finish shutdown") from None
        finally:
            reader.join(timeout=5)
            process.stdout.close()
        if process.returncode:
            raise RuntimeError(f"LMCache server exited with code {process.returncode}")


if __name__ == "__main__":
    main()

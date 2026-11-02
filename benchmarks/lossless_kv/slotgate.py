# SPDX-License-Identifier: Apache-2.0
"""Client-side request slots for the AgentX harness (closed-loop admission).

Drop-in ``--agentx-bin`` (run it with the AgentX environment's Python):
starts an HTTP proxy in front of the vLLM URL that admits at most ``SLOTS``
chat-completion requests at a time (FIFO), rewrites ``--url`` to the proxy
and runs the real ``agentx`` as a child. A request waits in the proxy until a
slot frees, like ``vllm bench serve --max-concurrency``; the engine then sees
at most ``SLOTS`` requests.

Per request it logs arrival / admit / first byte / end (ns) keyed by
``X-Request-ID`` to ``<output-artifact-dir>/slotgate.jsonl``, so engine TTFT =
aiperf TTFT - (admit - arrival).

Env: SLOTS (required), AGENTX_REAL (default: agentx next to this python).
"""

# Standard
from collections.abc import Callable
import asyncio
import json
import os
import signal
import socket
import sys
import time

# Third Party
from aiohttp import web
import aiohttp

HOP = {"host", "content-length", "transfer-encoding", "connection"}


def _free_port() -> int:
    """Return an unused local TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_app(
    upstream: str, slots: int, log: Callable[[dict], None]
) -> web.Application:
    """Build the gating proxy app.

    Args:
        upstream: vLLM base URL.
        slots: Max concurrently admitted chat-completion requests.
        log: Sink for one record per gated request.

    Returns:
        aiohttp application.
    """
    sem = asyncio.Semaphore(slots)
    state: dict = {}

    async def on_start(app: web.Application) -> None:
        """Open the upstream session."""
        state["s"] = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None),
        )

    async def on_stop(app: web.Application) -> None:
        """Close the upstream session."""
        await state["s"].close()

    async def handle(req: web.Request) -> web.StreamResponse:
        """Proxy one request, holding a slot for chat/completions POSTs."""
        gated = req.method == "POST" and req.path.endswith("/completions")
        body = await req.read()
        rec = {"id": req.headers.get("X-Request-ID"), "arrive": time.time_ns()}
        if gated:
            await sem.acquire()
        rec["admit"] = time.time_ns()
        resp = None
        try:
            headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
            async with state["s"].request(
                req.method, upstream + req.path_qs, data=body, headers=headers
            ) as up:
                resp = web.StreamResponse(status=up.status)
                for k, v in up.headers.items():
                    if k.lower() not in HOP and k.lower() != "content-encoding":
                        resp.headers[k] = v
                await resp.prepare(req)
                async for chunk in up.content.iter_any():
                    if "first" not in rec:
                        rec["first"] = time.time_ns()
                    await resp.write(chunk)
                await resp.write_eof()
        finally:
            if gated:
                sem.release()
                rec["end"] = time.time_ns()
                log(rec)
        return resp

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


async def _main(argv: list[str]) -> int:
    """Run the proxy and the real harness; return the harness exit code."""
    slots = int(os.environ["SLOTS"])
    real = os.environ.get(
        "AGENTX_REAL", os.path.join(os.path.dirname(sys.executable), "agentx")
    )
    i = argv.index("--url")
    upstream = argv[i + 1].rstrip("/")
    out = argv[argv.index("--output-artifact-dir") + 1]
    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, "slotgate.jsonl"), "a", buffering=1)
    port = _free_port()
    runner = web.AppRunner(
        _make_app(upstream, slots, lambda r: f.write(json.dumps(r) + "\n")),
        access_log=None,
    )
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    argv = argv[: i + 1] + [f"http://127.0.0.1:{port}"] + argv[i + 2 :]
    child = await asyncio.create_subprocess_exec(real, *argv)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, child.send_signal, sig)
    rc = await child.wait()
    await runner.cleanup()
    f.close()
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))

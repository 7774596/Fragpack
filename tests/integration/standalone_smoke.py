# SPDX-License-Identifier: Apache-2.0
"""Exercise an already running standalone vLLM service using only the stdlib.

The service must allow at least 2326 total tokens. The client submits up to three
requests concurrently; actual engine scheduling remains controlled by the server.
To require a repeated-prefix cache hit, start it with APC and prompt-token usage
details enabled, then pass --require-apc-hit. This client never starts a service,
changes server configuration, or measures performance. It checks serving behavior,
not bit-exact KV reconstruction or equivalence to an uncompressed model.
Use --require-uncached-prefill-over 2048 to require the long request's reported
uncached input to exceed a server batch-token limit of 2048.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send JSON and retain HTTP errors as reviewable response records.

    Args:
        url (str): Endpoint URL.
        payload (dict[str, Any]): JSON request body.
        timeout (float): Socket timeout in seconds.

    Returns:
        dict[str, Any]: HTTP status and decoded body, or a transport error.

    Async/thread-safety:
        Blocking CLI I/O; each invocation owns its request and response objects.
        Safe for independent calls from the client's worker threads.
    """
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        try:
            response = urlopen(request, timeout=timeout)
        except HTTPError as exc:
            response = exc
        with response:
            status = response.code
            text = response.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = {"unparsed_body": text}
        return {"http_status": status, "body": body}
    except (OSError, ValueError) as exc:
        return {"http_status": None, "error": f"{type(exc).__name__}: {exc}"}


def complete(
    root_url: str,
    model: str,
    case: str,
    prompt: list[int],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    """Run one completion and check response status, finish, and exact lengths.

    Args:
        root_url (str): Service URL without a trailing /v1.
        model (str): Served model identifier.
        case (str): Stable case label for the JSON artifact.
        prompt (list[int]): Fixed token IDs obtained from this service's tokenizer.
        max_tokens (int): Required generated-token count; EOS stopping is disabled.
        timeout (float): Per-request socket timeout in seconds.

    Returns:
        dict[str, Any]: Request, response, individual checks and overall validity.

    Async/thread-safety:
        Blocking CLI I/O; no shared mutable state, suitable for worker threads.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
    }
    response = post_json(root_url + "/v1/completions", payload, timeout)
    body = response.get("body")
    body = body if isinstance(body, dict) else {}
    choices = body.get("choices")
    choices = choices if isinstance(choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    text = choice.get("text")
    checks = {
        "http_200": response["http_status"] == 200,
        "one_choice": len(choices) == 1,
        "finished_at_requested_length": choice.get("finish_reason") == "length",
        "exact_prompt_tokens": usage.get("prompt_tokens") == len(prompt),
        "exact_completion_tokens": usage.get("completion_tokens") == max_tokens,
        "nonempty_text": isinstance(text, str) and bool(text.strip()),
    }
    return {
        "case": case,
        "request": payload,
        "response": response,
        "cached_tokens": cached_tokens,
        "checks": checks,
        "valid": all(checks.values()),
    }


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """Populate the report with sequential and concurrently submitted requests.

    Args:
        args (argparse.Namespace): Parsed command-line options.
        report (dict[str, Any]): Mutable report persisted by the CLI entry point.

    Returns:
        None; failed requests remain recorded and make the report invalid.

    Async/thread-safety:
        Synchronous CLI orchestration. Three independent HTTP calls run in threads;
        only the calling thread updates the report after collecting their results.
    """
    root_url = args.base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]
    # Use varied natural text rather than a repeated token, which can produce
    # pathological KV distributions unrelated to normal serving behavior.
    text = "Research notebook for the following completion task.\n" + "\n".join(
        f"Record {i}: The assistant compared the observations, retained their "
        "context, and described how the project changed after the latest review."
        for i in range(120)
    )
    tokenization = post_json(
        root_url + "/tokenize",
        {"model": args.model, "prompt": text, "add_special_tokens": False},
        args.timeout,
    )
    report["tokenization"] = tokenization
    body = tokenization.get("body", {})
    tokens = body.get("tokens") if isinstance(body, dict) else None
    if (
        tokenization["http_status"] != 200
        or not isinstance(tokens, list)
        or len(tokens) < 2309
        or not all(type(token) is int and token >= 0 for token in tokens)
    ):
        raise RuntimeError("/tokenize did not return at least 2309 valid token IDs")
    context_limit = body.get("max_model_len")
    if isinstance(context_limit, int) and context_limit < 2326:
        raise RuntimeError("The service needs max_model_len >= 2326 for these cases")

    def send(case: str, length: int, generated: int) -> dict[str, Any]:
        return complete(
            root_url, args.model, case, tokens[:length], generated, args.timeout
        )

    results = report["results"]
    # Submit the long prompt first so this client has not already warmed its
    # prefix. A reused server can still have it cached; validate usage explicitly
    # when a caller requires an uncached length above the server's batch limit.
    prefill = send("prefill_over_2048_tokens", 2309, 17)
    results.append(prefill)
    prefill_cached = prefill["cached_tokens"]
    uncached = (
        2309 - prefill_cached
        if type(prefill_cached) is int and 0 <= prefill_cached <= 2309
        else None
    )
    threshold = args.require_uncached_prefill_over
    prefill_requirement_met = threshold is None or (
        uncached is not None and uncached > threshold
    )
    report["prefill"] = {
        "prompt_tokens": 2309,
        "cached_tokens": prefill_cached,
        "uncached_tokens": uncached,
        "required_uncached_tokens_over": threshold,
        "requirement_met": prefill_requirement_met,
        "scope": (
            "Uncached input is derived from reported usage. The client does not "
            "inspect the server's batch-token limit or scheduler execution."
        ),
    }
    # 259 is deliberately greater than 128 and unaligned. Generating 137 tokens
    # crosses a block boundary in decode even without counting the prompt tail.
    results.append(send("unaligned_prompt_long_decode", 259, 137))
    repeated = send("repeat_259_token_prefix", 259, 33)
    results.append(repeated)
    cases = [
        ("concurrent_short", 131, 33),
        ("concurrent_shared_prefix_medium", 389, 65),
        ("concurrent_shared_prefix_long", 519, 137),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(send, *case) for case in cases]
        results.extend(future.result() for future in pending)
    results.append(send("repeat_519_token_prefix", 519, 33))

    cached = repeated["cached_tokens"]
    observed_hit = type(cached) is int and cached > 0
    report["prefix_cache"] = {
        "required": args.require_apc_hit,
        "repeat_259_cached_tokens": cached,
        "positive_cached_tokens_observed": observed_hit,
        "scope": (
            "Usage reports cached tokens; APC attribution assumes this is the "
            "requested standalone service without an external KV connector. "
            "Missing usage details mean unknown, not zero."
        ),
    }
    report["valid"] = (
        all(row["valid"] for row in results)
        and (observed_hit or not args.require_apc_hit)
        and prefill_requirement_met
    )


def main() -> int:
    """Parse CLI arguments, persist a JSON report and return success or failure.

    Returns:
        int: Zero when all requested checks pass, otherwise one.

    Async/thread-safety:
        Synchronous CLI entry point; owns the output file and orchestrator.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Service root or /v1 URL")
    parser.add_argument("--model", required=True, help="Served model identifier")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="Socket timeout (s)"
    )
    parser.add_argument(
        "--require-apc-hit",
        action="store_true",
        help="Require positive cached_tokens on the sequential repeated prefix",
    )
    parser.add_argument(
        "--require-uncached-prefill-over",
        type=int,
        default=None,
        help="Require known uncached tokens strictly above this count on long prefill",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if (
        args.require_uncached_prefill_over is not None
        and args.require_uncached_prefill_over < 0
    ):
        parser.error("--require-uncached-prefill-over must be nonnegative")
    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Functional standalone serving smoke; no performance conclusions.",
        "arguments": {**vars(args), "output": str(args.output)},
        "valid": False,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args, report)
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"valid={report['valid']} report={args.output}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

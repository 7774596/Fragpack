# SPDX-License-Identifier: Apache-2.0
"""Poll vLLM ``/metrics`` and append queue, KV usage and TTFT-split rows.

Usage: ``python -m benchmarks.lossless_kv.metrics_sampler OUT.tsv [URL] [S]``.
Exits once the server has been unreachable for 10 minutes.
"""

# Standard
import sys
import time
import urllib.request

KEYS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:kv_cache_usage_perc",
    # Engine-side TTFT split (cumulative): queue = arrival to first schedule.
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum",
)


def sample(url: str) -> dict[str, float] | None:
    """Return the tracked metrics summed over label sets.

    Args:
        url: vLLM base URL.

    Returns:
        Metric name to value, or None when the server is unreachable.
    """
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=5) as resp:
            text = resp.read().decode()
    except OSError:
        return None
    out = dict.fromkeys(KEYS, 0.0)
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in out and not line.startswith("#"):
            out[name] += float(line.rsplit(" ", 1)[1])
    return out


def main() -> None:
    """Append one TSV row per interval until the server stays unreachable."""
    path = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8047"
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    missing_since = time.monotonic()
    with open(path, "a") as f:
        f.write("t\t" + "\t".join(k.split(":")[1] for k in KEYS) + "\n")
        while time.monotonic() - missing_since < 600:
            row = sample(url)
            if row is not None:
                missing_since = time.monotonic()
                values = "\t".join(f"{row[k]:g}" for k in KEYS)
                f.write(f"{time.time():.1f}\t{values}\n")
                f.flush()
            time.sleep(interval)


if __name__ == "__main__":
    main()

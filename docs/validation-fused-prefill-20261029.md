# Direct fused prefill: 2026-10-29

Direct prefill works without a decoded BF16 KV workspace, but this prototype
is slower than gather plus FlashAttention 3 in all nine CUDA Graph cases.
Keep `FRAGPACK_PREFILL=gather` as the default; `fused` is an experimental
startup option, not a performance recommendation.

## Configuration and evidence

All GPU runs used an exclusively available NVIDIA H800 (physical GPU 4),
Python 3.10.20, PyTorch 2.13.0+cu129, vLLM 0.30.0+cu129 and CUDA toolkit 12.9.
The source was the uncommitted `perf/fused-prefill` branch based on
`d40ef9a8d03ec5d98b2b7a4207a44b87d220b033`. Each archived run includes source
copies, hashes, configuration, logs and GPU ownership monitoring.

The measured kernel source SHA-256 is
`7cbf56ec57fc8140b0054e33f8332a96f758b4d6c88313db24fe4fa97bac829e`.
After the timing and serving runs, a host-side check was added to reject query
views with an unaligned data pointer; the device computation was unchanged.
The later correctness run also covers this rejection and empty request rows.

The experiment archive identifier is `20261029-fused-prefill-gpu4`.
Large raw artifacts remain outside the repository. `bench-01` contains eager
timing and `bench-02` CUDA Graph timing, including every sample. `tests-01`
contains the initial correctness run and compiled resource report;
`tests-02` contains the final correctness run, and `serve-fused-02` the
successful model-serving check.

## Functional coverage

The final run passed 43 CPU tests, 18 existing GPU tests and 15 new fused
prefill GPU tests. These cover compressed and raw overflow records, unfinished
tails poisoned with NaNs, right-aligned causal attention, query and block-table
strides, GQA groups 1/2/4/8, empty request rows, unaligned query rejection,
cold prefill and direct-kernel CUDA Graph replay.
Attention outputs are checked numerically against an independent FP32
reference; this does not assert bitwise-identical attention arithmetic.

The TP=1 eager serving check passed seven requests using
Qwen3-30B-A3B-Instruct-2507-FP8, **BF16 KV**, a 4,294,967,296-byte KV arena,
8,192 maximum context, four concurrent sequences, 2,048 batched-token limit,
128-token blocks, APC and async scheduling. The model's FP8 suffix refers to
weights, not KV precision. Observed cases include:

- A cold 2,309-token prompt, above the configured per-step token budget.
- Unaligned prompts and generation crossing block boundaries.
- Three concurrent requests and repeated prefixes with 256/512 cached tokens.

The validation process replaced `HbmKvStore.gather` with a function that raises
on any call and asserted zero `prefill_workspace` bytes in the published
capacity snapshot. All seven requests succeeded with these guards active.
DaseR imports were forbidden in all validation processes. This is a functional
check; no serving TTFT comparison or generation-quality claim follows from it.

## Timing method

The benchmark is one layer, one request, four KV heads, GQA=8, head size 128
and 128-token blocks. It uses synthetic normal BF16 KV with independently
sampled calibration data. All three arms share the same queries and physical
page mapping. Raw FA3 reads the original BF16 KV; gather + FA3 materializes
the same FragPack records consumed by the fused arm.

Each arm warms up for at least 20 ms, then runs ten rounds of five calls,
alternating forward and reverse arm order. Encoding, setup and validation are
outside timing. Pages are repeatedly reused without a hardware-cache flush;
these are warm-cache measurements, not cold-HBM results. CUDA Graph captures
five calls per replay; eager CUDA-event timing also includes host launch gaps.

All 512 encoded records compressed successfully with no fallbacks, spills,
staging errors or overflow errors. The largest absolute attention difference
from raw FA3 was 0.00048828125, below the recorded 0.02 absolute/relative test
tolerances. The raw tensors and both comparison workspaces coexist in this
controlled benchmark; its allocation total is not a serving-capacity result.

## CUDA Graph single-layer timing

Median microseconds per call; a ratio above 1 means fused is slower.

| Context | Query | Raw FA3 | Gather + FA3 | Fused | Fused / gather |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 15.08 | 18.68 | 47.15 | 2.52 |
| 1024 | 128 | 19.09 | 22.93 | 49.30 | 2.15 |
| 1024 | 512 | 26.84 | 31.80 | 125.72 | 3.95 |
| 4096 | 16 | 18.20 | 24.12 | 170.45 | 7.07 |
| 4096 | 128 | 33.44 | 40.69 | 207.60 | 5.10 |
| 4096 | 512 | 81.04 | 92.87 | 475.87 | 5.12 |
| 8192 | 16 | 21.46 | 33.34 | 334.96 | 10.05 |
| 8192 | 128 | 56.67 | 70.92 | 394.40 | 5.56 |
| 8192 | 512 | 157.86 | 174.04 | 886.14 | 5.09 |

## Eager single-layer timing

Median microseconds per call; a ratio above 1 means fused is slower.

| Context | Query | Raw FA3 | Gather + FA3 | Fused | Fused / gather |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 44.16 | 57.04 | 49.48 | 0.87 |
| 1024 | 128 | 43.62 | 56.38 | 50.22 | 0.89 |
| 1024 | 512 | 37.37 | 49.44 | 106.90 | 2.16 |
| 4096 | 16 | 44.44 | 56.84 | 171.26 | 3.01 |
| 4096 | 128 | 45.96 | 58.07 | 183.57 | 3.16 |
| 4096 | 512 | 91.31 | 106.99 | 443.04 | 4.14 |
| 8192 | 16 | 44.41 | 56.78 | 333.94 | 5.88 |
| 8192 | 128 | 63.39 | 77.87 | 406.72 | 5.22 |
| 8192 | 512 | 167.43 | 182.27 | 884.95 | 4.86 |

## Workspace and interpretation

For an 8,192-token context with four KV heads, the eliminated global BF16 K/V
workspace is 16,777,216 bytes (about 16.8 MB). This is one shared prefill
workspace, not that amount multiplied by model layers. New KV still uses the
store's existing writable staging, and store scratch, overflow, metadata and
decode-attention workspace remain.

The compiled prefill kernel uses 183 registers per thread, 1,056 bytes of
static shared memory and 120,416 bytes of dynamic shared memory per CTA.
The resource dump reports zero local memory and zero stack bytes. At GQA=8,
each CTA covers eight query tokens and scans its history serially; the
16-query, four-head benchmark therefore exposes only eight CTAs. These facts
suggest insufficient parallelism for short suffixes and restricted CTA
residency. They are design/resource deductions, not a profiling attribution
of elapsed time.

Two small eager cases improve by about 11–13%, but both regress under CUDA
Graph. Removing launches therefore does not establish a faster GPU kernel.
A follow-up design should consider splitting long histories across CTAs and
merging small attention partials, then revisiting query tiling and shared-memory
use. It can still avoid writing decoded KV to global memory. Those changes
are not implemented in this prototype.

Full-model CUDA Graph serving, TP>1, a real-workload performance matrix and
end-to-end TTFT gains remain unverified.

## Failed checks and cleanup

The first serving attempt, `serve-fused-01`, failed before model execution:
the generated UNIX socket path exceeded the 107-character limit. The local
validation launcher was changed to use a shorter RPC directory, after which
`serve-fused-02` passed. No third-party source was modified.

Compute Sanitizer memcheck is **not clean**. With CUDA 12.9.1 / Sanitizer
2025.2.1.0 and driver 575.57.08, each of three attempts reported one
`CUDA_ERROR_INVALID_HANDLE` from `cuKernelGetFunction`, reached through the
existing `hbm_prepare` launch. The tool returned exit code 42 even though the
selected pytest case passed:

| Archive | Selected check | Outcome |
| --- | --- | --- |
| `memcheck-01` | Mixed-record GQA=8 fused prefill | One CUDA API error |
| `memcheck-02` | Same case, both CUDA module loading settings forced to `EAGER` | Same CUDA API error |
| `memcheck-03` | Existing store tail-inheritance test; no fused-prefill invocation | Same CUDA API error |

No invalid device read/write was reported. API error reporting remained
enabled with no suppressions. Reproduction in the store-only test narrows the
issue, but does not establish a harmless tool false positive. Sanitizer
validation therefore remains unresolved; the functional passes above are a
separate result. Logs and all failed attempts are retained.

The successful vLLM run also emitted a multiprocessing resource-tracker warning
about one semaphore during shutdown. The launcher stopped its owned process
groups, verified no owned GPU process remained, and removed its RPC directory,
per-run temporary files and shared compiler caches. The final GPU-4 process
list was empty. Benchmark results, source snapshots and logs were preserved.

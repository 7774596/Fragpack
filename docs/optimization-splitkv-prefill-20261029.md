# Split-KV fused prefill: 2026-10-29

Split-KV removes most of the short-suffix regression identified in the
[profiling report](profiling-fused-prefill-20261029.md). At context 8192 / query
16, the final CUDA Graph run reduces fused attention from 330.38 to 34.06 us,
a 9.70x improvement within the same four-arm experiment. Gather + FA3 remains
slightly faster at 32.52 us. Long-query performance remains a limitation, so
`FRAGPACK_PREFILL=gather` stays the default.

## Change and scope

The experimental fused path partitions each query tile's visible history,
computes independent softmax/output partials, then combines them. One-split
calls bypass both partial storage and combination. The automatic host planner
uses existing query/context metadata and targets one wave of active CTAs,
with at most 32 splits and no more splits than pages in the longest active
context. It performs no GPU metadata readback or hot-path allocation.

The representation, codec, open-block staging, overflow behavior, write/encode
ordering and third-party sources are unchanged. Reconstructed KV still never
passes through a global BF16 prefill workspace. A separate FP32 attention
workspace is owned by the adapter and reused across layers on the same stream.
Its fixed H800 allocation is **4,392,960 bytes**. The previous BF16 workspace at
8192 tokens / four KV heads was 16,777,216 bytes; the difference is 12,384,256
bytes for that workspace pair, not a whole-model or whole-store capacity claim.

The native source SHA-256 is
`cf55c15c9de76a92dbc576141edcefa71c65e22d390fd89fb7a64b11fc465787`.
This work remains on the uncommitted `perf/fused-prefill` branch based on
`d40ef9a8d03ec5d98b2b7a4207a44b87d220b033`.

## Validation and measurement

All completed GPU runs used exclusive physical GPU 4, NVIDIA H800 with 132 SMs.
Environment: Python 3.10.20, PyTorch 2.13.0+cu129, vLLM 0.30.0+cu129 and CUDA
12.9. No GPU clocks were modified. The existing source was snapshotted before
editing and rerun on this same card. Each experiment preserves its actual
source, hashes, arguments, full samples and GPU process monitoring.

- 36 fused-prefill GPU tests and 18 existing GPU regression tests passed.
  Tests cover raw/compressed KV, GQA 1/2/4/8, strided tensors, causal tails,
  more splits than pages, empty requests and poisoned partial buffers.
  Graph replay changes query offsets/context lengths while reusing buffers.
- 50 CPU tests passed, including the automatic planner's partial-capacity bound
  across ragged batches. The final planner adjustment was CPU-tested again.
- Seven TP=1 eager model requests passed with gather forbidden. The model was
  Qwen3-30B-A3B-Instruct-2507-FP8 with **BF16 KV**, a 4,294,967,296-byte KV arena,
  context limit 8192, sequence limit 4, 2048 batched tokens, APC and async
  scheduling. The run observed a three-split prefill call, zero BF16 prefill
  workspace and 4,392,960 partial-workspace bytes. This is a functional serving
  check, not a TTFT or quality benchmark.

Timing is single-layer synthetic BF16 KV, batch 1, four KV heads, GQA=8,
head dimension 128, block size 128, independent calibration samples, and warm
reused pages without a hardware-cache flush. Four arms share each case's
queries and pages: raw FA3, gather + FA3, unsplit fused, and split fused.
Each arm warms for 20 ms; ten rounds of five calls alternate forward/reverse
order. Allocation, encoding and validation are outside timing. All 512 records
compressed with no fallbacks/spills/errors. Maximum absolute deviation from
raw FA3 was 0.00048828125; numerical checks passed. This does not assert
bitwise-identical attention arithmetic.

## Final CUDA Graph timings

Median us per call. S is the automatic split count; S=1 executes the same
unsplit kernel in both fused arms, so their small differences are timing noise.

| Context | Query | S | Raw FA3 | Gather + FA3 | Unsplit fused | Split fused | Speedup vs unsplit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 8 | 15.35 | 19.14 | 47.06 | 16.17 | 2.91x |
| 1024 | 128 | 2 | 20.42 | 24.46 | 53.35 | 37.17 | 1.44x |
| 1024 | 512 | 1 | 24.50 | 28.91 | 112.39 | 111.83 | 1.00x |
| 4096 | 16 | 16 | 17.95 | 23.74 | 168.22 | 23.63 | 7.12x |
| 4096 | 128 | 2 | 32.35 | 39.55 | 198.03 | 108.99 | 1.82x |
| 4096 | 512 | 1 | 82.83 | 92.20 | 417.30 | 413.97 | 1.01x |
| 8192 | 16 | 16 | 20.71 | 32.52 | 330.38 | 34.06 | 9.70x |
| 8192 | 128 | 2 | 53.94 | 65.72 | 384.16 | 204.24 | 1.88x |
| 8192 | 512 | 1 | 163.35 | 171.82 | 790.51 | 790.20 | 1.00x |

At Q=16, the fused path is about 15.5% faster than gather + FA3 at context
1024, essentially equal at 4096, and 4.7% slower at 8192 in this run. The Q=128
rows improve by 1.44–1.88x versus unsplit fused but remain slower than gather +
FA3. Q=512 already has enough query tiles for the current planner and keeps
one split; its large gap is not addressed by this change.

## Split-count ablation

Median us of the split arm in the separate six-choice sweep. Every choice
uses the same queries/KV within a geometry and reruns all four interleaved arms.

| Context | Query | S=1 | S=2 | S=4 | S=8 | S=16 | S=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 47.25 | 30.07 | 20.16 | 15.84 | 16.61 | 30.18 |
| 1024 | 128 | 47.98 | 34.22 | 42.60 | 58.08 | 97.17 | 164.87 |
| 4096 | 16 | 166.68 | 88.57 | 48.92 | 30.37 | 22.64 | 31.90 |
| 4096 | 128 | 194.41 | 108.04 | 115.34 | 136.07 | 180.38 | 256.78 |
| 8192 | 16 | 326.65 | 175.17 | 92.38 | 50.63 | 34.01 | 44.10 |
| 8192 | 128 | 356.56 | 205.64 | 215.07 | 232.58 | 272.60 | 355.88 |

The initial planner required at least two pages per split. The 1024/16 sweep
favored one-page chunks (S=8), so that floor was removed and the complete
nine-case matrix rerun. Increasing splits beyond the useful CTA wave adds
partial traffic, redundant Q loading and combine work; it did not help the
4096/8192 short-suffix cases. This is a measured heuristic on this setup, not
a universal optimal schedule.

## Eager timings

Median us per call. These include host launch gaps and must not be substituted
for the CUDA Graph result when discussing GPU execution efficiency.

| Context | Query | S | Raw FA3 | Gather + FA3 | Unsplit fused | Split fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 8 | 45.69 | 58.09 | 49.76 | 19.85 |
| 1024 | 128 | 2 | 46.30 | 59.23 | 51.82 | 38.33 |
| 1024 | 512 | 1 | 38.89 | 52.30 | 107.99 | 107.42 |
| 4096 | 16 | 16 | 45.38 | 59.48 | 171.10 | 27.12 |
| 4096 | 128 | 2 | 47.32 | 61.03 | 184.08 | 104.43 |
| 4096 | 512 | 1 | 92.45 | 105.75 | 416.33 | 412.61 |
| 8192 | 16 | 16 | 46.73 | 59.30 | 332.73 | 37.73 |
| 8192 | 128 | 2 | 63.93 | 79.84 | 379.71 | 203.26 |
| 8192 | 512 | 1 | 174.37 | 183.75 | 798.28 | 794.32 |

## Remaining work and artifacts

Nsight Systems confirms the intended launch change for 8192/16/S=16:

| Kernel | Grid | CTAs | Captured median duration |
| --- | --- | ---: | ---: |
| Unsplit prefill | `(2, 1, 4)` | 8 | 321.83 us |
| Split prefill | `(2, 1, 64)` | 128 | 28.94 us |
| Combine | `(16, 32, 1)` | 512 | 3.36 us |

These are instrumented component medians, not paired end-to-end sums. Use the
unprofiled four-arm table above for speed comparisons. The trace confirms that
the changed GPU work distribution, rather than only a host launch difference,
accounts for the large short-suffix improvement. It does not measure achieved
SM utilization or provide a hardware-counter stall breakdown.

The binary resource dump reports 233 registers/thread for split prefill,
183 for unsplit, zero local-memory/stack bytes for both, and unchanged 120416
bytes of dynamic shared memory. Splitting increases available work; it does
not resolve per-CTA resource pressure or the long-query tile/softmax pipeline.
Further work should target long-query tiling and on-chip reuse. Full-model
CUDA Graph serving, TP>1 and end-to-end performance comparisons remain unrun.

Archive identifier: `20261029-fused-prefill-splitkv`.

| Archive | Purpose |
| --- | --- |
| `baseline-source`, `baseline-bench-01` | Original implementation and same-card rerun |
| `splitkv-tests-01`, `final-cpu.xml` | GPU/CPU correctness and final host planner tests |
| `splitkv-bench-01` | Initial two-page-floor automatic planner |
| `splitkv-bench-02` | Complete split-count sweep |
| `splitkv-bench-04` | Final automatic planner, CUDA Graph |
| `splitkv-bench-05` | Final automatic planner, eager |
| `splitkv-bench-06` | Successful Nsight Systems trace, SQLite export and kernel summary |
| `splitkv-serve-fused-01` | Seven serving requests and allocation/execution guards |
| `splitkv-memcheck-01` | Sanitizer attempt; unresolved CUDA API error |

## Failed attempts and cleanup

`splitkv-bench-03`, the first profiling attempt, was stopped by the launcher's
ownership guard. Nsight starts the Python target in another process group;
the original group-only check classified that PID as foreign. Its trace
identifies the PID as the validation Python executable. The launcher was
changed to verify PID/start-time identities and parent ancestry, rather than
trusting executable names or users. The successful rerun records the complete
ancestry of the separate-group target back to the launched profiler. The
aborted attempt and its partial trace are retained and excluded from timings.

Compute Sanitizer again returned exit 42 for a single
`cuKernelGetFunction / CUDA_ERROR_INVALID_HANDLE` reached through the existing
`hbm_prepare` launch. The mixed-record, GQA=8, S=8 case itself passed, and no
invalid device read/write was reported. No API errors were suppressed. As in
the initial validation, this is **not a clean Sanitizer pass** and the cause
remains unresolved. The seven-request service also retained its shutdown
warning about one multiprocessing semaphore.

Ruff and `git diff --check` passed. All owned services and GPU processes were
stopped; the final GPU-4 process list was empty. Per-run temporary files,
CPU-test scratch and shared compiler caches were removed. Source snapshots,
raw reports, logs and timing samples remain in the data archive.

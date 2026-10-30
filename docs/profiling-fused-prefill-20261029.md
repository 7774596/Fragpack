# Fused prefill profiling: 2026-10-29

The original slowdown reproduces. Nsight Systems confirms that short-suffix
latency is inside the fused GPU kernel, rather than primarily host launch gaps.
A raw-record control removes only about 15–16% of fused latency in this
experiment: avoiding codec work alone does not close the gap to FA3.

## Scope and evidence

These are single-layer synthetic experiments, not serving TTFT measurements.
Source is the current uncommitted `perf/fused-prefill` working tree based on
`d40ef9a8d03ec5d98b2b7a4207a44b87d220b033`. Source copies, SHA-256 hashes,
commands, GPU process monitoring, logs, samples and failed attempts are retained
under archive identifiers:

- `20261029-fused-prefill-profile-01`: GPU 4 H800; unprofiled timing,
  Nsight Compute attempts, Nsight Systems report and exported SQLite database.
- `20261029-fused-prefill-profile-02`: GPU 3 H800; raw-record and batch controls.

GPU 4 was idle during the first run. Another user's service appeared after it
finished; the next preflight refused to run there. The supplementary experiment
used idle GPU 3 instead. No other workload was stopped and the H800 PCIe card
was not used. The two completed runs recorded 62 and 3 process-monitor samples,
respectively, with no foreign process on their selected GPU.

Environment: Python 3.10, PyTorch 2.13.0+cu129, vLLM 0.30.0+cu129, CUDA 12.9,
Nsight Systems 2025.4.1 and Nsight Compute 2025.1.1. CUDA device selection used
UUIDs. Both H800s have 132 SMs. Driver clocks were not modified.

## Unprofiled reproduction

Context 8192, four KV heads, GQA=8, BF16 KV, head dimension 128. The existing
benchmark uses independent synthetic calibration, warm repeated pages, ten
alternating-order rounds and five calls per CUDA Graph replay. Setup and
encoding are outside timing. Median microseconds per call on GPU 4:

| Query tokens | Raw FA3 | Gather + FA3 | Fused | Fused / gather |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 20.90 | 33.48 | 321.23 | 9.59 |
| 128 | 53.61 | 67.41 | 397.24 | 5.89 |
| 512 | 157.25 | 172.01 | 875.85 | 5.09 |

These reproduce the original regression. Profiled benchmark timings are kept
in separate files and are not substituted into this table.

## Timeline findings

The Nsight Systems trace records Q=16 fused launches with grid `(2,1,4)`:
eight CTAs for a 132-SM GPU. Captured calls have a median kernel duration of
319.78 microseconds. The host-only explanation is therefore insufficient.

The corresponding FA3 short-query path launches a 132-CTA persistent main
kernel and a separate 64-CTA `FlashAttnFwdCombine` kernel. Across captured
short-query calls, their individual medians are 14.88 and 7.20 microseconds;
these include both raw and gathered arms and must not be summed as a paired
end-to-end measurement. The trace establishes a split/combine execution path
that the current fused kernel lacks. Grid size is not itself an achieved-SM-
utilization measurement.

Fused launches use 256 threads, 183 registers per thread and 120416 bytes of
dynamic shared memory. Nsight reports 32 bytes of static shared memory; the
earlier binary resource dump reported 1056 bytes. These are different tool
reports and are not silently equated. Either way, dynamic shared memory alone
precludes two such CTAs sharing one H800 SM. The register allocation also
constrains residency. This is a resource-bound deduction, not measured achieved
occupancy. FA3 itself uses substantial per-CTA resources, so high allocation
alone does not explain the performance gap.

The Q=512 fused trace contains 256 CTAs. Its captured kernel median under
instrumentation is 1022.53 microseconds, above the unprofiled per-call median;
use the unprofiled table for speed comparisons.

## Raw-record and parallelism controls

On GPU 3, both fused arms use identical queries, physical block mapping and KV
values. One store contains encoded records; the other holds raw records in an
unencoded step's scratch, kept alive for the whole experiment. Both execute
the same fused attention kernel. Raw FA3 over the original KV is the numeric
reference. This changes both decoding and record payload size, so the delta
is not an isolated codec timer.

The same graph timing protocol is used. Median microseconds per batch:

| Batch | Query tokens per request | Fused CTAs | Raw FA3 | Fused compressed | Fused raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 8 | 21.00 | 320.04 | 269.09 |
| 1 | 128 | 64 | 54.79 | 344.32 | 293.99 |
| 1 | 512 | 256 | 164.93 | 795.53 | 673.62 |
| 4 | 16 | 32 | 39.35 | 340.68 | 285.11 |
| 16 | 16 | 128 | 120.62 | 421.31 | 354.43 |

All controls passed the benchmark's numeric checks; the largest absolute
difference from raw FA3 was 0.0001220703125. This is not full-model validation.

Replacing encoded records with raw records reduces fused latency by about
15–16%, while leaving a large gap to FA3. This rules out a codec-only remedy
for these cases. At Q=16, increasing batch from one to sixteen multiplies the
work by sixteen but increases compressed fused batch latency only 1.32 times.
This supports substantial unused parallelism in the single-request case.
All batch rows intentionally share the same physical history, making them
warm-cache controls rather than disjoint-history serving throughput results.
Do not compare absolute times across the two GPUs as an optimization A/B.

## Remaining attribution and next steps

1. Prioritize split-KV for short suffixes, retaining per-split softmax/output
   partials and a combine step instead of materializing global BF16 KV.
2. For longer queries, investigate query tiling and the decode/MMA/softmax
   pipeline. At GQA=8, each CTA handles only eight query tokens; different CTAs
   decode overlapping history independently. Repeated decode work is real,
   but repeated reads may hit L2 and are not equivalent to repeated HBM traffic.
3. Measure barrier, scoreboard, tensor-pipeline and shared-memory stalls before
   attributing the residual slowdown to a specific instruction or resource.

All three Nsight Compute attempts failed with `ERR_NVGPUCTRPERM` (exit 1).
The driver reports `RmProfilingAdminOnly: 1`, and noninteractive sudo is not
available. No hardware performance counters, achieved occupancy, bandwidth or
stall breakdown were obtained. No driver settings were changed. Nsight Systems
and the unprofiled controls succeeded independently of that failure.

No production kernel was modified. Per-run compiler caches were removed after
the processes exited; raw reports, source snapshots, scripts, logs and samples
were retained.

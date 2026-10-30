# Direct-block preparation: two optimization iterations

Two changes reduce the direct path's prepare kernel from **40.320 to
18.448 us** for 48 layers and 4096 aligned new tokens on H800. Cooperative
coverage checking provides most of the reduction. Reordering metadata writes
adds a repeatable **0.416–0.465 us (2.20–2.46%)** in that case. These are
prepare-kernel measurements. In the real 48-layer model, paired median
request-latency reductions are only 0.59–1.92%, with mixed directions across
repetitions in four of the six cases; direct writes remain opt-in.

## Changes and ownership

**Round 1:** replace each candidate's serial 128-slot check with one warp
checking four slots per lane. Candidates still require a fresh, successful
claim, a closing block and 128 available source rows. Every slot must match;
arbitrary source-row offsets work, while reordered slots and old tails retain
staging. Existing CTA barriers delimit validation.

**Round 2:** visit final metadata in `(layer, claim, K/V head)` order instead
of `(claim, layer, K/V head)`. Every record still releases its old overflow
slot and resets its address/length; each block-layer clears arena fill once.
Those resets cannot be omitted: leaving an address to freed overflow could
later free a slot reassigned to another block.

Neither round adds buffers, kernel launches, CTA barriers, device readbacks or
hardware requirements. Record format, codec, staging allocation, overflow
ownership, tail inheritance and commit behavior are unchanged. The final
prepare kernel uses 54 registers/thread versus 53 in baseline/Round 1; all
versions launch one 1024-thread CTA. `FRAGPACK_WRITE=staged` remains the default;
direct writes still require `FRAGPACK_PREFILL=gather`.

## Measurement

All five profiling runs completed on exclusive physical GPU 4, an H800 with
132 SMs, PyTorch 2.13.0+cu129 and CUDA 12.9. The matrix uses 4 KV heads,
128-dimensional BF16 records, 128-token blocks, 4 persistent staging slots,
469 physical blocks and either 1 or 48 layers. The 469-block geometry follows
a 4,294,967,296-byte arena budget for the 48-layer model shape.

Patterns are: one aligned token sequence; four ragged sequences with partial
tails; and complete blocks whose interior slots 31/32 are swapped to require
staged fallback. Each case has 20 warmup calls per arm and 10 rounds of five
samples, alternating staged/direct order. Every invocation uses a new prepare
step stamp, then commits. Allocation shapes are real, but this workload writes
no KV payload, performs no encoding or attention, and uses no overflow records.
It is not a full model execution.

Values below come from `prepare-kernels.json`, using Nsight CUDA kernel
timestamps correlated through the launch API to the measured NVTX labels.
Host submission and CUDA-event stream spans are archived separately and are
not substituted for kernel durations. Independent SQL audits passed for all
five traces: 1800 measured launches matched per run, 720 warmup launches were
excluded, source hashes matched, and every sample's map, claims, closes, commit
clearing and slot accounting checks were reverified. Correlation does not
require GPU completion to fall inside the CPU NVTX time range.

## Complete matrix

Microseconds; lower is better. Baseline is one process median. Round 1 and
Round 2 are the median of their two process medians, each based on 50 samples
per arm/case. Process order was baseline, R1, R2, R2, R1. Columns S/D denote
staged/direct preparation; all use NB=469.

| Layers | Tokens | Pattern | Baseline S | Baseline D | R1 S | R1 D | R2 S | R2 D |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | aligned | 4.432 | 8.192 | 4.496 | 4.896 | 4.472 | 4.896 |
| 1 | 128 | ragged | 4.353 | 4.576 | 4.352 | 4.832 | 4.312 | 4.792 |
| 1 | 128 | permuted | 4.432 | 7.713 | 4.496 | 4.896 | 4.480 | 4.904 |
| 1 | 1024 | aligned | 4.896 | 10.352 | 4.976 | 5.120 | 4.960 | 5.112 |
| 1 | 1024 | ragged | 5.504 | 9.888 | 5.536 | 5.560 | 5.584 | 5.535 |
| 1 | 1024 | permuted | 4.912 | 10.304 | 4.960 | 5.136 | 4.952 | 5.104 |
| 1 | 4096 | aligned | 10.912 | 32.416 | 10.816 | 11.168 | 10.848 | 11.064 |
| 1 | 4096 | ragged | 10.944 | 26.720 | 10.848 | 11.200 | 10.928 | 11.064 |
| 1 | 4096 | permuted | 10.912 | 32.064 | 10.824 | 11.152 | 10.832 | 11.111 |
| 48 | 128 | aligned | 4.385 | 8.160 | 4.432 | 4.736 | 4.400 | 4.704 |
| 48 | 128 | ragged | 5.264 | 5.488 | 5.184 | 5.664 | 5.184 | 5.648 |
| 48 | 128 | permuted | 4.400 | 7.680 | 4.448 | 4.760 | 4.399 | 4.712 |
| 48 | 1024 | aligned | 6.816 | 12.640 | 6.784 | 7.344 | 6.880 | 7.312 |
| 48 | 1024 | ragged | 8.288 | 13.232 | 8.224 | 8.872 | 8.240 | 8.800 |
| 48 | 1024 | permuted | 6.785 | 12.544 | 6.768 | 7.312 | 6.880 | 7.344 |
| 48 | 4096 | aligned | 18.881 | 40.320 | 18.752 | 18.888 | 18.440 | 18.448 |
| 48 | 4096 | ragged | 19.712 | 35.169 | 19.553 | 19.872 | 19.288 | 19.416 |
| 48 | 4096 | permuted | 18.864 | 39.889 | 18.800 | 18.912 | 18.400 | 18.416 |

At 48 layers/4096 aligned tokens, the direct-minus-staged gap falls from
21.439 us to approximately 0.136 us after Round 1. Round 2 then improves
the shared metadata pass; it is a small additional gain, not another large
reduction. The two ordered comparisons are:

| Comparison order | R1 direct us | R2 direct us | Reduction |
| :--- | ---: | ---: | ---: |
| R1 then R2 | 18.8965 | 18.4320 | 2.458% |
| R2 then R1 | 18.8800 | 18.4640 | 2.203% |

Round 2 is retained, but smaller cases show mixed changes: for example,
48-layer/1024-token permuted direct preparation increases from 7.312 to
7.344 us. The 128-token ragged workload has no complete block to bypass;
its direct time also remains above baseline. Two process repetitions do not
establish a universal speedup or a serving-level confidence interval.

## Correctness and limits

`final-tests-01` passed **26 direct GPU tests, 54 GPU regressions and 55 CPU
tests**, with no failures, errors or skips. Coverage includes arbitrary source
rows, internal slot permutations, empty claims, old tails and heirs, prepare
without a preceding commit, graph replay, exact record reconstruction, raw
fallback and explicit exhaustion. The added five-layer/four-head metadata
case checks nonsequential reused blocks, untouched records and overflow reuse.
CPU traversal checks also verified each record and fill reset exactly once.

The prepare measurements include profiling instrumentation and real metadata
sizes, but do not measure TTFT, throughput, output quality or hardware-counter
cache behavior. No clean sanitizer result is claimed: the
[previous direct-encoding attempt](optimization-direct-encode-20261030.md)
exited 42 on `cuKernelGetFunction / CUDA_ERROR_INVALID_HANDLE` through prepare.
That unresolved API error was not suppressed or retested by this comparison.

## Real 48-layer serving comparison

The service runs **Qwen3-30B-A3B-Instruct-2507-FP8 weights with BF16 KV**:
all 48 model layers, 4 KV heads, 32 query heads, head dimension 128, TP=1,
eager execution, async scheduling and gather/FA3 prefill. Both arms use
vLLM 0.30.0+cu129, max length 8192, max sequences 4, a 2048-token scheduling
budget, 128-token blocks and the same 4,294,967,296-byte KV arena (469 blocks).
APC is disabled. No profiler or functional hot-path instrumentation is active.

Three fresh processes per arm execute the same saved token-ID prompt plan:
inputs 128/1024/4096, concurrency 1/4, eight measured requests per case and
64 generated tokens per request. Temperature is zero, seed is fixed, and EOS
is ignored. Each case has one warmup batch. The primary comparison contains
288 measured and 90 warmup requests. Model loading and prompt construction
are excluded. TTFT ends at the first generated token-ID delta; TPOT spans
first-to-last output token; request latency ends at SSE DONE. Throughput uses
active batch intervals, excluding gaps. These are synthetic natural-language
prompts with a real model, not AgentX or a capacity experiment: even the
largest burst uses about 132 of 469 logical blocks. Chunked and mixed batches
can leave blocks on staging; a 4096-token prompt does not guarantee that all
32 blocks use direct encoding.

**Execution corrections, retained in the evidence:** the first direct launch
failed its local-port preflight (`EADDRINUSE`), before model startup or any
requests. Subsequent launches used separate loopback ports. In the first
completed pair, direct's 1024/C4 measured batch produced two new MoE Triton
binaries and a latency spike (about 7.5% higher process-median request latency).
Both entire arms of that pair were retained as diagnostic data and rerun
after cache priming; no individual request was removed. The other two pairs
had no observed measured-window binary writes. The primary chronological
order is therefore BA, AB, AB, with the replaced pair last. This correction
was recorded before the rerun and was based on observed compilation, not on
which performance values looked better.

All eight completed processes passed the request, source/configuration,
prompt-plan, zero-cache-hit and 48-layer checks. Across the six primary runs,
recorded JIT warnings/cache writes did not fall in measured windows. The
38 cached binary hashes and timestamps are archived; this does not prove
absence of every possible initialization cost. The original eight-process
data (384 measured plus 120 warmup requests), the failed preflight and both
initial/final summaries are retained.

Tables show the median of three process-level observations for each arm.
Paired changes are `(direct/staged - 1)` within each pair, followed by the
median and full range of those three changes. Thus a paired percentage need
not equal the ratio of the two displayed medians. There are three process
repetitions, not 24 independent repetitions per arm/case.

### TTFT (ms)

| Input / concurrency | Staged | Direct | Paired change, median [min, max] |
|---|---:|---:|---:|
| input128-c1 | 108.925 | 107.690 | -0.77% [-3.35%, +0.46%] |
| input128-c4 | 170.582 | 168.306 | -1.33% [-3.88%, -0.27%] |
| input1024-c1 | 113.134 | 111.996 | -1.01% [-1.23%, -0.08%] |
| input1024-c4 | 206.057 | 201.608 | -2.42% [-6.88%, -2.16%] |
| input4096-c1 | 169.897 | 168.593 | -0.39% [-1.62%, +0.72%] |
| input4096-c4 | 428.610 | 427.842 | -0.18% [-0.21%, -0.02%] |

### TPOT (ms/token)

| Input / concurrency | Staged | Direct | Paired change, median [min, max] |
|---|---:|---:|---:|
| input128-c1 | 47.644 | 46.724 | -1.94% [-2.47%, +0.33%] |
| input128-c4 | 48.043 | 47.348 | -1.90% [-2.33%, -1.09%] |
| input1024-c1 | 47.454 | 46.792 | -0.94% [-1.40%, +0.22%] |
| input1024-c4 | 48.049 | 48.007 | -0.82% [-1.99%, +1.47%] |
| input4096-c1 | 47.423 | 46.966 | -0.95% [-2.02%, +0.81%] |
| input4096-c4 | 49.122 | 48.580 | -0.65% [-1.89%, -0.19%] |

### Submission-to-DONE latency (ms)

| Input / concurrency | Staged | Direct | Paired change, median [min, max] |
|---|---:|---:|---:|
| input128-c1 | 3110.615 | 3051.019 | -1.92% [-2.48%, +0.35%] |
| input128-c4 | 3184.575 | 3141.023 | -1.73% [-2.12%, -1.37%] |
| input1024-c1 | 3103.846 | 3060.460 | -0.98% [-1.40%, +0.21%] |
| input1024-c4 | 3230.799 | 3211.115 | -0.97% [-1.92%, +0.95%] |
| input4096-c1 | 3158.314 | 3128.655 | -0.91% [-1.96%, +0.81%] |
| input4096-c4 | 3524.234 | 3487.014 | -0.59% [-1.70%, -0.23%] |

### Active-batch output throughput (token/s)

| Input / concurrency | Staged | Direct | Paired change, median [min, max] |
|---|---:|---:|---:|
| input128-c1 | 20.574 | 20.984 | +2.00% [-0.23%, +2.57%] |
| input128-c4 | 80.334 | 81.471 | +1.71% [+1.41%, +2.17%] |
| input1024-c1 | 20.596 | 20.921 | +1.06% [-0.18%, +1.58%] |
| input1024-c4 | 79.129 | 79.649 | +0.94% [-0.91%, +2.04%] |
| input4096-c1 | 20.265 | 20.460 | +1.08% [-0.80%, +2.02%] |
| input4096-c4 | 70.687 | 71.434 | +0.59% [+0.22%, +1.71%] |

The observed end-to-end gains are small: paired median total latency improves
0.59–1.92% and output throughput 0.59–2.00%. Four of six latency cases change
direction across process pairs. The two C4 edge cases (128 and 4096 tokens)
improve in all three pairs, but this small experiment supports no universal
or statistical-significance claim. A roughly 22-us prepare reduction cannot
be read as a 54% request speedup; most of these 64-token requests' time is
after the first token. Default mode remains staged.

Generated token IDs differ in 50/189 cross-arm comparisons and 57/252
within-arm repeat comparisons, counting warmup and measured requests. Every
difference is in concurrency 4; all concurrency-1 outputs agree. Same-arm
variability shows that these serving outputs are not a deterministic oracle
for KV losslessness. Scheduling/numerical variability is a plausible factor,
not an isolated cause established here. Byte-level evidence comes from the
GPU reconstruction/record tests; these runs make no quality or full-model
bitwise-equivalence claim.

## Evidence

Artifact ID: `20261030-prepare-iterations`. Runs are `baseline-bench-01`,
`round1-bench-01`, `round2-bench-01`, `round2-bench-02`, `round1-bench-02`, and
`final-tests-01`. Each profiling run retains its raw trace/SQLite database,
`prepare-kernels.json`, independent `prepare-audit.json`, status, per-sample
state checks and source manifest. The benchmark source hash is identical
across all five runs. Native `hbm_kv.cu` SHA-256 values are:

| Version | SHA-256 |
| :--- | :--- |
| Baseline | `ece7f07cbdec3612ae30beb2018ad61c6670ecbc379ead3861021fabe9ca7188` |
| Round 1 | `5064ff9df41b828075dfe1650b9e0260b0752d7f7238ae097c5c28a403110ff6` |
| Round 2 | `17a51331bccc9dc925d94fe914e6eda7746823e15ff1e43964736dcdcf89c56d` |


Serving evidence: the primary six runs are `final-serve-gather-direct-perf-03`,
`final-serve-gather-staged-perf-02`, `final-serve-gather-staged-perf-03`,
`final-serve-gather-direct-perf-04`, `final-serve-gather-staged-perf-04`, and
`final-serve-gather-direct-perf-05`. `serving-summary.json` retains every paired
value and request; `serving-jit-audit.json` retains compilation attribution.
The first pair remains in `serving-summary-initial.json`; preflight failure
`final-serve-gather-direct-perf-01` is separate. The shared prompt plan SHA-256
is `31d6a8921c87ec44f7bbedb96f7bbd1bb7030e3edce7cded97e517cbed98ba41`.

# Direct encoding of complete new KV blocks — 2026-10-30

## Result and scope

`FRAGPACK_WRITE=direct` now bypasses kernel-order scratch for complete blocks
created in the current step. Original BF16 K/V feed both the existing prefill
gather and the encoder. Incomplete blocks retain staging. The record format,
codec tables, raw fallback, FA3 reader and attention-before-encode order are
unchanged. Enable it with `FRAGPACK_PREFILL=gather`; direct writes with fused
prefill are rejected at initialization.

The final implementation passed 50 CPU tests, 69 GPU tests and seven real-model
serving requests on an exclusively allocated H800. In the single-layer CUDA
Graph benchmark, 4096 new tokens reduce write + encode from 39.65 to 20.62 us;
the attention data path falls from 466.50 to 448.91 us. Preparation adds work,
small eager cases regress, and total-step results do not show a consistent
improvement. The default therefore remains `FRAGPACK_WRITE=staged`. There is
no full-model latency or throughput claim.

## Data flow and costs

```text
Complete new blocks: original K/V -> BF16 workspace -> FA3
                                 -> encoder -> arena / raw overflow
Partial / existing open blocks: original K/V -> staging -> existing path
```

GPU prepare marks a fresh block only when all 128 physical slots appear in
consecutive source rows and its ordinary staging claim succeeded. Write skips
that block, gather copies from the original source, and encode reconstructs
the same kernel-order registers directly from the source strides. The existing
claim, tail inheritance, overflow allocation and commit lifecycle remain in
place. Nonconsecutive slot layouts fall back to staging.

The adapter passes the whole layer batch's original K/V to each gather group
and encoder. The caller owns those tensors and keeps them alive and unchanged
through encoding on the same CUDA stream. Device state, including clearing
closed-block source rows in commit, governs graph replay; a Python step flag
is not used. Missing originals for a selected direct block cause an explicit
device assertion. Encode requires both originals at the public API boundary.

The aligned gather branch copies `uint4` vectors; two-byte-offset and other
supported positive-stride inputs use the scalar fallback. This introduces no
new hardware requirement beyond the existing supported SM90 configuration.

Costs and retained allocations:

- A source-row map adds four bytes per physical block, including staged stores
  in this implementation. It remains device-resident; production hot paths add
  no device readback or synchronization.
- Prepare checks complete-block coverage once per model step. Its result is
  reused across layers; the encoder still processes each layer's own KV.
- Scratch reservations, the BF16 prefill workspace, tables, overflow storage
  and allocator policy remain. This change does not increase usable capacity
  under the existing reservation scheme.
- At four KV heads, a full block contains 262,144 bytes of BF16 K/V. Bypassing
  32 full blocks avoids an 8.389 MB scratch write and its corresponding read:
  16.777 MB of logical transfers. This is layout arithmetic, not measured DRAM
  traffic; the attention workspace transfer remains.

## Measurement configuration

- Branch: `perf/fused-prefill`, uncommitted source snapshot on base
  `d40ef9a8d03ec5d98b2b7a4207a44b87d220b033`.
- Final `fragpack/ops/hbm_kv.cu` SHA256:
  `ece7f07cbdec3612ae30beb2018ad61c6670ecbc379ead3861021fabe9ca7188`.
- Physical GPU 4: NVIDIA H800, UUID
  `GPU-910de65d-2e94-591b-557e-4b29047bf2fc`, exclusive during each run.
- Python 3.10.20, PyTorch 2.13.0+cu129, vLLM 0.30.0+cu129,
  CUDA toolkit 12.9, nvcc 12.9.86.
- Synthetic timing: batch one, one layer, four KV heads, GQA group eight,
  head dimension 128, 128-token blocks, BF16 K/V, model-like fused-QKV strides.
  Calibration inputs are separate from timed inputs; timed inputs produce no
  raw fallback. All tokens are newly generated KV with no historical prefix.
- Seed 314; ten rounds, five samples per arm per round, alternating arm order,
  20 ms warmup. Buffers are reused and hardware caches are warm, without cache
  flushing. Each call prepares a fresh step and commits it; encoded-block
  counters and pre/post reconstruction checks exclude a replayed no-op.
- CUDA Graph captures the layer call, with ordinary prepare/commit outside.
  `write_encode` measures write through encode. `write_gather_fa3_encode` also
  includes BF16 gather and FA3. Neither includes model projections or MLPs.
  Prepare and total-step event windows are recorded separately.

## Final timing results

CUDA Graph medians in us (`direct-bench-04`). Positive reductions mean less time.
1031 tokens comprise eight complete blocks and a seven-token staged tail.

| New tokens | Write + encode staged | Direct | Reduction | Attention data path staged | Direct | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 16.99 | 17.95 | -5.65% | 31.76 | 31.82 | -0.20% |
| 1024 | 22.32 | 17.98 | 19.43% | 53.33 | 49.02 | 8.07% |
| 4096 | 39.65 | 20.62 | 47.98% | 466.50 | 448.91 | 3.77% |
| 1031 | 22.16 | 19.04 | 14.08% | 53.47 | 49.70 | 7.06% |

Eager medians in us (`direct-bench-05`), retaining host submission effects:

| New tokens | Write + encode staged | Direct | Attention data path staged | Direct |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 32.74 | 33.70 | 94.61 | 97.06 |
| 1024 | 32.78 | 33.79 | 100.51 | 104.10 |
| 4096 | 46.02 | 21.57 | 460.74 | 449.42 |
| 1031 | 31.73 | 32.99 | 99.44 | 102.74 |

The full recorded step spans prepare through commit, including host enqueue
gaps and measurement events. These are the CUDA Graph run's medians in us:

| New tokens | Write-only step staged | Direct | Attention step staged | Direct |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 61.70 | 61.90 | 66.26 | 66.14 |
| 1024 | 61.30 | 61.71 | 71.30 | 69.60 |
| 4096 | 71.22 | 70.27 | 497.66 | 518.77 |
| 1031 | 61.79 | 61.94 | 68.74 | 70.40 |

Thus the shorter layer windows do not imply a faster complete step. These
single-layer steps also charge all prepare work to one layer, whereas serving
reuses one prepare across model layers. Neither summing component medians nor
multiplying the measured savings establishes full-model performance.

All final cases reconstruct KV bit-for-bit, have zero staging/overflow errors,
and match the comparison attention output exactly for these inputs. This is
not a general claim of identical model outputs across attention algorithms.

## Profiling and remaining bottlenecks

The first direct gather used scalar copies. A diagnostic Nsight trace showed
about 11.8 us versus 3.6 us for a one-block staged gather, motivating aligned
vector copies. The final trace (`direct-bench-06`) records about 2.94 versus
3.58 us for that geometry, and 3.14 versus 3.87 us at eight blocks. At 32 blocks
direct gather remains slower in this diagnostic aggregate: 7.34 versus 6.08 us.
These aggregates include warmup and validation launches, so the tables above
remain the performance comparison.

Actual prepare-kernel durations in that trace expose the added coverage check:

| New tokens | Staged median us | Direct median us |
| ---: | ---: | ---: |
| 128 | 4.42 | 8.48 |
| 1024 | 5.15 | 11.01 |
| 4096 | 12.06 | 32.00 |

Prepare labels were assigned from the same stream's following write geometry
and encode template. Sample counts include warmup/checks and differ between
arms. Unlike event windows around ordinary prepare, these kernel durations do
not include host enqueue gaps. The complete-block coverage scan is a concrete
remaining cost. FA3 dominates the long-query attention window, limiting the
fraction saved by removing the scratch round trip. Graph replay reduces launch
overhead; eager small cases still show regressions. A controlled full-model
comparison is needed before selecting this path by default.

## Correctness and serving

`direct-tests-03` passed 50 CPU tests, 15 new direct-encoding GPU cases and all
54 prior GPU cases. Coverage includes CPU-reference encoded bytes and BF16
reconstruction, compressed/raw/spill records, poisoned bypassed scratch,
strided and two-byte-offset inputs, reordered blocks and nonzero source rows,
mixed historical/full/tail blocks over two layers, tail inheritance, padding,
nonconsecutive-slot fallback, overflow exhaustion, explicit missing-input
errors, and graph replay with commit captured or outside the graph. The
missing-gather-input device assertion runs in an isolated subprocess.

The final serving run (`direct-serve-gather-direct-03`) used
Qwen3-30B-A3B-Instruct-2507-FP8 weights with BF16 KV, TP=1, eager execution,
asynchronous scheduling, a 4,294,967,296-byte arena, maximum length 8192,
four sequences, and a 2048-token per-step budget. Seven requests passed,
including a 2309-token uncached prompt, chunked prefill, unaligned input,
decode across block boundaries, concurrency and prefix reuse.

Functional instrumentation checked that write, gather and encode received the
same full-layer K/V and observed 16 direct blocks in an actual request, with
source rows 0, 128, ..., 1920. Its one-time GPU readback belongs only to the
validation guard; this run is not a timing comparison. Full-model CUDA Graph,
TP > 1, and external cache integration were not validated here.

## Failures, artifacts and cleanup

- `direct-bench-01` stopped after a CUDA capture error. The benchmark retained
  a graph through a closure cycle; replacing the closure with a method and
  explicitly managing graph lifetime removed the failure in subsequent complete
  runs. Its partial samples are excluded from final results.
- `direct-memcheck-01` passed six reference/striding pytest cases but Compute
  Sanitizer exited 42 with `CUDA_ERROR_INVALID_HANDLE (400)` in
  `cuKernelGetFunction` through prepare. No invalid device read/write was
  reported, but this is **not a clean sanitizer pass**. The API error remains
  unresolved and was not suppressed.
- An early serving guard printed diagnostics to stdout, contaminating a
  `cpuinfo` JSON subprocess and causing a nonfatal usage-reporting exception.
  Routing guard output to stderr passed a CPU-only JSON probe and the final
  serving rerun without that exception. No third-party source was changed.
  The existing one-semaphore shutdown warning remains recorded.

Archive ID: `20261030-direct-encode`. The archive retains the pre-change
snapshot, immutable per-run source and hashes, commands, environment reports,
GPU ownership logs, pytest XML, complete benchmark samples and serving replies.
Final timing uses `direct-bench-04` (Graph) and `direct-bench-05` (eager);
`direct-bench-06` contains the final Nsight trace and kernel summaries.
`direct-bench-02/03` are intermediate scalar-gather runs, not final evidence.
Earlier test and serving attempts remain separately labeled.

Owned services were stopped and GPU 4 released after each run. Per-run scratch
and retained build caches were removed after validation. No commit or push is
part of this task.

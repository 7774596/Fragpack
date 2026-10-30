# Validation and evidence boundary

This document describes the checks needed for the standalone extraction.
It does not certify that an unrun check passed. GPU execution requires separate
authorization and an idle supported GPU.

## Prepare optimization and real-model status on 2026-10-30

The [two-round prepare report](optimization-prepare-iterations-20261030.md)
records 55 CPU tests, 26 direct GPU tests and 54 GPU regressions passing, with
zero skips. All 48 real model layers execute in TP=1 eager staged/direct
serving: six primary processes, three pairs, 288 measured plus 90 warmup
requests, BF16 KV and no APC. Two earlier completed processes remain as
diagnostic data after measured-window MoE JIT triggered a whole-pair rerun;
one preflight failure also remains recorded. No primary measured-window new
Triton binary writes were observed. This is not a clean sanitizer result.

The 48-layer/4096-token prepare kernel improves 40.32 to 18.45 us. Serving
latency changes are much smaller (paired median reductions 0.59–1.92%), with
mixed directions across repetitions in four of six cases. This is a synthetic
prompt workload on the actual model, not AgentX, a capacity-pressure test,
or a quality evaluation. Concurrent generated outputs vary even within one
mode; exact KV reconstruction is checked by the GPU tests separately.
The reader remains gather/FA3; full-model graphs and TP > 1 are unverified.

## Initial direct-write status on 2026-10-30

The [direct-encoding report](optimization-direct-encode-20261030.md) records
`FRAGPACK_WRITE=direct` with the default gather/FA3 prefill reader. It passed
50 CPU tests, 15 new GPU cases plus 54 GPU regressions, and seven TP=1 eager
serving requests with actual direct-block execution observed. Kernel graphs
include commit replay; full-model graph serving and TP > 1 remain unverified.

Both eager and CUDA Graph write/attention matrices completed. Large full-block
cases reduce write/encode time, but prepare adds work and complete-step timing
does not improve consistently. Direct writes remain opt-in. The new sanitizer
attempt passed six pytest cases but exited with an unresolved CUDA
function-handle API error; it is not a clean pass.

## Status on 2026-10-29

See the [initial extraction report](validation-20261029.md) and the
[fused-prefill report](validation-fused-prefill-20261029.md) for their separate
configurations, artifacts and observed limits. The subsequent
[split-KV report](optimization-splitkv-prefill-20261029.md) records the current
experimental prefill implementation.

| Layer | Status |
| --- | --- |
| Source and CLI inspection | vLLM 0.30 flag definitions and source boundaries inspected |
| Standalone packaging and CPU suite | Initial wheel/sdist and 24 CPU tests passed; current split-KV checkout passes 50 CPU tests |
| Standalone CUDA compilation and correctness | 18 existing and 36 split/unsplit prefill GPU tests passed on H800, including kernel graph replay |
| Compute Sanitizer | Unresolved CUDA function-handle API error in fused-prefill and existing store-only tests; not a clean pass |
| Standalone vLLM eager serving | TP=1; seven requests passed; split-KV run observes multi-split execution, forbids gather, and reports zero BF16 workspace plus 4,392,960 partial bytes |
| Full-model CUDA Graph and TP>1 serving | Not run |
| Standalone performance | Fused prefill single-layer eager/Graph matrix completed; no end-to-end performance comparison |
| External cache service integration | Outside this extraction |

Historical DaseR test counts and performance tables do not establish these
standalone results. Preserve historical version/configuration labels when
using old evidence to explain why a regression test exists.

## Ordered checks

1. **Packaging and CPU contracts.** Check an installed package away from the
   source directory, required native source resources, core imports without
   vLLM, and plugin metadata. Run reference record round trips with escapes and
   raw fallback, malformed-input rejection, open-block state, and host-only
   allocation accounting. Check actual executed/skipped tests.
2. **CUDA store and kernels.** Compile on the target toolchain. Validate writes,
   block closure, tail-slot inheritance, overflow reuse, blob round trips, and
   failed imports. Reconstruct KV bit-for-bit, including raw fallback.
3. **Decode and graphs.** Check numerical outputs against the independent
   attention reference across ragged lengths, wave boundaries, padding, empty
   rows, and replanning across layers. Exercise eager and CUDA Graph paths.
4. **Real serving.** Start with TP=1/eager, perform prefill and enough decode
   steps to close and reopen blocks, then exercise request completion, reuse,
   and mixed batches. Confirm the selected plugin and actual allocation.
   Extend to graph/TP only as separate recorded configurations.
5. **Performance.** Compare extracted and source implementations under the
   same corrected harness before making new serving claims. Include uniform
   and skewed workloads, real encoding work, warmup, and all samples. Then
   measure full-model decode without capacity pressure and admission under a
   stated memory budget.

The transferred GPU files cover store lifecycle and decode scheduling. They
are selected with `pytest -q tests/gpu`; default CPU CI must not discover or
initialize them. Sanitizer results, when collected, must preserve warnings and
tool limitations rather than reporting only successful subchecks.

## Measurement rules

- Verify writes and encodes actually occurred; a closed block can otherwise
  turn an intended write benchmark into a no-op.
- Separate eager launch gaps from graph replay timings. A difference between
  component medians is an estimate, not an isolated kernel measurement.
- Preserve slower workloads. Historical scheduling changes improved skewed
  cases but also showed uniform-load regressions; migration does not erase
  those evidence boundaries.
- Keep calibration pages separate from evaluation pages where possible.
- Report auxiliary allocations and synchronization, and distinguish measured
  values from layout arithmetic or configured reservations.
- After a run, save results and manifests, stop owned services, verify resource
  release, and delete generated stores and per-run scratch.

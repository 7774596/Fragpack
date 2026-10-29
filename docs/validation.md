# Validation and evidence boundary

This document describes the checks needed for the standalone extraction.
It does not certify that an unrun check passed. GPU execution requires separate
authorization and an idle supported GPU.

## Status on 2026-10-29

See the [standalone validation report](validation-20261029.md) for the
configuration, artifacts and observed limits.

| Layer | Status |
| --- | --- |
| Source and CLI inspection | vLLM 0.30 flag definitions and source boundaries inspected |
| Standalone packaging and CPU suite | Wheel and sdist built; installed-package check and 24 CPU tests passed |
| Standalone CUDA compilation and correctness | 18 GPU tests passed on H800, including kernel graph replay |
| Standalone vLLM eager serving | TP=1; seven requests passed, including cold prefill above budget, long decode, concurrency and APC |
| Full-model CUDA Graph and TP>1 serving | Not run |
| Standalone performance | Not run |
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

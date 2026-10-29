# Agent instructions

FragPack is an experimental lossless KV representation, GPU execution module,
and vLLM adapter. Read [docs/development.md](docs/development.md), the relevant
design documentation, and [CONTRIBUTING.md](CONTRIBUTING.md) before changes.

## Repository boundaries

- Production Python and native sources belong under `fragpack/`.
- `fragpack/compression/` defines the reference representation and tables;
  `fragpack/ops/` owns GPU storage and kernels;
  `fragpack/integrations/vllm/` owns inference-engine adaptation.
- Keep the representation and GPU core independent of vLLM, DaseR, LMCache,
  HTTP services, and external tier policies. Integrations depend on the core,
  not the reverse.
- Cross-module calls use public interfaces. Document ownership and stream
  ordering for GPU tensors, staging, metadata, and imported records.
- Do not add device synchronization, device readback, or blocking I/O to hot
  paths without an explicit design justification and appropriate measurement.
  Offline calibration and initialization may use blocking operations.
- Update the design before changing record formats, block lifecycle invariants,
  or ownership boundaries. Preserve explicit handling of overflow and invalid
  records; never silently drop KV data.
- Do not modify vLLM, LMCache, or other third-party source without explicit user
  authorization. The existing adapter wraps the V2 runner at runtime; changes
  to that compatibility boundary require documented validation.

## Implementation and validation

- Prefer targeted, maintainable changes. Avoid speculative compatibility layers.
- Python files begin with `# SPDX-License-Identifier: Apache-2.0`.
- Add type hints and public docstrings covering inputs, outputs, and relevant
  concurrency or CUDA-stream behavior.
- Use the package logger in production code. Explain non-obvious invariants,
  memory ownership, and hardware constraints in comments.
- Test observable behavior and public contracts. Do not add tests that only
  restate constants or mechanical renames.
- Run appropriate CPU checks after implementation. GPU work needs explicit
  authorization and exclusive access to the selected GPU; do not interfere
  with other users' processes.
- Run authorized benchmarks to completion and report failures as well as
  successes. Separate functional checks, kernel timing, and serving results.
- Verify command flags against the supported dependency version before writing
  examples. Installation or import success is not GPU runtime validation.
- Keep large tests, model data, JIT caches, and benchmark scratch on an approved
  data filesystem outside the repository. Preserve requested results, stop
  owned services, and delete generated per-run stores and scratch afterward.
- Never commit credentials, machine-specific private paths, model files, or
  generated experiment stores. Historical performance evidence must retain its
  original version, configuration, and scope.

## Workflow

The default branch is `main`. Develop on a named branch and follow the scoped
commit format in `CONTRIBUTING.md`. Do not initialize, commit, push, or change
remotes unless the task authorizes it. Do not add references to workflow tools
or managed directories that are absent from this repository.

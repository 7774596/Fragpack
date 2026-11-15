# Agent instructions

FragPack is an experimental lossless KV representation, GPU execution module,
and vLLM adapter. Read [docs/development.md](docs/development.md), the relevant
design documentation, and [CONTRIBUTING.md](CONTRIBUTING.md) before changes.
直接给出结论和可执行步骤。省略客套、重复解释和泛泛建议，保留必要的依据、验证结果和限制,不是很有必要的也不要展开解释.


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

## Reports and experiment records

- Write new reports, experiment notes, result summaries, table headings, and
  figure labels in clear, standard technical English. Use English when
  substantively revising these documents. Preserve original raw logs and data.
- Summarize progress and results briefly in Simplified Chinese in chat. Lead
  with the outcome, key measurements, and limitations that affect the conclusion.
- Keep current reports and reproducible result summaries easy to find. Archive
  superseded runs and duplicate snapshots instead of accumulating loose records.
  Preserve source revisions, configurations, measurements, and failure evidence.

## Manuscript writing style

- Use formal but natural academic English. Follow the user-approved TAPE_A
  exemplar for paragraph flow and cohesion, without copying its wording or
  scientific claims.
- Connect the problem, existing approaches, their limitations, and the proposed
  solution through clear reasoning. Use causal, conditional, or concessive
  transitions where they help the argument; avoid formulaic linking phrases.
- Explain relevant prior-work mechanisms simply before discussing their limits.
  Keep enough background to make the comparison understandable, without turning
  the Introduction into a catalogue of systems.
- Group paragraphs by shared purpose and merge closely related points. Avoid
  fragmented short paragraphs, but do not impose a fixed paragraph count. The
  four-paragraph KVFold Introduction was a local editorial choice.
- Prefer precise, familiar words and varied sentence lengths. Remove redundant
  explanations, inflated claims, and mechanical phrasing; avoid colloquial slang.
- Preserve numbers, results, citations, technical meaning, and evidence scope.
  Keep measured findings, interpretations, and planned work distinct.
- For the current KVFold manuscript, end the Introduction with the system and
  its approach, rather than demonstration activities or console operations. The
  Abstract may briefly summarize the demonstrated capabilities and benefits.
- Apply this style within the requested editing scope. Approval of a style does
  not authorize applying a chat-only draft to manuscript files.

## Workspace cleanup

- Before retiring a worktree or local branch, check active processes and services,
  Git ancestry, and unique uncommitted files; age alone does not prove inactivity.
- During authorized cleanup, preserve unique work and required evidence in a
  verified archive with a restoration index before removing obsolete worktrees,
  branches, or experiment directories. Keep active development and services intact.
- Remove disposable test stores and build caches separately from scientific
  evidence. Do not clean shared datasets, model weights, or unrelated repositories.

# Contributing to FragPack

Discuss changes to the record format, GPU memory lifecycle, or engine interface
in the design documentation before implementation. Keep the core independent
of a particular cache service.

## Branches and commits

The default branch is `main`. Use a branch for each change, such as
`feat/<topic>`, `fix/<topic>`, `perf/<topic>`, `refactor/<topic>`,
`test/<topic>`, `docs/<topic>`, `chore/<topic>`, or `revert/<topic>`.

Commit subjects and PR titles use:

```text
<type>(<scope>): <short description>
```

Types match the branch categories above. Useful scopes include `codec`,
`kernel`, `memory`, `vllm`, `build`, `tests`, `ci`, and `docs`.
Every commit body describes its changes as `- ` bullets. For documentation
tooling or restructuring use `chore(docs)`; ordinary documentation changes may
use `docs(<scope>)`.

PRs target `main` and use squash merges. Do not commit directly or force-push to
`main`. Preserve source attribution and license headers when adapting code.

## Review and validation

Explain the problem, resulting behavior, and validation performed. Include the
tested environment for GPU changes and clearly identify unrun checks. Keep
implementation fixes separate from changes to experimental claims.

CPU checks cover reference encoding, lifecycle logic, and host-side accounting.
GPU checks cover actual kernels and engine integration; they require suitable
hardware and explicit allocation. Use [docs/validation.md](docs/validation.md)
to select the necessary checks without running unrelated benchmarks.

Before submitting, run relevant tests, Ruff checks, and `git diff --check`.
Do not include local data paths, credentials, generated stores, model weights,
or compiler caches in commits or PR descriptions.

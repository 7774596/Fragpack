# Migration provenance

This project extracts FragPack from
[DaseR](https://github.com/GentleCold/DaseR) via its
[demo fork](https://github.com/7774596/DaseR), source revision
`9f113cd0bc973e9d84aa538610343c7c18e9171c` on the demo experiment line.
The original Apache-2.0 license and source SPDX headers are retained.

| Source area | Standalone responsibility |
| --- | --- |
| `daser/compression/attn_page.py`, `attn_tables.py` | Reference format and tables under `fragpack/compression/` |
| FragPack/HBM/fused attention files in `daser/ops/` | GPU core under `fragpack/ops/` |
| FragPack backend and plugin in `daser/attention/` | `fragpack/integrations/vllm/` |
| Focused codec, state, capacity, and GPU tests | `tests/unit/` and `tests/gpu/` |
| `benchmarks/research/fragpack_step_bench.py` | `benchmarks/fragpack_step_bench.py` |

The extraction renames package imports, plugin registration, extension build
identity, and the table/overflow environment variables. It separates CPU test
collection from GPU and service dependencies. The new variables are
`FRAGPACK_TABLES` and `FRAGPACK_OVERFLOW`; the plugin name is `fragpack`.

The DaseR server, IPC protocol, external tier manager, document interfaces,
dashboard, and broad service benchmark suite are not copied. Export/import
operations remain core capabilities; a new external cache adapter is separate
work. No vLLM or LMCache source files are changed by this extraction.

The independent repository was initialized by the user with default branch
`main`; migration work is prepared on `feat/standalone-vllm`. Repository
creation does not imply a commit, publication, or GPU validation.

Historical experiments remain attributed to their original implementation.
Standalone results must name the new revision, dependency builds, input data,
and measurement scope. See [validation](validation.md) for completed and outstanding checks.

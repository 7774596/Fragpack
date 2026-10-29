# Development

## Environments

The GPU migration target is Python 3.10.20, official vLLM 0.30.0 with its CUDA
12.9 build, PyTorch 2.13.0+cu129, and CUDA toolkit 12.9 with nvcc 12.9.86.
Use the requirements file to select the matching inference packages before
installing FragPack:

```bash
python -m pip install -r requirements/vllm-cu129.txt
python -m pip install -e '.[dev]'
```

Set `CUDA_HOME` externally to your toolkit directory. The first GPU operation
JIT-compiles the C++/CUDA extension using PyTorch's extension loader. A working
compiler and CUDA toolkit are needed in addition to runtime wheels. Keep the
extension cache on a suitable data filesystem using `TORCH_EXTENSIONS_DIR`.

Do not substitute an arbitrary PyTorch wheel into a vLLM environment: the
engine, PyTorch, CUDA build, and native extensions must agree. A successful
editable installation only checks Python packaging.

For CPU development and CI, use a separate environment:

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

The CPU PyTorch wheel is needed for allocation-accounting tests. vLLM and CUDA
are not needed for the CPU core suite; do not silently skip accounting checks
and report the remaining tests as complete validation.

## Layout

| Path | Responsibility |
| --- | --- |
| `fragpack/compression/` | CPU reference records and calibrated lookup tables |
| `fragpack/ops/` | HBM store, native extension, fused decode and gather |
| `fragpack/integrations/vllm/` | Backend, plugin registration, engine adaptation |
| `tests/unit/` | CPU reference, state, and accounting contracts |
| `tests/gpu/` | Explicit CUDA correctness and graph tests |
| `benchmarks/` | Measured workloads and result collection |

The vLLM plugin entry point is `fragpack`, calling
`fragpack.integrations.vllm.plugin:register`. It registers
`fragpack.integrations.vllm.backend.FragPackAttentionBackend` as `CUSTOM`.
Set `VLLM_PLUGINS=fragpack` to select this plugin explicitly, especially when an
environment also contains another custom attention backend.

## Checks

```bash
ruff check fragpack tests benchmarks
ruff format --check fragpack tests benchmarks
pytest -q tests/unit
```

The default `pytest` selection is `tests/unit`. GPU tests must be requested
explicitly after authorization and GPU allocation:

```bash
pytest -q tests/gpu
```

Use a unique per-run directory on an approved data filesystem for test scratch
and benchmark output. When using `--basetemp`, remember that pytest manages
that directory: never point it at shared results or another active run.
Keep requested evidence separately, stop owned services, and delete temporary
stores and scratch when finished.

The serving client calls an already running service and saves its requests,
responses, and functional checks:

```bash
python tests/integration/standalone_smoke.py \
    --base-url http://127.0.0.1:8000 --model your-served-model \
    --output /path/to/results/serving.json --require-apc-hit \
    --require-uncached-prefill-over 2048
```

For this check, start a fresh service with `--enable-prefix-caching`,
`--enable-prompt-tokens-details`, and `--max-num-batched-tokens 2048`.
The client begins with a 2309-token input and checks that uncached prefill
exceeds the stated budget; a long prompt alone can fit one step after a cache
hit. It also covers unaligned prompts, generation across 128-token boundaries,
concurrent requests, and repeated prefixes. The positive
cached-token assertion applies to this standalone setup without an external
KV connector. It is a functional check, not a latency benchmark.

The first serving check uses the eager TP=1 example in the README. Validate
CUDA Graph execution and TP head sharding separately before advertising them
as verified configurations. No GPU checks are implied by CI success.

## Measurements

From the repository root, `python -m benchmarks.fragpack_step_bench --help`
describes the transferred single-layer benchmark. This tool is included in the
source checkout, not the installed wheel, and needs the serving environment. Its `--dump` input contains captured uint16 BF16 bits;
`--out` selects the retained JSON result. Choose calibration and held-out pages
explicitly. Keep all measured samples and record versions, source revision,
input hash, GPU identity, geometry, warmup, and graph/eager mode.

Kernel timing is not TTFT or full-model TPOT. Capacity reports must include
staging, overflow, tables, metadata, and workspace, and state which allocations
remain excluded. See [validation](validation.md).

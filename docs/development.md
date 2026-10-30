# Development

The [system design](design/README.md) is the current architecture reference.
Read its overview and the affected component/flow or topic document before
changing interfaces, KV lifecycle, ownership or record formats. Keep dated
validation reports separate from the current design and preserve their original
measurement scope.

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
| `fragpack/ops/` | HBM store, native extensions, independent page codec, fused decode and gather |
| `fragpack/integrations/vllm/` | Backend, plugin registration, engine adaptation |
| `fragpack/integrations/lmcache/` | Optional 0.5.6 plugin, prefix identity and bounded record transfer |
| `tests/unit/` | CPU reference, state, and accounting contracts |
| `tests/gpu/` | Explicit CUDA correctness and graph tests |
| `benchmarks/` | Measured workloads and result collection |

The vLLM plugin entry point is `fragpack`, calling
`fragpack.integrations.vllm.plugin:register`. It registers
`fragpack.integrations.vllm.backend.FragPackAttentionBackend` as `CUSTOM`.
Set `VLLM_PLUGINS=fragpack` to select this plugin explicitly, especially when an
environment also contains another custom attention backend.

The optional LMCache extension has its own hash-pinned requirement file and
[integration guide](lmcache-plugin-056.md). It uses the stock server-module
factory; it is not a replacement for vLLM's KV connector configuration. Keep its
real-service and GPU tests opt-in so CPU core development does not require
LMCache or start services.

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

The independent native-page codec uses `fragpack.ops.native_codec` and a separate
JIT extension; it does not initialize the HBM store or an attention backend.
Its [round-trip benchmark and report](optimization-native-codec-20261030.md)
compare actual pinned-host transfers. Initialization and JIT belong outside the
steady-state timing window. The [design contract](design/native-attention-external-codec.md)
defines native-page ownership, operation ordering and the unimplemented serving boundary.

The expanded read/restore comparison accepts a JSON list of cases. Each case
specifies `name`, per-request `contexts` and `queries`, `shared_prefix` in tokens,
and `reads` after one restore. It compares raw/native FA3, compressed/native FA3,
gather/FA3, and the current fused reader. All-one-token batches use the production
decode kernel. For example:

```json
[{"name":"shared","contexts":[8192,8192],"queries":[128,128],"shared_prefix":6144,"reads":1}]
```

```bash
python -m benchmarks.native_codec_comparison_bench \
    --dump /path/to/captured-kv.npy --cases /path/to/cases.json \
    --out /path/to/results/comparison.json --rounds 24 --warmup 3 \
    --input-description 'Describe the actual capture and its provenance'
python -m benchmarks.native_codec_scale_bench \
    --dump /path/to/captured-kv.npy --out /path/to/results/roundtrip.json \
    --source-pattern capture-repeat --blocks 1,16,64,256 --page-tokens 128 \
    --rounds 24 --warmup 3
```

These drivers repeat captured blocks for explicitly synthetic scale experiments.
The first is a 48-layer read-only replay; the second includes encoding and actual
pinned D2H/H2D. Neither measures full-model serving. `fallback25` and `fallback100`
construct finite out-of-table records for the roundtrip cost of raw fallback.
Use `--order reverse` in a separate process for the matching order control.

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

## Experimental fused prefill

Set `FRAGPACK_PREFILL=fused` before starting a worker to read FragPack records
directly inside prefill attention. The normal open-block staging still receives
new KV, but this mode neither allocates a BF16 prefill K/V workspace nor calls
the store's gather operation during prefill. `FRAGPACK_PREFILL=gather` is the
default comparison path. Both modes share the same record format and decode
attention implementation.

After reserving an idle SM90 GPU, select `tests/gpu/test_fragpack_prefill.py`
for direct-attention correctness, including causal prefix handling, GQA, raw
records, partial tails, and graph replay. Run a serving check as well: a passing
kernel test does not validate mixed-batch metadata or model-runner integration.

The four-arm synthetic microbenchmark compares raw FA3, gather + FA3,
unsplit fused prefill, and split-KV fused prefill. Keep output on the approved
data filesystem and compare
eager and CUDA Graph execution separately:

```bash
python -m benchmarks.fragpack_prefill_bench \
    --contexts 1024,4096,8192 --queries 16,128,512 \
    --heads 4 --group 8 --rounds 10 --inner 5 --warmup-ms 20 \
    --out /path/to/results/prefill-eager.json
python -m benchmarks.fragpack_prefill_bench \
    --contexts 1024,4096,8192 --queries 16,128,512 \
    --heads 4 --group 8 --rounds 10 --inner 5 --warmup-ms 20 \
    --cuda-graph --out /path/to/results/prefill-graph.json
```

`--splits auto` uses the serving planner. `--splits 1,2,4,8,16,32` runs a
forced-count ablation using the same inputs for every choice; these larger
forced workspaces may exceed the serving planner's allocation bound.

The [split-KV results](optimization-splitkv-prefill-20261029.md) improve short
suffixes over the [initial implementation](validation-fused-prefill-20261029.md),
but long-query cases remain slower. This mode is still experimental.

## Direct encoding of complete new blocks

To bypass the kernel-order scratch round trip for complete new blocks, set
these options before starting the same vLLM service:

```bash
export FRAGPACK_PREFILL=gather
export FRAGPACK_WRITE=direct
```

Use `FRAGPACK_WRITE=staged` for the same-configuration control. Direct writes
currently require the gather reader. They retain FA3, the BF16 prefill
workspace, scratch allocation and the existing tail/overflow lifecycle.
The caller supplies the complete layer batch's original K/V to gather and
encode on the same stream; input slices for individual gather groups are not
valid substitutes. See [architecture](architecture.md).

After reserving an idle SM90 GPU, run
`pytest -q tests/gpu/test_fragpack_direct_encode.py` and the existing GPU
regressions. The new checks compare reconstructed bits and encoded records
with the CPU reference, poison bypassed scratch, and exercise mixed tails,
strided inputs, raw overflow and graph replay. Follow with the serving smoke
client described above. A passing standalone kernel test does not establish
the runtime adapter's source-row mapping or an end-to-end speedup.

The source-checkout benchmark compares staged and direct writes with fresh
prepare/commit work and records separate layer, prepare and total-step windows:

```bash
python -m benchmarks.fragpack_write_bench \
    --tokens 128,1024,4096,1031 --heads 4 --group 8 \
    --rounds 10 --inner 5 --warmup-ms 20 --seed 314 \
    --cuda-graph --out /path/to/results/write-graph.json
```

Omit `--cuda-graph` for the eager comparison. This is a synthetic single-layer
benchmark with warm hardware caches, not a serving latency benchmark. Preserve
all three windows: faster encode/gather can be offset by preparation or launch
costs. See the [measured results](optimization-direct-encode-20261030.md).

## Prepare and real-model A/B measurements

The prepare benchmark covers fresh claims across one or 48 layers, including
unaligned request boundaries and permuted interior slots. Each sample calls
public prepare and commit with a new step stamp and validates device state
outside the timing window. It does not write KV or run model inference:

```bash
nsys profile --trace=cuda,nvtx --sample=none \
    --output /path/to/results/prepare \
    python -m benchmarks.fragpack_prepare_bench \
    --layers 1,48 --tokens 128,1024,4096 --blocks 469 --heads 4 \
    --rounds 10 --inner 5 --warmup 20 \
    --out /path/to/results/prepare.json
```

The JSON's host submission and CUDA event spans are separate measurements;
neither is kernel-only latency. Recover kernel durations from CUDA timestamps,
linking the measured NVTX launch ranges through runtime correlation IDs.
Exclude warmup ranges. This benchmark rejects CUDA Graph capture because a
fixed host step stamp would stop exercising fresh block detection.

For serving, start a fresh TP=1 eager service for each write mode, with the
same model, tables, gather reader and KV budget. The measured configuration
uses `--no-enable-prefix-caching --enable-prompt-tokens-details`,
`--max-num-seqs 4 --max-num-batched-tokens 2048 --max-model-len 8192`,
`--block-size 128 --kv-cache-memory-bytes 4294967296`, BF16 KV, and
`--async-scheduling`. Keep functional hot-path instrumentation disabled.
Run the streaming client against that service:

```bash
python -m benchmarks.fragpack_serving_bench \
    --base-url http://127.0.0.1:8000 --model your-served-model \
    --arm staged --replicate 0 \
    --input-tokens 128,1024,4096 --concurrency 1,4 \
    --requests-per-case 8 --warmup-batches 1 --output-tokens 64 \
    --seed 314 --plan /path/to/results/shared-prompts.json \
    --out /path/to/results/staged-0.json
```

Reuse the exact plan for both modes. Use `--arm direct` only with a service
started with `FRAGPACK_WRITE=direct`; this label does not reconfigure a running
server. Use three fresh process pairs in staged/direct, direct/staged,
staged/direct order. The client checks prompt/output token counts and zero
cached tokens, retains per-request streaming events, and reports TTFT, TPOT,
request latency and active-batch output throughput. Its synthetic prompts run
the real full model; they are not AgentX, a capacity-pressure workload, or a
quality evaluation. Requests within a process are nested samples, not
independent process repetitions. See the
[prepare and serving report](optimization-prepare-iterations-20261030.md).

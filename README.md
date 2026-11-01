# FragPack

FragPack is an experimental lossless BF16 KV representation and GPU codec with
two vLLM integration paths.

The native-attention path keeps active GPU KV in standard BF16 pages.
`FragPackNativeConnector` uses public scheduler/worker hooks to save complete
prompt blocks through `NativeBlockTransfer` into LMCache CPU L1 after normal
request completion. Recall decodes directly into native pages before prefill
can be skipped. Attention remains native, without a CUSTOM backend, runner
wrapper or third-party source changes. This path does not expand active GPU KV
capacity.

The HBM-resident path keeps completed blocks compressed in GPU memory using
the CUSTOM attention adapter. Decode reconstructs operands inside its kernel;
prefill uses a temporary BF16 workspace by default, with fused prefill as an
experimental option. Incomplete blocks remain writable until they reach
128 tokens. This path's external-cache API remains explicitly caller-driven.

This standalone extraction comes from [DaseR](https://github.com/GentleCold/DaseR)
via the [demo fork](https://github.com/7774596/DaseR) at revision
`9f113cd0bc973e9d84aa538610343c7c18e9171c` under Apache-2.0. It contains
the representation, GPU store, kernels, and integration adapters. The optional
LMCache adapter uses the external service's CPU L1 and placement policy; the
service itself is not bundled, and the representation/core do not depend on it.

## Status and scope

The native-attention path completed real CPU-L1 functional checks and four
fixed-budget offline model runs on H800 PCIe with
Qwen3-30B-A3B-Instruct-2507-FP8, BF16 KV, native FA3, TP=1 and eager execution.
With 32,768-token histories and 56 GB L1, the request-plus-publication window
fell by 25.43% across the two processes per mode. External restores increased
TTFT; compression also used about 100.8 MB more GPU workspace than Raw. The
result does not establish faster user requests or concurrent serving capacity.
See the [2026-11-01 report](docs/validation-native-codec-serving-20261101.md).
Raw uses a distinct FPNR envelope and the same byte service as compressed FPKB;
it is a matched-interface control, not an optimized stock-LMCache Raw transport.
HTTP, NVMe, TP>1 and full-model CUDA Graph execution remain unverified for this
path.

The following extraction and HBM-optimization results retain their original
dated configurations and evidence boundaries; their test counts are historical.

The initial standalone wheel passed 24 CPU tests, 18 GPU tests, and a
seven-request TP=1 eager serving check on H800. See the
[extraction report](docs/validation-20261029.md) and
[migration provenance](docs/migration.md).

The split-KV fused-prefill checkout passed 50 CPU tests, 54 GPU tests, and a
seven-request TP=1 eager serving check with prefill gather forbidden and zero
BF16 prefill workspace. At 8192 context / 16 query tokens, split-KV reduces
single-layer fused attention from 330.38 to 34.06 us; gather + FA3 takes
32.52 us in the same CUDA Graph run. Long-query performance still trails FA3,
so the default remains `FRAGPACK_PREFILL=gather`. See the
[split-KV report](docs/optimization-splitkv-prefill-20261029.md) and
[initial fused-prefill report](docs/validation-fused-prefill-20261029.md) for
complete matrices and limits. No end-to-end speedup is claimed.

The initial optional direct-new-block write implementation passed 50 CPU
tests, 69 GPU tests and seven TP=1 eager serving requests. It reduces the 4096-token write + encode
window by 48% in a single-layer CUDA Graph test, but adds prepare work and does
not consistently improve complete-step timing. See the
[direct-encoding report](docs/optimization-direct-encode-20261030.md).

Two subsequent prepare optimizations passed 55 CPU tests and 80 GPU tests.
At 48 layers/4096 aligned tokens, prepare falls from 40.32 to 18.45 us.
Three real-model staged/direct process pairs show much smaller request-latency
changes: paired median reductions of 0.59–1.92%, with mixed directions across
repetitions in four of six cases. Direct writes remain opt-in. See the
[prepare and serving report](docs/optimization-prepare-iterations-20261030.md)
for all configurations, compilation-related reruns and evidence limits.

| Component | Target |
| --- | --- |
| Python | 3.10.20 |
| vLLM | Official `0.30.0+cu129` build |
| PyTorch | 2.13.0+cu129 |
| CUDA toolkit | 12.9; nvcc 12.9.86 |
| GPU/kernel geometry | SM90, BF16 KV, head dimension 128; 128-token logical objects; serving validated with 128-token physical pages |
| HBM attention | GQA group 1, 2, 4, or 8; no speculative decoding |
| Native connector | TP=PP=1, eager, one full-attention cache group, text decoder; no LoRA, multimodal inputs or speculative decoding |

Serving was validated with TP=1 and eager execution. Kernel CUDA Graph tests
passed; full-model graph serving and TP head sharding remain unverified.
Exact KV reconstruction does not imply identical generated tokens across
different attention implementations.

## Install and run

Run these commands from the repository root in a fresh Python 3.10.20
environment. Configure `CUDA_HOME` for your externally installed CUDA 12.9
toolkit before compiling GPU operations.

```bash
python -m pip install -r requirements/vllm-cu129.txt
python -m pip install -e '.[dev]'
```

Supply a compatible model and calibrated tables. The calibration CLI accepts
captured BF16 bit patterns as a uint16 array shaped
`[samples, layers, 2, 128, kv_heads, 128]`:

```bash
python -m fragpack.compression.attn_tables /path/to/kv.npy /path/to/tables.npz
```

For the native-attention external codec, install the optional pinned LMCache
dependency and use the connector configuration and offline validation entry in
[development](docs/development.md). It selects `FragPackNativeConnector` through
`kv_connector_module_path`, with matching tables/identity and an existing CPU L1
endpoint; `mode` is fixed to `compressed` or `raw` at startup. Raw and compressed
use separate identities. See the [connector contract](docs/design/native-codec-vllm-connector.md)
for lifecycle and supported-layout constraints.

The following command starts the **HBM-resident CUSTOM route**, after reserving
an idle supported GPU:

```bash
export VLLM_PLUGINS=fragpack
export VLLM_USE_V2_MODEL_RUNNER=1
export FRAGPACK_TABLES=/path/to/tables.npz

python -m vllm.entrypoints.cli.main serve /path/to/model \
    --attention-backend CUSTOM \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --block-size 128 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --enforce-eager
```

TP=1 eager serving was tested with Qwen3-30B-A3B-Instruct-2507-FP8 and BF16 KV;
other models require their own validation. `FRAGPACK_OVERFLOW` sets the number
of raw overflow records, not bytes. Its default is 8,192 records. Model files,
calibration data, and raw experiment artifacts are not bundled.

The experimental `FRAGPACK_PREFILL=fused` startup option consumes compressed
records directly inside prefill attention, without allocating or gathering into
a BF16 prefill K/V workspace. Open-block write staging remains part of the store.
Short-query batches can split history across CTAs and combine FP32 attention
partials; their separately reported workspace is reused across layers.
The default `FRAGPACK_PREFILL=gather` retains FlashAttention over a decoded
workspace for comparison. Select the mode before starting the worker; runtime
switching is not supported. See [HBM design](docs/design/hbm-resident-kv.md) for the
ownership and causal-attention contracts.

The independent opt-in `FRAGPACK_WRITE=direct` option works with
`FRAGPACK_PREFILL=gather`. Complete blocks generated in one step go directly
from the model's original K/V tensors to the encoder, without a kernel-order
scratch round trip. Gather also reads those original tensors, so FA3 still
receives BF16 pages without decoding newly encoded data. Partial tails keep
staging. This saves transfers, not the current preallocated scratch capacity.
`FRAGPACK_WRITE=staged` remains the default and the comparison path; combining
direct writes with experimental fused prefill is rejected at startup.

## Development

The [system design documentation](docs/design/README.md) records the current
architecture, component ownership, complete data flows, HBM/attention paths and
LMCache integration. It follows the original DaseR design-document structure;
implemented behavior and the remaining integration roadmap are kept explicit.

The optional LMCache 0.5.6 integration supplies the CPU L1 byte service used by
both routes. The [HBM explicit API](docs/lmcache-plugin-056.md) exports/imports
existing compressed records and still needs scheduler integration. The
[native connector](docs/design/native-codec-vllm-connector.md) already manages
real request saves and restores through public vLLM hooks. Neither route enables
disk-tier recovery, and installing the server module alone does not configure a
model's connector.

Start with [development](docs/development.md), [system design](docs/design/README.md),
and [contributing](CONTRIBUTING.md). The default test suite is CPU-only; GPU tests
are selected explicitly. All source remains under [Apache-2.0](LICENSE).

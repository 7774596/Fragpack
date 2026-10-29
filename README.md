# FragPack

FragPack is an experimental lossless BF16 KV-cache representation with a fused
decode-attention kernel and a vLLM adapter. Completed blocks stay compressed in
GPU memory; decode reconstructs operands inside the attention kernel. Prefill
uses a temporary BF16 workspace, and incomplete blocks remain writable until
they reach 128 tokens.

This standalone extraction comes from [DaseR](https://github.com/GentleCold/DaseR)
via the [demo fork](https://github.com/7774596/DaseR) at revision
`9f113cd0bc973e9d84aa538610343c7c18e9171c` under Apache-2.0. It contains
the representation, GPU store, kernels, and vLLM integration. External cache
services and their placement policies are outside this package.

## Status and scope

The standalone wheel passed 24 CPU tests, 18 GPU tests, and a seven-request
TP=1 eager serving check on H800. Validation blocked DaseR imports in all
processes. These are functional results; standalone performance comparisons
remain unrun. See the [validation report](docs/validation-20261029.md) and
[migration provenance](docs/migration.md).

| Component | Target |
| --- | --- |
| Python | 3.10.20 |
| vLLM | Official `0.30.0+cu129` build |
| PyTorch | 2.13.0+cu129 |
| CUDA toolkit | 12.9; nvcc 12.9.86 |
| GPU/kernel geometry | SM90, BF16 KV, head dimension 128, block size 128 |
| Attention | GQA group 1, 2, 4, or 8; no speculative decoding |

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

After reserving an idle supported GPU, the initial serving configuration is:

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

## Development

Start with [development](docs/development.md), [architecture](docs/architecture.md),
and [contributing](CONTRIBUTING.md). The default test suite is CPU-only; GPU tests
are selected explicitly. All source remains under [Apache-2.0](LICENSE).

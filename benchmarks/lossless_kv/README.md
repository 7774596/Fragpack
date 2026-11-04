# Lossless KV evaluation harness

End-to-end and kernel benchmarks for the lossless KV codec (`tier` mode)
and FragPack attention (`pool` mode) on LMCache's MP server and vLLM.
Run every module from the repository root; requires vLLM >= 0.30.

## Arms

| arm           | vLLM attention     | LMCache                                   |
|---------------|--------------------|-------------------------------------------|
| `native`      | FLASH_ATTN         | none (GPU prefix cache only)              |
| `lmcache`     | FLASH_ATTN         | raw L1 (+ bounded fs_native L2)           |
| `tier`        | FLASH_ATTN         | lossless codec in L1 and L2               |
| `pool`        | CUSTOM (FragPack)  | FragPack records as blobs in L1 (+ L2)    |
| `pool-native` | CUSTOM (FragPack)  | none                                      |

`--kv-cache-dtype fp8` runs any arm with an FP8 E4M3 KV cache (use FP8
codebooks and tables). Every arm gets the same `--gpu-kv-size`; CUSTOM arms
give vLLM 858 MiB less, which FragPack allocates after KV sizing. LMCache
pins L1 non-lazily, and vLLM starts only after LMCache's `/status` reports
the full L1 and a healthy L2 adapter. Each arm's scratch store is deleted
when its services stop.

## Calibration

```bash
# KV dump from real text (or --trajectories agent.parquet)
python -m benchmarks.lossless_kv.capture_kv --model MODEL --text corpus.txt \
    --out kv.npy --tokens 4096
# tables of both arms: tier takes books/, pool books/tables.npz (for an FP8
# cache add --fp8 --rope-theta THETA --max-positions N)
python -m benchmarks.lossless_kv.calibrate kv.npy books/ --model-id NAME
# stored ratios of both formats and the order-0 bound
python -m benchmarks.lossless_kv.kernels.kv_ratio kv.npy NAME
```

## End-to-end runs

`run_arm.sh` waits for idle GPUs and runs one arm with the watchdog, the
metrics sampler and an nvidia-smi log:

```bash
# AgentX closed loop, at most C requests admitted at a time (slotgate.py
# runs under the AgentX environment's Python and starts its agentx)
SLOTS=C GPUS=0,1 TP=2 benchmarks/lossless_kv/run_arm.sh tier OUT SCRATCH \
    --model MODEL --gpu-kv-size 54gib --l1-size 400gib --l2-size 2tib \
    --codebooks books/ \
    --agentx-bin "AGENTX_PYTHON benchmarks/lossless_kv/slotgate.py" \
    --agentx-args "AGENTX_PROFILE_OPTIONS"

# cold/warm long-context trace replay
python -m benchmarks.lossless_kv.w2_trace w2.jsonl
GPUS=0 benchmarks/lossless_kv/run_arm.sh pool OUT SCRATCH --model MODEL \
    --fragpack-tables tables.npz --trace w2.jsonl --l1-size 32gib --l2-size 0

# RULER-style accuracy, cold then warm through the tiers
GPUS=0 benchmarks/lossless_kv/run_arm.sh tier OUT SCRATCH --model MODEL \
    --codebooks books/ --agentx-bin "python -m benchmarks.lossless_kv.accuracy" \
    --agentx-args "--tokenizer MODEL --haystack wikitext.txt --n 40"
```

`python -m benchmarks.lossless_kv.arms --help` lists every option.
Summaries:

```bash
python -m benchmarks.lossless_kv.report agentx OUT/tier OUT/lmcache
python -m benchmarks.lossless_kv.report cached OUT/tier
python -m benchmarks.lossless_kv.report queue OUT/tier.metrics.tsv OUT/tier
python -m benchmarks.lossless_kv.report w2 OUT/pool OUT/lmcache
```

## Kernels

```bash
# codec encode/decode throughput and ratio versus general-purpose codecs
python -m benchmarks.lossless_kv.kernels.codec_pareto_bench --help
# FragPack decode attention step versus FLASH_ATTN
python -m benchmarks.lossless_kv.kernels.fragpack_step_bench --help
# sustained (power-capped) decode attention throughput and energy
python -m benchmarks.lossless_kv.kernels.fused_power --dump kv.npy --help
# fused attention error against an fp64 reference
python -m benchmarks.lossless_kv.kernels.fused_error --dump kv.npy --help
```

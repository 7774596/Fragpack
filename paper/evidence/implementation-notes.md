# Implementation and measurement notes

These notes document the implementations and measurement protocol behind the
recorded results. The current `kvfold_update4_b` retains the Table I and II
values from the earlier drafts. Source IDs are relative to the original `daser_bench` archive;
raw logs are not bundled with this paper. The historical JSON is unchanged.

## Demonstration setup

The saved prefix-reuse interaction used Qwen3-4B-Instruct-2507 in eager pool
mode on one H800 PCIe GPU, with BF16 weights and KV, TP=1, 128-token blocks,
a 4 GiB GPU KV budget, and 1 GiB of LMCache DRAM.

The recorded capacity experiments used Qwen3-30B-A3B-Instruct with FP8 weights
and BF16 KV, TP=2, and two H800 SXM GPUs connected over PCIe. vLLM used
128-token blocks and 8,192-token chunked prefill. SemiAnalysis's InferenceX
AgentX benchmark replayed Claude Code session traces, including subagent
requests and recorded think times. The harness version was 1.0.7, the trace
dataset was `cc_traces_weka_062126_256k`, and the random seed was 42.
The workload used snapshot priming and ten additional warmup requests per
lane. In the paper, P counts live root sessions; a FIFO proxy admits at most
S concurrent requests. The measurement window, completion grace, exclusions,
and metric definitions are specified below.

## AgentX implementation mapping

Revisions come from `tp2/<run>/status`, rather than inferred commit dates.
Each arm contains `agentx/profile_export.jsonl`, `agentx/slotgate.jsonl`,
`agentx/logs/aiperf.log`, and service logs; its queue samples are in the sibling
`<arm>.metrics.tsv`. The checked values reproduce the printed table rounding.

| Table row | Run / arm / backend | LMCache | Completed | Cancelled | Included span (s) |
| --- | --- | --- | ---: | ---: | ---: |
| I Raw S=8; II Raw 2 TiB | `m11/lmn-c64-s8/lmcache` | `8419292` | 596 | 74 | 1827.936 |
| I Pool S=8 | `m80/lmp-c64-s8/lmcache` | `91ec45c` | 615 | 80 | 1823.089 |
| I Raw S=12 | `m87/lmn-c64-s12/lmcache` | `91ec45c` | 623 | 82 | 1828.079 |
| I Pool S=12 | `m88/lmp-c64-s12/lmcache` | `91ec45c` | 668 | 79 | 1827.317 |
| II Raw 640 GiB | `m34/lmn-c64-s8/lmcache` | `37b3ca6` | 281 | 96 | 1815.332 |
| II Tier 640 GiB | `m18/lmz-c64-s8/lmcache-codec` | `37b3ca6` | 536 | 84 | 1829.517 |
| II Tier 2 TiB | `m15/lmz-c64-s8/lmcache-codec` | `37b3ca6` | 578 | 83 | 1821.684 |

The harness was `696d555` for m11, `9827ba1` for m15/m18, and
`9827ba1+dirty` for m34/m80/m87/m88. LMCache had no dirty marker in these logs.
The pool backend at `91ec45c`, `lmcache/v1/lossless_kv/fragpack/backend.py::forward`,
gathers prefill KV before FlashAttention and uses fused `decode_attention` for
decode. The recorded CUSTOM configuration and vLLM 0.30 SM90 default select FA3.

The harness sends for 1800 s, allows 30 s completion grace, then cancels
unfinished requests. Metrics use completed profiling exports, including grace
completions. Engine TTFT subtracts proxy wait; per-request ITL is decode duration
divided by output tokens minus one. Goodput counts requests meeting both limits
(2 s TTFT, 40 ms mean ITL), divided by the included-record span, times 60.
No exported completed record lacked TTFT in these seven runs. Cancelled requests
are absent from that export; they are not equivalent to requests without a first
token. See `tp2/slotstat.py` and LMCache `benchmarks/lossless_kv/report.py::agentx`,
`cached`, and `queue`. The earlier local audit JSON is retained in the external workspace archive;
the mapping, counts, and definitions are retained here.

## Codec revisions and correctness scope

- `c0b2273` (2026-11-05) changed tier records from per-plane tables (v1) to
  per-head tables (v2) and shared offline calibration with pool records. Pool
  already used per-head tables; its BF16 calibration algorithm, record format,
  backend, and attention kernels were unchanged. Its calibration entry point moved.
- `d096273` (2026-11-07) introduced tier grouped tile indices (v3), changing
  index size and decoding. It did not change the pool record or attention path.
  Tier v1 ratios, kernel rates, and correctness checks are not v3 measurements;
  pool results remain measurements of `91ec45c`, not a rerun of `d096273`.
- The 1229/376 GB/s comparison comes from `tab3/moe_r1.jsonl`, lines 15/36:
  the pre-migration DaseR codec and SplitZip, respectively. `tab3/run.sh` names
  DaseR's `benchmarks.research.codec_pareto_bench`; the run SHA was not recorded.
  This per-plane/tile-prefix implementation preceded the LMCache package added
  in `fa5aada`. Both restored byte-exactly; the numbers are not current KVFold rates.
- The 192 matching outputs come from `tp2/m28` (160 GiB DRAM) and `m29`
  (220 GiB), both LMCache `37b3ca6`, harness `9827ba1+dirty`. Comparing raw/tier
  `requests.jsonl` by sample ID gives 96/96 matching generated-text hashes per
  run, with no errors. These are cold/warm tier comparisons, not pool versus FA3.
- `live-run.json` records the November 8 interaction at `d096273`. Its After
  Build Inspector checks export/import integrity of an already encoded model
  block through host memory and an independent GPU store. It does not compare
  the initial encoding with the original model-produced BF16 or validate Restore.
  Separately, `record-example.json` checks CPU encode/decode of the synthetic
  Figure 2 record; it is not a model-wide GPU first-encoding check.

## Budget interpretation and remaining gaps

Qwen3-30B-A3B uses 96 KiB/token overall, or 48 KiB per TP=2 rank. Pool logs
for m80/m88 reserve 1048 MiB staging/scratch/overflow plus 256 MiB prefill
workspace: 1304 MiB (about 1.3 GiB) inside the 54 GiB KV budget, leaving
1,585,280 tokens (the printed 1.59M). Raw holds 1,179,648 tokens. The Industry
source reports about 1.0 GiB/GPU of tier-codec buffers outside this budget;
pool transfer buffers are separate and were not verified to have that size.
Weights, activations, and cache-server CUDA contexts are also outside the KV budget.

Unresolved: the dirty harness patchsets, the microbenchmark run SHA, and exact
no-first-token counts among cancelled requests. Proxy first-byte timestamps
cannot establish the last item. Different calibration samples or auxiliary
allocations require fresh capacity/performance checks even when pool code is unchanged.

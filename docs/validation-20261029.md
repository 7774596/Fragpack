# Standalone migration validation

## Environment and scope

Validated on 2026-10-29 with one exclusively allocated H800 (SM90), Python
3.10.20, vLLM 0.30.0+cu129, PyTorch 2.13.0+cu129, CUDA toolkit 12.9
(nvcc 12.9.86), and Qwen3-30B-A3B-Instruct-2507-FP8. Model weights are FP8;
KV remains BF16. Serving used TP=1, the V2 runner, eager execution, APC,
128-token blocks, a maximum sequence length of 8192, four sequence slots,
and a 2048-token scheduling budget.

A dedicated local virtual environment installed the FragPack wheel while
reusing the existing reference environment's inference dependencies. This
validates package isolation from DaseR, not a fresh download/install of the
entire pinned stack. A startup import guard rejected all `daser` imports in
the parent and spawned workers. Only the `fragpack` plugin was enabled.
No external cache service or KV connector was configured.

## Results

| Check | Result |
| --- | --- |
| Wheel and source distribution | Built; wheel includes all four native source/header resources, source distribution includes requirements and test tools |
| Installed package away from checkout | Passed; entry point discovered, resources found, DaseR import blocked |
| CPU contracts | 24 passed, 0 skipped |
| CUDA store and decode contracts | 18 passed, 0 skipped; native extension compiled from the installed wheel |
| Kernel CUDA Graph replay | Passed within the CUDA suite, including ragged/padded batches and GQA groups 1/2/4/8 |
| Real TP=1 eager serving | Seven requests passed on each of two fresh server runs; cold-prefill budget and APC assertions passed |
| Full-model CUDA Graph serving / TP>1 | Not run |
| Performance comparison | Not run |
| External cache integration | Outside this extraction |

In the final serving run the first request had 2309 input tokens, zero cached
tokens, and 17 generated tokens. Its uncached input exceeded the configured
2048-token scheduling budget. Other requests exercised 131/259/389/519-token
prompts, up to 137 generated tokens, three concurrent HTTP calls, and repeated
prefixes. The repeated 259- and 519-token prompts reported 256 and 512 cached
tokens respectively. All seven returned HTTP 200 and the requested token counts.

The core suite checks exact KV round trips, raw fallback, block handoff,
blob import rejection, and fused decode against an independent numeric
reference. It does not establish identical generated tokens across attention
implementations.

Serving used a configured KV arena of 4,294,967,296 bytes. The adapter reported
48 layers and 469 blocks, with 190,464 bytes per block-layer versus 262,144 raw
bytes. Staging, overflow, metadata, tables and attention workspaces are extra;
this run is not an equal-total-memory performance comparison.

The initial isolated-environment probe failed before CUDA because a nested
venv did not inherit the reference venv's dependencies. A local site-directory
link corrected this; no inference dependency or third-party source was edited.

## Reproduction and retained evidence

Run the CPU/GPU checks and the standalone HTTP client described in
[development](development.md). The local evidence records the wheel and source
hashes, model/table identity, dependency versions, launch arguments, GPU process
samples, JUnit output, server logs, and full HTTP request/response reports.
The initial run is identified as `20261029-standalone-gpu4-02`; the cold-prefill
verification is `20261029-standalone-gpu4-03`. Hardware paths and model data are
kept outside the repository.

The first run's 2309-token request reused 384 cached tokens, leaving 1925
uncached tokens, so it did not establish chunked prefill under a 2048-token
budget. The follow-up starts with that request on a fresh server and explicitly
checks the uncached-token count. Preserve this distinction when interpreting
the initial JSON report.

Both services were stopped, GPU processes released, and generated JIT caches
and scratch removed. Shutdown logs retain vLLM's forced-engine-stop and
multiprocessing semaphore warnings; these are not hidden as a clean-shutdown
claim. No DaseR service or third-party source files were changed.

Full-model graph execution, tensor parallelism, sanitizer coverage, clean
network installation and raw-versus-FragPack performance remain separate
checks. Historical DaseR performance results are not standalone measurements.

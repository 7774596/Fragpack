# KVFold update 4 B: revision ledger

This ledger maps the 15 requirements in `paper/修改方案.txt` to changes from
`paper/kvfold_update3_b.tex` to `paper/kvfold_update4_b.tex`. The new version
uses the Fig. 3 B layout. The update3 baseline and retired figure variants are
preserved in the external archive recorded in `evidence/workspace-cleanup.json`.
Source line numbers refer to the completed revision;
section names, figure labels, and quoted text provide stable anchors.
The accompanying `kvfold_update4_b.diff` contains the exact manuscript, SVG,
TikZ, and added-bibliography differences.

The changes revise presentation, terminology, units, and related-work coverage.
They do not report new experiments. The recorded numerical results and their
implementation boundaries remain in place. Acknowledgment remains empty.

## 01. Use KVFold consistently

**Location:** `kvfold_update4_b.tex:133,156,172,355`; Fig. 1 source
`images/fig1_architecture_update4.svg:16,82`.

| Before | After |
| --- | --- |
| `Pool mode stores FragPack records` | `pool mode exports its compressed GPU records unchanged` |
| `Each FragPack record holds 128 tokens` | `Each KVFold record holds 128 tokens` |
| `An illustrative FragPack record` | `An illustrative KVFold record` |
| `DaseR's pre-migration codec restored directly into pages at 1,229 GB/s.` | `An earlier KVFold prototype restored directly into pages at 1,229 GB/s.` |
| Fig. 1: `FragPack pool`; `Same FragPack records` | `KVFold pool`; `Same KVFold records` |

Historical repository names and identifiers in the evidence files are preserved
for traceability. The prototype rate is still a recorded microbenchmark, not a
measurement of the current implementation.

## 02. Expand LLM and BF16 on first use

**Location:** `kvfold_update4_b.tex:40,45,73`, Abstract and Introduction opening.

- **Before:** `LLM agents often reuse ...`; `the original BF16 bits`;
  `particularly useful for LLM agents`.
- **After:** `Large language model (LLM) agents often reuse ...`;
  `the original bfloat16 (BF16) bits`;
  `particularly useful for large language model (LLM) agents`.

The Abstract and main text each define LLM independently, as requested.

## 03. Specify when tier encoding begins

**Location:** `kvfold_update4_b.tex:232`, Section II-C, paragraph beginning
`Tier mode integrates encoding and recovery`.

- **Before:** `After the producing step, the server encodes full chunks into ...`
- **After:** `After the corresponding model step completes, the server encodes
  full chunks into ...`

The wording follows the producer-event ordering in LMCache revision `d096273`:
`lmcache/integration/vllm/lmcache_mp_connector.py:983,1011–1018` records a producer
event at the forward-context save boundary;
`lmcache/v1/multiprocess/transfer_context/worker_transfer.py:546,560–561` records
that event on the current stream;
`lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py:798,872` waits on the
imported producer event before submitting the store; and
`lmcache/v1/multiprocess/lossless_kv_transfer.py:720` calls
`encode_engine_blocks` on that ordered stream. Completion here is enforced by
CUDA stream dependencies; this edit does not imply an added host synchronization.

## 04. Make the Inspector comparison explicit

**Location:** `kvfold_update4_b.tex:342`, Scenario 1, `After Build` paragraph.

- **Before:** `comparing decodes of an already encoded block before and after
  transfer through host memory into an independent GPU scratch store`.
- **After:** `comparing the decoded KV values before and after transferring an
  encoded block through host memory into an independent GPU scratch store`.

The preceding term `export/import integrity` is retained. This remains a
comparison between two decodes of an already encoded block, not a claim that
the Inspector compares the first encode/decode against model-produced BF16.
The existing CPU sample check and historical output check remain separate.

## 05. Put units and queue meaning in the table headers

**Location:** `kvfold_update4_b.tex:284–293,305–314`, tables `tab:gpu` and
`tab:tiers`.

| Before | After |
| --- | --- |
| Table I: `Wait` | `Mean queue` |
| Tables I and II: `TTFT` | `TTFT (s)` |
| Tables I and II: `Goodput` | `Goodput (req/min)` |
| Table I caption: `Wait is mean vLLM queue length` | `Mean queue is the average vLLM queue length` |

The TTFT and goodput headers place units on a second line to preserve column
widths. Table values are unchanged, except for the equivalent capacity labels
documented in item 12. TTFT remains mean engine TTFT; goodput retains its existing
request-level latency thresholds.

## 06. State that Fig. 3 shows selected console elements

**Location:** `kvfold_update4_b.tex:267`, caption of `fig:views`.

- **Before:** `Recomposed console views. (1) Saved Qwen3-4B interaction: ...`
- **After:** `Recomposed console views. Selected console elements are shown.
  (1) Saved Qwen3-4B interaction: ...`

The caption now distinguishes the displayed selection from the console's full
set of TTFT, checksum, and mismatch-count fields described in Scenario 1.

## 07. Remove the Fig. 1 subfigure letters

**Location:** `images/fig1_architecture_update4.svg:7,92`;
`kvfold_update4_b.tex:132` includes the new rendered PDF.

- **Before:** `(a) Pool mode`; `(b) Tier mode`.
- **After:** `Pool mode`; `Tier mode`.

The source is a new SVG derived from `fig1_architecture_optimized.svg`, rendered
as `images/figure1-kvfold_update4.pdf` and its PNG preview. Fig. 1's original
two-path organization is retained. The previous SVG and update 3 figure remain
available in that archive.

## 08. Define P and S precisely

**Location:** `kvfold_update4_b.tex:261`, Demonstration Plan setup paragraph.

- **Before:** `P counts live root sessions and S limits concurrent requests.`
- **After:** `P denotes the number of live root sessions, and S is the maximum
  number of concurrent requests.`

The notation and all recorded P/S configurations remain unchanged.

## 09. Limit the shared bar scale to each metric

**Location:** `kvfold_update4_b.tex:273`, caption of `fig:views`.

- **Before:** `Paired bars share a zero-based scale.`
- **After:** `Within each metric, paired bars use the same zero-based scale.`

This describes the existing bar pairs without suggesting that tokens, queue
length, TTFT, hit share, and stored-size ratio share one cross-metric scale.
No bar values or scales are changed by this wording edit.

## 10. Expand FA3 in the main text

**Location:** `kvfold_update4_b.tex:194`, Section II-B, prefill paragraph.

- **Before:** `an unmodified FA3 kernel`.
- **After:** `an unmodified FlashAttention-3 (FA3) kernel`.

The existing Fig. 1 caption expansion and citation are retained. The prefill
path remains gather/decode followed by the unmodified attention kernel.

## 11. Use Fig. consistently in prose

**Location:** `kvfold_update4_b.tex:144,164,249,383`, references to
`fig:architecture`, `fig:record`, and `fig:views` in System Description, the
Demonstration Plan opening, and Scenario 3.

- **Before:** `Figure~\ref{fig:architecture}`;
  `Figure~\ref{fig:record}`; `Figure~\ref{fig:views}`.
- **After:** `Fig.~\ref{fig:architecture}`;
  `Fig.~\ref{fig:record}`; `Fig.~\ref{fig:views}`.

The LaTeX labels are unchanged. Fig. 1 is referenced as a complete two-path
diagram after removal of its letters; Fig. 3 retains numbered scenarios
`(1)`, `(2)`, and `(3)` in its caption and `Scenario 3` in the related discussion.

## 12. Use binary capacity units at readable scales

Following the author's clarification, capacities use KiB, MiB, GiB, or TiB
according to magnitude rather than being restricted to KiB and GiB.

**Location:** `kvfold_update4_b.tex`, Fig. 2 caption, pool-page example,
Fig. 3 caption, setup paragraph, both tables, restoration microbenchmark, and
Scenario 3; `figures/record_layout_update4.tex:51–52`;
`figures/fig3_b.js:145–156` under `--update4`.

| Before | After | Basis |
| --- | --- | --- |
| Pool page: `190,464 B` versus `262,144 B` | `186 KiB` versus `256 KiB` | Exact division by 1,024; `docs/design/hbm-resident-kv.md:63`. |
| Fig. 2 fixed streams: `16,384 B`; `6,144 B` | `16 KiB`; `6 KiB` | Exact division by 1,024. |
| Fig. 2 record limit: `2,048 B` | `2 KiB` | Exact division by 1,024. |
| Fig. 3 Inspector: `18.00 MiB`; `12.73 MiB` | `18.00 MiB`; `12.73 MiB` (retained) | Formatted from exact archived bytes with two decimal places. |
| NVMe capacity: `2 TiB` | `2 TiB` (retained) | The original scale already uses a binary unit. |
| Restoration input: `3.2 GB` | `3 GiB` | Capture geometry and benchmark expansion give 3,221,225,472 bytes. |

For the Inspector, `evidence/live-run.json:59–61` records 18,874,368 raw bytes
and 13,344,768 encoded bytes. The encoded value includes the blob header and
4,096-byte alignment. `fig3_b.js --update4` validates these byte counts and
formats them in MiB with two decimal places for the separate update 4 outputs.

For the restoration benchmark, archived `tab3/run.sh:9,12` identifies the
Qwen3-30B capture and historical benchmark invocation. The capture header has
shape `(32, 48, 2, 128, 4, 128)` and dtype `u2`. The available runtime copy,
`benchmarks/lossless_kv/kernels/codec_pareto_bench.py:732–737,812–816`, defaults
to 256 evaluation blocks and cyclically selects evaluation slots. Its timed
payload is therefore:

```text
256 × 48 × 2 × 128 × 4 × 128 × 2 bytes = 3,221,225,472 bytes = 3 GiB.
```

The old `3.2 GB` was a rounded decimal label. The original run SHA remains
unrecorded, as documented in `evidence/implementation-notes.md:69–73`; the
conversion is not a new benchmark or proof that the current runtime reproduces
the historical rates.

The 64 B variable region, 48 B header, and 3 B/8 B padding lengths remain in
bytes because they describe the record's byte layout. The existing GB/s
throughput units also remain unchanged: the requirement concerns capacities,
and replacing GB/s with GiB/s would require converting the reported rates.

## 13. Add five relevant primary-source references

**Location:** `kvfold_update4_b.tex:84–103`, Introduction;
`references_update4.bib:12,25,38,51,65`;
`kvfold_update4_b.tex`, `\bibliography{references,references_update4}`.

**Before:** The related-work discussion covers PagedAttention, LMCache, KIVI,
CacheGen, nvCOMP, and SplitZip, without the following five works.

**After:** It adds compact mechanism-level coverage of prefix reuse, token
selection, offloading, and quantization:

> SGLang's RadixAttention uses a radix tree to reuse cached prefixes across
> requests. Other methods reduce the retained token set: H2O keeps recent and
> high-attention tokens, while StreamingLLM retains initial attention sinks with
> a recent-token window. FlexGen combines GPU/CPU/disk offloading with low-bit
> compression, and KVQuant tailors KV quantization to channel distributions and
> outliers.

The connecting claim now refers specifically to sharing, offloading, and
token-selection mechanisms, which do not by themselves reduce bytes per
retained KV value. FlexGen is discussed with quantization because it also
compresses the attention cache. No external performance numbers are added.

| New key | Work and verified primary source |
| --- | --- |
| `sglang` | Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, NeurIPS 2024. [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html). |
| `flexgen` | Sheng et al., *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU*, ICML 2023. [Official proceedings](https://proceedings.mlr.press/v202/sheng23a.html). |
| `kvquant` | Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*, NeurIPS 2024. [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html). |
| `h2o` | Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*, NeurIPS 2023. [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html). |
| `streamingllm` | Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, ICLR 2024. [Author paper](https://arxiv.org/abs/2309.17453); the [authors' project page](https://hanlab.mit.edu/projects/streamingllm) confirms the conference year. |

The additions contain five new scientific references and one bibliography
display-control record. The shared `references.bib` is unchanged by this
revision, and ZipFusion is not introduced.

## 14. End the Abstract with the demonstration actions

**Location:** `kvfold_update4_b.tex:50–53`, Abstract ending.

**Before:**

> Interactive tasks illustrate prefix reuse, while recorded workloads show how
> added capacity improves history retention and reduces time to first token
> under constrained cache budgets.

**After:**

> We demonstrate KVFold through an interactive coding session and recorded trace
> replays from the InferenceX AgentX benchmark: users inspect prefix reuse, GPU
> capacity, and history retention, and compare raw and compressed runs under
> fixed cache budgets.

This retains the established term `users` and identifies AgentX as a benchmark.
It separates the interactive coding session from the recorded trace results.

## 15. Add the demo setting and planned walkthrough

**Location:** `kvfold_update4_b.tex:252–260`, opening of Demonstration Plan.

**Before:**

> The interactive scenario uses Qwen3-4B in pool mode, with BF16 weights and KV.
> The recorded capacity experiments replayed Claude Code session traces using
> SemiAnalysis's InferenceX AgentX benchmark. They used Qwen3-30B-A3B with FP8
> weights and BF16 KV on two H800 SXM GPUs (TP 2).

**After:**

> The live prefix-reuse view runs Qwen3-4B in pool mode on one H800 PCIe GPU, with
> BF16 weights and KV; the capacity and retention views display recorded results
> from Claude Code session traces replayed with SemiAnalysis's InferenceX AgentX
> benchmark, using Qwen3-30B-A3B with FP8 weights and BF16 KV on two H800 SXM GPUs
> (TP 2). A planned eight-minute walkthrough covers Build/Append/Edit/Restore,
> then S=8 or 12, and finally 640 GiB or 2 TiB of NVMe capacity. The console
> reports cached-token counts, KV sizes, and TTFT for the live interaction, plus
> waiting requests and latency for the selected recorded runs.

The hardware and workload settings follow
`evidence/implementation-notes.md:9–19`. Eight minutes is a proposed walkthrough
duration, not a timed user study or measured run. Changing the historical
selectors displays archived results; it does not launch a new AgentX run.

## Figure and source changes

| Artifact | Update 4 B change |
| --- | --- |
| `images/fig1_architecture_update4.svg` | Separate editable Fig. 1 source with KVFold labels and no `(a)/(b)` letters; renders to `figure1-kvfold_update4.pdf` and PNG. |
| `figures/record_layout_update4.tex` | Separate Fig. 2 source; only the two fixed-stream capacities change from B to KiB. Scalar values, escape codes, byte-field lengths, and record layout remain unchanged. |
| `figures/fig3_b.js --update4` | Separate Fig. 3 B output mode reading exact Inspector bytes from the saved interaction. It generates a PPTX candidate; checked PDF, PNG, and PPTX exports are published as `images/fig3_b_update4.*`. Default Fig. 3 B output is unchanged. |
| `kvfold_update4_b.tex` | Uses the three update 4 figure inputs above and the additional bibliography file. |
| `references_update4.bib` | Five scientific additions and compact-display controls, isolated from the shared bibliography. |

## Layout and build integration

`Makefile` now defaults to `kvfold_update4_b` and includes the new bibliography
and versioned figures as dependencies. `render_architecture.py --update4`
exports the new Fig. 1 without replacing update3 assets. `README.md` identifies
the current manuscript and its archived predecessors.

Fig. 3 is declared after the Demonstration Plan setting and placed at the top
of page 4; the two aligned result tables appear at the bottom of page 3, after
Demonstration Plan and the Scenario 2/3 introductions. The shared table float
uses `[!b]` to keep the results below their narrative context. The five-page
version's manual reference-column break is removed. Figure sizes,
IEEE text size, and page margins are unchanged. The final revision is **four
pages including references**, with an empty Acknowledgment body and a positive
`3\baselineskip` reserve for three lines of text.

### Approved compression

The approved compression pass removed 227 words without changing the Abstract,
Introduction, result tables, scenarios, or evidence boundaries. The counts below
describe that pass, before the local wording refinements listed afterward.

| Location | Before → after (words) | Retained content |
| --- | --- | --- |
| System Description opening, lines 143–148 | 102 → 51 | Serving-stack responsibilities, versions, and plugin integration. |
| Section II-A, lines 151–154 | 72 → 41 | Open-block staging, encoding after attention, and immutable reuse. |
| Section II-B, lines 192–195 | 85 → 43 | Current-layer BF16 gathering/decoding and unmodified FA3. |
| Section II-C closing, lines 242–244 | 73 → 31 | Encoded-length storage/prefetch allocation, rank ownership, and 4 KiB alignment. |
| Demonstration Plan opening, lines 248–250 | 52 → 29 | Live prefix interaction and recorded capacity/history views. |
| Fig. 1 caption, lines 133–137 | 76 → 38 | Pool/tier paths, process boundaries, LMCache tiers, and FA3 expansion. |

Bibliography display uses the first author plus `et al.` for works with more
than six authors and suppresses optional URL fields. All 13 cited works and
full metadata remain in the BibTeX sources; online resources retain their
`howpublished` links. No reference is removed.

The preceding capacity-unit clarification and exact Scenario 2 ending remain:
`At S=12, pool mode cuts mean TTFT from 5.16 s to 1.94 s and raises goodput from
10.6 to 14.9 versus raw.`

### Local wording refinements

Four suggested language edits are applied without changing technical content:

| Location | Before | After |
| --- | --- | --- |
| Demonstration opening, line 249 | `offers live prefix reuse and recorded ... views` | `offers a live prefix-reuse view and recorded ... views` |
| Setup metrics, line 260 | `interaction, and ... from the selected recorded runs` | `interaction, plus ... for the selected recorded runs` |
| System Description opening, line 143 | `changes KV representation and recovery` | `changes how KV is represented and recovered` |
| Prefill, line 194 | `copying open-block values` | `copying values from open blocks` |

The setup retains the explicit subject `The console` and describes recorded
results rather than a live trace replay. The allocation explanation and the
related-work transition retain their existing wording and scope. Tables stay
at the bottom of page 3; Figure 3 and the acknowledgment reserve remain on page 4.

## Validation status

- The ledger maps all 15 requested items to existing source edits and stable
  textual or figure anchors.
- The five bibliography keys are distinct and do not duplicate existing keys.
  Primary sources support the bibliographic metadata and added mechanism claims.
- Exact-byte conversions above have been checked arithmetically. No GPU
  experiment or fresh performance measurement was performed for this revision.
- The final `latexmk` build succeeds. All 13 references resolve; the log contains
  no warnings, undefined references, overfull boxes, or underfull boxes.
- All four rendered pages were inspected. Figure 1 labels, Figure 2 byte fields,
  Figure 3 Inspector sizes, and the two-line table headers are legible and do
  not overflow. Table captions, header baselines, and data rows align.
- The Acknowledgment reserve spans three body-text baselines (about 12.65 mm).
  A separate trial replacing the spacer with three actual text lines also
  compiles to four pages; the delivered manuscript retains the blank reserve.
- All 23 PDF fonts are embedded; the document contains no raster image objects.
- All eight result-table rows match update3 B. No performance value changed.
- Fig. 3 retains the original MiB labels, geometry, fonts, colors, and bar
  lengths. Its two Inspector labels are regenerated from archived byte counts.
- SHA-256 checks confirm that both update3 manuscript sources/PDFs, shared
  bibliography, original Figure 1 and Figure 3 B assets, the original Fig. 2
  source, and the supplied modification plan are unchanged.

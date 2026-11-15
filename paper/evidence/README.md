# Recorded evidence and figure provenance

These files preserve the evidence used by the KVFold demonstration paper.
Packaging and rebuilding the paper do not execute the model or create new
experimental results. `provenance.json` preserves the original source mappings
and asset checksums, including retired manuscripts, screenshots, and capture
records. Those files remain in the verified external archive identified by
[workspace-cleanup.json](workspace-cleanup.json); they are not build dependencies.
The current Figure 3 B uses the same saved-session and AgentX evidence, with
an interaction-focused first panel and six metric cards. Each paired bar uses
a shared zero baseline and scale.

| File | Scope |
| --- | --- |
| `live-run.json` | Saved November 8, 2026 Qwen3-4B interaction and model-block export/import integrity check |
| `paper-evidence.json` | Transcribed Qwen3-30B AgentX capacity results and measurement definitions from the author-supplied Industry manuscript |
| `evidence.json` | Additional historical native-codec experiment summaries; distinct from the AgentX results used in Figure 3 |
| `record-example.json` | CPU validation of the synthetic BF16 encoding example used in Figure 2 |
| `provenance.json` | Portable source identifiers, scope boundaries and asset checksums |
| `implementation-notes.md` | Full demonstration configurations and replay protocol, plus run revisions, metric definitions, correctness checks, and memory accounting |

The four input JSON files are copied byte for byte from the validated console
exports and encoding-example record. No fields, numeric values or scope notes
have been removed or rewritten. They contain no account credentials, service
addresses or private absolute filesystem paths.

## Historical source names

`paper-evidence.json` calls its result source `industry.pdf`. Its recorded
SHA-256 identifies **`../materials/main(11).pdf`**, dated November 4, 2026.
The separate `../materials/industry.pdf` is another author-supplied draft and
must not replace this source. The JSON's version-3 manuscript identifier and
the `docs/data/...` source identifiers in `evidence.json` describe historical
provenance; they are not dependencies for rebuilding the retained papers.

The Scenario 1 codec comparison is transcribed from Table II (page 7) of
`../materials/main(11).pdf`, which names the codec ExactKV. The original run
used the pre-migration DaseR implementation, as documented in
[implementation-notes.md](implementation-notes.md). It uses one H800
SXM, 3.2 GB of Qwen3-30B-A3B KV, and 256 chunks per call. Throughputs count
original bytes; the paged measurements include restoring into scattered
destination blocks, and every restore was checked byte-exact. These are
historical component measurements, separate from both the saved interactive
session and the AgentX serving comparisons. No codec-switching console
control or new benchmark run is implied by this manuscript addition.

The superseded Chinese briefing, original console screenshots, and capture
records are preserved in the external archive. Their original hashes remain
in `provenance.json`. No screenshot was recaptured during cleanup.

## Interpretation

- The saved functional session is not a repeated latency benchmark. Its
  Inspector checks export/import of one already encoded model block after
  Build, independently of Restore; it does not verify the initial encoding
  against the model-produced BF16. The synthetic CPU encode/decode check is separate.
- The packaged AgentX JSON contains aggregates, not request time series or
  replicate samples. A subsequent audit identified the underlying run revisions
  and checked their request records; see [implementation-notes.md](implementation-notes.md).
- Recorded tier runs used per-plane tables and the earlier tile index. Shared
  calibration and grouped indices changed the tier implementation; the pool's
  BF16 codec and attention path were unchanged by those updates. Rebuilding
  the paper does not rerun either implementation.
- Cached-token share includes GPU and external hits. Engine TTFT excludes FIFO
  proxy waiting. The historical JSON retains its original scope notes; the
  subsequent audit clarifies completion grace and cancellation in
  [implementation-notes.md](implementation-notes.md).
- Image and PDF hashes identify the distributed artifacts. A rebuild may
  change PDF metadata; text, geometry and rendering are the relevant checks.

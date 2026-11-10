# Recorded evidence and figure provenance

These files preserve the evidence used by the KVFold demonstration paper.
Packaging and rebuilding the paper do not execute the model or create new
experimental results. `provenance.json` records source mappings, quantitative
figure scales, layout dimensions and SHA-256 hashes for the distributed files.

| File | Scope |
| --- | --- |
| `live-run.json` | Saved November 8, 2026 Qwen3-4B interactive session and model-block validation |
| `paper-evidence.json` | Transcribed Qwen3-30B AgentX capacity results and measurement definitions from the author-supplied Industry manuscript |
| `evidence.json` | Additional historical native-codec experiment summaries; distinct from the AgentX results used in Figure 3 |
| `record-example.json` | CPU validation of the synthetic BF16 encoding example used in Figure 2 |
| `provenance.json` | Portable source identifiers, scope boundaries and asset checksums |

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

The original screenshot bytes required by versions 8 and 14 and the Chinese
introduction are in `../images/console-*.png`. They were restored from a
verified archive. Their hashes and the original capture-record hashes are
included in `provenance.json`. No screenshot was recaptured during packaging.

## Interpretation

- The saved functional session is not a repeated latency benchmark. Its
  Inspector validates one model block after Build independently of Restore.
- The AgentX results are historical aggregates. They do not provide request
  time series or replicate samples. Exact run commits were not identified in
  the supplied result manuscript.
- Recorded tier runs predate shared per-head calibration and grouped tile
  indices. Rebuilding this paper does not validate the current codec revision.
- Cached-token share includes GPU and external hits. Engine TTFT excludes FIFO
  proxy waiting; the measurement window and exclusion rules remain in the JSON.
- Image and PDF hashes identify the distributed artifacts. A rebuild may
  change PDF metadata; text, geometry and rendering are the relevant checks.

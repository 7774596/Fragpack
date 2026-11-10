# KVFold demonstration paper

The current manuscript is [fragpack_demo(16).pdf](fragpack_demo%2816%29.pdf),
with editable source in `fragpack_demo(16).tex`. It presents GPU-resident
FragPack records and the external-cache codec as separate integration paths.
The paper uses gather + FA3 for pool-mode prefill. Historical measurements
and the saved functional demonstration retain their original evidence scope.

## Package contents

| Location | Contents |
| --- | --- |
| `fragpack_demo(16).tex` / `.pdf` | Current four-page paper with the KV data-flow Figure 3; empty Acknowledgment body |
| `fragpack_demo(15).tex` / `.pdf` | Four-page alternative with the prompt, prefix strips and response view |
| `fragpack_demo(14).tex` / `.pdf` | Retained interaction-view revision |
| `fragpack_demo(8).tex` / `.pdf` | Retained screenshot-based baseline |
| `kvfold-demo-introduction.tex` / `.pdf` | Four-page Chinese paper and demonstration introduction |
| `figures/` | Editable TikZ architecture, record example and demonstration figures |
| `images/` | Figure PDFs, PNG previews, captions and original console captures |
| `references.bib` | Bibliography shared by the retained manuscripts |
| [evidence/](evidence/README.md) | Recorded evidence, provenance and record-example validation |
| `materials/` | Author-supplied design references and implementation notes |

Earlier retired manuscripts are not part of this package. Some retained
figure filenames contain older version numbers because they are shared
inputs, not obsolete files. The screenshot assets required by versions 8
and 14 and the Chinese introduction are included with their original bytes.

## Build

Building the paper requires Python 3, Make, `latexmk`, pdfLaTeX, BibTeX,
IEEEtran and the packages imported by the manuscript preamble. Standalone
figure export also uses `preview.sty` and Poppler's `pdftoppm`.
The Chinese introduction requires XeLaTeX, `ctex`, Noto Sans CJK SC and
DejaVu Sans / Sans Mono. Model weights, GPUs and running services are not required.

From the repository root:

```sh
make -C paper                    # Current version 16
make -C paper VERSION=15         # Alternative Figure 3
make -C paper VERSION=14         # Retained interaction revision
make -C paper VERSION=8          # Screenshot baseline
make -C paper briefing           # Chinese introduction
make -C paper figures            # All three figures from version 16
make -C paper VERSION=15 figures FIGURES=3
make -C paper dataflow           # Standalone full-height data-flow figure
```

Build intermediates stay in ignored `paper/build/` subdirectories. Successful
builds update the corresponding reading PDF or figure exports in place.
The complete package can also be copied out of the repository and built with
`make -C <copied-paper-directory>`; no external worktree is needed.

## Figure variants

Versions 15 and 16 share the architecture and record example. Version 15 uses
`figures/demo_views(15).tex`, keeping the prompt, complete saved response,
four request controls and prefix strips. Version 16 uses the data-flow
composition in `figures/demo_views_dataflow.tex` through
`figures/demo_views(16).tex`.

The independent `images/figure3-dataflow.pdf` has a 178 by 180 mm canvas.
The version-16 wrapper changes vertical coordinates to a 178 by 158.4 mm
canvas while retaining font sizes and horizontal scales. Version 15 uses a
178 by 154 mm canvas. All three panels share the same outer right edge;
right-side cards retain their original padding. The Inspector represents
an independent After Build check, not verification of the Restore request.

Both current manuscripts remain four pages. Demonstration begins near the
end of page 2, Figure 3 appears on page 3, and both result tables and all
references remain on page 4. The opening paragraph uses normal pagination,
so it does not leave a large blank area on page 2.

## Evidence and supporting materials

See [evidence/README.md](evidence/README.md) for immutable source hashes,
experiment configurations, scope limitations and the mapping from historical
source identifiers to files in this package. The saved Qwen3-4B interaction
and model-block byte check are distinct from the recorded Qwen3-30B AgentX
capacity experiments. Packaging or rebuilding the paper runs no experiment.

| Material | Role |
| --- | --- |
| `materials/main(11).pdf` | Author-selected Industry manuscript supplying the recorded results |
| `materials/industry.pdf` | Additional author-supplied Industry draft; does not replace the evidence source above |
| `materials/lmcache-lossless-kv.pdf` | Historical Chinese implementation notes for revision `c0b2273` |
| `materials/KVFold_Fig3_Workflow_Comparison.pdf` | Author-supplied Figure 3 composition reference |

The third-party demonstration-paper library, editing-session records,
machine-specific restoration archives and build caches remain outside this
package. The committed sources and assets are sufficient to rebuild every
retained manuscript and the standalone figures.

Validation on November 10, 2026 rebuilt all four manuscript versions and
the Chinese introduction from an independent copy of this directory. All
five PDFs remain four pages; their 20 rendered pages and extracted text
match the distributed copies. Standalone figure exports also match their
distributed renderings. Final LaTeX logs contain no warnings or box errors.

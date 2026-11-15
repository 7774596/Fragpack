# KVFold demonstration paper

The current manuscript is [kvfold_update4_b.pdf](kvfold_update4_b.pdf), with
editable source in [kvfold_update4_b.tex](kvfold_update4_b.tex). It is four pages
including references and reserves three body-text lines for Acknowledgment.
The two result tables appear at the bottom of page 3, after their demonstration
context; Figure 3 appears at the top of page 4.

The [revision ledger](kvfold_update4_b-changes.md) records the 15 requirements
in `修改方案.txt`, approved compression, and subsequent wording refinements.
The [source diff](kvfold_update4_b.diff) compares this manuscript with the
archived update3 B baseline. Experiment values and evidence boundaries are
unchanged. The manuscript uses the accepted Figure 3 B interaction layout.

## Contents

| Location | Contents |
| --- | --- |
| `kvfold_update4_b.tex` / `.pdf` | Current manuscript and reading PDF |
| `kvfold_update4_b-changes.md` / `.diff` | Revision record and source comparison |
| `references.bib`, `references_update4.bib` | All 13 cited works, full metadata, and bibliography display controls |
| `images/fig1_architecture_update4.svg` | Editable architecture source |
| `images/figure1-kvfold_update4.pdf`, `.png` | Current architecture exports |
| `figures/record_layout_update4.tex` | Editable TikZ encoding example included by the manuscript |
| `images/figure2-kvfold_update4.pdf`, `.png` | Standalone encoding-example exports |
| `figures/fig3_b.js`, `fig3_b_metrics.json` | Native PowerPoint generator and the six recorded metric-card inputs |
| `images/fig3_b_update4.pdf`, `.png`, `.pptx` | Current demonstration figure, preview, and editable PowerPoint |
| [evidence/](evidence/README.md) | Recorded results, implementation mapping, correctness scope, and provenance |
| `materials/` | Author-supplied Industry manuscripts, figure reference, and abstract/Introduction inputs |
| `package.json`, `package-lock.json` | Optional, pinned PowerPoint-generation dependency |

## Build the manuscript

Run from the repository root:

```sh
make -C paper
```

The build requires `latexmk`, pdfLaTeX, BibTeX, IEEEtran, and the imported LaTeX
packages. It uses the checked figure assets and needs no GPU, Node.js,
LibreOffice, or model files. Intermediate files stay in ignored `paper/build/`.
A successful build updates `paper/kvfold_update4_b.pdf`.

For Overleaf, upload this directory with its relative paths intact, select
`kvfold_update4_b.tex` as the main document, and use pdfLaTeX. The figure PDFs
are included, so figure-generation dependencies are unnecessary.

To add acknowledgments, replace `\par\vspace*{3\baselineskip}` with the final
paragraph. Do not retain the spacer in addition to the text; it reserves three
lines. Recompile after editing to confirm pagination.

## Edit and regenerate figures

Figure 1 is an SVG; its exporter requires Python with PyGObject, pycairo, and
the Rsvg 2.0 typelib. Figure 2 uses the manuscript's TikZ preamble and requires
`preview.sty`. Both preview renderers require Poppler's `pdftoppm`.

```sh
make -C paper figure1
make -C paper figure2
```

The explicit figure targets regenerate their current PDF/PNG exports. Figure 2
can also be exported as editable SVG using `latex` and `dvisvgm` with WOFF2:

```sh
python3 paper/render_figures.py --figures 2 --svg
```

Figure 3 uses native editable PowerPoint shapes, lines, and text. Install its
optional dependency and generate a candidate:

```sh
npm ci --prefix paper
make -C paper figure3
```

This writes `paper/build/kvfold_update4_b/fig3/fig3_b_update4.pptx`; it does not
replace the checked exports under `images/`. The generator reads saved prefix
counts and exact Inspector bytes from `evidence/live-run.json`, and metric
values and bar scales from `figures/fig3_b_metrics.json`. It does not depend on
old manuscripts, old figure sources, or a pre-existing build directory.

A candidate can be rendered with LibreOffice and Poppler from the repository
root. Use a separate LibreOffice profile and retain the candidate until checked:

```sh
soffice -env:UserInstallation=file:///tmp/kvfold-figure3-render \
  --headless --convert-to pdf \
  --outdir paper/build/kvfold_update4_b/fig3 \
  paper/build/kvfold_update4_b/fig3/fig3_b_update4.pptx
pdftoppm -r 400 -singlefile -png \
  paper/build/kvfold_update4_b/fig3/fig3_b_update4.pdf \
  paper/build/kvfold_update4_b/fig3/fig3_b_update4
```

Use Arial/Courier New or the compatible Liberation fonts, plus DejaVu Sans for
circled numbers. Inspect text, alignment, and bar scales before replacing the
three matching exports under `images/` and rebuilding the manuscript.

## Evidence and retired versions

The saved Qwen3-4B interaction is distinct from the Qwen3-30B AgentX capacity
runs. The Inspector checks export/import integrity of already encoded KV;
it does not compare the first encoding with model-produced BF16. See
[implementation notes](evidence/implementation-notes.md) for revisions,
measurement definitions, memory accounting, and remaining unverified details.
Rebuilding the paper does not rerun experiments.

The current workspace retains only update4 B and its required materials.
Earlier manuscripts, alternative figures, the version-8 Chinese briefing,
and writing intermediates were preserved in a verified external archive before
removal. [The cleanup index](evidence/workspace-cleanup.json) records the archive
identifier, checksums, retired paper files, and retained author-source moves.
The archive includes a per-file manifest and restoration instructions; restore
into a separate directory before selecting files to bring back.

The historical `evidence/provenance.json` remains byte-for-byte unchanged.
Its references to retired assets describe the original evidence package;
they are not dependencies for building this manuscript.

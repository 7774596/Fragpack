# KVFold demonstration manuscript

Open [kvfold_1127.pdf](kvfold_1127.pdf) to read the paper. Copy this entire
directory, keeping the filenames and subdirectories intact, to edit or rebuild it.

## Build

Use a TeX installation with pdfLaTeX, BibTeX, latexmk, and the IEEEtran class
and bibliography style. A full TeX Live installation includes these and the
packages used by the manuscript. Run the following command in this directory
to update `kvfold_1127.pdf`.

```sh
latexmk -pdf -halt-on-error kvfold_1127.tex
```

For Overleaf, upload this directory's contents together, choose
`kvfold_1127.tex` as the main document, and select pdfLaTeX.

## Figures

The manuscript reads [references.bib](references.bib) and the three PDFs in
[images/](images/). The same directory holds their editable SVGs and PNG
previews. The recorded values for Figure 3 are in
[figures/fig3_update6_metrics.json](figures/fig3_update6_metrics.json).

If you edit an SVG, use Python 3 and Inkscape to export the corresponding PDF
and PNG. For example, to update Figure 3, run the following command in this
directory and then rebuild the manuscript.

```sh
python figures/generate_update6_figs.py 3
```

Omit the figure number to export all three. On Linux, the script also supports
PyGObject with librsvg, Pycairo, and Poppler's `pdftocairo`.

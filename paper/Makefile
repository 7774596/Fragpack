LATEXMK ?= latexmk
PYTHON ?= python3
SVG_PYTHON ?= /usr/bin/python3
DOCUMENT := kvfold_update4_b
BUILD := build/$(DOCUMENT)

.PHONY: all figures figure1 figure2 figure3
all: $(DOCUMENT).pdf

$(DOCUMENT).pdf: $(DOCUMENT).tex references.bib references_update4.bib \
                figures/record_layout_update4.tex \
                images/figure1-kvfold_update4.pdf images/fig3_b_update4.pdf
	mkdir -p "$(BUILD)"
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir="$(BUILD)" "$(DOCUMENT).tex"
	cp "$(BUILD)/$(DOCUMENT).pdf" "$@"

# Figure regeneration is explicit; normal paper builds use the checked assets.
figures: figure1 figure2 figure3

figure1:
	$(SVG_PYTHON) render_architecture.py

figure2:
	$(PYTHON) render_figures.py --figures 2

# Install the pinned optional dependency first: npm ci --prefix paper
# This creates a candidate PPTX; inspect its PDF render before publishing it.
figure3:
	node figures/fig3_b.js

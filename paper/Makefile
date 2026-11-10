LATEXMK ?= latexmk
PYTHON ?= python3
VERSION ?= 16
FIGURES ?= 1 2 3
BRIEFING_VERSION ?=
DOCUMENT := fragpack_demo($(VERSION))
BUILD := build/v$(VERSION)
BRIEFING := kvfold-demo-introduction$(if $(BRIEFING_VERSION),($(BRIEFING_VERSION)))
BRIEFING_BUILD := build/briefing$(if $(BRIEFING_VERSION),/v$(BRIEFING_VERSION))

.PHONY: all figures dataflow briefing
all: $(DOCUMENT).pdf

$(DOCUMENT).pdf: $(DOCUMENT).tex references.bib $(wildcard figures/*.tex) $(wildcard images/*.png)
	mkdir -p "$(BUILD)"
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir="$(BUILD)" "$(DOCUMENT).tex"
	cp "$(BUILD)/$(DOCUMENT).pdf" "$@"

figures:
	$(PYTHON) render_figures.py --version "$(VERSION)" --figures $(FIGURES)

dataflow:
	$(PYTHON) render_figures.py --version "$(VERSION)" --dataflow

briefing:
	mkdir -p "$(BRIEFING_BUILD)"
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir="$(BRIEFING_BUILD)" "$(BRIEFING).tex"
	cp "$(BRIEFING_BUILD)/$(BRIEFING).pdf" "$(BRIEFING).pdf"

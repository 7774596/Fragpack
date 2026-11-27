LATEXMK ?= latexmk
SVG_PYTHON ?= /usr/bin/python3
PACKAGE := kvfold-demo
DOCUMENT := kvfold_1127
FIGURE_EXPORT := $(PACKAGE)/figures/generate_update6_figs.py

.PHONY: all figures figure1 figure2 figure3
all:
	cd "$(PACKAGE)" && $(LATEXMK) -pdf -halt-on-error -interaction=nonstopmode -file-line-error -outdir=build "$(DOCUMENT).tex"
	cp "$(PACKAGE)/build/$(DOCUMENT).pdf" "$(PACKAGE)/$(DOCUMENT).pdf"

figures:
	$(SVG_PYTHON) "$(FIGURE_EXPORT)"

figure1 figure2 figure3:
	$(SVG_PYTHON) "$(FIGURE_EXPORT)" $(@:figure%=%)

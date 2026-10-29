#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -f figures/risk_ratios.pdf ]; then
    python3 experiments/make_figure.py
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if command -v bibtex >/dev/null 2>&1; then
    bibtex main
elif command -v bibtex.original >/dev/null 2>&1; then
    bibtex.original main
elif [ -f main.bbl ]; then
    printf '%s\n' 'Using the included v21 main.bbl (bibliography unchanged).'
else
    printf '%s\n' 'BibTeX or the included main.bbl is required.' >&2
    exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

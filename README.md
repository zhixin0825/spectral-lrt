# Rate-Optimal Spectral Likelihood-Ratio Clustering in Stochastic Block Models

**Zhixin Zhou — October 28, 2026 — manuscript revision v21**

The current ordinary-SBM TeX manuscript is in [paper/sbm](paper/sbm/README.md). Its entry point is [paper/sbm/main.tex](paper/sbm/main.tex). This research branch contains the v21 manuscript synchronized from the conversation source archive on October 29, 2026.

The upload preserves the agreed title, simple-first exposition, complete degree-plus-spectrum algorithm, oracle-risk theorem, seven proof appendices, 35-reference bibliography, and Section 5 experiments. It is a repository synchronization, not a new mathematical or experimental revision.

## Build the current manuscript

```sh
cd paper/sbm
sh build.sh
```

The build script generates the risk-ratio figure from the saved original simulation summary if needed, then runs PDFLaTeX. It uses the included `main.bbl` when BibTeX is unavailable. Figure generation needs Python, pandas and matplotlib; it does not run the experiments again.

**The pre-existing PDF and source ZIP under `paper/sbm` are historical compiled artifacts and were not replaced by this TeX synchronization.** Use the v21 PDF/source ZIP provided in the conversation, or compile the current TeX. Do not identify those older repository binaries with the current manuscript.

Companion notes, older manuscript snapshots, and unrelated experiment directories are retained unchanged. The separate `main` branch is not modified by this research-branch synchronization.

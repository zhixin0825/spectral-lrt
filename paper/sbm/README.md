# Current manuscript: v21

**Rate-Optimal Spectral Likelihood-Ratio Clustering in Stochastic Block Models**  
**Zhixin Zhou**  
**October 28, 2026**

These are the manuscript sources from `sbm_spectral_likelihood_source_20261028_v21.zip`, synchronized on October 29, 2026. The source entry point is `main.tex`; `article.tex` contains the introduction, model, algorithms and main results. `experiments.tex` contains Section 5. All seven proof appendices and both bibliography files are included.

The TeX, bibliography and experiment descriptions retain the v21 content. Only `sharp_tail.tex` and `references.bib` have one additional final newline relative to the source archive. No statement, equation, algorithm step, reference, experiment value or manuscript date was changed during synchronization.

## Build

```sh
sh build.sh
```

Requirements: PDFLaTeX with the packages used by `main.tex`; Python 3 with pandas and matplotlib if `figures/risk_ratios.pdf` is absent. The figure-generation script uses the unchanged v21 `experiments/production/summary.csv`. It reconstructs the original plot from saved values and does not rerun or fit simulations. Only its output-directory path is adapted to this repository layout.

The script runs BibTeX when available; otherwise the included v21 `main.bbl` is used. The output is `main.pdf`.

## Provenance and boundaries

The title and terminology are the agreed v21 version. Whole-partition recovery is called clustering; vertexwise decisions and the oracle retain classifier/classification terminology. The full algorithm retains raw degrees and one regularized adjacency eigensystem. The oracle-risk guarantee and its qualifications are unchanged.

The old files `sbm_spectral_likelihood.pdf`, `sbm_spectral_likelihood_source.zip`, and earlier build/audit/checksum records are historical artifacts, not a v21 publication. This synchronization publishes the current TeX and the data/script needed for its figure. The complete experiment-by-experiment data and revision history remain in the v21 conversation archive; this upload does not claim that all archive files were separately published.

Other directories and the separate main branch are not overwritten.

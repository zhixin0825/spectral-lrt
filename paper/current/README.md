# Current spectral likelihood-ratio manuscript

This is the complete English research manuscript dated 2026-10-09, following the agreed six-section structure. It supersedes the outline as the current prose manuscript; the historical draft remains in `paper/previous_draft/`.

The working title is **Spectral Likelihood Ratios for Nearly Optimal Community Recovery in the Stochastic Block Model**.

The paper contains the algorithm, related work, logarithmic-density conditional recovery theorem, lower-density extension, the existing 240-graph experiments, conclusion, full proofs of the stated results, and references. The target statistical rate is `exp(-(1+o(1)) I_n)`. Zhou�CLi is cited as prior work studying optimal rates, without restating its refined rate.

The end-to-end leading-exponent guarantee for the executable clipped-profile EM procedure remains unproved. The proved recovery result is conditional on explicit likelihood-score and exceptional-event bounds. Exact identities and optimization statements are proved separately; empirical convergence is not used as a substitute for a recovery proof.

## Read and compile

- `manuscript.pdf`: compiled reading version.
- `manuscript.tex`: main source, which includes the section files.
- `references.bib`: verified bibliography.
- `figures/initialization_failure_repair.pdf`: vector diagnostic figure.
- `selected_experiment_results.csv` and `selected_experiment_models.csv`: numerical summaries used in the tables.

Run from this directory:

```sh
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error manuscript.tex
```

The repository workflow compiles changes to the manuscript and commits its PDF with build provenance. The primary algorithm corresponds to `global_gain_no_update` in the existing experiment archive: subset-first seed, frozen-candidate global gains, then joint EM. `method='global_gain'` in the experiment API is a different growing-EM control.

## Existing evidence

Across 240 graphs, the primary deterministic branch has mean mislabeled fraction 0.0946% and exact recovery on 163 graphs; the ten-start spectral k-means baseline has mean mislabeled fraction 0.3194% and exact recovery on 139 graphs. This is a paired recovery comparison with unequal fitting budgets, not an equal-runtime benchmark or an asymptotic exponent experiment.

The detailed experiment reports and runnable scripts remain under `experiments/20261009_residual_likelihood_seeding/`. The numerical records are not duplicated in this manuscript directory.

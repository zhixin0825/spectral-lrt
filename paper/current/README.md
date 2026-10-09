# Current spectral likelihood-ratio manuscript

This English research manuscript follows the six-section structure and includes proof appendices, references, and the archived 240-graph evaluation.

The working title is **Spectral Likelihood-Ratio Decoding for Community Recovery in the Stochastic Block Model**.

## Current algorithm and notation

Algorithm 1 is the already tested growing global-gain branch: start at the global profile mean, add a node profile by all-node likelihood gain, refit all components after each addition, and repeat to K components. There is no top-fraction or 1.5K parameter. It is recorded as growing_global_gain_EM and exposed by method='global_gain'. The subset-first global_gain_no_update algorithm is a historical control.

The manuscript follows Zhou--Li notation: P is the K-by-K block probability matrix, p_{kj}=P_{k,z_j}, Mis is the permutation-invariant error, and D_alpha denotes Chernoff information. Omega is the n-by-n graph mean. I_n is the minimum exact pairwise Chernoff information. Zhou--Li is cited as optimal-rate prior work without restating its refined rate.

## What is proved

At logarithmic density with fixed full-rank B and an admissible fixed floor c_0 rho_n, the revised proof establishes:

- Full empirical spectral reconstruction is uniformly o(log n) from oracle block-count compression in row L1.
- Clipped oracle comparisons retain the leading Chernoff exponent.
- Actual spectral-profile EM from a sufficiently reliable weak start achieves exp(-(1+o(1)) I_n), through further exact updates and finite adaptive stopping.
- A provable spectral warm-start variant supplies an end-to-end recovery corollary.
- Exact contrast preservation, fixed-profile greedy properties, and within-fit objective convergence hold with their stated assumptions.

The unresolved logarithmic-density step for the primary Algorithm 1 is its high-probability weak-start guarantee. This is not implied by the frozen-profile greedy guarantee or objective convergence. The theoretical amplified k-means++ initializer is distinct from the archived fixed ten-start Lloyd baseline.

## Read and compile

- manuscript.pdf: compiled reading version.
- manuscript.tex: main source including the section files.
- references.bib: bibliography.
- figures/initialization_failure_repair.pdf: vector diagnostic figure.
- selected_experiment_results.csv and selected_experiment_models.csv: archived numerical summaries.

Run PDFLaTeX, BibTeX, and two further PDFLaTeX passes from this directory. The repository workflow compiles source changes and commits the PDF with build provenance.

## Existing evidence

Across 240 graphs, the growing global-gain branch has mean mislabeled fraction 0.0946% and exact recovery on 163 graphs; the ten-start spectral k-means baseline has mean mislabeled fraction 0.3194% and exact recovery on 139 graphs. This is a paired recovery comparison with unequal fitting budgets, not an equal-runtime benchmark or an asymptotic exponent experiment.

Detailed reports and scripts remain in experiments/20261009_residual_likelihood_seeding/. Original numerical records are unchanged.

# Spectral likelihood decoding: current manuscript

[Compiled PDF](manuscript.pdf) · [Full proofs](theory.tex) · [Review and corrections, 2026-10-10](REVIEW_20261010.md)

## Model and theorem scope

`P` is the K-by-K Bernoulli block probability matrix, with `p_kj=P_{k,z_j}` following Zhou–Li. The main results assume fixed known K, positive limiting community proportions, and fixed entrywise-positive symmetric full-rank `(n/log n)P`. There is no separate density or connectivity-shape matrix.

This probability scale allows unequal expected degrees. For a node in class a, the expected degree is `sum_b (n_b - 1{a=b})P_ab`; the observed degree is random. The eigenpairs are computed from the original adjacency matrix.

The implemented profile floor is `epsilon=1e-4 log n/n`. The fitting theorems require `min_ab nP_ab/log n > 1e-4`. This is an additional condition, not a consequence of positivity alone. The floor makes logarithms of candidate reconstructed rows well defined; it does not modify the adjacency matrix used for eigendecomposition.

## Proven algorithms

- Theorem 3.8 proves the original global-mean growing branch for K=2.
- Theorem 3.9 proves the general fixed-K algorithm with `ceil(log n)` independent likelihood-residual initializations, likelihood selection, and one mandatory final E/M update.
- The expected error is at most `exp(-(1+o(1)) I_n)`, attaining the leading oracle Chernoff exponent.
- Exact recovery follows when `liminf I_n/log n > 1`.
- The unchanged growing branch is not proved for general K. The theorem does not claim a refined multiplicative risk or a new global minimax lower bound.

Algorithm 1 uses only the retained eigenpairs after compression. Residual weights use the explicit profile divergence `D(x||y)=sum_j[x_j log(x_j/y_j)-x_j+y_j]`. Fitting optimizes the spectral working objective `ell_i(p)=sum_j q_ij log p_j-sum_j p_j`; its relation to the exact Bernoulli LLR is proved by a sparse Taylor expansion.

The explicit API option is `method='global_gain_certified'`, calling `Work.certified_growing()` in [residual_seed.py](../../experiments/20261009_residual_likelihood_seeding/residual_seed.py). The default `method='repair'` is a historical alternative. The original growing arm is `method='global_gain'`.

## Numerical evidence

The archived 240-graph comparison concerns the original growing branch: average error 0.0946%, exact recovery 163/240, versus the spectral K-means baseline 0.3194%, 139/240. It does not evaluate the complete safeguarded algorithm on that collection.

[Twenty-four new implementation checks](../../experiments/20261010_certified_likelihood/) verify fit selection, the mandatory final update, and numerical constraints. They do not establish a finite-sample accuracy advantage or estimate an asymptotic exponent.

## Build

The active manuscript has five main sections, proof appendices, and references. The speculative lower-density extension is excluded from this version. The previous discussion remains in Git history.

From this directory, run PDFLaTeX, BibTeX, and PDFLaTeX twice. GitHub Actions records the source commit and checksums in BUILD.txt and BUILD_SHA256SUMS.txt. Raw experiment records and stopping rules remain unchanged.

# Adjacency spectral likelihood classification in stochastic block models

This is the independent ordinary-SBM article, revised on 21 October 2026 to use adjacency spectral information and raw degrees. Its 22 pages contain the complete algorithm, theorem and proofs.

- [Article PDF](sbm_spectral_likelihood.pdf)
- [Complete, compilable source package](sbm_spectral_likelihood_source.zip)
- [Companion working notes: DC-SBM and additional material](../companion/README.md)

## Main result

For arbitrary fixed K, comparable community sizes, and a general positive symmetric block-probability matrix P whose entries are comparable to p and whose smallest singular value is at least kappa p, the classifier satisfies

    E Mis(z_hat, z) <= C R_oracle

under d = np -> infinity and np^2 <= C1. There is no lower bound proportional to log n. R_oracle is the actual average error probability of the true-parameter, raw incident-row Bernoulli oracle, evaluated at the true labels with a fixed tie rule. The bound gives a constant risk ratio; it does not claim that the ratio tends to one. Exact recovery follows when n R_oracle -> 0.

The finite algorithm retains raw degrees and one eigensystem of the high-degree-regularized adjacency matrix A_tilde = W A W. Its threshold is tau = b0 times the observed average degree, with b0 sufficiently large and fixed. The weight is W_i = 1 when D_i <= tau, and W_i = sqrt(tau / D_i) otherwise. If the cap is inactive, these are precisely the eigenpairs of the original adjacency matrix A. The theorem over every diverging degree scale uses the stated cap.

The decoder performs exact degree correction and hollowing, initializes by deterministic radius counts, and uses symmetric Bernoulli block fits followed by ordinary, unprofiled likelihood updates. The full nonedge intercept is retained. Any finite positive number of the specified hard refits, including finite adaptive stopping, is covered.

The main text uses the familiar eigenpair notation U, Lambda and F for the rank-K reconstruction on the original adjacency scale. Without an active cap, F = A_K = U Lambda U^T. A single off-diagonal row formula defines the final degree-preserving reconstruction A_hat. The proof auxiliaries Z, Omega, G, H, Q and R1 appear only in the appendices. This notation cleanup leaves the algorithm and theorem unchanged.

## Established writing requirements

- State the abstract, introduction and main result directly in terms of oracle risk. Keep the exponential expansions and Chernoff calculations in the proof appendices.
- Write R_oracle in full. Do not reintroduce the abbreviation "or" or the combined-paper suffix R_SBM,*. Preserve the current average oracle-risk definition; do not replace it with a worst-pair equal-prior error or an affinity.
- Follow Zhou--Li for A, P, z, z_hat, K and Mis. Keep U, Lambda, F and A_hat stable. State the structural assumptions directly on P and p; do not introduce a second block matrix or a new sparsity parameter. The scale p has not been redefined.
- Use adjacency spectral information and retain the raw degrees. Describe the high-degree cap explicitly; the all-sparsity theorem uses that cap.
- Preserve arbitrary fixed K, comparable group sizes, the constant oracle-risk comparison, np -> infinity and np^2 <= C1. Do not add a logarithmic degree lower bound.
- Keep ordinary SBM self-contained in this first article. DC-SBM and overflow material belong in the companion working notes.

The former R_SBM,* notation came from the combined ordinary/DC manuscript. In this separate ordinary-SBM article, the model suffix and star are unnecessary. The completed-mean matrix and other proof-only definitions are introduced where they are first used in the appendices.

## Contents

| File | Content |
| --- | --- |
| `article.tex` | Introduction, ordinary model and oracle, complete algorithm, main theorem, proof outline |
| `sharp_tail.tex` | Sharp Bernoulli transfer and exceptional-coefficient payment |
| `geometry.tex` | Capped-adjacency aggregate geometry and weighted cap energy |
| `row_tail.tex` | Same-threshold truncated spectral-row energy |
| `local_loo.tex` | Local scalar leave-one-out comparison and exact-degree hollowing |
| `fitting.tex` | Initializer, block fitting, score replacement, invariant refinement basin |
| `high_range.tex` | Exponentially accurate aggregate geometry |
| `risk_proof.tex` | Ordinary tilted cancellation and completion over every diverging degree scale |
| `references.bib` | References cited by this article |

The article is self-contained and has no proof dependency on the companion. Degree correction, the degree-normalized formulation, historical alternative decoders, counterexamples, optional derivations and the internal review record are collected in the companion. The preceding normalized ordinary article is preserved as a source snapshot in `paper/companion/archive/ordinary_normalized_20261020/`. The original 68-page combined note remains archived in `paper/notes/normalized_oracle_risk_20261020/`.

## Build and checks

Run the commands in `BUILD.txt` from this directory. The source package includes the generated `main.bbl`, so its bibliography is also available without re-running BibTeX. The checked build uses pdfLaTeX and BibTeX, has no undefined references, duplicate labels or overfull boxes, and embeds all fonts. Both revised manuscripts were rendered and visually inspected.

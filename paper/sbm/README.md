# Adjacency spectral likelihood classification in stochastic block models

## 24 October 2026: simple first, complete sparse version retained

The article now introduces a spectrum-only likelihood construction before the complete degree-plus-spectrum classifier. The rewrite changes exposition and adds an explicitly separate motivating procedure. It does not replace the complete algorithm, change the oracle, weaken the main theorem, or remove any proof appendix.

A new 24-page PDF and a complete source ZIP were compiled and supplied with the conversation. The repository's existing `sbm_spectral_likelihood.pdf`, `sbm_spectral_likelihood_source.zip`, `BUILD.txt`, and their checksum records remain the 21 October compiled snapshot unless replaced in a separate artifact publication. Do not describe those historical binary files as the 24 October build. The current editable sources are `main.tex` and `article.tex`.

## Reading order

1. Model and one common incident-row oracle.
2. Spectrum-only likelihood: raw adjacency eigenpairs, reconstructed block counts, symmetric block fitting and likelihood updates. It does not receive the observed degree vector. This motivating procedure has no asserted constant-oracle-risk theorem in the present article.
3. Complete sparse algorithm: high-degree regularization, original-scale reconstruction, exact degree correction and hollowing, radius-count initialization, and Bernoulli block fits/LRT updates. Exact count and score identities explain its relation to the simple construction.
4. Main theorem, its scope, proof outline, discussion, and all seven proof appendices.

An inactive cap does not make degree correction inactive. Neither the simple procedure nor a historical free-profile EM decoder is silently identified with the complete estimator.

## Preserved main result

For arbitrary fixed K, comparable community sizes and a general symmetric block matrix P satisfying

    c_- p <= P_ab <= c_+ p,   sigma_min(P) >= kappa p,

with fixed positive structural constants, the complete classifier satisfies

    E Mis(z_hat, z) <= C R_oracle

under np -> infinity and np^2 <= C1. No lower bound proportional to log n is added. R_oracle remains the actual average error probability of the true-parameter raw incident-row Bernoulli oracle at the true labels, with a fixed tie rule. The constant is independent of n and of the number of specified hard refits. The statement is a bounded risk ratio, not convergence of the ratio to one; exact recovery follows when n R_oracle -> 0.

Absence of a logarithmic density floor is not absence of the relative entrywise lower bound c_- p or the signal-conditioning assumption. The clipping definition of a data-driven procedure does not itself extend the theorem beyond its stated model class.

The complete encoder retains raw degrees and one eigensystem of A_tilde = W A W. With tau = b0 times the observed average degree, W_i is 1 below tau and sqrt(tau / D_i) above tau. The decoder reconstructs F = W^-1 U Lambda U^T W^-1, then applies the same exact degree-preserving hollowing formula as before. Its full nonedge intercept, initializer, symmetric block fit and finite hard-update schedule are unchanged.

## Writing requirements

Simple-first exposition is encouraged. Build it around the retained complete version rather than rebuilding the manuscript from the simple version. Every required general-algorithm step and every proof dependency must remain identifiable in the ordinary-SBM article.

- State results in terms of R_oracle; keep Chernoff expansions in the proof appendices.
- Preserve A, P, z, z_hat, K, Mis, U, Lambda, F and A_hat. The superscript (0) is local to the simple construction. Do not introduce a new sparsity parameter or redefine p.
- Keep arbitrary fixed K, comparable group sizes, the constant oracle-risk comparison, np -> infinity and np^2 <= C1. Do not reintroduce a logarithmic degree floor.
- Distinguish degree correction in ordinary SBM from the DC-SBM companion. All ordinary-SBM theorem dependencies stay here.
- Do not attach the general theorem to the raw, uncorrected spectral version. Changes of estimator or mathematical scope must be identified, not described as editorial cleanup.
- Check preservation by file hashes and theorem/equation comparisons. This is not a substitute for independent mathematical review.

## Source map

| File | Content |
| --- | --- |
| `article.tex` | Introduction; model/oracle; simple spectrum-only version; complete algorithm; main theorem; proof outline; discussion |
| `main.tex` | Preamble, abstract and complete input list |
| `sharp_tail.tex` | Sharp Bernoulli transfer and exceptional-coefficient payment |
| `geometry.tex` | Capped-adjacency aggregate geometry and weighted cap energy |
| `row_tail.tex` | Same-threshold truncated spectral-row energy |
| `local_loo.tex` | Local scalar leave-one-out comparison and exact-degree hollowing |
| `fitting.tex` | Initializer, block fits, score comparison and invariant refinement basin |
| `high_range.tex` | Exponentially accurate aggregate geometry |
| `risk_proof.tex` | Tilted cancellation and completion over all diverging degree scales |
| `references.bib`, `main.bbl` | Editable bibliography and retained generated bibliography |
| `EDITORIAL_AUDIT_20261024.md` | Preservation checks and precise revision scope |

The seven proof files, structural assumptions, oracle definition, complete-algorithm equations and main theorem are unchanged from base commit `8f3962e913bf32cf6c93d9325b37de4faf171b68`. The companion remains separate. No proof needed by the ordinary-SBM main theorem has been moved to it.

## Build

The supplied source package includes `main.bbl`. Compile `main.tex` with three PDFLaTeX passes. If bibliography entries are edited, run BibTeX between the first and subsequent PDFLaTeX passes. The 24 October local build used the retained bibliography, produced 24 pages, and had no undefined references/citations or overfull/underfull boxes. These are build and editorial checks, not a new certification of the inherited statistical proofs.

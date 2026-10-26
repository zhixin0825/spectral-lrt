# Adjacency spectral likelihood classification in stochastic block models

## Current source: 26 October 2026 notation revision

The manuscript keeps the simple-first exposition introduced on 24 October: a spectrum-only motivating procedure is followed by the complete degree-plus-spectrum classifier. This revision harmonizes the notation with Zhou–Li and the author's requested singular-value notation; it does not replace either procedure or weaken the sparse theorem.

### Core notation

- `p^* = max_ab P_ab`, rather than an unspecified probability scale.
- `n_a >= n/(beta K)` for community imbalance; no separate `pi_-`.
- `p^*/omega <= P_ab <= p^*` for probability comparability; no separate `c_-`, `c_+`.
- `sigma_K(P)` for the smallest singular value of the K-by-K block matrix; no `sigma_min` and no separately named `kappa`.
- `d = n p^*` is a frequently reused abbreviation, not a second model parameter.
- Permutations use `pi in S_K`; Chernoff quantities use `D_alpha` and `alpha^*`. Indexed row probabilities `p_j` retain their local meaning.
- The actual averaged incident-row risk remains `R_oracle`, as explicitly requested. It is not replaced by the affinity or by an exponential-rate expression.

The main theorem retains `n p^* -> infinity` and `n (p^*)^2 <= C1`, with no logarithmic density lower bound. Positivity, probability comparability and signal nondegeneracy remain explicit assumptions. Normalizing the old scale to `p^*` rescales fixed constants, including the numerical upper bound C1; see the audit for both directions of the conversion.

### What is unchanged

Both algorithm sections and 13 protected model/oracle/algorithm/risk equations are byte-for-byte unchanged from `0fe40da1fee92f7bec3ea3f7f6fdd53faae8c442`. All 95 labels, 17 theorem-like statement environments and seven proof inputs are retained. The main theorem changes only through the explicit scale normalization. `local_loo.tex` is byte-for-byte unchanged. Other proof changes are notation substitutions, the equivalent Chernoff definition, and removal of the now-redundant auxiliary scale `d_*`.

The complete procedure still retains raw degrees, caps high-degree vertices, reconstructs on the original adjacency scale, preserves degrees exactly, removes the diagonal, initializes by radius counts, and uses symmetric Bernoulli block fits and full edge/nonedge LRT updates. Its guarantee is not transferred to the preliminary spectrum-only procedure. Ordinary-SBM degree correction and all required lemmas remain in this first paper.

### Files

`main.tex` includes `article.tex`, then the seven proof files `sharp_tail.tex`, `geometry.tex`, `row_tail.tex`, `local_loo.tex`, `fitting.tex`, `high_range.tex`, `risk_proof.tex`. `references.bib` and the retained `main.bbl` provide the bibliography.

`NOTATION_AUDIT_20261026.md` records the primary-source comparison, conversion of assumptions and checks. `NOTATION_CHECKS_20261026.json` records the preservation checks. `BUILD_20261026.txt` records the actual local build. The conversation source ZIP also includes `notation_changes.patch`, the source-only patch relative to the 24 October sources.

### Build and artifact provenance

The current source was compiled locally into a 25-page PDF on 26 October 2026. Run PDFLaTeX three times from this directory using the included `main.bbl`; after editing the bibliography, run BibTeX between the first and subsequent PDFLaTeX passes.

The revised PDF and matching source ZIP are supplied as conversation artifacts. Unless updated in a separate binary publication, the repository files named `sbm_spectral_likelihood.pdf` and `sbm_spectral_likelihood_source.zip`, and their older BUILD/checksum records, remain historical snapshots; this source revision does not claim to have replaced those binaries.

### Fixed editing rules

Start from the complete manuscript. Keep the simple explanation first without deleting the general algorithm or proofs. Follow the existing Zhou–Li model notation unless a genuinely new mathematical object requires a distinct name. Any reparameterization must state its effect on constants. Do not silently add a log-degree floor, remove a structural assumption, change the oracle, introduce an algorithmic step, or move an ordinary-SBM proof dependency to the companion. Archive the preservation checks with the edit. Compilation and notation checks are not a new independent certification of the inherited statistical proof.

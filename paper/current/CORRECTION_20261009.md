> Historical record. Superseded for current model, algorithm and theorem scope by [README.md](README.md) and [REVIEW_20261010.md](REVIEW_20261010.md). The original content below is retained as a dated record.

# Notation, algorithm and proof correction (2026-10-09)

The current manuscript corrects the previous mismatch between the tested growing algorithm and the subset-first frozen control.

## Notation

The SBM symbols follow Zhou--Li: P is K-by-K block probability, p_{kj}=P_{k,z_j}, Mis is the permutation-invariant error fraction, and D_alpha is Chernoff information. Omega is the n-by-n no-self-loop mean, and tilde A is the signed empirical spectral reconstruction. I_n is the minimum exact incident-row pairwise Chernoff information, with I_n=d_n J_*+o(d_n). The refined Zhou--Li rate is not restated.

## Main algorithm

The primary branch is growing_global_gain_EM, implemented by Work.growing() / method='global_gain': global-mean first component, K-1 all-node likelihood-gain additions, and joint EM after every addition. It has no top-fraction size. The existing 240-graph recorded results already test this exact branch. The frozen global_gain_no_update branch is a historical control; its subset sizes are confined to the implementation appendix.

Within-stage EM monotonicity does not assert monotonicity across stage changes that reset weights. Frozen submodularity and noiseless coverage are explicitly properties of the frozen control.

## New proved results

1. General full-rank logarithmic SBM, including negative signal eigenvalues: AFWZ first-order eigenspace analysis plus Lei--Rinaldo arbitrary polynomial confidence gives uniform empirical spectral reconstruction row L1 error o(log n) relative to true block-count projection. The no-self-loop mean and both sign groups are handled explicitly.
2. An admissible fixed floor c_0 rho_n with 0<c_0<b_- preserves the leading clipped-oracle Chernoff exponent through a direct moment comparison.
3. Weak responsibilities imply fitted profile L1 error o(log n), without block-constraining the fitted profiles or requiring row independence.
4. A deterministic fixed weak basin is mapped into a vanishing-error basin, so subsequent exact EM iterations and finite adaptive stopping preserve the leading recovery exponent.
5. A provable spectral warm-start variant yields an end-to-end recovery corollary using only the retained eigenpairs.

The practical growing initializer's entrance into that basin with failure probability at the oracle error scale remains unproved. The standard ten-start Lloyd experiment is not automatically the amplified initializer in the theoretical corollary. No new simulations or replaced experimental numbers are claimed in this correction.

## Primary theory references checked

- Zhou--Li, arXiv:1812.11269, Section 3.1 definitions of P, p_{k*}, Mis; the published citation remains EJS 2020.
- Abbe--Fan--Wang--Zhong, arXiv:1709.09565, Theorem 2.1, assumptions A1--A4, Bernoulli Lemma 7.
- Lei--Rinaldo, arXiv:1312.2050, Theorem 5.2 and deterministic clustering bounds.

Proof review specifically checked the sign split, polynomial-confidence adjustment, matrix row concentration, row L1 algebra, floor moment comparison, no-self-loop corrections, and basin invariance.

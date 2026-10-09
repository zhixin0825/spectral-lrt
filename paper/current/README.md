# Nearly optimal spectral likelihood decoding

The current manuscript proves end-to-end recovery at the leading error exponent, under fixed known K, fixed positive full-rank B, positive limiting community proportions, density log(n)/n, and an admissible positive profile floor.

## Complete algorithm

Algorithm 1 retains the original global-mean growing fit. Alongside it, run ceil(log(n)) conditionally independent likelihood-residual++ initializations: uniform first node, subsequent seeds sampled proportional to min-seed Poisson deviance, all K seeds chosen before joint fitting. Select the highest final working mixture likelihood across these fits and the growing fit. Perform one final E-step and profile/weight M-step before decoding.

The exact implementation is Work.certified_growing() in ../../experiments/20261009_residual_likelihood_seeding/residual_seed.py, exposed by method='global_gain_certified'. The original growing_global_gain_EM / method='global_gain' is retained separately. Neither uses a top-fraction parameter or a k-means decoder.

## End-to-end results

- Theorem 3.8 (label thm:two-community-growing) proves the original growing branch itself for K=2.
- Theorem 3.9 (label thm:certified-likelihood) proves the complete likelihood-restart algorithm for every fixed K.
- The guarantee is E Mis <= exp(-(1+o(1)) I_n). It does not assume that the algorithm has already initialized correctly.
- The proof establishes reliable seed coverage, likelihood-based fit selection, and weak posterior responsibilities, then applies the empirical spectral EM theorem.
- For I_n/log(n) -> J_* > 1, exact recovery follows.
- Further finite exact EM updates preserve the guarantee. A prescribed polynomial iteration bound gives polynomial complexity.

The unsafeguarded growing branch is not claimed to be proved for general K. The general-K theorem explicitly includes the safeguard. It is fully likelihood-based; the separate spectral k-means warm-start corollary is not used in this proof.

## Notation and scope

P is the K-by-K block probability matrix, p_{kj}=P_{k,z_j}, Mis is permutation-invariant error, D_alpha is the Chernoff quantity, and Omega is the n-by-n graph mean. I_n is the minimum exact pairwise information. Zhou–Li is cited as optimal-rate prior work without repeating its refined rate. The claim is equality of the leading exponent.

## Numerical evidence

The archived 240-graph comparison remains unchanged and concerns the original growing branch: average error 0.0946%, exact recovery 163/240, versus the archived spectral k-means baseline 0.3194%, 139/240. It is not a full evaluation of the newly safeguarded combination.

The new code was separately verified on 24 fresh SBM cases: likelihood selection never falls below the original growing objective; the final M-step is mandatory; profiles and floored weights satisfy their constraints. These checks do not prove finite-sample superiority in misclassification. See ../../experiments/20261010_certified_likelihood/.

## Build and read

manuscript.pdf is the compiled paper. Run PDFLaTeX, BibTeX, and two additional PDFLaTeX passes from this directory. GitHub Actions records the source commit and SHA256 checksums.

The source retains six main sections, complete proof appendices, the historical experiment implementation, and references. All raw experiment records remain unchanged.

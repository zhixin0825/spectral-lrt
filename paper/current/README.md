# Degree and spectral likelihood decoding for the SBM

[Compiled PDF](main.pdf) · [Complete LaTeX source](main.tex) · [Bibliography](reference.bib) · [Compilable sources](latex-source.zip)

The manuscript first introduces the easier unregularized case and then, in Section 4, the general algorithm retaining both observed degrees and regularized spectral information. Both use the oracle likelihood-ratio test as the benchmark and the complete Bernoulli row criterion for initialization and refinement.

## Easier case: Algorithm 1 and its sharp theorem

The partition is a fixed unknown parameter. For every fixed known K, P is the symmetric Bernoulli block probability matrix and p=max_ab P_ab. The unconditional sharp theorem assumes fixed model constants with:

- n_a >= n/(beta*K), beta >= 1;
- kappa*p <= P_ab <= p and sigma_min(P) >= kappa*p;
- p <= 1-eta;
- p = Omega(log n/n).

Neither P/p nor community proportions need converge. Unequal community sizes and expected degrees, either sign of signal eigenvalues, and logarithmic through dense probabilities are allowed.

Algorithm 1 retains the raw-adjacency eigenpairs with largest absolute eigenvalues. It reconstructs clipped probability rows, initializes by global-mean likelihood growing, and uses at most ceil(log n) rounds of component replacement. Every deletion position is checked with its best all-node candidate, trial weights start uniformly, each fit takes at least one exact M-step, and the old endpoint participates in the objective comparison. One final E-step and profile/weight M-step precedes classification.

The theorem states Zhou–Li’s benchmark:

`E Mis <= C eta*`.

Here eta* is the maximum total variation affinity between the pairwise incident-edge product-Bernoulli laws, twice the maximum equal-prior binary oracle risk. Under these model bounds, eta* is equivalent to exp(-I_n)/sqrt(np), with the absent self-loop excluded in I_n. Exact recovery follows from n eta* -> 0. The result does not claim risk-ratio convergence to one, exact integer count recovery, or a new global minimax lower bound.

The [unregularized implementation](../../experiments/20261010_general_density/spectral_likelihood.py) receives only its eigenpairs. The theorem and all previously reported graph benchmarks apply to this easier case.

## General case: retain degrees and regularized eigenpairs

Compute D=A*1 and Dbar=mean(D). Set W_ii=min(1,2*Dbar/D_i) when D_i>0 and W_ii=1 for isolates. Retain the raw degrees and the K signed eigenpairs of W A W with largest absolute eigenvalues:

`(D,U,Lambda)`.

W is recoverable from D. The exact identities

`A(WU)=W^-1 U Lambda, A*1=D`

give the original adjacency action on the augmented features. With

`X=[1,WU], Y=[D,W^-1 U Lambda], R=Y pinv(X)`,

we have R=A Proj(col(X)) and **R*1=D exactly**. For every candidate grouping e, the real-valued block counts R Z(e) sum to each observed degree. If the true membership space is contained in col(X), its edge counts are preserved exactly; empirical approximate containment carries no rounding guarantee. Section 4 also records the earlier centroid-inverse block decoder and its exact degree correction.

Algorithm 2 uses the fixed reconstructed rows R in the complete Bernoulli criterion:

`ell_i(mu)=sum_j {R_ij log(mu_j)+(1-R_ij) log(1-mu_j)}`.

R can be signed and nonsymmetric. It is neither clipped nor symmetrized, preserving its exact degree identity. Candidate and fitted probability profiles lie in [epsilon,1-epsilon], with epsilon=c0*max(1,Dbar)/n. Their exact M-step is the clipped weighted mean of R. Growing, replacement, soft responsibilities and final classification all use this same row criterion; no K-means initializer is introduced. This is a working decoding criterion, not the likelihood of independent Bernoulli observations with data R.

The [degree-plus-spectrum implementation](../../experiments/20261018_degree_spectral/degree_spectral_likelihood.py) accepts only D,U,Lambda. [Reproducible checks](../../experiments/20261018_degree_spectral/verify_degree_spectral.py) and [saved results](../../experiments/20261018_degree_spectral/CHECKS.json) verify exact retained-information identities, redundant feature directions, signed reconstruction, and EM objective updates. They establish implementation properties, without a sparse statistical performance claim.

## General-case theorem scope

The general algorithm is defined without a logarithmic density floor. The sparse target is np -> infinity. Section 4 gives a sharp transfer proposition with explicit vertex-deleted affine-score, conditional cumulant, tilted local-concentration and bounded-moment remainder hypotheses. Under these hypotheses its conclusion is again **E Mis <= C eta***.

Verifying these sharp hypotheses uniformly for the full degree-regularized algorithm below logarithmic expected degree remains open. The unconditional theorem of Sections 2–3 is not asserted for Algorithm 2. The old exponent-only rate is not used as a substitute for the Zhou–Li bound.

## Notation and evidence

The paper uses community detection as the task name and recovery for accuracy statements. The conventional imbalance normalization is n/(beta*K). Following Amini et al. (2013), b_ik(e) denotes a node-to-block edge count. Total ordered block counts are entries of Z(e)' A Z(e), with within-block undirected edges counted twice on the diagonal.

- The [60-graph benchmark](../../experiments/20261010_general_density/) evaluates the unregularized complete Bernoulli decoder for K=2–6. Replacement changed no classification errors in that collection; weak general signals had substantial finite-sample errors.
- The [five-community population certificate](../../experiments/20261010_general_density/population_audit/) exhibits finite-stage growing failure and replacement repair.
- The [240-graph sparse-score experiment](../../experiments/20261009_residual_likelihood_seeding/) is an earlier, separately labelled collection.
- The [144-configuration tilted-row diagnostic](../../experiments/20261018_sharp_rate/) checks the unregularized arrowhead identities and constant-scale remainders.

## Build and historical files

The six main sections are introduction, unregularized algorithm, its sharp guarantees, general degree-plus-spectrum algorithm, experiments, and conclusion. All sections and proofs are in main.tex. Compile with PDFLaTeX, BibTeX and two further PDFLaTeX passes. GitHub Actions builds main.pdf, the identical compatibility copy manuscript.pdf, and latex-source.zip, with source checksums and provenance in BUILD.txt and BUILD_SHA256SUMS.txt.

Earlier section files and dated notes are historical snapshots, not active build inputs. The [standalone sharp-rate note](../notes/spectral_sharp_rate_20261018.pdf) records the earlier unregularized derivation.


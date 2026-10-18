# Rate-optimal community detection by spectral likelihood decoding

[Compiled PDF](main.pdf) · [Complete LaTeX source and proofs](main.tex) · [Bibliography](reference.bib) · [Download compilable sources](latex-source.zip)

The October 18, 2026 manuscript, revised for conventional notation and vanishing probabilities, starts from the oracle likelihood-ratio test. Its nodewise evidence consists of edge counts to the true communities. Empirical spectral reconstruction gives approximate, generally noninteger counts. The theorem controls the resulting loss in likelihood-tail information strongly enough to preserve the oracle misclassification rate up to a constant factor. It does not require exact recovery of each count by rounding.

## Model and rate

The partition is a fixed unknown parameter. For every fixed known K, P is the symmetric Bernoulli block probability matrix and p = max_ab P_ab. Fixed positive model constants satisfy:

- n_a >= n/(beta*K), with fixed beta >= 1;
- kappa*p <= P_ab <= p and sigma_min(P) >= kappa*p;
- p <= 1-eta;
- p = Omega(log n/n).

The model allows unequal community sizes and expected degrees, either sign of signal eigenvalues, and logarithmic through dense probability scales. Neither P/p nor the community proportions need converge.

The fixed-floor sharp theorem for Algorithm 1 uses Zhou–Li’s affinity benchmark:

`E Mis <= C eta* <= C exp(-I_n)/sqrt(np)`.

Here eta* is the maximum total variation affinity between the pairwise incident-edge product-Bernoulli laws; it is twice the corresponding maximum equal-prior binary oracle risk. The paper also states `C exp(-I_n)/sqrt(I_n)` and the common leading-exponent consequence `exp(-(1-o(1))*I_n)`, plus the usual homogeneous planted-partition Rényi-rate corollary.

I_n is the minimum exact incident-edge product-Bernoulli Chernoff information, including the self-loop correction. Under the model bounds, the Zhou–Li binary oracle scale is `exp(-I_n)/sqrt(np)`. Thus both the exponential rate and the `1/sqrt(np)` factor are preserved up to constants. This strengthens the earlier leading-exponent theorem. It does not assert identical risks, a risk ratio tending to one, statistical sufficiency, or a new global minimax lower bound. Exact recovery follows if `n exp(-I_n)/sqrt(np) -> 0`.

## Removing the minimum-entry condition

The separate vanishing-floor theorem drops the lower bound on every P_ab and permits zero entries. It keeps fixed K, n_a >= n/(beta*K), sigma_min(P) >= kappa*p, p <= 1-eta, and np >= c log n. Change only the floor to

`epsilon = c0*(s/n)*exp(-sqrt(log(e+s)))`.

The same growing/replacement/final-E/M procedure then satisfies

`E Mis <= exp(-I_n + o(np)) = exp(-(1-o(1))*I_n)`.

The full proof handles the slowly growing likelihood coefficients, the floor’s dependence on the same graph, smoothing of zero probabilities, and the conditional spectral likelihood tails. This is a leading-exponent guarantee. The sharp C eta* theorem remains tied to comparable probabilities and its fixed floor; no universal 1/sqrt(np) factor is asserted at zero probabilities.

The existing constructor parameter `floor` is the coefficient multiplying s/n. Passing `floor=c0*exp(-sqrt(log(e+s)))` implements the new option without changing the fitting engine. All archived experiments use their recorded fixed floor; no new accuracy claim for this option is inferred from them.

## Conventional notation

The title and general discussion use community detection; recovery describes the accuracy statements. The imbalance normalization n/(beta*K) is standard, with beta=1 corresponding to balanced sizes. The manuscript needs only this lower size bound. Following Amini et al. (2013), b_ik(e) denotes the node-to-block edge count, and its spectral counterpart is tilde b_ik(e). Total ordered block counts are entries of Z(e)' A Z(e), with diagonal entries counting within-block undirected edges twice.

## Algorithm

Retain the K raw-adjacency eigenpairs with largest absolute eigenvalues. Every later operation uses only these eigenpairs.

With `s=max(1,||Lambda||op)`, set `epsilon=c0*s/n` and `q=clip(U Lambda U',epsilon,1-epsilon)`, where `0<c0<min(kappa,eta,1)/8`. The complete row score is

`ell_i(mu)=sum_j {q_ij log(mu_j)+(1-q_ij)log(1-mu_j)}`.

Free connection profiles are fitted by weighted-mean EM updates. Global-mean growing selects candidates using positive hard gains of this row score. At most `ceil(log n)` component-replacement rounds test every deletion position and its best all-node candidate, refit from uniform weights, and compare endpoints by the common soft mixture objective. Every fit takes at least one M-step. A full unsuccessful round permits early stopping. One final E-step and profile/weight M-step is mandatory. Further finite exact EM updates preserve the theorem.

Initialization and refinement use the same row score; hard candidate gains and the soft mixture objective are different aggregations of it. The general-K theorem covers the complete replacement algorithm, without an assumed K-means initializer or a claim that growing alone always succeeds.

## Why the loss is small enough

1. An oracle membership projection preserves the community edge counts exactly. The empirical projection estimates this space and introduces count errors; the proof makes no sub-1/2 rounding claim.
2. A deterministic replacement reduces population hard loss by a factor at most 1-1/K. Its empirical objective certificate gives the endpoint E-step responsibility error O(1/(np)). The existing final M-step supplies the required fitting accuracy.
3. A vertex-deleted arrowhead expansion expresses spectral contrasts as linear statistics of independent incident edges plus bounded remainders. Projected weights change the conditional log moment generating function by O(1).
4. A finite-branch treatment of clipping and local concentration under the oracle Chernoff tilt retain the 1/sqrt(np) factor. Spectral exceptional events have probability at most C_H exp(-H np) for arbitrary fixed H.

This controls the testing tails; ordinary relative consistency of counts alone would not establish the sharp rate.

## Implementation and evidence

The current entry point is [SpectralLikelihood.decode()](../../experiments/20261010_general_density/spectral_likelihood.py). It accepts the eigenpairs, not the original graph or degrees. Defaults include the complete Bernoulli score, 100 growing updates per stage, one update per replacement trial, the stated replacement budget, and one mandatory final E/M update. The default floor coefficient 1e-4 requires the explicit admissibility condition above.

- The [60-graph benchmark](../../experiments/20261010_general_density/) covers K=2–6, two density scales, and three matrix families. Replacement changed no classification errors in that collection; weak general signals had high finite-sample error. These observations are retained without claiming universal superiority over K-means.
- The [five-community population certificate](../../experiments/20261010_general_density/population_audit/) exhibits a finite-stage growing failure and its replacement repair.
- The earlier [240-graph sparse-score experiment](../../experiments/20261009_residual_likelihood_seeding/) evaluates a different historical score and remains separately labeled.
- The [144-configuration tilted-row diagnostic](../../experiments/20261018_sharp_rate/) checks arrowhead identities and constant-scale remainders. It is not an empirical proof of an exponentially small risk bound.

## Build and historical files

Edit main.tex and reference.bib. The manuscript contains all sections and proofs in one TeX file; its figure is figures/initialization_failure_repair.pdf. Run:

```sh
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

GitHub Actions builds main.pdf, the identical compatibility copy manuscript.pdf, and latex-source.zip, recording source provenance in BUILD.txt and BUILD_SHA256SUMS.txt. manuscript.tex is a compatibility wrapper. Earlier section files and dated audit documents are historical snapshots and are not build inputs. The [standalone sharp-rate note](../notes/spectral_sharp_rate_20261018.pdf) records the preceding derivation; its proof is now integrated into main.tex.


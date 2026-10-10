# Spectral likelihood decoding: current manuscript

[Compiled PDF](main.pdf) · [LaTeX source](main.tex) · [Bibliography](reference.bib) · [Download sources with figures](latex-source.zip) · [Proof explanation](END_TO_END_20261010.md) · [Audit and corrections](REVIEW_20261010.md)

## Model and guarantee

For every fixed known K, `P` is the actual symmetric Bernoulli block probability matrix and `p = max_ab P_ab`, with `p_aj=P_{a,z_j}` following Zhou–Li. The assumptions have fixed positive constants:

- `n_a >= pi0*n`;
- `P_ab >= kappa*p` for every a,b and `sigma_min(P) >= kappa*p`;
- `p <= 1-eta`;
- `p = Ω(log n/n)`.

The expected degree in class a is `sum_b (n_b - 1{a=b})P_ab`. These expectations may differ, and each is `Θ(np)` under the uniform probability bounds, so the density condition implies an `Ω(log n)` lower bound. There is no fixed `(n/log n)P` assumption, no upper bound of logarithmic order on degree, and no convergence requirement on `P/p` or community proportions. The model includes sparse, intermediate and dense sequences with probabilities bounded away from one. Quantitative signal separation remains necessary; degree alone does not identify communities.

Theorem **Nearly optimal spectral likelihood recovery** gives

`E Mis <= exp(-(1+o(1)) I_n)`

for Algorithm 1, with exact recovery if `liminf I_n/log n > 1`. Here `I_n` is the minimum exact product-Bernoulli Chernoff information after removing the self-loop coordinate. The rate is the leading oracle exponent, not a refined multiplicative risk claim or a new global minimax lower bound.

## Actual algorithm

The method retains the K eigenpairs of the original adjacency matrix with largest absolute eigenvalues. It uses no raw-adjacency refinement after this step.

With `s=max(1,||Lambda||op)`, define `epsilon=c0*s/n` and `q=clip(U Lambda U',epsilon,1-epsilon)`. The chosen constant must satisfy `0<c0<min(kappa,eta,1)/8`. The default implementation uses `c0=1e-4`; that value is not claimed to cover every arbitrarily small allowed signal constant.

The complete score is

`ell_i(mu)=sum_j {q_ij log(mu_j)+(1-q_ij)log(1-mu_j)}`.

The dependent fractional profiles make this a working likelihood. Its relation to the exact edge likelihood is proved, rather than asserted as an independence model. The weighted-mean profile M-step is unchanged.

Global-mean growing supplies K profiles. Algorithm 1 then performs at most `ceil(log n)` deterministic replacement rounds. Each round tests all K deletion positions and all node candidates, uniformly initializes each trial's weights, runs at least one exact M-step, and chooses the greatest common working likelihood among the trials and old fit. A full round with no improvement permits early stopping. The final E/M update is mandatory. Any further finite exact EM updates preserve the theorem.

This repeated replacement is an explicit change to the original procedure. The unchanged growing-only path is not claimed to have a general-K guarantee. The proof requires neither a random restart nor an assumed weak initializer.

## Proof structure

The population replacement lemma reduces hard loss by at least a factor `1-1/K`. Uniform average spectral-profile approximation transfers it to the empirical soft loss:

`F_(t+1) <= (1-1/K) F_t + C a_n n^2 p + n log K`, with `a_n -> 0`.

After `ceil(log n)` rounds, `F=o(n^2 p)`. Exactly K separated true profiles and K fitted profiles then imply a common matching and weak responsibilities. The mandatory final M-step supplies the required local likelihood accuracy.

The spectral and profile bounds hold on an event with failure at most `C_H exp(-H np)` for every fixed H. A direct clipped Chernoff bound uses deterministic envelopes for the random spectral threshold. These bounds cover information scales above `log n`; inverse-polynomial confidence alone would not suffice.

## Implementation and empirical scope

The current entry point is [SpectralLikelihood.decode()](../../experiments/20261010_general_density/spectral_likelihood.py). Defaults are the full score, 100 growing updates per stage, one update per replacement trial, `ceil(log n)` rounds with valid early stopping, and one mandatory final update. The companion sparse module now defaults to `global_gain_certified`, a deterministic threshold-certificate algorithm with its own general-K proof and 24-graph checks. Its random-residual option is `global_gain_restarts`. Neither is the full-Bernoulli Algorithm 1.

The [60-graph audit](../../experiments/20261010_general_density/) covers K=2–6, two density scales and three matrix families. It retains every input, candidate score and objective trace. Replacement did not change any graph's error, and weak general matrices gave high error at n=320. The [population certificate](../../experiments/20261010_general_density/population_audit/) rigorously identifies a K=5 one-update-per-stage growing failure and demonstrates a replacement repair.

The older [240-graph sparse-score experiment](../../experiments/20261009_residual_likelihood_seeding/) remains separately labeled. Its favorable results do not evaluate the current full-score algorithm.

## Build and historical files

Edit `main.tex` and `reference.bib`. The entire manuscript, including all sections and proofs, is contained in `main.tex`; it has no external LaTeX input files. The figure lives in `figures/initialization_failure_repair.pdf`. [latex-source.zip](latex-source.zip) contains the two editable files and the figures, ready to compile after extraction.

From this directory, run:

```sh
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

GitHub Actions compiles `main.tex`, packages the editable sources, and records provenance in `BUILD.txt` and `BUILD_SHA256SUMS.txt`. Its outputs are `main.pdf` and `latex-source.zip`. The previous PDF URL `manuscript.pdf` is kept as an identical compatibility copy; `manuscript.tex` is a wrapper around `main.tex`.

The earlier section files and `references.bib` remain reference snapshots and are not build inputs. `sparse.tex` and `iterative_swap_proof.tex` are historical companion material. The latter is not an input to this full-Bernoulli manuscript. Dated correction/outline documents retain their historical scope.

## Concurrent sparse revision preserved

The merge retains the sparse general-K implementation and [24-graph checks](../../experiments/20261009_residual_likelihood_seeding/general_k_fresh_checks.json) from revision `452ca1d`. That version already proved arbitrary fixed K at expected degree Ω(log n), with `p -> 0` and a threshold stopping certificate. The present version extends the score and proof to dense probabilities and uses a direct geometric soft-loss contraction. The [earlier audit](REVIEW_20261010_GENERAL_K.md) remains a dated companion record, including its later notation updates.

This merge also incorporates the common notation `p = max_ab P_ab` from `6cbb58e`, the shortened roadmap from `ca4f330`, and the final companion build audit `f250231`. The full-Bernoulli manuscript has been recompiled and all 31 pages visually verified after these changes.

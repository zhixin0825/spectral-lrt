# Oracle-risk scaling experiment for v11

The complete experiment has **300 graphs**: three ordinary-SBM models, five graph sizes, and 20 independent graphs per condition. It uses v11 Section 3.2, not the older algorithm in paper/current. All primary runs completed; no runs or failed initializations were discarded.

| n | Balanced K=2 | Unequal K=3 | Below-log degree K=2 |
| --- | --- | --- | --- |
| 1,000 | 2.90 | 2.01 | 6.22 |
| 2,000 | 2.47 | 2.37 | 6.06 |
| 4,000 | 2.67 | 2.08 | 6.43 |
| 8,000 | 2.54 | 1.98 | 5.92 |
| 16,000 | 2.57 | 2.06 | 5.98 |

Entries are mean misclassification / exact Bernoulli oracle risk. Graphwise 95% intervals, risks, degrees, and initialization errors are in [summary.csv](summary.csv). [section5.tex](section5.tex) is the manuscript-ready section; [main.tex](main.tex) compiles it as a standalone report. This commit does not replace the newer v11 manuscript with the older GitHub paper/current source.

## Reproduce

Only Node.js and its standard library are required:

~~~sh
node validate.cjs
node run_experiment.cjs ./reproduced
~~~

The numerical kernels used for the archived runs executed directly in a JavaScript V8 runtime. The supplied standalone Node.js script contains those same kernels. The first model uses the original block subspace eigensolver; the other two use a faster fully reorthogonalized Lanczos solver. Both select the K largest eigenvalues in absolute value. An independent dense Jacobi eigensolver and matched-graph cross-checks validate them. The driver preserves this solver choice when reproducing the archive and checkpoints each graph. The exact count-sum oracle is computed from the true parameters for evaluation only. True labels are never passed to initialization, fitting or classification.

## Fixed design

- Sizes: 1000, 2000, 4000, 8000, 16000; 20 graphs each.
- Balanced K=2: proportions (0.5,0.5), P = 4 log(n)/n * [[1,.35],[.35,1]].
- Unequal K=3: proportions (.4,.35,.25), P = 14 sqrt(log(n))/n * (.8 I + .2 J).
- Below-log degree K=2: P = 1.6 log(n)/(1.05 n) * [[1,.05],[.05,1]]. Its exact expected degree is below .8 log(n).
- b0=4; c0=.02; radius-count initialization as written in v11; ten completed symmetric block fits and simultaneous label updates.
- Fixed round-robin fallback if an operation is undefined, independent of true labels. It did not occur on the main runs.
- Seed = 270000000 + 1000000*modelIndex + 100*n + replicate, with zero-based model index and replicate. Community membership is randomly permuted before sampling.
- Exact Bernoulli graphs are sampled by geometric gaps over unordered block pairs. No Poisson approximation is used.

## Oracle and uncertainty

The oracle is the **average multiclass row LRT error with true parameters and the other labels**, including self-exclusion and full nonedge terms. It matches v11 Section 2, rather than the historical maximum pairwise binary error in the older GitHub manuscript. Binomial-product sums are deterministic. A geometric decreasing-ratio bound controls the discarded count tail; this does not certify floating-point rounding. Intervals use independent graph risks and t(19)=2.093024054408263, not independent vertex indicators.

## Diagnostics and scope

All 300 graphs and all 15 conditions are archived. Maximum relative eigenpair residual: 9.94e-8; maximum degree residual: 4.62e-13. There were no main numerical failures or fallback partitions. The cap was inactive in the main simulations; an active-cap star graph separately checks the cap, isolated vertices and a negative informative eigenvalue. [validation.json](validation.json) and [validate.cjs](validate.cjs) include the independent dense and exhaustive checks.

The symmetric two-community models show little improvement after the initializer; this is retained in per_graph.json. The unequal three-community model benefits from the likelihood update. The low-degree family has a clearly larger constant loss, near six. These finite model families support a bounded-ratio trend over n up to 16000, not a proof of an asymptotic or uniform guarantee.

## Pilot transparency

Pilot parameters were adjusted before any main seed was run. An initial unequal-three-community design with proportions (.5,.3,.2) and P = 6 log(n)/n * [[1,.25,.2],[.25,.9,.18],[.2,.18,.8]] did not reach the eigensolver tolerance after 700 subspace iterations at n=2000; its residual was 2.53e-6, so no clustering risk was recorded. A stronger pilot with proportions (.4,.35,.25) and P = 8 log(n)/n * (.9 I+.1 J) had oracle risk 1.55e-5 at n=2000, too small for the intended 20-graph budget; the final three-community design was chosen to keep the oracle denominator estimable. The final three families and seeds were then frozen. Pilot records are separate from primary records in [pilots.json](pilots.json).

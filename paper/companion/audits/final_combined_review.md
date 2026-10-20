# Internal review of the restored sparse proof interfaces

**Date:** 20 October 2026  
**Reviewer:** scope / row-tail subagent  
**Disposition:** The reviewed interfaces and the current primary statements are consistent. No additional unresolved mathematical obstruction was found in this final inspection. This is an internal derivation and interface review, not a claim of external peer review or a fresh independent verification of every earlier section of the note.

## Files and scope reviewed

The substantive review covered `restored_local_loo.tex`, `restored_dc_fit.tex`, `sparse_restoration.tex`, and `restored_high_range.tex`. The final statement check also compared the primary theorem in `statements.tex`, the abstract in `main.tex`, and the scope table in `scope_audit.tex`. The reviewer derived the truncated-row module in `restored_row_tail.tex`; that module is therefore a supplied proof component, rather than an independently authored component being certified by its own author.

The older sparse draft, `spectral_audit_history/old_20261009.tex`, supplies historical motivation for the same-threshold recursion. Its former scalar errors were only of order square-root degree or smaller than degree. Those errors alone do not imply a bounded oracle-risk ratio. The new argument must, and does, use the new scalar and fitting estimates.

## 1. Final theorem scope is aligned

The primary `thm:regularized` and the two detailed theorems `restore:dc-theorem` and `restore:sbm-theorem` now have the same relevant range:

- Fixed arbitrary number of groups, comparable group sizes, fixed positive compact full-rank block profiles, and bounded positive degree parameters where applicable.
- Average-degree scale `d = np` diverges; `np^2` remains bounded by a fixed constant.
- There is no lower bound proportional to `log n`. Splitting the proof at a fixed multiple of `log n` selects a probability estimate and does not restrict the algorithm or theorem.
- Ordinary SBM fixes all degree parameters at one and uses the full, unprofiled Bernoulli likelihood. Its reference is the true-parameter raw-row Bernoulli oracle.
- DC-SBM uses the specified clamped, within-group-normalized nuisance estimate and target profiling. Its general reference is the profile oracle. Comparison to the oracle supplied with the target's true degree parameter retains the finite-sample equal-exposure assumption.
- At least one fit-and-assignment update is required. Any finite sequence of the specified hard refits, including finite adaptive stopping, is covered with the same fixed reconstruction.

The new ordinary theorem is for the explicitly specified radius-count and block-fitting decoder. It does not assert the unchanged older free-profile mixture / soft-EM schedule in the sparse regime. The earlier ordinary theorem can retain its separate probability range beyond bounded `np^2`.

The last remark of `dc_oracle_bridge.tex` was flagged for editorial clarification: its warning about the earlier algorithm's logarithmic range should name the earlier `thm:dc` and point to the subsequently completed regularized argument. The underlying oracle comparison remains valid; this was not a mathematical obstruction to the new theorem.

## 2. High-range probability closure removes the former ordinary scope mismatch

`restore:high-aggregate` uses the raw centered spectral norm bound with an arbitrarily prescribed fixed exponential failure rate when `d` is at least a fixed multiple of `log n`. It does not intersect that norm event with a minimum-raw-degree event.

On the total-degree event, the empirical threshold is comparable to `d`. A sufficiently small structural floor multiplier places the threshold below all population degrees, while still giving a lower bound of order `d` on every floored degree. Consequently the raw norm estimate directly gives normalized noise of order `d^{-1/2}` and weighted degree energy of order `n`. Flooring reduces the distance to the population degree; it need not be inactive at observed low-degree vertices.

The signal gap and block residual follow from the same deterministic Gram and projection arguments as the sparse-floor proposition. No upper bound on every observed degree is needed for those arguments.

The true block-total threshold is a fixed large constant times `n`. Bernstein's exponent is of order `n/d`; bounded `d^2/n` lets its fixed constant be chosen to give any prescribed exponential rate in `d`. Full-graph raw norm bounds persist on principal submatrices. Removing a fixed number of vertices changes a raw block total by at most a fixed multiple of `n`, and changes total degree by at most the same order. Thus the event covers all required fixed-depth minors without a degree-wise union.

The tail-energy recurrence uses only deterministic aggregate consequences on its event. Its innovation and degree-tail estimates are applied before restricting to that event. Replacing polynomially accurate sparse geometry by the exponentially accurate high-range event is therefore legitimate. No polynomial failure probability is being relabeled as exponential.

This closes all diverging degree scales in the bounded-`np^2` class for general positive compact ordinary profiles. The old information-degree margin and entry-ratio-two restriction are not needed for this new regularized decoder.

## 3. The scalar full-to-deleted comparison uses only local row control

The generalized-eigenvector proof in `restore:loo-scalar` retains the empirical threshold change. Its lower-block perturbation has a neighbor-supported part and a global diagonal part of order `d/n`; the resulting `1/n` term in the complementary-space bound is kept.

The top resolvent denominator contains the floored target degree, which is of order `d`, rather than the possibly zero or small raw target degree. The denominator is therefore bounded away from zero at the required scale. The informative eigenvalues may have either sign; only separation from the complementary spectrum is used.

The row bound is required on the tested row's hit coordinates only. The remaining factors use total floored degree and weighted norms. Expanding the scalar action gives a bounded error when the tested degree is at most a fixed multiple of `d`. The exact rank-one degree correction and simple hollowing add bounded errors as well. For an isolated target, the reconstructed row is zero, so no whole-graph fallback is required.

For the LOO hit condition, the relevant exceptional set is a set of eigenvector rows, not merely a coefficient-error set. On training geometry, `Delta_j >= c d`, so a violation of the weighted-row bound implies a sufficiently large value of `sqrt(n) * ||U_B[j]||`. The truncated-energy lemma bounds the expected number of such rows. Conditioning on the training graph and multiplying by the incident probabilities, at most a constant times `d/n`, gives the required expected hit probability with arbitrarily prescribed fixed exponential rate. This implication does not require reversing the coefficient-to-row inclusion.

## 4. Fitting, initialization, and the hard-refit basin

The aggregate fitting module does not use a global spectral-row bound. In particular, every possibly data-dependent error set of size at most a fixed multiple of `n/d` has total floored degree of order `n` and total weighted spectral-row norm of order `sqrt(n/d)`. These deterministic estimates simultaneously control all partitions in the count basin.

They justify the radius-count initializer, block-total accuracy of order `n`, and the local wrong-label score error of order one. The clamped nuisance estimator remains bounded and has exact mean one within each fitted group. The signed Bernoulli block-root argument uses aggregate absolute reconstructed mass; it does not silently assume that every reconstructed entry lies between zero and one.

For the signed target profile, the derivative bound confines every maximizer to a fixed interval. Expanding on that interval costs order `np^2`, which is bounded in the stated theorem. This is the visible reason for retaining the bounded-`np^2` hypothesis.

The invariant-basin proof is valid with a fixed input radius `M`. The output error-count constant is independent of `M`; only the sufficiently-large-sample threshold can depend on `M`. Choosing `M` above the initializer and output constants gives a deterministic invariant basin under one common label matching. The auxiliary diagonal/hollowing count is used only to prove basin invariance; it is not charged as an oracle-scale probability event.

For every locally regular target, the same canonical true-contrast event contains errors from every partition in that fixed basin. This is why arbitrary finite hard refits and adaptive finite stopping need no union over iteration number or independence assertion about stopping.

## 5. Ordinary SBM retains the information in the nonedge intercept

The ordinary counterpart explicitly fixes theta at one and does not profile the target. Its symmetric block Bernoulli fit estimates each probability with additive error of order `1/n`. The full likelihood contrast consists of a bounded block-constant log-odds action and a deterministic nonedge intercept.

Parameter replacement in the log-odds action costs order `1/d` times a reconstructed absolute row mass of order `d`. Wrong labels cost order one by the weighted wrong-set estimate. In the intercept, probability errors sum to order one, and the `n/d` incorrect group assignments each cost order `p`, again giving order one. The true population margin is a sum of Bernoulli KL divergences and is uniformly of order `d` under the compact full-rank assumptions. The ordinary invariant-basin proof therefore has the same structure.

Under the ordinary exact Bernoulli error tilt, the tilted probabilities are block constant and admit the required block-constant population-degree representation. The existing weighted cancellation and quadratic error estimates apply. No degree information is lost through an unintended target profile.

## 6. Independence and averaging repairs are present in the current integration

The full common event depends on the tested incident row. It cannot be used as a conditioning event that preserves that row's independence. The current sparse proof now explicitly introduces a separate event formed only from the deleted training graph and its further deletions. It first uses deterministic error inclusion on the full graph, then conditions on the training graph, paying the full and training geometry failures separately. The high-range module gives the analogous construction.

Fitted coefficients are replaced by deterministic true block contrasts before the independent-row sharp transfer is invoked. Uniformity of the algebraic fitted bounds is not confused with independence of fitted coefficients.

The full-target spectral tail is an averaged statement. The current risk display first averages the local target exceptions and then gives the expected misclassification bound. It no longer implicitly claims a pointwise uniform tail at every heterogeneous-degree vertex. The independent deleted-graph coefficient-hit estimate remains available for each fixed tested target with uniform structural constants.

For either degree range the final additive geometry and local-exception costs can be made smaller than the lower oracle scale `d^{-1/2} exp(-C_I d)`. Below a fixed logarithmic scale this uses an arbitrary fixed polynomial geometry exponent; above it this uses an arbitrary fixed exponential geometry exponent. Local hit and target exceptions have arbitrarily large fixed exponential rates after choosing fixed thresholds. There is no union over all tested targets in the risk accounting.

## 7. Reference check and limits of this review

A final static scan found **66 `restore:` labels, no missing `restore:` references, and no duplicated `restore:` labels** in the current TeX files. The earlier stale `restore:tail-hit` reference has been removed. Compilation and final rendered-document checks are delegated to the integrating parent agent.

The checked conclusion is a bounded risk ratio with constants depending on the fixed structural class and bounded `np^2`. It is not a ratio tending to one, a small numerical multiplier, a result for vanishing signal singular value, an unrestricted known-target-degree DC oracle result, or a claim that earlier experimental curves already implement this new regularized decoder.


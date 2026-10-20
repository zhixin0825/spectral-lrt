# Independent review: closing the remaining degree scales without a minimum-degree event

## Verdict and exact scope

The proposed high-degree-scale replacement is valid. For any fixed A > 0 and H > 0, on d = np >= A log n one can construct a training-measurable aggregate event with failure probability at most C_H exp(-H d), using the raw centered operator norm and true block totals. This event supplies the new degree-floor geometry, including all one- and two-vertex deletions, without requiring a lower bound on any individual raw degree. Thus the old information-versus-minimum-degree margin is not needed for this regularized decoder.

Combined with the newly proved local row-tail, full-to-LOO, and fitting modules, this closes the high-scale payment step for general fixed positive compact full-rank ordinary block profiles under np^2 <= C_1. It is not a claim about the original unchanged finite growing/replacement/free-profile EM algorithm. That separate algorithm-scope obstruction remains as documented in ordinary_algorithm_scope_review.md.

This note audits this closure and its interfaces. It is not a replacement for the separate mathematical reviews of the row-tail recurrence and fitted-parameter proof.

## The cited raw operator-norm tail really has an arbitrary fixed exponential rate

I independently opened the original paper by Bandeira and van Handel, arXiv:1408.6185v4, PDF pages 20–21 (zero-based 19–20):

https://arxiv.org/pdf/1408.6185

Corollary 3.12 and the centered-entry extension in Remark 3.13 give, for independent centered upper-triangular entries of a symmetric matrix X,

P{ ||X|| >= (1+epsilon) 2 sqrt(2) sigma + t }
  <= n exp[-t^2/(c_epsilon sigma_star^2)],

where sigma^2 = max_i sum_j E X_ij^2 and sigma_star = max_ij ||X_ij||_infinity. The sqrt(2) is the extension from symmetric distributions to centered distributions. The manuscript's normalized_geometry.tex lines 10–23 cites the correct result.

For X = A - E A, sigma^2 <= C d and sigma_star <= 1. If d >= A log n, choosing t = sqrt(c_epsilon (H + 1/A)) sqrt(d) gives the desired exp(-H d) bound. The completed-mean diagonal has norm O(p) and is absorbed into the fixed multiple of sqrt(d). No minimum-degree event enters this calculation.

## Deterministic aggregate consequences

Let X = A - Omega and suppose ||X||op <= L_H sqrt(d). Put D0 = Omega 1 and barD = n^{-1} 1' A 1. Then

|barD - n^{-1}1'Omega1| <= ||X||op <= L_H sqrt(d).

Because population degrees lie in [c d, C d], this yields barD comparable to d for all sufficiently large n. Choose the fixed empirical floor coefficient a sufficiently small for the structural class. Then tau = a barD satisfies c_a d <= tau < min_i D0_i, independently of whether individual observed degrees vanish. With G_i = max(D_i,tau),

G_i >= c_a d,    sum_i G_i <= sum_i D_i + n tau <= C n d,

||G^{-1/2} X G^{-1/2}||op <= L_H/(c_a sqrt(d)),

sum_i (D_i-D0_i)^2/G_i
 <= ||X1||_2^2/(c_a d) <= (L_H^2/c_a) n.

Since the floor threshold is below D0_i, |G_i-D0_i| <= |D_i-D0_i| coordinatewise. Thus the second weighted degree-energy bound follows from the same inequality. The population signal gap and the weighted block residual now follow from the deterministic argument already written in prop:sparse-floor. No uniform upper or lower raw-degree premise is needed.

Every principal submatrix of X has operator norm at most ||X||op. Therefore all one- and two-vertex-deleted graphs inherit this raw norm bound simultaneously, without an n^2 probability union. Their completed means are the corresponding principal submatrices of Omega; their population degrees and average degrees change by only the appropriate bounded deletion corrections. The same empirical-floor argument applies on each graph.

## True block totals at the same probability strength

For fixed K, a true centered block total is an independent-edge sum with coefficients at most two and variance O(n d). At threshold L_B n, Bernstein gives

P{ max_ab |[Z'(A-Omega)Z]_ab| > L_B n }
 <= C_K exp[-c L_B^2 n/(d+L_B)].

Under d^2 <= C_1 n and eventually d >= L_B, the exponent is at least c' L_B^2 d/C_1. Increase the fixed L_B to obtain any chosen exponential rate H. Finite prefactors can be absorbed by starting with H+1. The completed diagonal costs O(d), which is absorbed by the n-scale threshold.

The same block-total inequalities for one- and two-vertex deletions can be obtained from the full event by enlarging the fixed threshold: deleting a fixed number of vertices changes such a total by O(n), because each matrix entry is bounded. Alternatively one can derive the same estimate for each training graph directly when establishing its own measurable event. One does not need a minimum-degree failure payment.

## Interface with the new row-tail proof

The recurrence in restored_row_tail.tex uses only these deterministic premises: recomputed floors comparable to d, total degree O(n d), centered weighted noise O(sqrt(d)), fixed informative gap on the d scale, and discarded norm O(sqrt(d)), for the required deletion family. The innovation moment estimate and its high-degree remainder are unconditional estimates and do not use the n^{-r} failure bound.

Consequently the recurrence can be rerun with this high-scale event E_H in place of E_r, giving the same expected truncated energy and local hit-mass conclusions, now with P(E_H^c) <= C_H exp(-H d). This substitution should be stated explicitly. One cannot merely reinterpret the previously written n^{-r} probability bound as exponential in d.

Use each deleted graph's own raw-norm and block-total event when conditioning on training data. That event is measurable from the training graph alone; the full event supplies it where needed. This preserves the independence interface of the local LOO and sharp-transfer lemmas.

## Paying the remaining failure probability

On either fixed compact model class, the reference average risk has the uniform lower bound c d^{-1/2} exp(-C_I d). Choose H > C_I + 1 for the raw norm, block totals, local degree tails, and exceptional coefficient/target tails. Then

exp(-H d) / [d^{-1/2} exp(-C_I d)]
 = sqrt(d) exp[-(H-C_I)d] -> 0.

The low-scale proof already handles d <= A log n by choosing a fixed polynomial probability exponent r sufficiently large. Joining the two regimes covers every d -> infinity under np^2 <= C_1. The split at A log n is a proof device and is not a lower-density hypothesis of the resulting theorem.

## Theorem statement consequences

The restored ordinary theorem may therefore use general positive compact full-rank P=pC with comparable group sizes, d -> infinity, and np^2 <= C_1. It need not retain the entry-ratio-two restriction or an information-degree margin solely to pay minimum-degree events.

The DC profile-oracle statement has the same all-scale closure under its bounded positive theta and exact within-block normalization assumptions. Its comparison to the known-target-theta oracle still requires the stated equal-exposure condition. This closure does not remove that separate oracle distinction.

The new decoder remains the explicitly declared degree-floor encoder, metric reconstruction, degree correction, simple hollowing, ball-count initializer, block-Bernoulli fit, and profile or ordinary block LRT with finite hard refits. The original free-profile growing schedule is not established in arbitrary sparsity by this argument.

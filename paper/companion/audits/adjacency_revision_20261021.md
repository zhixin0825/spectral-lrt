# Ordinary adjacency revision, 21 October 2026

This is an internal source-review record. The revised first article and
the current companion statements control the mathematical scope.

## Encoded information and theorem

The first article retains raw degrees and the K largest-absolute
eigenpairs of A_tilde = W A W, where tau = b0 times the observed average
degree, b0 is sufficiently large and fixed, and
W_i = sqrt(tau / max(D_i, tau)). Vertices below the threshold have weight
one. With no active cap, these are the eigenpairs of the original A.
The all-scale theorem uses this specific square-root cap.

The guarantee remains E loss <= C R_SBM,* for fixed arbitrary K,
comparable group sizes, positive compact full-rank block profiles,
d = np -> infinity and np^2 <= C1. No logarithmic lower bound is added.
It is the literal risk comparison with the true-parameter incident-row
Bernoulli oracle, including the d^(-1/2) prefactor.

## Reviewed proof changes

- On the aggregate event, tau is above all completed population degrees.
  The new energy sum (G_i - tau)^2 / G_i is at most the weighted degree
  fluctuation sum. The population residual uses tau G^(-1/2) Z.
- The LLV comparison factor is sqrt(max(D_i, 2d*) / G_i); the former
  small-floor assertion that it equals one above 2d* is not used.
- The informative adjacency gap has order d. Q = A_tilde / tau is only
  an auxiliary scalar rescaling in the proof.
- The same-threshold spectral-row recurrence and the local scalar
  comparison retain the global O(d/n) empirical-threshold perturbation.
- Wrong-set mass, initialization and diagonal trace estimates use the
  new cap energy. The true block totals and KL margins still use Omega.
- For the exact tilted probabilities, r = tau h with block-constant
  h = O(1/n). Weighted orthogonality and cap energy bound both terms
  of the tilted mean by O(1), and the quadratic mass is O(1).
- Independent source review found no new gap in these interfaces.
  The polynomial and exponential event arguments still cover every
  diverging expected-degree scale.

## Companion and retained material

The DC algorithm keeps its small-floor normalized eigensystem. Its
geometry and high-range event now have complete local proofs in
Sections 3.4 and 3.7. Sharing the sharp transfer, row-energy recursion
and deterministic local comparison is justified by their stated
generic assumptions. A large-cap DC extension is not claimed.
The preceding ordinary normalized source is archived separately.

## Artifact checks

Both manuscripts compile cleanly (22 and 59 pages), with no undefined
references, duplicate labels or overfull boxes, and all fonts embedded.
All pages were rendered; the changed formulas and algorithm pages also
received full-page visual inspection. The source packages include the
bibliographies and compile independently.

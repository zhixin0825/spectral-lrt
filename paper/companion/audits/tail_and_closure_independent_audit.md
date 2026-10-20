# Independent audit of the sparse row tail and the final sharp-risk bridge

20 October 2026. Reviewed by the local fitting/exception-repair agent.

## Conclusion

The same-threshold row-energy argument in `sparse_tail_work.md`, as converted
to `restored_row_tail.tex`, closes under its stated aggregate geometry
premises. I found no hidden simultaneous row-incoherence assumption, no
degree-wise union that would force `d >= c log n`, and no invalid use of a
row-dependent Procrustes rotation in the conditional Bennett inequality.

The fitted DC and ordinary contrasts, the exact-degree/simple-hollow
full-to-LOO action, and the existing sharp Bernoulli transfer compose with
bounded score cost. The remaining edits needed for a formally clean
composition are explicit separation of full-graph and training-measurable
events and completion of the proof-only weight definition on an all-zero
training graph. These are specified below.

I also supplied `restored_high_range.tex`: it proves an arbitrarily accurate
exponential aggregate event above any fixed logarithmic scale from the raw
norm inequality. It avoids the old raw minimum-degree event altogether.
Combining it with the polynomial aggregate event below that scale closes
the whole `d -> infinity, np^2 <= C` range for general positive compact
full-rank ordinary block profiles as well as DC profiles. No
information--degree margin is needed for this regularized algorithm.

## 1. The perturbation estimate does not assume row incoherence

Writing the weighted full matrix as

`M = [[0,m_i'],[m_i,B_i]]`,

the floor gives `||m_i|| <= sqrt(tau)` and `||B_i|| <= tau`. The deleted
lower matrix is `(I+Gamma_i) B_i (I+Gamma_i)`. Its residual on the full
orthonormal signal basis is bounded by

`C d (||U_i|| + ||Gamma_i U||_F)`.

For example `Gamma_i B_i U_J = Gamma_i U_J Xi - Gamma_i m_i U_i`, by the
full eigenvector equation. The top residual is exactly `-U_i Xi`.
Thus this residual calculation never substitutes an upper bound on any
other spectral row. Separation of the full signal eigenvalues from the
deleted complementary spectrum gives the displayed Frobenius angle and
Procrustes bound. No separation inside the informative spectrum is needed.

The perturbation of empirical square-root weights satisfies

`|Gamma_ij| <= C (A_ij/G_j + (d+D_i)/(n d))`.

The first term is neighbor-supported; the second is the empirical-floor
change. Keeping the second term is essential. The bound `||Gamma_i|| <= C/d`
uses only `G_j >= c d`, `D_i <= n`, and comparability of full/deleted total
degree. It is valid without a maximum raw-degree event.

## 2. The two weight estimates and the contraction

For the clipped basis `V=f_t(U)`, with `t=R/sqrt(n)`, the bound is

`(C beta_i/d)||Gamma_i V||_F <= C t/d + C t sqrt(D_i)/d^(3/2)`.

Here `beta_i = W_i(sqrt(D_i)+C d/sqrt(n))` has square at most `C d`.
The square-root cap does not give `W_i D_i <= C d`, and the proof correctly
does not use that false inequality. The second term can exceed a small
multiple of `t` only on `D_i >= c d^3`; its squared excess is charged as
`C t^2 D_i/d^3` on that event.

For `T=U-V`, changing summation order gives

`sum_i beta_i^2 ||Gamma_i T||_F^2 <= C ||T||_F^2`.

The local coefficient at a fixed row `j` is bounded by
`C d D_j/G_j^2 <= C`. The global coefficient is bounded by
`C (n^2 d)^(-1) sum_i(d+D_i)^2 <= C`, using total degree and
`sum_i D_i^2 <= n sum_i D_i`. These estimates hold simultaneously for
the realized graph; they do not require independence of `Gamma_i` and `T`.

After comparing the clipped row innovation with its genuinely deleted
counterpart, the pointwise inequality can be truncated at exactly the
same `t`. On `||U_i||>t`, the population term, the small bounded-weight
term, and innovations below `c d t` can all be absorbed into a fixed
fraction of `||U_i||`. Squaring the remaining terms and summing gives

`H_R 1_E <= C d^-2 sum_i ||K_i||^2 1{||K_i||>c d t}
              + (C/d) H_R 1_E
              + C t^2 d^-3 sum_i D_i 1{D_i>=c d^3}`.

The remainder has `H_R`, not `H_(R/c)`. Consequently taking expectations
really gives an absorbable `(C/d) E[H_R 1_E]`. The indicator on the
recursive term is retained; the innovation and raw-degree terms may be
bounded without it.

## 3. Conditional innovation concentration is legitimate

The innovation used for Bennett is the unrotated vector

`K_i=(A_iJ-Omega_iJ) W^(-i) f_t(U^(-i))`.

Conditional only on `A_JJ`, all its coefficients are fixed and the tested
Bernoulli variables are independent and centered. The coordinate jump
bound is `t`, and the conditional variance is at most `C d/n` for every
training graph. The possibly row-dependent rotation `O_i` is used only
through the identity `||K_i O_i||=||K_i||`; no concentration is applied to
coordinates after that rotation.

At `x=c d t`, Bennett's exponent is

`(v/t^2) h(t x/v)`, with `v=C d/n` and `t x/v` of order `R^2`.

It is at least `c d log R` for fixed sufficiently large `R`. Beyond that
point its derivative is at least `c log(R)/t`. Integrating the squared
tail therefore gives `C d^2 t^2 exp(-c d log R)`, with constants uniform
over `R>=R0`; the lower threshold on `d` may depend on fixed `R`.

The raw degree expectation above `c d^3` is at most
`C d^3 exp(-c d^3 log d)`, or zero if the threshold exceeds the graph
size. Multiplication by `t^2/d^3` and summation over vertices leaves a
factor `n t^2=R^2`. For each fixed `R`, this rate is eventually stronger
than `exp(-c d log R)`. Thus the claimed truncated energy follows.

**Definition to add.** On an all-zero training graph set `W^(-i)=I` and
use the specified deterministic orthonormal fallback basis. The formula
`sqrt(tau) G^-1/2` otherwise gives `0/0` there. This completion keeps
`||W^(-i)||<=1` and makes the innovation defined on every training graph;
it changes no computation on the geometry event or any implemented
all-zero output.

## 4. Full-graph and training events must be kept separate

Let `E` collect aggregate geometry for the full graph and all deletions
of at most three vertices, together with the full block-total event.
For target `i`, let `E_i` collect the analogous geometry for the training
graph `A_(-i,-i)` and its deletions of at most two further vertices.
Then `E_i` is measurable from training alone, and `E` implies its
geometric requirements. Fixed-depth polynomial unions cost no density
condition: choose the probability parameter larger before taking the
union. In the high range the raw norm, total degree and block-total
events actually imply all fixed-depth minor events deterministically.

For the probability argument, use the following inclusion and relaxation:

`P(error_i, E, local_i) <= P(tilde_T_i <= C, E, local_i)
                       <= P(tilde_T_i <= C, E_i)`.

Now condition on `A_(-i,-i)` and apply the sharp transfer on `E_i`.
Do not write a conditional product-Bernoulli assertion given the full
event `E`, which depends on the incident row. Pay the full aggregate
failure and, if not already covered by a chosen inclusion, the training
aggregate failure separately. Their average has the same order as one
failure probability; there is no union over targets in the final risk.

Similarly, the neighbor-hit exception is bounded by dropping full-graph
restrictions and conditioning on training alone:

`P(hit exceptional training row, E) <= E[1_(E_i) sum_bad_j p_ij]`

when the common-event inclusion is used. This is at most
`C d exp(-H d)` by the training-row count. The target spectral-row
exception is controlled by the full-graph truncated energy after
averaging over targets. No theta exchangeability is used in either step.

The tail proof itself needs only one additional deletion at any invocation;
three levels are a convenient way to match literally the already stated
two-level training events. One can reduce the depth by restating those
events, but doing so is unnecessary.

## 5. Bounded-cost DC profile bridge

For every partition in the invariant count basin, degree clamping and
aggregate block fitting give relative block error `O(1/d)`, exact fitted
mean theta one, and fitted neighbor-mass error `O(n/d)`. Consequently
each fitted log-profile coefficient differs by `O(1/d)` from its true
self-excluded counterpart. At a regular target the reconstructed row
has absolute mass `O(d)`, so this replacement costs `O(1)`. A wrong
label set of size `O(n/d)` has reconstructed row mass `O(1)`, which pays
the label replacements uniformly even when the partition is selected
from the same graph.

For a signed reconstructed row, the exact target-profile derivative has
nonedge contribution at least

`c n p - C p sum_j |R^h_ij| >= c d - C d p >= c' d`

throughout the fixed probability cap. Thus every maximizing target
parameter is bounded. Uniform Bernoulli expansion on that bounded
interval costs `O(np^2)=O(1)`; no entrywise assertion `R^h_ij<1` is
needed. The coefficient of `log t` is the exact preserved raw degree.
The candidate-independent neighbor-theta logarithms cancel. This gives
the true block-profile contrast claimed in the fitting module.

The simple hollow/degree-corrected LOO formula then changes this true
full-graph contrast by another `O(1)` to

`x' [H_B(w-bar(w)1)+bar(w)1]`.

Relative to the true raw linear profile contrast `x'w`, its error is
`x'e`, where `e=(H_B-I)(w-bar(w)1)`. The centered vector remains bounded
and block constant. On the training aggregate event, the previously
proved weighted cancellation therefore gives bounded tilted mean and
quadratic cost. The row-tail lemma supplies the expected exceptional
coefficient hit mass. The sharp transfer applies at a fixed threshold.

The existing raw profile-oracle lemma uses only a single-target
upper-degree cutoff and the exact Bernoulli saddle. Its cutoff failure
has arbitrarily large exponential rate after enlarging a fixed constant.
It does not require the old global raw-degree event. Therefore its
comparison of the true linear profile tail with the actual capped raw
profile oracle is valid at every diverging degree scale. The same is
true for its equal-exposure comparison with the known-target-theta
oracle.

## 6. Ordinary Bernoulli bridge

The sparse ordinary decoder fixes all theta values to one and does not
profile the target. Its block fit has absolute error `O(1/n)`. Incident
log-odds coefficients therefore change by `O(1/d)`, with total row cost
`O(1)`. The nonedge intercept changes by `n O(1/n)=O(1)`; replacing the
wrong group labels adds `(n/d) O(p)=O(1)`. Wrong-label incident mass is
again `O(1)`.

The true incident log-odds vector is bounded and block constant, and its
deterministic Bernoulli nonedge intercept is retained. After the same
degree-corrected LOO step, the exact Bernoulli tilted probabilities are
block constant. They can be expressed exactly as `D0 h` for bounded
block-constant `h=O(1/n)` on the deleted graph. Thus the weighted
cancellation and sharp transfer work on a general fixed positive compact
full-rank block class. No information--degree margin is used in this
local transfer; that margin was only needed by the obsolete global
minimum-degree probability closure.

## 7. High-range replacement and scope

The complete high-range proof is in `restored_high_range.tex`. For any
fixed `A0,H>0`, the raw norm inequality at `d>=A0 log n` has failure
`exp(-H d)` after enlarging its fixed norm constant. With the floor
`G>=c d`, it directly implies normalized noise `O(1/sqrt(d))` and
weighted degree energy `O(n)`. No lower bound on each raw degree enters.
The full block-total threshold `L n` has Bernstein exponent at least
`c L^2 n/(d+L)` and hence arbitrarily large fixed rate in `d` under
`np^2<=C1`. Total-degree failure is `exp(-c n d)`.

The same-threshold proof is a deterministic-on-geometry argument plus
unconditional conditional-innovation moments, so it also applies to
this exponentially accurate event. Choose its rate and the local
exception rates larger than the uniform oracle information constant.
Together with the below-log polynomial closure, this proves the desired
all-scale probability payment.

The result concerns the explicitly restored floor encoder, exact-degree
simple-hollow reconstruction, radius initializer, bounded theta fit,
and hard block-LRT refinements. It does not silently establish the old
unmodified ordinary free-profile growing/replacement/soft-EM schedule.


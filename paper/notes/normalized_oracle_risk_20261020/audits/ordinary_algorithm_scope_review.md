# Ordinary sparse decoder: established algorithm scope and an unadopted safeguard idea

Independent review for the spectral-lrt sparse repair. This note distinguishes:
(1) the original finite growing/replacement plus free-profile EM schedule;
(2) a ball-count initializer plus block-parameter Bernoulli LRT;
(3) an unadopted hybrid safeguard idea, recorded below only for possible later work.

The requested repair uses item (2). The safeguard is not part of the manuscript theorem, has not been separately integrated and checked end to end, and must not be reported as a completed result. The established scope conclusion is the non-transfer result in the next two sections.

## Verdict

The new degree-floor aggregate geometry does NOT directly establish the original sbm:box lemma. The original free-profile algorithm theorem therefore cannot simply be relabeled as a no-logarithmic-degree theorem. A proved ball-count + block-fit + block-LRT route is a different declared decoder.

Two separate obstructions occur:
- Unweighted reconstruction Frobenius error O(d) is not a deterministic consequence of the current weighted geometry.
- The original uniform entry bound |R_ij|<=Cp is incompatible with retaining all raw degrees on sublogarithmic graphs that have D_max/d unbounded. Hence the generated common parameter box and uniform free-profile fitting derivative bounds also require new arguments or an explicit algorithm change.

One possible future safeguard for PARTITIONS is to refit every candidate using bounded comparable block probabilities, compare their classification Bernoulli working likelihoods on the same R, then make one common block-LRT update. The argument sketched below suggests how including the ball-count partition could certify an O(1/d) basin. This proposal preserves the original growing output as one candidate but does not prove the unchanged free-profile endpoint; it is not an adopted or completed result.

## Why the existing sbm:box proof does not transfer

The original proof writes R=A Pi and uses
||A-Omega||op<=C sqrt(d),
||Pi-Pi0||op<=C/sqrt(d),
D_i asymp d uniformly,
and ||U_i||<=C/sqrt(n) uniformly.

These imply both ||R-Omega||F^2<=Cd and max_ij |R_ij|<=Cp. The latter also gives:
- every candidate and every exact EM mean lies in a common Cp probability box;
- max_i ||R_i||_1<=Cd;
- bounded derivatives of every fitted row score;
- a uniform score error O(1) from O(n/d) label/responsibility mistakes.

The floor geometry only controls
||G^{-1/2}(A-Omega)G^{-1/2}||op,
weighted degree energy, and weighted block residuals.
Multiplying a normalized rank-K error by G^{1/2} on both sides cannot be bounded by Cd without more work.

A deterministic counterexample mechanism is a regular degree-d background graph with one additional universal hub. The added normalized star has operator norm O(d^{-1/2}), total G remains O(nd), and the weighted degree energy is O(n). Yet the hub has D_i about n. For ANY degree-preserving R,
||R-Omega||F^2 >= (1/n) sum_i (D_i-D0_i)^2,
which is of order n for this graph, much larger than d.
The event is extraordinarily rare under the SBM and may be paid separately, but it is allowed by the existing aggregate premises.

Even for typical sublogarithmic SBM graphs, a fixed global entry bound is unavailable:
R1=rawdeg and max_ij|R_ij|<=Cp imply D_max<=Cnp=Cd.
The sparse theorem cannot impose that consequence by silently treating every high-degree vertex as global failure.

## Future work only: proposed partition selector from weighted geometry

### Inputs and explicit algorithm

Let R be the new degree-corrected, simply hollowed reconstruction, shared by all candidates.
Candidates include the ball-count partition e0 and optionally the original growing/EM output and any other finitely computed partitions.
For every candidate e, maximize the ordered Bernoulli block criterion
Q_R(e,B) = sum_{i != j} [R_ij log B_{e_i,e_j}
                       +(1-R_ij)log(1-B_{e_i,e_j})]
over a fixed compact comparable-probability box
c_- p <= B_ab <= c_+ p < 1/2.
One can use observed average degree / n to set the box, with fixed constants covering the model class; the good event makes it comparable to p.
For symmetric B, pool the corresponding ordered block-pair contributions. The exact maximizer is a clipped block average, even for signed R.

Select the candidate with largest fitted Q_R.
Then perform the already proved block nuisance refit and one block-LRT classification.
Any optional class-proportion term contributes only O(n), so it does not change the basin argument.

This is an explicit additional safeguard. It does not compare arbitrary free-profile mixtures using the original objective.

### Deterministic premises

Assume d=np->infinity, fixed K and separated compact positive block profiles, comparable true group sizes, and
G_i>=cd,
sum_i G_i<=Cnd,
sum_i (G_i-D0_i)^2/G_i<=Cn,
||G^{-1/2}(R-Omega)G^{-1/2}||F<=C/sqrt(d).
Here D0_i<=C0 d, and the candidate e0 makes at most M0 n/d errors.
These premises are enough for the selector; no uniform upper degree or unweighted R Frobenius bound is used.

### Heavy-degree mass

Choose L0>2C0 and T={i:G_i>L0 d}.
Then
sum_{i in T} G_i
<=4 sum_i (G_i-D0_i)^2/G_i
<=Cn.

### Uniform empirical/population comparison

For a candidate fitted matrix M_ij=B_{e_i,e_j}, define
K(M)=sum_{i!=j} KL(Ber(Omega_ij) || Ber(M_ij)),
L_ij=logit(M_ij)-logit(Omega_ij).

The comparable probability box gives
|L_ij|<=C,
p L_ij^2<=C KL(Ber(Omega_ij)||Ber(M_ij)).

Let E=R-Omega. Split pairs into T-free pairs and pairs incident to T.
Then
sum_ij G_i G_j L_ij^2
<=C d^2 sum_ij L_ij^2
  +C(sum_{i in T}G_i)(sum_jG_j)
<=C n d K(M)+C n^2 d.

Weighted Cauchy-Schwarz therefore gives the uniform bound
|<E,L>|
<=C sqrt(n K(M))+Cn
<=eta K(M)+C_eta n
for every fixed eta>0.

This is the needed objective comparison. A crude absolute O(n sqrt(d)) comparison would only certify weak consistency and is not sufficient for the O(1/d) basin.

### Selection certificate

Let
F_R(M)=Q_R(true Omega)-Q_R(M).
Exactly F_R(M)-K(M)=-<E,L>.
The ball-count partition with the true block matrix has population loss
K(M0)<=C p n * (M0 n/d)<=Cn,
because only pairs incident to mislabeled vertices differ.

Its fitted empirical score is no worse than its score at the true B.
The selected candidate therefore has F_R<=Cn.
The lower comparison gives K(M_selected)<=Cn.

Since all candidate probabilities are comparable to p,
||M_selected-Omega||F^2<=Cp K(M_selected)+O(np^2)<=Cd.
This is a bound on the SELECTED FITTED BLOCK MATRIX, not on R itself.

The true row profiles are separated by c p sqrt(n). At most Cn/d rows can have fitted profile error above one quarter of that separation. Every true class contains a good row, different true classes must use different fitted labels, and fixed K gives a common permutation. Thus the selected partition has at most M1 n/d errors.

The final block-LRT update is essential for the sharp oracle risk. The selector alone certifies an O(1/d) initialization basin, not exponentially small error. Apply the independently proved uniform-in-basin block-fit and LRT theorem after selection.

## The weighted reconstruction premise survives degree correction and simple hollowing

For R0=G^{1/2} S_K G^{1/2},
||G^{-1/2}(R0-Omega)G^{-1/2}||F
=||S_K-M||F<=C/sqrt(d),
using rank <=2K and the normalized noise bound.

For c=rawdeg-R0 1=A(I-H)1,
G^{-1/2}c=S_G(I-Pi)G^{1/2}1.
The discarded-spectrum norm and weighted block residual give
||G^{-1/2}c||<=C sqrt(n/d).
Therefore the normalized Frobenius norm of c1'/n is at most C/d.

For simple hollowing, use the signal angle:
U0 has rows O(n^{-1/2}), and ||U-U0O||op<=C/sqrt(d), so
sum_i ||U_i||^4<=C/d.
Thus the normalized diagonal of R0 has squared Frobenius norm <=C/d.
The correction c/n contributes less.
Redistributing the removed diagonal uniformly over off-diagonal columns has normalized squared norm O(d^{-2}), using the trivial G_max<=Cn.
Consequently the simply hollowed, degree-corrected R obeys the weighted premise above.

## Boundaries

This safeguard certifies partitions produced by the explicit block-likelihood selection rule. It does not certify:
- the unchanged original free-profile EM final estimator;
- the same arbitrary free-profile mixture likelihood as a safe selector;
- sharp risk for the selected partition before its final block-LRT update;
- a fitted-score theorem merely from a lower global objective.

These distinctions should be stated in the new theorem and algorithm description.


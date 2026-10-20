# DC-SBM oracle alignment audit

Date: 20 October 2026.

## 1. Direct answer and necessary correction of scope

The current proof note does account for degree heterogeneity, but its general
profile-LRT reference should not be identified without qualification with
the classical DC-SBM information-theoretic benchmark.

In the classical balanced planted-partition DC-SBM, the exact local
known-parameter oracle considered by Gao, Ma, Zhang and Zhou is covered by
the note's equal-exposure corollary. A direct bridge, proved below, also
connects its risk to the familiar degree-dependent error exponent.

For arbitrary full-rank block matrices, the note's profile oracle is a
separately defined generalized-LR reference. The classical minimax theorem
is a different claim: it takes a supremum over an assortative parameter
space for a fixed degree-parameter vector. It is not a pointwise constant
ratio statement to a target-known-theta oracle on every general block
matrix.

The existing counterexample matrix
\[
\begin{pmatrix}4&2\\2&1.1\end{pmatrix}
\]
is not in the classical paper's assortative class: its second diagonal is
smaller than its off-diagonal. Its no-go theorem therefore must not be
presented as an obstruction to the standard balanced planted-partition
DC-SBM objective.

## 2. Primary source checked

Chao Gao, Zongming Ma, Anderson Y. Zhang and Harrison H. Zhou,
“Community Detection in Degree-Corrected Block Models,”
Annals of Statistics 46(5), 2153–2185 (2018), DOI 10.1214/17-AOS1615.

Full mathematical text checked against the public author manuscript:
https://arxiv.org/pdf/1607.06993 (the served copy is arXiv v1, 44 pages).
The published bibliographic record is
https://doi.org/10.1214/17-AOS1615.
The direct published-PDF fetch was unavailable during this audit; equation
and theorem numbers below refer to the inspected author manuscript.

The inspected locations are parameter space (3), information quantity (6),
Theorems 1–2 and Corollary 1, local testing problem (10), Algorithm 2,
Theorem 4 and (13)–(15), and Section 5.1, especially (18)–(20) and Lemmas 2–3.

Retrieval refs: turn81view1; targeted passages turn82view1, turn82view4,
turn83view0, turn83view2, turn83view3. Parent must open a source itself
before using a child retrieval ref in the final user citation.

## 3. What the source's oracle and minimax benchmark mean

The fundamental local test holds the same target parameter theta_0 fixed
under the two hypotheses:
\[
H_a:\quad
\bigotimes_{j\in G_a}\operatorname{Bern}(\theta_0\theta_jp)
\otimes
\bigotimes_{j\in G_b}\operatorname{Bern}(\theta_0\theta_jq),
\]
\[
H_b:\quad
\bigotimes_{j\in G_a}\operatorname{Bern}(\theta_0\theta_jq)
\otimes
\bigotimes_{j\in G_b}\operatorname{Bern}(\theta_0\theta_jp).
\]
This is a simple-versus-simple Bernoulli experiment. Its NP optimum uses
the true parameters; its adaptive neighbor-voting substitute avoids their
estimation. The cited result matches the leading exponent, not an exact
risk ratio including a sharp prefactor.

For equal-size groups, the classical degree-dependent exponent is
\[
J_i=\theta_i\frac nK(\sqrt p-\sqrt q)^2.
\]
Its reported minimax scale is
\[
\left[\frac1n\sum_i e^{-J_i}\right]^{1+o(1)}.
\]
In the more general parameter-space statement, the displayed exponent
uses n/2 for K=2 and n/(beta K) for K>=3, under the theorem's other
conditions. Those are classwise minimax quantities; they are not the
exact oracle error at each particular block matrix.

The parameter class is assortative:
\[
\max_{a\ne b}B_{ab}\le q<p\le\min_a B_{aa}.
\]
Its vector theta is fixed while taking the parameter-space supremum, with
approximate within-group normalization and allowed group-size variation.

## 4. A sharp bridge that can be added to the current note

### Proposition: classical balanced DC-SBM oracle bridge

Let K>=2 be fixed and n=Km. Suppose
\[
P_{ab}=
\begin{cases}p,&a=b,\\q,&a\ne b,\end{cases}
\qquad
0<r_-\le q/p\le r_+<1,
\]
with fixed r_-,r_+. Suppose each group has m vertices,
\[
\theta_-\le\theta_i\le\theta_+,\qquad
\sum_{z_j=a}\theta_j=m,
\]
with fixed positive bounds, and
\[
np\longrightarrow\infty,\qquad np^2\le C_1.
\]
Let R_known be the actual raw-row known-parameter local LRT risk, R_prof
the raw-row profile reference defined in the current note, and R_vote
the raw-row classifier that knows the neighbor groups and selects the
largest block count. Resolve ties by any fixed ordering.

Then, uniformly over this parameter class,
\[
R_{\rm known}\asymp R_{\rm prof}\asymp R_{\rm vote}
\asymp
\frac1n\sum_{i=1}^n
\frac{1}{\sqrt{np}}
\exp\left\{-\theta_i\frac nK(\sqrt p-\sqrt q)^2\right\}.
\tag{B1}
\]
Equivalently, the prefactor can be written as
\((\theta_i m(p+q))^{-1/2}\), changing only fixed constants.

This proposition is solely an oracle comparison. It does not require
np >= C log n and is therefore valid for np growing arbitrarily slowly.
It does not by itself establish the sparse spectral algorithm's error
bound.

### Proof

Fix target i in its true group a and a competitor b. Let
\[
N_c=\sum_{j\ne i:z_j=c}A_{ij},\qquad
M_c=\sum_{j\ne i:z_j=c}\theta_j
=m-\theta_i\,\mathbf1\{c=a\}.
\]
Put L=log(p/q), d=np, and
\[
T=L(N_a-N_b).
\]

**Exact count cumulant.** Under the actual true Bernoulli row law,
\[
K_i(s)=\log\mathbb E e^{-sT}
=\sum_{j\in G_a\setminus\{i\}}
\log\{1+\theta_i\theta_jp(e^{-sL}-1)\}
+\sum_{j\in G_b}
\log\{1+\theta_i\theta_jq(e^{sL}-1)\}.
\]
Uniformly for s in [0,1], including its first two derivatives,
\[
K_i(s)
=m\theta_i\{p(e^{-sL}-1)+q(e^{sL}-1)\}
+O(np^2+p).
\tag{B2}
\]
The O(p) accounts for the one self-excluded theta_i in group a.
The leading expression is minimized at s=1/2 and its minimum is
\[
-m\theta_i(\sqrt p-\sqrt q)^2.
\]
The derivative signs at the two endpoints and the second derivative
are uniformly separated at scale d because q/p stays in a fixed
compact subinterval of (0,1). Thus the exact saddle remains interior,
its tilted variance is comparable to d, and
\[
-\min_s K_i(s)=J_i+O(np^2+p)=J_i+O(1).
\tag{B3}
\]
The Bernoulli coordinates stay independent under this exact tilt.
Their weighted centered third moments sum to O(d).
The sharp fixed-threshold tail lemma in the current manuscript gives,
for every fixed bounded x,
\[
\Pr(T<x)\asymp\Pr(T\le x)
\asymp d^{-1/2}e^{-J_i}.
\tag{B4}
\]

**Known-parameter contrast.** Write
\[
\mathcal T_c=\sum_{j\ne i}\theta_jP_{c,z_j}.
\]
Here
\[
\mathcal T_a=m[p+(K-1)q]-\theta_ip,\qquad
\mathcal T_b=m[p+(K-1)q]-\theta_iq.
\]
For each fixed L_0, on the single-target event D_i <= L_0 d,
direct expansion of the exact Bernoulli nonedge terms gives
\[
\ell_{\rm known}(a)-\ell_{\rm known}(b)
=T-\theta_i(\mathcal T_a-\mathcal T_b)
+O(D_ip+np^2)
=T+O(np^2+p).
\tag{B5}
\]
There is no change in sampling model.

**Profile contrast.** For a candidate c, the raw Bernoulli likelihood as
a function of the target parameter t has its maximizer in a fixed
bounded interval on D_i <= L_0 d. Indeed its derivative is
\[
\frac{D_i}{t}
-\sum_{j\ne i}(1-A_{ij})
\frac{\theta_jP_{c,z_j}}{1-t\theta_jP_{c,z_j}},
\]
and the sum of the nonedge intensities is at least c d for all large n.
The displayed derivative is negative for t above a fixed C(L_0).
The cap at candidate edge probability 1/2 is then inactive.

Uniformly on this bounded t interval, expansion followed by maximization
gives the exact profile contrast
\[
\ell_{\rm prof}(a)-\ell_{\rm prof}(b)
=T-D_i\log(\mathcal T_a/\mathcal T_b)+O(np^2).
\]
Since the exposure difference is O(p), the logarithm is O(1/n), and
\[
\ell_{\rm prof}(a)-\ell_{\rm prof}(b)
=T+O(np^2+p).
\tag{B6}
\]
The formula includes D_i=0 with the usual continuous interpretation
of D_i log D_i.

**Exceptional rows.** For any prescribed fixed H, choose L_0 so that
\[
\Pr(D_i>L_0d)\le e^{-Hd}.
\]
Every J_i <= C_J d uniformly. Hence choosing H>C_J+1 makes this
single-target exception negligible relative to the lower bound in (B4).
There is no union over n in this oracle comparison.

Equations (B4)–(B6) bound both the strict and nonstrict directional error
probabilities above and below by fixed constants times
\(d^{-1/2}e^{-J_i}\). A strict pairwise defeat forces a multiclass
error; every multiclass error loses weakly to a competitor. Fixed K
therefore gives the same per-target scale for each of the three
references. Averaging over targets proves (B1).

If the true-group vote is instead normalized by the self-excluded group
sizes m or m-1, multiply its contrast by m. The difference from
\(N_a-N_b\) is at most \(C D_i/n=O(p)\) on the same event, so the
same constant comparison also covers that normalized-voting reference.

### Consequence for the current algorithmic theorem

On the overlap with the current algorithmic theorem's parameter range,
the equal-exposure corollary yields
\[
\mathbb E\ell(\widehat z_D,z)
\le C R_{\rm known}.
\]
This is a constant comparison to the classical exact local DC oracle
in the balanced planted-partition class. It should be stated explicitly
as an alignment result, rather than suggesting that the profile reference
is universally identical to all DC-SBM oracle notions.

The leading minimax expression from the literature omits the sharp
prefactor. Equation (B1) sharpens the actual local reference on this
narrower fixed-separation, bounded-theta class. It does not establish a
constant-ratio minimax theorem over the entire larger parameter class
of the literature.

## 5. Unequal exposures and general block profiles

For arbitrary positive P with P entries comparable to p, define
\[
\mathcal T_c=\sum_{j\ne i}\theta_jP_{c,z_j}.
\]
On D_i <= L_0 np, the two exact contrasts have the expansions
\[
\ell_{\rm known}(a)-\ell_{\rm known}(b)
=\sum_cN_c\log(P_{ac}/P_{bc})
-\theta_i(\mathcal T_a-\mathcal T_b)+O(np^2),
\]
\[
\ell_{\rm prof}(a)-\ell_{\rm prof}(b)
=\sum_cN_c\log(P_{ac}/P_{bc})
-D_i\log(\mathcal T_a/\mathcal T_b)+O(np^2).
\]
If the two exposures differ at leading order, the difference between
these contrasts can be of order np. There is no general O(1) bridge.
This is why the existing general-B profile theorem does not
automatically establish the exact known-target oracle comparison.

Even in an unequal-size planted-partition model, the profile comparison
is not generally the same rule as the source's normalized neighbor vote.
For K=2, writing M_a,M_b for neighbor theta masses and
\(\mathcal T_a=M_ap+M_bq,\ \mathcal T_b=M_aq+M_bp\), the profile score is
\[
(N_a-N_b)\log(p/q)
-(N_a+N_b)\log(\mathcal T_a/\mathcal T_b),
\]
to bounded Bernoulli error in the stated sparse regime.
Normalized neighbor voting instead compares N_a/M_a with N_b/M_b
(and the paper estimates masses by group sizes under its normalization).
Those thresholds coincide for equal masses and need not coincide
otherwise.

## 6. Recommendation for the parent's answer

It is accurate to say that the DC-specific likelihood and nuisance
parameter issue were considered. It is not accurate to say the general
profile theorem already covers the classical DC minimax benchmark in all
its parameter ranges.

The helpful correction is to make the standard balanced DC reference
primary, add the bridge above, and explain that the previously emphasized
general-B known-target counterexample concerns a broader and different
comparison. The bridge itself already extends below log n expected
degree; the separate sparse algorithmic work must then preserve the
sharp risk-transfer conditions.

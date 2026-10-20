# Regularized encoder: exact identities, a naive-regularization obstruction, and a degree-floor route

Independent mathematical audit, 20 October 2026.

This note addresses the encoder/decoder interface only. It does not claim
that the full sparse constant-oracle-risk theorem has been completed.
The existing sparse section in **previous_main.tex**, especially the
degree-preserving reconstruction and the conditional sharp-transfer
proposition, has been read and is the starting point.

## 1. The existing positive-diagonal scheme already has exact raw-edge action

For any strictly positive diagonal matrix \(W\), suppose
\[
WAWU=U\Lambda,\qquad U^\top U=I.
\]
Set
\[
V=[\mathbf1,WU],\qquad
Y=[D,W^{-1}U\Lambda],\qquad D=A\mathbf1.
\]
Then exactly, for every graph,
\[
AV=Y,\qquad
R_E=YV^\dagger=A\Pi_V,\qquad
R_E\mathbf1=D.
\]
These identities require no \(np\gtrsim\log n\) assumption. The old
soft-degree-capping choice of \(W\) is therefore a valid computational
route for retaining degrees and spectral information without later
access to individual edges.

More generally, with \(G=W^{-2}\), the \(G\)-orthogonal projector onto
\(\operatorname{col}(V)\) is
\[
H_V=V(V^\top GV)^\dagger V^\top G.
\]
The retained data determine
\[
R_G=Y(V^\top GV)^\dagger V^\top G=AH_V,\qquad R_G\mathbf1=D.
\]
Here \(H_V^2=H_V\) and \(GH_V=H_V^\top G\). If every leverage
\((H_V)_{ii}<1\), then
\[
R_G^\circ
=R_G+\operatorname{diag}\!\left(
  \frac{(R_G)_{ii}}{1-(H_V)_{ii}}\right)(H_V-I)
\]
has zero diagonal, retains its degree vector, and retains its action
on \(\operatorname{col}(V)\).

These are algebraic facts, not a spectral accuracy or risk guarantee.
In particular, changing \(W\) can change the generalized population
space against which a block-likelihood weight is projected.

## 2. Exact removal of a uniform rank-one regularizer

Consider
\[
A_\tau=A+\alpha\mathbf1\mathbf1^\top,\quad
\alpha=\tau/n,\quad G=D_{\rm diag}+\tau I,
\quad S_\tau=G^{-1/2}A_\tau G^{-1/2}.
\]
Let \(S_\tau U=U\Lambda\), with the degree eigenvector
\(G^{1/2}\mathbf1/\|G^{1/2}\mathbf1\|\) retained, and put
\(X=G^{-1/2}U\). Then \(\mathbf1\in\operatorname{col}(X)\), and
\[
AX=G^{1/2}U\Lambda
-\alpha\mathbf1(\mathbf1^\top G^{-1/2}U).
\]
Thus the original adjacency action is recoverable exactly.

For \(H=G^{-1/2}UU^\top G^{1/2}\), the correct raw-edge reconstruction is
\[
AH
=G^{1/2}U\Lambda U^\top G^{1/2}
-\alpha\mathbf1\mathbf1^\top H.
\]
It obeys \(AH\mathbf1=D\). The second term is generally not
\(\alpha\mathbf1\mathbf1^\top\): \(H\mathbf1=\mathbf1\) does not imply
\(\mathbf1^\top H=\mathbf1^\top\). The distinction disappears for the
Euclidean orthogonal projector onto \(X\), because that projector fixes
the constant vector on both sides.

The deterministic population rank also matters. In ordinary SBM,
\(\mathbf1\in\operatorname{col}(Z)\), so adding this rank-one term does
not increase the population rank beyond \(K\). In general DC-SBM,
\(\mathbf1\notin\operatorname{col}(\Theta Z)\), and the rank typically
increases to \(K+1\). Retaining only \(K\) directions can lose a
nonvanishing signal direction. Retaining \(K+1\) alone does not solve
the likelihood-efficiency issue below.

## 3. Why adding \(\tau\asymp np\) and reusing the old decoder is unsafe

Write \(d=np\), \(D_0=\operatorname{diag}(\Omega\mathbf1)\), and
\[
\Omega=\Theta Z(pB)Z^\top\Theta,\qquad
(D_0)_{ii}=dS_{z_i}\theta_i,\qquad
S_a=\sum_b(n_b/n)B_{ab}.
\]
For the degree-only regularized matrix
\(G^{-1/2}AG^{-1/2}\), \(G=D_0+\tau I\), its generalized population
feature space is
\[
\operatorname{col}(G^{-1}\Theta Z).
\]
Within group \(a\), its nonzero feature is proportional to
\[
f_a(\theta_i)=\frac{\theta_i}{dS_a\theta_i+\tau}.
\]
Unlike the unregularized case, this is not constant inside a group
when \(\tau/d\) stays positive and degree parameters vary.

Adding a constant direction does not normally recover all block-constant
vectors. Every vector in the augmented space has, within group \(a\),
the form \(c+\beta_a f_a(\theta_i)\). If each group has two degree
parameter values occurring with positive proportions, a block-constant
contrast with distinct group values is outside that space. Its weighted
projection residual has squared norm of order \(n\), and hence
\[
\sum_j r_j\{(H-I)w\}_j^2=\Theta(d)
\]
for tilted probabilities \(r_j\asymp d/n\). The bounded-quadratic-error
condition needed for literal constant oracle risk is lost.

This does not mean that the population denoised mean is wrong:
\(\Omega H=\Omega\) can still hold. Correct population mean reconstruction
and correct large-deviation likelihood coefficients are different
requirements.

### A concrete population projected-score calculation

Take balanced two groups,
\[
B=\begin{pmatrix}4&1\\1&4\end{pmatrix},
\]
and, within each group, let half of the vertices have \(\theta=1/2\)
and half have \(\theta=3/2\). Thus each group has mean degree parameter
one, and \(S_1=S_2=5/2\). Put \(\tau=(5/2)d\).

For the anti-symmetric block contrast, either Euclidean or degree-metric
population projection onto the regularized generalized signal space,
with a constant direction added, produces a score with weights
\[
+f(\theta_j)\quad\text{on group 1},\qquad
-f(\theta_j)\quad\text{on group 2},
\quad f(\theta)=\frac{\theta}{1+\theta}.
\]
A common positive multiplier does not change its decision.
Thus the two within-group weights are \(1/3\) and \(3/5\).
The same anti-symmetric direction arises if uniform rank-one
regularization is used and the full \(K+1\) population range is retained.

For a true group-1 target with fixed degree parameter \(\theta_i\), the
exact Bernoulli log transform of this weighted-count score is
\[
K_{\rm reg}(s)
=\frac{d\theta_i}{2}
 \left\{4m(-s)+m(s)-5\right\}+O(np^2+p),
\quad
m(s)=\frac14e^{s/3}+\frac34e^{3s/5}.
\]
This follows directly by expanding the Bernoulli log transform; it is
not a replacement of the graph model by a Poisson model. In the
sublogarithmic range \(d\to\infty\), \(d=o(\log n)\), the remainder is
\(o(1)\).

The unweighted block-count oracle has information
\[
I_{\rm oracle}=\frac12d\theta_i+o(1).
\]
The weighted projected score instead has information
\[
J_{\rm reg}
=\frac{d\theta_i}{2}
\left[5-\min_s\{4m(-s)+m(s)\}\right]+o(1).
\]
Pointwise arithmetic-geometric mean gives
\[
4e^{-sf}+e^{sf}\ge4.
\]
Equality requires \(sf=\log 2\). Since the two positive values of \(f\)
differ, no common \(s\) attains equality for both. Therefore the minimized
mixture is strictly greater than \(4\), establishing a strictly positive
loss proportional to \(d\).

For numerical orientation only,
\[
s_{\rm opt}=1.2424384787291793,\quad
\min_s\{4m(-s)+m(s)\}=4.0432802796946,
\]
so
\[
J_{\rm reg}=0.47835986015269993\,d\theta_i+o(1),
\]
and the information gap is
\[
0.021640139847300066\,d\theta_i+o(1).
\]
These numbers are obtained by bisection on the derivative of the displayed
one-dimensional transform; the strict gap was proved analytically above.

This example concerns the naive projected-score decoder. It is not an
impossibility theorem for all regularized encoders or all decoders of
their retained real-valued coordinates.

## 4. A population-preserving degree floor is more promising

Use
\[
G_i=\max(D_i,ad),\qquad
S_G=G^{-1/2}AG^{-1/2},
\]
where \(a>0\) is below the fixed minimum expected-degree multiplier.
An implementable version replaces \(ad\) by a sufficiently small fixed
multiple of the observed mean degree; the latter is comparable to \(d\)
on a much stronger global edge-count event.

At the deterministic mean, \(G_i=(D_0)_{ii}\), so this floor does not
create the systematic within-group transformation of Section 3. It
changes only unusually small observed degrees.

Here is a useful deterministic consequence of the root's proposed
normalized-noise argument. Suppose
\[
\|G^{-1/2}(A-\Omega)G^{-1/2}\|\le C/\sqrt d,\qquad
\sum_iG_i\le Cnd,\qquad ad<(D_0)_{ii}\quad\forall i.
\]
Applying the normalized noise to \(G^{1/2}\mathbf1\) gives
\[
\sum_i\frac{(D_i-(D_0)_{ii})^2}{G_i}\le Cn.
\]
Because flooring toward a threshold below the mean cannot increase
the distance from that mean,
\[
|G_i-(D_0)_{ii}|\le |D_i-(D_0)_{ii}|,
\]
and hence
\[
\sum_i\frac{(G_i-(D_0)_{ii})^2}{G_i}\le Cn.
\]

The population nonzero spectral gap remains fixed even without a
uniform degree upper bound. Indeed,
\[
\sum_{i:z_i=a}\frac{\theta_i^2}{G_i}
\le Cn/d,
\qquad
\sum_{i:z_i=a}\frac{\theta_i^2}{G_i}
\ge
\frac{(\sum_{i:z_i=a}\theta_i)^2}{\sum_{i:z_i=a}G_i}
\ge cn/d.
\]
The last inequality uses comparable group sizes and the total-degree
bound. The fixed full-rank condition on \(B\) then supplies the gap.

Let \(\Pi=UU^\top\) be the informative sample projector of \(S_G\).
The normalized noise and the gap give
\[
\|(I-\Pi)G^{1/2}Z\|_F\le C\sqrt n.
\]
To see this, compare \(G^{1/2}Z\) to
\(G^{-1/2}D_0Z\), which belongs to the population signal space.
Their difference has norm at most \(C\sqrt n\) by the preceding
aggregate degree estimate. The sample-versus-population projector
error contributes another
\(O(d^{-1/2})\|G^{1/2}Z\|_F=O(\sqrt n)\).

Consequently, for \(H=G^{-1/2}\Pi G^{1/2}\) and a fixed bounded
block-constant vector \(w\),
\[
\|(I-H)w\|_G\le C\sqrt n,\qquad
\|(I-H)w\|_2^2\le Cn/d.
\]
For the exact sparse Bernoulli tilt,
\(r=D_0h+\varepsilon\), where \(h\) is block constant with entries
\(O(1/n)\) and \(\|\varepsilon\|_\infty=O(p^2)\), one has
\[
\begin{split}
r^\top(H-I)w
={}&-\langle(I-H)h,(I-H)w\rangle_G\\
&+((D_0-G)h)^\top(H-I)w
+\varepsilon^\top(H-I)w.
\end{split}
\]
The first two terms are \(O(1)\) from the residual and aggregate degree
bounds; the last is \(O(np^2)\). Also
\[
\sum_jr_j\{(H-I)w\}_j^2\le Cp\|(H-I)w\|_2^2=O(1).
\]
Thus the crucial DC tilted mean and variance calculations survive this
floor construction once the stated normalized-noise event is established.

### Connection to the LLV adjacency regularization theorem

Let \(d_*=\max_{ij}n\,\mathbb E A_{ij}\asymp d\) and
\[
(W_h)_{ii}=\sqrt{\min(1,2d_*/D_i)}.
\]
The row squared Euclidean norms of \(W_hAW_h\) are at most \(2d_*\).
LLV Theorem 2.1, with its row-norm extension, controls
\(W_hAW_h-\mathbb EA\) by \(O(\sqrt d)\), with probability
\(1-n^{-r}\) for fixed \(r\), on the event that the changed vertex set
has size \(O(n/d)\). Their Remark 2.2 supplies the usual high-degree
set choice. The constants may depend on fixed \(r\).

There is a bounded diagonal \(F\) satisfying
\[
G^{-1/2}=(2d_*)^{-1/2}FW_h.
\]
For \(D_i>2d_*\), \(F_i=1\); otherwise
\(F_i=\sqrt{2d_*/G_i}\le C_a\).
The population change
\(\|W_h\Omega W_h-\Omega\|\le C\sqrt d\) follows from
the changed-set size and \(\max_{ij}|\Omega_{ij}|\le Cd/n\).
These identities yield
\[
\|G^{-1/2}(A-\Omega)G^{-1/2}\|\le C/\sqrt d.
\]
If the model's scale \(d=np\) is not itself the maximum probability
scale, the high-degree threshold must be \(2d_*\), or a sufficiently
large structural multiple of \(d\), rather than automatically \(2d\).

Primary source: Le, Levina and Vershynin,
*Concentration and regularization of random graphs*, Theorem 2.1,
Remark 2.2, and the row-norm extension; arXiv:1506.00669v2,
https://arxiv.org/pdf/1506.00669 .
The theorem is a regularized spectral norm result. It is not by itself
a sharp oracle-risk theorem.

## 5. Preserve degrees without an unstable extra orthonormal direction

One option is to augment \(\operatorname{col}(G^{-1/2}U)\) by the
constant vector and use its \(G\)-orthogonal projector. This does
preserve degrees and does not increase the block residual norm.
However, the added direction may concentrate on floored or isolated
vertices. A row bound on \(U\) does not automatically give a row bound
on that augmented orthonormal basis.

A useful alternative is the degree correction already present in the
older manuscript, now applied without an estimated coordinate inverse.
Take a fixed probability vector \(\pi=\mathbf1/n\), and define
\[
R_0=AH=G^{1/2}U\Lambda U^\top G^{1/2},\qquad
R=R_0+(D-R_0\mathbf1)\pi^\top.
\]
Every quantity is retained-data computable, and \(R\mathbf1=D\)
exactly. For a block-constant likelihood weight \(w\), put
\(\bar w=\pi^\top w\). Then
\[
Rw
=A\left[H(w-\bar w\mathbf1)+\bar w\mathbf1\right],
\]
so the effective coefficient error against raw evidence is exactly
\[
e=(H-I)(w-\bar w\mathbf1).
\]
The centered weight is still bounded and block constant. Therefore all
tilted mean and quadratic-error estimates in Section 4 apply directly.
There is no need to prove incoherence of an added basis vector.
This degree-corrected reconstruction is not claimed to be an orthogonal
projection reconstruction.

For a hollow version, write \(h_i=R_{ii}\) and subtract \(h_i\) from
the diagonal while adding \(h_i/(n-1)\) to every other entry in row \(i\).
This preserves row sums exactly. Its action on a bounded contrast
changes by at most \(2|h_i|\). Thus a local \(h_i=O(p)\) bound suffices.
It does not preserve every original retained-space action, and that
additional identity is not needed for this proposed correction.

## 6. What remains unproved

The arguments above complete useful algebraic and deterministic
ingredients, not the sparse statistical theorem. In particular:

- A sufficient row condition for bounded coefficients is the weighted
  bound \(\|U_{i\cdot}\|/\sqrt{G_i}\le C/\sqrt{nd}\), not necessarily
  \(\|U_{i\cdot}\|\le C/\sqrt n\). Uniformly obtaining such a bound
  below logarithmic degree remains nontrivial; an appropriate
  integrated exponential-tail bound may suffice instead.
- Recomputed-degree LOO must be redone for the floor. A deterministic
  floor is Lipschitz under deletion. An observed-mean-degree floor
  additionally changes its threshold globally, and that change must
  enter the score remainder.
- The full finite fitting certificate must be controlled without a
  uniform \(D_i=O(d)\) assumption. Large degrees affect entrywise
  reconstruction bounds and the contribution of mislabeled vertices.
- The sharp transfer requires the full exponential perturbation cost,
  not only the first two moments. A generic spectral consistency bound
  is insufficient.
- A global \(n^{-r}\) spectral failure probability is small enough
  compared with \(d^{-1/2}e^{-Cd}\) when \(d=o(\log n)\), and can also
  be paid on a fixed-width logarithmic window by choosing fixed \(r\)
  sufficiently large. This favorable comparison does not pay the
  still-unproved coefficient/LOO/fitting events.

Accordingly, the factor \(20\) is not a model threshold and the old sparse
framework remains relevant. But deleting the logarithmic assumption
from the new unconditional constant-risk theorem is a further proof
task. The degree-floor route preserves the desired population geometry
and supplies a concrete route for that task; uniform additive
regularization with an unchanged decoder need not.

# Sparse constant-oracle-risk audit

20 October 2026. Independent audit of the existing normalized-spectral proof and of the proposed regularized encoder. This note does not modify the manuscript.

## 1. Conclusions that can be stated now

1. For the ordinary SBM theorem on its stated ratio-two parameter class, the same decoder and the same minimum-degree cutoff \(a_0=1/16\) work under **\(d=np\ge 5\log n\)**. This follows from the current good-event theorem and a sharper, fully uniform payment of the exceptional event. A complete proof and an insertable TeX paragraph appear below.
2. The number 20 is conservative. Removing the entire logarithmic scale is a separate change. The current proof pays a failure of any vertex by a whole-graph loss bound; that is the source of the unwanted factor \(n\). The explicit implementation that returns a fixed whole-graph partition when a zero degree makes the encoder undefined cannot simply be extended below logarithmic degree.
3. The previous sparse degree-plus-spectrum algorithm already exists. Its algebra is valid without a logarithmic lower bound. Its old sharp-transfer proposition explicitly leaves the necessary random coefficient, tilted moment, anti-concentration, and fitted-score remainder bounds conditional. The new theorem does not automatically close those conditions.
4. The floor \(G_i=\max(D_i,ad)\) has a real advantage over adding \(\tau\asymp d\) to every degree in DC-SBM: it preserves the relevant population block geometry. The proposed LLV-adjacency route does establish normalized noise, aggregate degree error, the weighted block residual, and bounded tilted mean and quadratic error. This chain is not circular.
5. Those ingredients still do not establish a literal constant-oracle-risk theorem below logarithmic degree. Effective coefficient tails, recomputed-degree LOO, diagonal corrections, and the actual finite fitting procedure remain to be controlled at the sharp testing scale.

Here and below all constants may depend on the fixed structural parameter bounds, but not on \(n\). All logarithms are natural.

## 2. A complete improvement from \(20\log n\) to \(5\log n\)

The current ordinary theorem assumes
\[
P_{ab}\in[p/2,p],\qquad p\le1/4,\qquad
\sigma_{\min}(P)\ge\kappa p,\qquad n_a\ge\pi_-n.
\]
Write \(q_{i,*}\) for the directional error probability of the raw-edge oracle at the fixed true vertex \(i\). Its average is \(R_{\mathrm{SBM},*}\).

### 2.1 The oracle lower bound is available at every vertex

The manuscript's exact Bernoulli contrast bound gives
\[
|w_j^{ab}|\le W=\log(7/3).
\]
The tilted Bernoulli probabilities are at most \(p\). Hence the second derivative of the log affinity is at most \(dW^2\). Since that convex log affinity vanishes at both endpoints, its minimum is at least \(-dW^2/8\). Thus
\[
I_{i,ab}\le \beta d,\qquad
\beta:=\frac{\log^2(7/3)}8
=0.08973920802709166.
\]
The sharp pairwise oracle lower bound already proved in the manuscript is uniform over every vertex and competitor. For any one competitor \(b\ne z_i\),
\[
q_{i,*}\ge c\,d^{-1/2}e^{-I_{i,z_i b}}
\ge c\,d^{-1/2}e^{-\beta d}.
\]
The strict pairwise error event is contained in the full oracle error event, so the tie rule causes no difficulty. Averaging over all vertices yields
\[
\boxed{
R_{\mathrm{SBM},*}\ge c\,d^{-1/2}e^{-\beta d}.
}
\tag{2.1}
\]
There is no factor \(1/n\). No further balancing assumption is being added: the same upper bound \(\beta d\) holds at every vertex on the existing parameter class.

### 2.2 Keep the exact lower-tail constant

For all sufficiently large \(n\), every raw expected degree is at least \(0.49d\), since it is at least \((n-1)p/2\). Keep the old proof's cutoff \(a_0=1/16\). The Bernoulli lower-tail transform gives
\[
\Pr(D_i<d/16)\le e^{-\alpha d},
\]
where
\[
\begin{aligned}
\alpha
&:=0.49-\frac1{16}
       -\frac1{16}\log(16\cdot0.49)\\
&=0.2987975728523552.
\end{aligned}
\]
The other good-event failures have probability at most \(C_Kn e^{-Hd}\), for any prescribed fixed \(H\), after increasing the fixed thresholds in the existing proof. Choose \(H>\alpha\). Consequently the same common good event satisfies
\[
\Pr(\mathcal G^c)\le Cn e^{-\alpha d}.
\tag{2.2}
\]
The raw norm estimate used to construct this event is available for \(d\ge c\log n\) with any fixed \(c>0\); changing 20 to 5 only changes the fixed norm threshold. The row bootstrap and deterministic LOO steps need \(d\to\infty\), not the specific coefficient 20.

### 2.3 Pay the event without wasting the square-root factor

Let
\[
\gamma:=\alpha-\beta
=0.20905836482526352.
\]
Combining (2.1) and (2.2),
\[
\frac{\Pr(\mathcal G^c)}{R_{\mathrm{SBM},*}}
\le Cn\sqrt d\,e^{-\gamma d}.
\]
The function \(f(t)=\sqrt t\,e^{-\gamma t}\) has derivative
\[
f'(t)=f(t)\left(\frac1{2t}-\gamma\right),
\]
and is decreasing once \(t>1/(2\gamma)\). Therefore, uniformly for \(d\ge5\log n\) and all sufficiently large \(n\),
\[
\begin{aligned}
\frac{\Pr(\mathcal G^c)}{R_{\mathrm{SBM},*}}
&\le Cn\sqrt{5\log n}\,e^{-5\gamma\log n}\\
&=C\sqrt{5\log n}\,
  n^{-0.0452918241263176}
\longrightarrow0.
\end{aligned}
\tag{2.3}
\]
Here \(5\gamma=1.0452918241263176>1\). It would be incorrect for this improvement to first replace \(\sqrt d\) by \(\sqrt n\): that coarser step discards the monotonicity which makes the coefficient 5 sufficient.

The existing on-good-event theorem gives
\[
\mathbb E[\ell(\widehat z,z)\mathbf1_{\mathcal G}]
\le C_{\rm good}R_{\mathrm{SBM},*}.
\]
Since the loss is at most one, (2.3) completes
\[
\boxed{
\mathbb E\ell(\widehat z,z)
\le (C_{\rm good}+o(1))R_{\mathrm{SBM},*},
\qquad d\ge5\log n.
}
\]
This is an asymptotic, \(n\)-independent constant-factor statement. It is not a claim that the ratio tends to one, nor a numerical finite-sample risk bound. The coefficient 5 is itself not a proposed intrinsic threshold.

### 2.4 Insertable TeX for the ordinary exceptional-event paragraph

The following replaces the paragraph beginning with the minimum-degree calculation and ending with the ordinary good-event payment. The theorem statement's 20 must also be changed to 5. No decoder change is needed.

```tex
For all sufficiently large $n$, every mean degree is at least $0.49d$.
Keep $a_0=1/16$ and put
\[
 \alpha=0.49-\frac1{16}
              -\frac1{16}\log(16\cdot0.49),\qquad
 \beta=\frac{\log^2(7/3)}8,\qquad \gamma=\alpha-\beta.
\]
The exact Bernoulli lower-tail transform gives
\[
 \PP(D_i<d/16)\le e^{-\alpha d}.
\]
Choosing $H>\alpha$ in \eqref{eq:other-bad-events}, the same common
good event therefore satisfies
\begin{equation}\label{eq:sbm-low-degree}
 \PP(\mathcal G^c)\le Cn e^{-\alpha d}.
\end{equation}
The sharp oracle lower bound in \eqref{eq:oracle-pairs} applies at
every vertex. Since \eqref{eq:sbm-information-upper} holds uniformly
in the vertex and its competitor, averaging these lower bounds gives
\[
 R_{\mathrm{SBM},*}\ge c d^{-1/2}e^{-\beta d}.
\]
Thus
\[
 \frac{\PP(\mathcal G^c)}{R_{\mathrm{SBM},*}}
 \le Cn\sqrt d\,e^{-\gamma d}.
\]
Here $\gamma=0.2090583648\ldots$, so $5\gamma>1$.
The function $t\mapsto\sqrt t\,e^{-\gamma t}$ is decreasing for
$t>1/(2\gamma)$. Consequently, for $d\ge5\log n$ and all sufficiently
large $n$,
\[
 \frac{\PP(\mathcal G^c)}{R_{\mathrm{SBM},*}}
 \le C\sqrt{5\log n}\,n^{\,1-5\gamma}=o(1).
\]
The good-event comparison in \cref{sec:sbm-proof}, together with the
loss bound by one on the complement, completes \cref{thm:sbm}.
```

## 3. Where the logarithmic scale really enters the current proof

| Current step | What is actually used | Consequence for a sparse extension |
|---|---|---|
| Raw adjacency norm | A tail with an \(n\) prefactor is evaluated at \(L\sqrt d\); \(d\gtrsim\log n\) absorbs it. | Replace raw adjacency concentration by a regularized norm bound. |
| Every degree lies in \([ad,bd]\) | A union of \(n\) one-vertex exponential tails. | Avoid requiring every low-degree vertex to be good; control local effects and training geometry separately. |
| Uniform row bootstrap | Conditional row events cost \(2Ke^{-Hd}\) each and are unioned over all vertices. | A weighted or integrated row bound is needed when \(n e^{-Hd}\) is large. |
| Recomputed-degree LOO | All training degrees are bounded below, and full/minor eigenvector rows are uniformly controlled. | Re-prove the actual deletion comparison for the regularized encoder and its data-dependent floor. |
| Entrywise reconstruction and fitting | Uniform degree/row bounds control entries, diagonal correction, and effects of mislabeled vertices. | A spectral angle alone does not replace this fitting certificate. |
| Final risk payment | Any failure of the common event is paid by whole-graph loss at most one. | This multiplies a local tail by \(n\). Average-risk proof should retain the local indexing. |

The number 20 is not present as a structural spectral threshold. The logarithmic scale appears because the current implementation and proof demand globally uniform regularity and then charge a failure globally.

### An explicit obstruction to retaining the zero-degree fallback

Take a balanced two-block SBM with within-block probability \(p\), between-block probability \(p/2\), and \(d=np\to\infty\), \(d=o(\log n)\). The expected degree is
\[
\lambda_i=(3/4+o(1))d.
\]
Let \(N_0=\sum_i\mathbf1_{\{D_i=0\}}\). Exact Bernoulli products give
\[
\mathbb E N_0=n\exp\{-(3/4+o(1))d\}\to\infty.
\]
For distinct \(i,j\),
\[
\frac{\Pr(D_i=D_j=0)}
     {\Pr(D_i=0)\Pr(D_j=0)}
=\frac1{1-P_{z_i z_j}}=1+O(p).
\]
It follows that
\[
\frac{\operatorname{Var}N_0}{(\mathbb E N_0)^2}
\le\frac1{\mathbb E N_0}+O(p)\to0,
\]
so a zero degree exists with probability tending to one.

Therefore a version that responds to this undefined normalization by returning a fixed whole-graph partition has nonvanishing error for admissible true partitions separated from that fixed fallback. The raw-edge oracle error tends to zero. Removing the logarithmic assumption while retaining that fallback would be false. Replacing the fallback by an independently successful sparse initializer is a change of algorithm whose guarantee must be established.

This counterexample is about the stated whole-graph fallback. It is not an information-theoretic impossibility for sparse spectral likelihood procedures. A vanishing fraction of low-degree nodes can coexist with a zero-degree node somewhere in the graph with overwhelming probability.

## 4. How local payment removes the unnecessary union factor

The desired assertion is an average:
\[
\frac1n\sum_i\Pr(\widehat z_i\ne z_i)
\le C\frac1n\sum_i q_{i,*}.
\]
The label matching must be the same common permutation throughout; nodewise alignment cannot be used to hide errors.

A sufficient architecture is a global training event \(\mathcal T\), with a valid common alignment, and vertex-indexed events \(E_i\), such that
\[
\Pr(\widehat z_i\ne z_i,\mathcal T\cap E_i)\le Cq_{i,*},
\]
\[
\Pr(\mathcal T^c)\le C R_*,
\qquad
\frac1n\sum_i\Pr(E_i^c)\le C R_*.
\]
No union over all \(E_i\) is required when paying the final loss. A global polynomial spectral failure probability \(n^{-r}\) is negligible compared with
\[
R_*\ge c d^{-1/2}e^{-C_I d}
\]
when \(d=o(\log n)\). On a fixed window \(d\le C_{\log}\log n\), the same payment holds by choosing a sufficiently large fixed \(r\).

### Ordinary SBM: the local degree tail already has enough margin

The constants in Section 2 imply
\[
\frac{n^{-1}\sum_i\Pr(D_i<d/16)}
     {R_{\mathrm{SBM},*}}
\le C\sqrt d\,e^{-(\alpha-\beta)d}\to0
\]
for every \(d\to\infty\). It is the global union, not this local statistical cost, which required \(d\gtrsim\log n\).

### DC-SBM: weight the local cost by its own profile information

For the profile reference on the fixed bounded structural class, the exact sparse Bernoulli calculation gives, uniformly in each true vertex \(i\) and competitor \(b\),
\[
J_{i,a\to b}
=\lambda_i(1-h^*_{ab})+O(np^2),
\qquad h^*_{ab}\ge h_0>0.
\]
Let \(J_i=\min_{b\ne z_i}J_{i,z_i\to b}\). In the sublogarithmic regime \(np^2=d^2/n=o(1)\); more generally the manuscript's \(np^2=O(1)\) suffices for a fixed margin. There is an \(\eta>0\) such that
\[
\lambda_i-J_i\ge\eta d.
\]
Choose \(a>0\) so small that
\[
a\{1+\log(\lambda_+/a)\}\le\eta/2,
\qquad a<\lambda_-.
\]
The Bernoulli degree lower-tail bound yields
\[
\Pr(D_i<ad)
\le \exp\{-J_i-\eta d/2\}.
\]
The sharp directional profile-reference lower bound is
\[
q_{i,\mathrm{prof},*}\ge c d^{-1/2}e^{-J_i}.
\]
Therefore, vertex by vertex,
\[
\Pr(D_i<ad)
\le C\sqrt d\,e^{-\eta d/2}q_{i,\mathrm{prof},*},
\]
and the averaged low-degree contribution is \(o(R_{\mathrm{prof},*})\).

This does not prove that bad training vertices have no effect on other targets. What matters for their influence is the incident-row tilted mass and coefficient size, for example
\[
\sum_j r_{ij}e_j^2,\qquad \sum_jr_{ij}e_j,
\]
not just the unweighted number of exceptional rows. With bounded degree parameters and bounded coefficients, an exceptional training set of size \(O(n/d)\) carries only \(O(1)\) expected tilted incident-edge mass. But declaring all \(O(n/d)\) exceptional *targets* misclassified costs \(O(1/d)\), which is much larger than an exponentially small oracle risk. They must still be decoded. Positive soft weights and retention of exact raw-degree action are relevant for this reason.

## 5. The earlier sparse algorithm is already part of the project

The recovered `previous_main.tex`, in its section “The general degree-plus-spectrum algorithm,” defines
\[
\tau=2\bar D,\qquad
W_{ii}=\min(1,\tau/D_i)
\]
with the stated positive-weight zero-degree conventions, retains \(WAWU=U\Lambda\), and constructs
\[
X=[\mathbf1,WU],\qquad
Y=[D,W^{-1}U\Lambda],\qquad R=YX^\dagger.
\]
Its exact identities are
\[
AX=Y,\qquad R=A\Pi_X,\qquad R\mathbf1=D.
\]
The complete Bernoulli score, nonedge contribution, and E/M algebra remain meaningful for this representation. They do not require a logarithmic expected degree.

That section explicitly distinguishes this algorithm from its proved logarithmic-density theorem. Its “sufficient sharp-transfer comparison” requires an actual deleted-graph score representation, a conditional cumulant bound at the oracle tilt, unit-interval tilted anti-concentration, and a bounded integrated exponential cost for the remainder. Those conditions were not proved for the actual sparse encoder. Thus the appropriate next step is to close that existing sharp-transfer task, adapting the encoder to preserve DC geometry. It would be inaccurate to say the project has no sparse construction, or to say its unconditional sparse constant-risk theorem has already been established.

## 6. Independent audit of `regularized_encoder.md`

### 6.1 The proposed LLV adjacency route is sound

Write \(d_*=\max_{ij} n\,\mathbb EA_{ij}\asymp d\), and define
\[
(W_h)_{ii}=\sqrt{\min(1,2d_*/D_i)}.
\]
Its modified matrix has squared row Euclidean norms at most \(2d_*\). The LLV adjacency regularization theorem applies uniformly to edge reductions incident to at most \(10n/d_*\) vertices, with an \(O(\sqrt d)\) bound and failure probability \(n^{-r}\), for each fixed \(r\). This is the adjacency theorem with its row-norm extension, not the full Laplacian theorem with an \(e^{-r}\) tail. Primary source: [Le, Levina, and Vershynin, Theorem 2.1 and Section 3.6](https://arxiv.org/pdf/1506.00669).

The required changed-set tail can be supplied directly, without reading an unspecified “with high probability” in Remark 2.2 as an arbitrary polynomial tail. Here is an elementary proof for \(d_*\to\infty\).

Let
\[
m=\lfloor10n/d_*\rfloor+1.
\]
If \(m>n\), the changed-set bound is automatic. Otherwise, if at least \(m\) vertices have degree greater than \(2d_*\), there is a set \(S\) of size \(m\) with
\[
T_S:=\sum_{i\in S}D_i>2m d_*.
\]
The variable \(T_S\) is a sum of independent upper-triangular Bernoulli edges with coefficients either one or two and mean at most \(m d_*\). For \(t\ge0\), convexity on the interval \([0,2]\) gives
\[
\log\mathbb E e^{tT_S}
\le \frac{m d_*}{2}(e^{2t}-1).
\]
At \(t=(\log2)/2\),
\[
\Pr(T_S>2m d_*)
\le \exp\{-(\log2-1/2)m d_*\}.
\]
Moreover
\[
\log\binom nm
\le m\log(en/m)=o(m d_*)
\]
as \(d_*\to\infty\). Since \(m d_*>10n\), the union over all such \(S\) has probability at most \(e^{-cn}\), for all sufficiently large \(n\). This is stronger than \(n^{-r}\) for every fixed \(r\).

The completed population diagonal does not cause a problem. If \(\Omega\) denotes the completed rank-\(K\) mean,
\[
\|\Omega-\mathbb EA\|=O(p).
\]
The changed mean \(\Omega-W_h\Omega W_h\) is supported on rows or columns in the changed vertex set, so its Frobenius norm is at most
\[
Cp\sqrt{n\,|S|}=O(\sqrt d).
\]
Consequently
\[
\|W_h(A-\Omega)W_h\|=O(\sqrt d).
\]

For \(G_i=D_i\vee ad\), where the fixed \(a\) is sufficiently small, there is a bounded diagonal \(F\) with
\[
G^{-1/2}=(2d_*)^{-1/2}F W_h.
\]
When \(D_i>2d_*\), one has \(G_i=D_i\) and \(F_i=1\); otherwise \(F_i=\sqrt{2d_*/G_i}\le C_a\). Thus
\[
\boxed{
\|G^{-1/2}(A-\Omega)G^{-1/2}\|
\le C/\sqrt d
}
\tag{6.1}
\]
with probability \(1-O(n^{-r})\). The diagonal completion alone contributes only \(O(p/d)=O(1/n)\) to this normalized bound.

### 6.2 The aggregate-degree step is not circular

Independently, a Chernoff bound for the total number of edges gives
\[
\sum_iD_i\le Cnd,\qquad
\sum_iG_i\le\sum_iD_i+nad\le Cnd
\tag{6.2}
\]
outside an event of probability \(e^{-cnd}\). Put \(D_0=\operatorname{diag}(\Omega\mathbf1)\). Applying the normalized noise matrix in (6.1) to \(G^{1/2}\mathbf1\) gives the exact identity and bound
\[
\begin{aligned}
\sum_i\frac{(D_i-(D_0)_{ii})^2}{G_i}
&=\|G^{-1/2}(A-\Omega)\mathbf1\|_2^2\\
&\le \frac{C}{d}\sum_iG_i
\le Cn.
\end{aligned}
\tag{6.3}
\]
This uses neither a uniform lower bound on the raw degrees nor an unproved raw adjacency norm.

Choose \(a<\lambda_-\) so that \(ad<(D_0)_{ii}\). Projection onto \([ad,\infty)\) cannot increase the distance to \((D_0)_{ii}\), and therefore
\[
\sum_i\frac{(G_i-(D_0)_{ii})^2}{G_i}\le Cn.
\tag{6.4}
\]
The order is: total-edge event and LLV event, then normalized noise, then (6.3), then (6.4). There is no dependence of the LLV event on a previously established version of (6.3).

### 6.3 The gap and \(G\)-weighted block residual follow

The observed-normalized completed mean is
\[
M_G=G^{-1/2}\Omega G^{-1/2}.
\]
For each block, the diagonal signal Gram matrix has entries
\[
\sum_{i:z_i=a}\frac{\theta_i^2}{G_i}.
\]
The floor gives an upper bound \(Cn/d\). Cauchy–Schwarz and (6.2) give
\[
\sum_{i:z_i=a}\frac{\theta_i^2}{G_i}
\ge
\frac{(\sum_{i:z_i=a}\theta_i)^2}
     {\sum_{i:z_i=a}G_i}
\ge cn/d.
\]
The fixed lower singular-value bound on the block matrix therefore yields a fixed nonzero spectral gap for \(M_G\), including when it has eigenvalues of both signs. By (6.1), the sample informative projector \(\Pi=UU^\top\) has angle \(O(d^{-1/2})\) to \(\operatorname{col}(G^{-1/2}\Theta Z)\).

Now \(G^{-1/2}D_0Z\) lies exactly in that signal space, and
\[
\|G^{1/2}Z-G^{-1/2}D_0Z\|_F^2
=\sum_i\frac{(G_i-(D_0)_{ii})^2}{G_i}\le Cn.
\]
Also \(\|G^{1/2}Z\|_F^2=\sum_iG_i\le Cnd\). The angle and the preceding identity imply
\[
\boxed{\|(I-\Pi)G^{1/2}Z\|_F\le C\sqrt n.}
\tag{6.5}
\]
This argument does not require every \(G_i=O(d)\).

### 6.4 Tilted mean and quadratic error are controlled

Let \(H=G^{-1/2}\Pi G^{1/2}\), let \(w\) be a bounded block-constant vector, and put \(e=(H-I)w\). The projector is self-adjoint in the \(G\) metric. Equation (6.5) gives
\[
\|e\|_G\le C\sqrt n,\qquad \|e\|_2\le C\sqrt{n/d}.
\tag{6.6}
\]
For the sparse exact Bernoulli tilt on the neighbor coordinates,
\[
r=D_0h+\varepsilon,\qquad
\|h\|_\infty\le C/n,\qquad h\text{ block constant},\qquad
\|\varepsilon\|_\infty\le Cp^2.
\]
Then
\[
\begin{aligned}
r^\top e
={}&-\langle(I-H)h,(I-H)w\rangle_G\\
&+((D_0-G)h)^\top e+\varepsilon^\top e.
\end{aligned}
\]
The first term is \(O(1)\), since the two residual norms are \(O(n^{-1/2})\) and \(O(n^{1/2})\). For the second, (6.4) implies
\[
\|(D_0-G)h\|_{G^{-1}}^2
\le \frac C{n^2}\sum_i\frac{((D_0)_{ii}-G_i)^2}{G_i}
\le C/n.
\]
It is therefore \(O(1)\) by weighted Cauchy–Schwarz and (6.6).

For the last term, **no unproved coefficient infinity bound is needed**:
\[
|\varepsilon^\top e|
\le \|\varepsilon\|_2\|e\|_2
\le Cnp^2/\sqrt d
\le Cnp^2.
\]
Thus \(r^\top e=O(1)\) when \(np^2=O(1)\), and in particular below logarithmic degree. Since \(r_j\le Cp\), (6.6) also gives
\[
\sum_jr_je_j^2\le Cp\|e\|_2^2=O(1).
\]
These are genuine sharp-scale mean and quadratic bounds. They do not by themselves bound the exponential perturbation cost.

### 6.5 Self-exclusion and reconstructed diagonal must remain distinct

There are three different diagonal issues.

1. **Completed population diagonal.** The norm difference from the actual mean is \(O(p)\), already accounted for in (6.1). Equations (6.3)–(6.5) use \(D_0=\Omega\mathbf1\), so their algebra is exact.
2. **The tested coordinate in the oracle tilt.** The clean relation \(\|\varepsilon\|_\infty=O(p^2)\) holds on the actual neighbor space. It is best to formulate the sharp comparison on the recomputed deleted graph. If instead one uses full \(n\)-dimensional \(G\) and pads \(r_i=0\), the artificial \(i\)th coordinate of \(D_0h\) is \(O(p)\), so \(\varepsilon_i\) is only \(O(p)\). This is still harmless for the present moment calculation: its separate contribution is at most
   \[
   Cp|e_i|\le Cp\|e\|_2\le C\sqrt p=O(1).
   \]
   It must be stated separately rather than claimed to be \(O(p^2)\).
3. **The sample reconstructed diagonal.** The new degree-preserving hollow correction changes the tested score by \(O(|R_{ii}|)\). The desired local \(R_{ii}=O(p)\) bound does not follow from (6.5) alone. It belongs to the remaining local row/LOO analysis. The fact that the population completed diagonal is small does not prove that the sample reconstructed diagonal is small.

### 6.6 The exact raw-degree correction is useful

The proposed formula
\[
R_0=AH=G^{1/2}U\Lambda U^\top G^{1/2},\qquad
R=R_0+(D-R_0\mathbf1)\mathbf1^\top/n
\]
is exactly computable from retained degrees and eigenpairs and obeys \(R\mathbf1=D\). For \(\bar w=n^{-1}\mathbf1^\top w\),
\[
Rw=A\{\bar w\mathbf1+H(w-\bar w\mathbf1)\}.
\]
The coefficient error is \((H-I)(w-\bar w\mathbf1)\), to which the preceding bounds apply because the centered contrast is still block constant.

This avoids normalizing a possibly tiny residual when adjoining \(\mathbf1\) to an almost \(K\)-dimensional signal space. Direct augmentation is algebraically valid, but the additional normalized direction can have high leverage on exceptional vertices. It does not inherit a row bound merely from the original \(K\) eigenvectors.

For an implementable random floor \(G_i=D_i\vee(a'\bar D)\), a separate total-edge event puts its threshold between fixed multiples of \(d\) and below every population degree, provided \(a'\) is chosen small enough. The norm argument is then obtained by bounded diagonal comparison with a fixed floor. In the LOO step its threshold changes globally by deleting a vertex, and that change is an additional term to control.

## 7. What still separates these ingredients from the requested sparse theorem

The missing step is not another ordinary consistency estimate.

- **Effective coefficient tails.** For an actual deleted-graph score with error coefficients \(e_j\), bounded first and second tilted moments do not control a rare large negative coefficient in the cumulant. A sufficient deterministic condition is \(\max_j|e_j|\le C\); an appropriate integrated exponential bound could replace it. For the floor, the relevant row scale is \(\|U_j\|/\sqrt{G_j}\), not necessarily \(\|U_j\|\).
- **Actual full-to-LOO score stability.** Recompute all retained normalizations on the minor, compare to the full encoder and degree correction, and retain an \(O(1)\) remainder or a bounded exponential cost. The floor is Lipschitz, but this fact alone does not supply the needed row bound or spectral coefficient stability.
- **Anti-concentration under the fitted tilt.** The score must retain variance of order \(d\) and a unit-interval tilted probability of order \(d^{-1/2}\). An operator-angle bound only describes geometry and does not automatically provide that local-limit control.
- **Actual fitting and exceptional vertices.** The finite growing/replacement or DC nuisance-fit procedure must reach and remain in its basin with the correct local score comparison. Removing all low-degree or high-degree targets and paying their count by \(O(1/d)\) would be insufficient. Raw degrees preserve one direction of evidence exactly; the other directions still require sharp control.

If these conditions are established, the old conditional sharp-transfer proof provides the correct route to a literal constant factor. A statement of the form
\[
\Pr(\text{error})\le e^{-(1-o(1))I}
\]
would still be too weak: when the oracle has size \(d^{-1/2}e^{-I}\), the ratio can diverge. The appropriate target is a bounded cumulant perturbation, a retained \(d^{-1/2}\) prefactor, and exceptions paid at the matching oracle scale.

For a fixed positive \(d\) and a fixed nondegenerate compact structural class, the oracle risk itself remains of positive order. A constant-ratio guarantee then becomes a weak criterion because the loss is bounded. The meaningful vanishing-risk extension in the present fixed-parameter setup is \(d\to\infty\), including \(d=o(\log n)\); it is not a removal of every lower signal requirement.

## 8. Source and scope record

Local sources audited:

- `spectral_proof_completion_20261020/manuscript/normalized_geometry.tex`
- `spectral_proof_completion_20261020/manuscript/probability_closure.tex`
- `spectral_proof_completion_20261020/manuscript/sbm_proof.tex`
- `spectral_proof_completion_20261020/manuscript/dcsbm_proof.tex`
- `spectral_proof_completion_20261020/manuscript/statements.tex`
- `dc_oracle_sparse_followup_20261020/previous_main.tex`, general degree-plus-spectrum section and conditional sharp-transfer proposition
- `dc_oracle_sparse_followup_20261020/regularized_encoder.md`

Primary external source: [Can M. Le, Elizaveta Levina, and Roman Vershynin, *Concentration and regularization of random graphs*, arXiv:1506.00669v2](https://arxiv.org/pdf/1506.00669), especially Theorem 2.1, Remark 2.2, and Section 3.6. The published article is in *Random Structures & Algorithms* 51(3), 538–561 (2017).

The concrete new completed result of this audit is the unchanged-decoder \(5\log n\) strengthening. The floor construction's normalized-noise and weighted moment chain is also justified above. The unconditional below-logarithmic constant-oracle-risk theorem remains a specific further proof task.

## Appendix A. Independent verification of an exponential-error sharp-transfer extension

The proposed replacement of a uniform coefficient bound by an exponential weighted error condition is valid. It supplies another completed analytic ingredient; it does not assert that the actual floor encoder satisfies its new hypothesis.

Retain the hypotheses on the oracle statistic from the manuscript's sharp-tail lemma:
\[
T=b_0+\sum_jw_jX_j,\qquad |w_j|\le W,\qquad
s_*\in[\tau,1-\tau],
\]
\[
K(s)=\log\mathbb E e^{-sT},\qquad
I=-K(s_*),\qquad
Q\propto e^{-s_*T},
\]
\[
\mathbb E_QT=0,\qquad
vd\le\operatorname{Var}_Q(T)\le V_0d,\qquad
\sum_jr_j\le V_0d.
\]
The Bernoulli parameters under \(Q\) are \(r_j\).
Condition on an independent training object so that \(b,e_j\) are fixed finite real numbers, and put
\[
\widehat T=T+b+\sum_je_jX_j.
\]
Suppose, on the training event,
\[
\left|b+\sum_jr_je_j\right|\le B,\qquad
\sum_jr_je_j^2e^{2|e_j|}\le V,
\tag{A.1}
\]
where \(V\) is a fixed constant independent of \(n\). Then for all sufficiently large \(d\) and every real \(x\),
\[
\boxed{
\Pr(\widehat T\le x\mid\text{training})
\le C e^{s_*x+B+V/2}\,d^{-1/2}e^{-I}.
}
\tag{A.2}
\]
The constant depends only on the fixed oracle constants and \(V\). It is uniform over the training event and over the individual coefficient magnitudes. A training-dependent \(B\) is allowed if its indicated exponential factor is integrated.

### A.1 The cumulant perturbation

By the Bernoulli product formula, \(\log(1+u)\le u\), and the Taylor bound for the exponential,
\[
\begin{aligned}
\log\mathbb E e^{-s_*\widehat T}
&=-I-s_*b
  +\sum_j\log\{1+r_j(e^{-s_*e_j}-1)\}\\
&\le-I-s_*\left(b+\sum_jr_je_j\right)
  +\frac{s_*^2}{2}\sum_jr_je_j^2e^{s_*|e_j|}\\
&\le-I+B+V/2.
\end{aligned}
\tag{A.3}
\]
No upper bound on \(\max_j|e_j|\) is used.

### A.2 The new tilted Bernoulli parameters

Let \(\widehat Q\) denote the tilt by \(e^{-s_*\widehat T}\), and write \(\widehat r_j=\widehat Q(X_j=1)\). With \(t_j=e^{-s_*e_j}\),
\[
\widehat r_j=\frac{r_jt_j}{1-r_j+r_jt_j},
\qquad
\frac{\widehat r_j(1-\widehat r_j)}
     {r_j(1-r_j)}
=\frac{t_j}{(1-r_j+r_jt_j)^2}.
\tag{A.4}
\]
The second formula is interpreted on nondegenerate coordinates; the variance inequality it implies also holds when a coordinate is degenerate.

In fact, the extra hypothesis \(r_j\le\bar r<1\) is unnecessary. For any \(r\in[0,1]\) and \(t>0\),
\[
\frac{t}{1+r(t-1)}\le\max(1,t).
\]
Consequently
\[
\widehat r_j\le r_je^{s_*|e_j|}\le r_je^{|e_j|}.
\tag{A.5}
\]
On the core \(C=\{j:|e_j|\le1\}\), \(t_j\in[e^{-1},e]\), so the variance ratio in (A.4) lies in \([e^{-1},e]\).

### A.3 A variance of order \(d\) survives on the core

On the complement of \(C\), \(|e_j|>1\), and (A.1) gives
\[
\sum_{j\notin C}r_j\le V.
\]
Thus the original variance lost outside the core is at most \(W^2V\):
\[
\sum_{j\in C}r_j(1-r_j)w_j^2\ge vd-W^2V.
\]
Also \(\sum_{j\in C}r_j(1-r_j)e_j^2\le V\). The triangle inequality in this weighted Euclidean norm gives
\[
\left(\sum_{j\in C}r_j(1-r_j)(w_j+e_j)^2\right)^{1/2}
\ge\sqrt{vd-W^2V}-\sqrt V.
\]
By (A.4),
\[
\operatorname{Var}_{\widehat Q}(\widehat T)
\ge e^{-1}\bigl(\sqrt{vd-W^2V}-\sqrt V\bigr)^2
\ge c d
\tag{A.6}
\]
for all sufficiently large \(d\). The cutoff used in this proof is deterministic and only a proof device; no coefficients are discarded from the implemented score.

### A.4 The third absolute moments remain \(O(d)\)

For a Bernoulli variable of mean \(\widehat r\),
\[
\mathbb E|X-\widehat r|^3\le \widehat r.
\]
Hence the sum of centered third absolute moments of \(\widehat T\) is at most
\[
\rho_3\le\sum_j\widehat r_j|w_j+e_j|^3.
\]
On the core this is bounded by \(e(W+1)^3\sum_jr_j=O(d)\). For \(u>1\),
\[
e^u(W+u)^3\le C_Wu^2e^{2u}.
\]
Equations (A.1) and (A.5) therefore bound the tail contribution by \(C_WV\). Thus
\[
\rho_3\le C(d+V)=O(d).
\tag{A.7}
\]
The same splitting, with the second power instead of the third, gives \(\operatorname{Var}_{\widehat Q}(\widehat T)\le C(d+V)\).

### A.5 Berry–Esseen still provides the sharp prefactor

Apply Berry–Esseen to the independent centered summands under \(\widehat Q\). From (A.6) and (A.7), the distribution-function approximation error is \(O(d^{-1/2})\). A normal distribution of variance at least \(cd\) puts at most \(C/\sqrt d\) probability in any interval of length one. Therefore
\[
\sup_y\widehat Q\{\widehat T\in[y,y+1]\}\le C/\sqrt d.
\tag{A.8}
\]
This is uniform in the location of the interval and does not require \(\mathbb E_{\widehat Q}\widehat T=0\). It also does not require bounded summand ranges: finite third absolute moments with the bound (A.7) are sufficient.

Decompose \((-\infty,x]\) into unit intervals. The identity
\[
dP/d\widehat Q
=\mathbb E e^{-s_*\widehat T}\,e^{s_*\widehat T}
\]
and (A.8) give
\[
\Pr(\widehat T\le x)
\le \mathbb E e^{-s_*\widehat T}\,
     \frac{C e^{s_*x}}{\sqrt d}
     \sum_{k\ge0}e^{-s_*k}.
\]
Since \(s_*\ge\tau\), the series is bounded by a fixed constant. Substitution of (A.3) proves (A.2).

### A.6 A slightly weaker exponential condition is sufficient

Because the original assumptions also contain \(s_*\le1-\tau\), the condition in (A.1) can be weakened to
\[
\sum_jr_je_j^2e^{|e_j|}\le V.
\]
Indeed, for \(u>1\),
\[
e^{s_*u}(W+u)^3
\le C_Wu^3e^{(1-\tau)u}
\le C_{W,\tau}u^2e^u.
\]
The cumulant and core variance proofs are unchanged. The version with \(e^{2|e_j|}\) is therefore safe, while the version with \(e^{|e_j|}\) may be easier to verify for an actual encoder. Either version still requires training independence of the coefficients and a fixed \(V\); the mere weighted quadratic bound proved in Section 6 does not imply either exponential condition.

## Appendix B. Review of the new manuscript sparse-extension section

I independently read the complete new sparse-extension section after the preceding note was written. Its floor-geometry proposition, the completed-diagonal treatment, the deleted-space tilted mean and quadratic calculation, exact raw-degree correction, exponential-error transfer with \(e^{|e_j|}\), and the expected hit-mass truncated transfer all check out.

In particular, the high-degree-set proof really gives \(e^{-cn}\); after combining it with LLV's theorem and the total-edge event, the claimed \(1-C_rn^{-r}\) probability follows for every fixed \(r>0\). The floor's signal matrix depends on \(G\), but the Gram-matrix gap and subsequent spectral perturbation are deterministic comparisons on the event. There is no need to make \(G\) independent of the adjacency matrix.

Three clarifications were sent to the author:

1. To replace \(a_0d\) by a fixed multiple of the observed average degree, explicitly use the two-sided total-edge event \(cnd\le\sum_iD_i\le Cnd\). The displayed proof initially states only its upper half. The lower half follows from the same Bernoulli Chernoff calculation, with failure \(e^{-cnd}\).
2. The truncated-transfer lemma should explicitly say that \(B,V\) are fixed constants. If either is training-random, the constant \(C_M\) cannot be pulled out of the training expectation without an additional integrated bound.
3. The tilted decomposition is for the bounded block-constant linear oracle contrast used in the DC proof. The full nonlinear profile GLRT should continue to enter through its separate bounded Bernoulli profile-reduction remainder; it should not be described as itself having a product-Bernoulli exponential tilt.

For the truncated transfer, the direct conditional hit probability is at most
\[
h_M=\sum_{j:|e_j|>M}p_j.
\]
Replacing \(e_j\) by zero only in the proof makes the two random scores identical unless one of those coordinates is hit. The truncated mean perturbation is bounded by \(B+V/M\), and its quadratic perturbation by \(V\). Thus the claimed bound
\[
\Pr(\widehat T\le x)
\le C_Md^{-1/2}e^{-I}
   +\mathbb E[\mathbf1_{\mathcal E}h_M]
   +\Pr(\mathcal E^c)
\]
is correct for bounded \(x\) and fixed \(M,B,V\). Bounded original oracle weights imply \(p_j\le e^W r_j\), so \(h_M\le e^W\eta_M/M^2\) is also valid. The direct expected hit-mass condition is the weaker sufficient hypothesis.

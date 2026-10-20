# Sparse sharp transfer by coupling after analytic coefficient truncation

Independent mathematical note, 20 October 2026.

This note proves a sharp-transfer lemma for an ideal leave-one-out score.
It does not claim the required truncated spectral-row theorem for the
degree-floor encoder, or the full-to-leave-one-out comparison for the
implemented fitted classifier, has already been proved.

The useful change is that a global bound on the coefficient exponential
remainder is unnecessary. Bounded total tilted quadratic error, together
with a sufficiently small expected number of large coefficients, suffices.
The truncation is used only in the proof; the retained-data decoder is
unchanged.

## 1. A sharp-transfer lemma without a maximum-coefficient assumption

Let the independent Bernoulli variables \(X_j\) have parameters \(p_j\),
and let the oracle contrast be
\[
T=b_0+\sum_jw_jX_j,\qquad |w_j|\le W.
\]
Write \(K(s)=\log\mathbb E e^{-sT}\). Suppose its saddlepoint satisfies
\(s_*\in[\tau,1-\tau]\), and let \(Q\) have density proportional to
\(e^{-s_*T}\). Under this tilt the coordinates remain independent, with
\[
r_j=Q(X_j=1)
=\frac{p_je^{-s_*w_j}}{1-p_j+p_je^{-s_*w_j}}.
\]
Assume the usual sharp-oracle conditions
\[
\mathbb E_QT=0,\qquad
v d\le \operatorname{Var}_Q(T)\le V_0d,
\qquad \sum_jr_j\le V_0d,
\]
where \(d\to\infty\). Put \(I=-K(s_*)\) and
\(q_0=d^{-1/2}e^{-I}\). For bounded decision thresholds, the actual
oracle tail is comparable to \(q_0\).

Let \(\mathcal W\) be independent of the entire tested row \((X_j)_j\).
Consider the possibly unbounded, training-dependent coefficient error
\[
\widehat T=T+b+\sum_je_jX_j.
\]
On a \(\mathcal W\)-measurable event \(\mathcal E\), assume only
\[
\left|b+\sum_jr_je_j\right|\le B,
\qquad \sum_jr_je_j^2\le V,
\tag{1}
\]
with fixed \(V\), and with \(B\ge0\) either fixed or training measurable.

For a fixed \(M>0\), define
\[
J_M=\{j:|e_j|>M\},\quad
m_M=\sum_{j\in J_M}p_j,\quad
\eta_M=\sum_{j\in J_M}r_je_j^2.
\tag{2}
\]

**Lemma.** For every fixed \(M,V\), for all sufficiently large \(d\),
and on \(\mathcal E\),
\[
\begin{split}
\mathbb P(\widehat T\le x\mid\mathcal W)
&\le
C\exp\{s_*x+B+V/M+e^M V/2\}\,q_0+m_M\\
&\le
C\exp\{s_*x+B+V/M+e^M V/2\}\,q_0
+\frac{e^W}{M^2}\eta_M.
\end{split}
\tag{3}
\]
Here \(C\) depends only on the fixed sharp-oracle constants and on
\(M,V,W\), never on \(n\), \(d\), or the largest \(|e_j|\).

Consequently, if \(x\) is fixed and \(B\) is bounded,
\[
\mathbb P(\widehat T\le x)
\le C_Mq_0+
\mathbb E[\mathbf1_{\mathcal E}m_M]
+\mathbb P(\mathcal E^c),
\tag{4}
\]
and hence also
\[
\mathbb P(\widehat T\le x)
\le C_Mq_0+
\frac{e^W}{M^2}\mathbb E[\mathbf1_{\mathcal E}\eta_M]
+\mathbb P(\mathcal E^c).
\tag{5}
\]
For random \(B\), replace the constant factor in front of \(q_0\)
by a constant times \(\mathbb E[\mathbf1_{\mathcal E}e^B]\).

### Proof

Use the analytically truncated coefficients
\[
e'_j=e_j\mathbf1_{\{|e_j|\le M\}},\qquad
\widehat T'=T+b+\sum_je'_jX_j.
\]
They satisfy
\[
\max_j|e'_j|\le M,\qquad
\sum_jr_j(e'_j)^2\le V,
\]
and
\[
\left|b+\sum_jr_je'_j\right|
\le B+\sum_{j\in J_M}r_j|e_j|
\le B+\eta_M/M\le B+V/M.
\tag{6}
\]
The Bernoulli exponential-moment calculation at the original saddlepoint
gives
\[
\log\mathbb E[e^{-s_*\widehat T'}\mid\mathcal W]
\le -I+B+V/M+e^M V/2.
\tag{7}
\]
To retain the \(d^{-1/2}\) prefactor, tilt the original row law by
\(e^{-s_*\widehat T'}\). The new odds are the oracle-tilted odds times
\(e^{-s_*e'_j}\), so their probabilities and variances are within fixed
factors of the oracle-tilted ones. Moreover, the weighted triangle
inequality gives
\[
\left[\sum_jr_j(1-r_j)(w_j+e'_j)^2\right]^{1/2}
\ge \sqrt{vd}-\sqrt V.
\]
Thus the new tilted variance is comparable to \(d\). Its sum of centered
third absolute moments is \(O(d)\), because \(|w_j+e'_j|\le W+M\)
and its total Bernoulli mass is \(O(d)\). Berry--Esseen therefore bounds
the probability of every interval of length one by \(C/\sqrt d\),
uniformly in its location; the mean need not be zero. Summing successive
unit intervals on \(( -\infty,x]\), using \(s_*\ge\tau>0\) and (7),
proves the first term of (3) for \(\widehat T'\).

On the event \(\{X_j=0\text{ for every }j\in J_M\}\),
\(\widehat T=\widehat T'\) exactly. Conditional independence of the
tested row from \(\mathcal W\) gives
\[
\mathbb P(\widehat T\ne\widehat T'\mid\mathcal W)
\le \mathbb P(\exists j\in J_M:X_j=1\mid\mathcal W)
\le m_M.
\tag{8}
\]
Finally,
\[
\frac{p_j}{r_j}=p_j+(1-p_j)e^{s_*w_j}\le e^W,
\]
so
\[
m_M\le e^W\sum_{j\in J_M}r_j
\le \frac{e^W}{M^2}\eta_M.
\tag{9}
\]
Combining (6)--(9) and integrating over training proves all claims.

The count form (4) is strictly weaker than (5). Once the *total* tilted
quadratic error is bounded, large coefficients matter for this comparison
only through the probability that the tested raw row meets their support.

## 2. Exact sufficient rates in the sublogarithmic regime

Suppose \(d\to\infty\), \(d=o(\log n)\), and \(I\le C_*d\).
Then every fixed polynomial failure probability is negligible relative
to the oracle tail:
\[
n^{-r}/q_0\le \sqrt d\exp(C_*d-r\log n)\longrightarrow0
\qquad(r>0).
\tag{10}
\]
For the truncation payment, any one of the following is sufficient:
\[
\mathbb E[\mathbf1_{\mathcal E}m_M]=O(q_0),
\quad\text{or}\quad
\mathbb E[\mathbf1_{\mathcal E}\eta_M]=O(q_0).
\tag{11}
\]
A simpler stronger condition is an exponential bound \(Ce^{-Hd}\)
with a fixed \(H>C_*\). Fixed powers of \(d\) multiplying
\(e^{-Hd}\) are harmless because of the strict exponent gap.

If \(p_j\le C_pd/n\), an expected-count statement alone suffices:
\[
\mathbb E[\mathbf1_{\mathcal E}|J_M|]
\le Cn e^{-Hd}
\quad\Longrightarrow\quad
\mathbb E[\mathbf1_{\mathcal E}m_M]
\le C'd e^{-Hd}=o(q_0).
\tag{12}
\]
Thus even a training graph containing many large coordinates can be
acceptable: the correct quantity is their fraction weighted by the
target row's edge probabilities, compared with the oracle error.

For average clustering risk, use (4) separately for each target row and
each of the finitely many competing labels, then average over rows.
There is no need to union-bound the large-coordinate event over every
target row. A union bound with an unnecessary extra factor \(n\) would
destroy (12) when \(d=o(\log n)\).

## 3. Reduction from degree-floor spectral rows

For the independent training graph, let
\[
G_j\ge ad,\qquad \sum_jG_j\le C_Gnd,\qquad
H=G^{-1/2}UU^\top G^{1/2},\qquad U^\top U=I.
\]
For a bounded block-constant contrast \(w\), set
\[
x=w-\bar w\mathbf1,\qquad
e=(H-I)x,\qquad \|x\|_\infty\le L_x.
\]
For the probability bounds in this section assume also
\(p_j\le C_pd/n\), as on the compact SBM/DC-SBM classes under discussion;
bounded oracle weights then imply \(r_j\le C_rd/n\).
The corresponding full-graph degree correction is
\[
R_0=G^{1/2}U\Lambda U^\top G^{1/2}=AH,\qquad
R=R_0+(D-R_0\mathbf1)\mathbf1^\top/n.
\]
It obeys the exact retained-data identity
\[
Rw=A\{H(w-\bar w\mathbf1)+\bar w\mathbf1\}.
\tag{13}
\]

Define the generalized row amplitude
\[
\zeta_j=\sqrt{nd/G_j}\,\|U_{j\cdot}\|_2.
\]
Cauchy--Schwarz gives the deterministic pointwise bound
\[
|(Hx)_j|
\le \frac{\|U_{j\cdot}\|_2}{\sqrt{G_j}}
       \|U^\top G^{1/2}x\|_2
\le L_x\sqrt{C_G}\,\zeta_j.
\tag{14}
\]
Write \(C_x=L_x\sqrt{C_G}\). For \(M\ge2L_x\),
\[
\{|e_j|>M\}\subseteq\{\zeta_j>M/(2C_x)\}.
\tag{15}
\]
Therefore the following **weighted spectral-row count theorem** would
be enough to close the truncation part of sharp transfer:
\[
\mathbb E\!\left[\mathbf1_{\mathcal E}
  \sum_j\mathbf1_{\{\zeta_j>L\}}\right]
\le Cn e^{-Hd},\qquad H>C_*.
\tag{16}
\]
Here \(L\) is a sufficiently large fixed constant, and \(M\) is then
a sufficiently large fixed constant. This conclusion is a reduction,
not an assertion that (16) has been established for the floor encoder.

A familiar truncated-row-norm theorem is more than enough. Indeed, since
\(G_j\ge ad\),
\[
\{\zeta_j>L\}
\subseteq\{\|U_{j\cdot}\|_2>L\sqrt{a/n}\}.
\]
If, for a sufficiently large fixed \(L\),
\[
\mathbb E\!\left[\mathbf1_{\mathcal E}
\sum_j\|U_{j\cdot}\|_2^2
\mathbf1_{\{\|U_{j\cdot}\|_2>L\sqrt{a/n}\}}\right]
\le Ce^{-Hd},
\tag{17}
\]
then Markov's inequality at this threshold gives (16), up to a fixed
factor. No uniform row-incoherence statement is needed.

For completeness, define
\[
\mathcal T_L=
\sum_j\frac d{G_j}\|U_{j\cdot}\|_2^2
\mathbf1_{\{\zeta_j>L\}}.
\]
For suitable fixed \(M,L\), (14) also gives directly
\[
\eta_M\le C\frac dn\sum_{\zeta_j>L}\zeta_j^2
=Cd\,\mathcal T_L,
\qquad
\mathcal T_L\le\frac1a\sum_j\|U_{j\cdot}\|_2^2
\mathbf1_{\{\|U_{j\cdot}\|_2>L\sqrt{a/n}\}}.
\tag{18}
\]
Thus (17) implies the requested truncated-coefficient quadratic-tail
bound too, with only an extra factor \(d\).

## 4. Why a global exponential remainder can be unnecessarily strong

The two already established bounds in (1) do not by themselves control
\(\sum_jr_j(e^{c|e_j|}-1-c|e_j|)\). Here is an exact illustration.
Choose two indices with equal \(r_j=r\asymp d/n\), and set
\[
e_1=(2r)^{-1/2},\qquad e_2=-(2r)^{-1/2},\qquad e_j=0
\quad(j\ne1,2),\qquad b=0.
\]
Then \(\sum r_je_j=0\), \(\sum r_je_j^2=1\), whereas
\[
\sum_jr_j(e^{c|e_j|}-1-c|e_j|)
=2r\{e^{c/\sqrt{2r}}-1-c/\sqrt{2r}\}\longrightarrow\infty.
\]
Nevertheless the original and truncated scores differ only when one
of two raw edges is present. This has probability \(O(d/n)=o(q_0)\)
in the sublogarithmic regime, so (4) still gives a constant-oracle
comparison. In this example \(\eta_M=1\), making the count form (4)
useful even where the stronger quadratic-tail criterion (5) is not.

This example does not purport to construct such coefficients from an
SBM eigenspace. It shows exactly why failure of the unrestricted MGF
bound is not, by itself, a risk obstruction.

## 5. Verified connection with Zhang's theorem

Primary source: Anderson Y. Zhang, *Fundamental Limits of Spectral
Clustering in Stochastic Block Models*, IEEE Transactions on Information
Theory 70(10), October 2024:

https://andersonyezhang.github.io/PDF/spectral_SBM.pdf

Theorem 3 is in Section III.B, printed page 7326 (PDF page 7), with
threshold (13)
\[
t_0=\frac{160k^2}{\beta}\sqrt{\frac{k}{\beta n}}.
\]
Under \(\beta^{-1},k=O(1)\), \(0<q<p\le1/2\), and
\(n(p-q)^2/p\to\infty\), it states
\[
\mathbb E\!\left[\mathbf1_{\mathcal F}
\sum_i\|U_{i\cdot}\|_2^2
\mathbf1_{\{\|U_{i\cdot}\|_2\ge t_0\}}\right]
\le t_0^2n\exp\!\left\{-\frac{3n(p-q)^2}{2p}\right\}.
\tag{19}
\]
Corollary 1 on the same page bounds the corresponding expected count by
\(n\exp\{-3n(p-q)^2/(2p)\}\). The proof starts in Section IV.B,
printed page 7334 (PDF page 15).

Its \(U\) is the eigenspace of a trimmed adjacency matrix, with high-degree
rows and columns set to zero; \(\mathcal F\) is its spectral event (4).
It is not the degree-floor normalized eigenspace. Formula (19) verifies
that precisely this kind of expected-tail statement is attainable without
uniform row incoherence, but it cannot be cited as already proving (16)
or (17) for DC-SBM or the present encoder.

## 6. Exact scope of the progress

The lemma is proved and can replace a blanket coefficient MGF requirement
in the conditional sparse sharp-transfer proposition. It also identifies
a weaker target than truncated \(\ell_2\): an expected count of overly
large generalized spectral rows, with a sufficiently strong exponent.

The ideal score must use a genuinely independent leave-one-out training
environment. The actual encoder uses the full graph, and its degrees,
floor, eigenspace, fitted partition, and fitted parameters depend on the
tested row. Their comparison with the ideal score still needs a separate
interface argument with bounded score cost or oracle-scale exceptional
probability. Conditioning on a full-graph spectral event does not create
the independence required in (8).

Entrywise clipping is not proposed as an unacknowledged change to the
algorithm. In general \(A\operatorname{clip}(v)\) is not determined merely
by knowing \(Av\) for vectors in the retained spectral space. Analytic
coupling avoids that computational issue entirely: the original decoder
continues to use the computable action (13).

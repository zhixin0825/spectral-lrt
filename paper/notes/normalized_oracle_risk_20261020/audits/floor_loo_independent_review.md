# Independent review: local degree-floor full-to-LOO scalar stability

This note audits only the deterministic scalar LOO step. It does not certify the row-tail probability, initializer, or nuisance-fit theorem.

## Verdict

The degree-metric action can be proved with a weighted row bound only on the support of the tested incident row. No uniform bound on every deleted eigenvector row and no global upper bound on every floored degree are needed. The changing empirical floor contributes terms of size 1/n in the normalized perturbation; after the correct weighted estimates these contribute bounded scalar errors, not sqrt(d).

The Euclidean action additionally needs lambda_min(U_B' G_0^{-1} U_B) >= c/d. This does not follow from totalG and orthonormality alone, but does follow from the established model signal-space angle and the population block geometry.

## Deterministic premises and notation

Write A = [[0,x'];[x,B]], k=sum x <= Ld, d->infinity, d<=Cn. Let Delta=G_0 be the recomputed deleted floor, Delta_j>=cd, sum Delta_j<=Cnd. Write the full floor as diag(g,Delta+E), with cd<=g<=Cd and
E=E_loc+E_glob,
0<=E_loc<=diag(x),
||E_glob||op<=Cd/n.
The decomposition is exact for G_j=max(D_j,a average(D)); split the effect of adding x while keeping the deleted threshold, then the effect of changing the threshold.

Put F=Delta^{-1/2} E Delta^{-1/2}. Then ||F||<=C/d, and ||F_glob||<=C/n.

Suppose
S_B=Delta^{-1/2}B Delta^{-1/2}=U_B Lambda_B U_B'+N,
N U_B=0, ||N||<=C/sqrt(d),
with fixed signal gap. Put V=Delta^{-1/2}U_B.

The only required row premise is
||V_j|| <= R/sqrt(nd) for each j with x_j=1.
Equivalently, sqrt(nd/Delta_j)||U_B,j||<=R on the hit support.
There is no row premise on j with x_j=0.

For any bounded w, ||w||infinity<=W, prove
(AH_G w)_i = x' H_B w_{-i}+O(1),
H_B=V V' Delta,
H_G=G^{-1/2}UU'G^{1/2}.

## Exact generalized equations

For full generalized eigenvectors gamma_r=G^{-1/2}u_r, write
gamma_r=(t_r, Delta^{-1/2}(U_B y_r+z_r)),
U_B' z_r=0.
Set h=Delta^{-1/2}x, h_V=U_B'h, h_Q=Qh.

The top equation is
h'(U_B y_r+z_r)=mu_r g t_r.
The factor is the floored target degree g, NOT the raw degree k.

The complementary equation is unchanged:
(mu_r I-N+mu_r QFQ)z_r
=t_r h_Q-mu_r QF U_B y_r.

Its inverse T_r has bounded norm. Substitution into the top equation yields
(mu_r g-h_Q'T_r h_Q)t_r
=h_V'y_r-mu_r h_Q'T_r QF U_B y_r.

The denominator is at least c d-C k/d >= c'd.
Here ||h||^2<=Ck/d.
The full normalized matrix differs from diag(0,S_B) by O(d^{-1/2}); this proves the fixed full signal gap and the bounded coordinate matrix Y without global row incoherence.

## Refined local bounds

The support premise gives
||h_V||<=C_R k/sqrt(nd),
||F U_B||_F<=C_R sqrt(k)/(d sqrt(n))+C/n.

Consequently,
||t||<=C_R/sqrt(nd),
||Z||<=C_R(sqrt(k)/(d sqrt(n))+1/n).

Write v_r=Delta^{-1/2}z_r, f_r=Vy_r+v_r. Then
||v||<=C_R(sqrt(k)/(d^{3/2}sqrt(n))+1/(n sqrt(d))),
||x'v||<=C_R(k/(d^{3/2}sqrt(n))+sqrt(k)/(n sqrt(d))).

## Gram normalization

The exact generalized normalization is
Y'Y+Z'Z+g tt'+f'Ef=I.

The local part of U_B'FU_B is bounded by R^2 k/(nd); its global part is O(1/n).
All remaining terms give
||YY'-I||<=C_R(k/(nd)+1/n).

Thus the leading scalar error is
||x'V|| ||YY'-I|| ||V'Delta w||
<=C_R[k^2/(nd)+k/n]=O(1).

The 1/n term caused by the changing floor is required. Omitting it is incorrect for a general adaptive floor, but including it does not spoil the conclusion.

## Weighted row expansion and all remaining factors

The exact row action is
(AH_G w)_i
=sum_r (x'Vy_r+x'v_r)
[g t_r w_i+y_r' V'Delta w+v_r'Delta w+f_r'Ew].

Useful bounds:
||x'V||<=C_R k/sqrt(nd);
||V'Delta w||<=C_W sqrt(nd);
||v'Delta w||<=C_RW[sqrt(k/d)+sqrt(d/n)];
||g t||<=C_R sqrt(d/n);
||f'Ew||<=C_RW[k/sqrt(nd)+sqrt(d/n)].

For the global part of the last expression, use
|f'E_glob w| <= (Cd/n) W ||f||_1
<= C_W(d/n) sqrt(n) ||f||
<=C_W sqrt(d/n).
Do not substitute an unavailable max Delta bound.

The potentially largest products are
k/d,
sqrt(k/n),
k^2/(nd),
k/n,
k sqrt(k)/(d sqrt(n)).
Every one is bounded because k<=Ld and d<=Cn.

The k=0 case has zero adjacency row action exactly; no singular convention is needed because g has a positive floor.

## Exact degree correction and hollowing

Let R_0=AH_G and
R=R_0+(rawdeg-R_0 1)1'/n.
Then for bar w=(1'w)/n:
(Rw)_i = [R_0(w-bar w 1)]_i+k bar w.
Apply the uniform bounded-w scalar lemma to w-bar w 1.
Changing the centering mean to the deleted mean costs O(k/n), since their difference is O(1/n) and H_B1 is bounded on the hit support.

The local t bound gives
||u_i||^2=g||t||^2<=C_R/n,
|(R_0)ii|<=C_R d/n.
Also, H_B1 is bounded on hit indices, so (R_01)_i=O_R(k+1).
Hence |R_ii|=O_R(d/n).
Removing R_ii and redistributing it over the remaining row changes a bounded contrast by O_W(d/n)=O(1), while retaining the raw degree.

## Euclidean Gram warning and repair

totalG<=Cnd and ||U_B||=1 alone do NOT imply lambda_min(V'V)>=c/d.
Include this explicitly in an abstract Euclidean scalar lemma, or derive it on the model event.

Let U_0 be an orthonormal basis for col(Delta^{-1/2}Theta Z).
Write s_a=sum_{j in a}theta_j^2/Delta_j.
The floor and totalG bounds give s_a asymp n/d.
On the block basis,
lambda_min(U_0' Delta^{-1} U_0)
= min_a (sum theta_j^2/Delta_j^2)/s_a
>=min_a s_a/(sum theta_j^2)
>=c/d.
The middle inequality is Cauchy-Schwarz.
If ||U_B-U_0 O||<=C/sqrt(d), then the Gram perturbation is O(d^{-3/2}), since ||Delta^{-1}||<=C/d.
This yields the empirical lower bound c'/d.

With this input, the old Euclidean proof works with
||E_basis||<=C_R[sqrt(k)/(d sqrt(n))+1/n].
The new 1/n term again causes bounded scalar error.

This does NOT justify orthogonally augmenting the floor spectral space by a nearly redundant constant vector. The exact rank-one degree correction above avoids that separate conditioning problem.

## Statistical scope

The no-hit event depends on the training graph and x. Its failure probability must be paid using the expected deleted bad-row count. The deterministic lemma alone does not establish that count.

Uniformity in bounded w permits algebraic use with data-dependent fits, but does not make their conditional score coefficients independent of x. A separate fitted-parameter argument is still required.


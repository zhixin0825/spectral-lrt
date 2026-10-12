# 谱信息关于 SBM 分组参数的频率学派近似充分性

2026-10-12。这里 **C 始终是固定参数**。不对 C 设先验，不使用 posterior。所有概率仅来自给定参数下随机图 A 的生成，以及允许的随机化决策。

## 本次证明的结论

对任意固定 K，在各组占比有正下界、实际 Bernoulli 概率矩阵 P 有统一谱分离、且 np >= M log n（M 足够大）的条件下，保留原始 adjacency 的 top-|lambda| K 个 eigenpairs，足以恢复分区，并且对所有有限行动集、只评价分组的有界统计决策，其信息损失一致地趋零。P 可以未知。

这个结论是 **关于分组的渐近决策充分性**。另证明：当 P 已知且标签约定一致时，可以得到分组参数实验的标准 Le Cam 渐近充分性。没有据此声称未知 (C,P) 的整个实验渐近等价，也没有宣称达到 sharp exact-recovery 常数。

## 1. 参数、模型和统计量

令 C 是 {1,...,n} 的 K 个非空 blocks 的分区，按标签置换识别。可按每个 block 的最小节点编号排序，选取唯一表示；P 与该顺序一致地索引。等价地可以使用 labelled membership，但所有关于分组的损失必须对同步标签置换不变。

给定固定 theta=(C,P)，A 是对称、无自环的 SBM：i<j 的 A_ij 独立，且

\[
 A_{ij}\sim\operatorname{Ber}(P_{c_i c_j}),\qquad A_{ii}=0.
\]

P 是实际的对称概率矩阵，p=max_ab P_ab。固定正数 pi_0,kappa,eta 和固定 K；参数类满足

\[
 n_a\ge\pi_0n,\quad
 \kappa p\le P_{ab}\le p\le1-\eta,\quad
 \sigma_{\min}(P)\ge\kappa p.
\]

不要求 P/p 或组比例收敛。令 T(A)=(U,Lambda)，其中 U 包含按绝对值最大的 K 个 adjacency 特征值对应的特征向量，Lambda 保留其符号。并列时采用确定的可测规则。

完整图和谱信息产生的统计实验为

\[
 \mathcal E_n=\{\mathsf P_{C,P}: (C,P)\in\Theta_n\},\qquad
 \mathcal F_n=\{\mathsf Q_{C,P}: (C,P)\in\Theta_n\},
\]

其中 Psf 是 A 的分布，Qsf 是 T(A) 的分布。参数是固定的；这两族分布不是参数的随机分布。

## 2. P 未知时，直接从统计决策定义分组的近似充分性

对任意有限行动集 D 和任意有界损失 0<=L(C,d)<=1，定义

\[
 R_{C,P}(\delta)=\mathbb E_{C,P}L(C,\delta).
\]

称 T 对分组具有 epsilon-决策充分性，如果对每个这样的 L 和每个使用完整 A 的（可随机化）规则 delta_A，都存在仅使用 T 的规则 delta_T，使

\[
 \sup_{(C,P)\in\Theta_n}
 \{R_{C,P}(\delta_T)-R_{C,P}(\delta_A)\}\le\epsilon.
\]

这里损失只评价 C，规则 delta_T 不允许使用未知 P。这是在标准统计实验比较中，将决策问题限制到目标参数 C；它不同于要求保留所有关于 (C,P) 的决策。

### 定理 1：一致完全恢复推出分组决策充分性

若存在谱估计器 Chat(T)，满足

\[
 \sup_{(C,P)\in\Theta_n}
 \mathsf P_{C,P}\{\widehat C(T)\ne C\}\le\epsilon_n,
\]

则 T 具有 epsilon_n-分组决策充分性。

**证明。** 对任意 L，取确定的并列规则，并定义

\[
 d^*(c)\in\arg\min_{d\in D}L(c,d),\qquad
 \delta_T(t)=d^*(\widehat C(t)).
\]

对每一个固定的 (C,P)，在 Chat=C 的事件上损失等于 min_d L(C,d)，在其余事件上损失至多为 1。因此

\[
 R_{C,P}(\delta_T)
 \le\min_d L(C,d)+\epsilon_n
 \le R_{C,P}(\delta_A)+\epsilon_n.
\]

最后一步因为任何完整图规则的期望损失也不可能小于已知真分区时的最低损失。取 supremum 即得结论。整个论证未将 C 随机化。证毕。

## 3. 任意固定 K 的谱恢复前提可以验证

### 定理 2：足够大常数的对数密度下，一致谱完全恢复

对每个 q>0，存在只依赖 K,pi_0,kappa,eta,q 的常数 M_q,n_0，使得

\[
 n\ge n_0,\qquad np\ge M_q\log n
\]

时，存在只使用 T 的分区估计器，满足

\[
 \sup_{(C,P)\in\Theta_n}
 \mathsf P_{C,P}\{\widehat C(T)\ne C\}\le n^{-q}.
\]

证明使用一个公开的逐行谱扰动定理作为概率引理，随后给出明确的确定性解码；不借用本仓库尚未在本次重新审计的最优指数证明。

**步骤 1：无自环均值的 signal space 恰为 block-constant space。** 令 Z 为 membership indicator，N=diag(n_a)，F=ZN^{-1/2}，D_ii=P_{c_i c_i}。则

\[
 M=\mathbb E_{C,P}A=ZPZ^\top-D.
\]

由于 D 在每个 block 上是常数，col(F) 是 M 的不变子空间，且

\[
 F^\top MF=N^{1/2}PN^{1/2}-\operatorname{diag}(P_{aa}).
\]

此 K 维矩阵的最小 singular value 至少为 pi_0 kappa np-p。在其正交补上，M 的 eigenvalues 是 -P_aa（各有 n_a-1 个）。因此 n 足够大时，K 个 signal eigenvalues 的绝对值和对其他谱的间隔均至少为常数乘 np，上界也是常数乘 np。

**步骤 2：population rows 的距离。** 令 V 是上述 K 维矩阵的 orthogonal eigenvector matrix，则 U_*=FV。真组 a 的中心是 V_a/sqrt(n_a)，所以

\[
 \|m_a-m_b\|_2=\sqrt{1/n_a+1/n_b}\ge\sqrt{2/n},\qquad
 \|U_*\|_{2\to\infty}\le(\pi_0 n)^{-1/2}.
\]

**步骤 3：概率输入和一致性。** 使用 Lei 的 Theorem 3.4（2020 arXiv version）。对正、负 signal tails 分别应用，并合并 orthogonal alignments，可以得到

\[
 \inf_{W\in O(K)}\|UW-U_*\|_{2\to\infty}
 \le \frac{B_q}{\sqrt n}\sqrt{\frac{\log n}{np}}
\]

以至少 1-n^{-q} 的概率成立（增大内部 confidence exponent 吸收常数）。这是该通用定理在本参数类上的一致化，而非直接将其固定-matrix Theorem 5.2 写成变动参数版本。

具体代入量为：lambda_min(abs),gap >= c np；p*,平均行概率 <=p；||M||_(2->infty)<=p sqrt(n)；||U_*||_(2->infty)<=1/sqrt(pi_0 n)；每个所选 signal tail 的维数 r<=K（并非 M 的 rank<=K）。取该定理的 delta=n^{-(q+2)} 和辅助 exponent 1/2，R=O(log n)、g=O(sqrt(np)+log n/log log n)。其各项一致地界于

\[
 \frac{B_q}{\sqrt n}
 \left\{\frac1{\sqrt{np}}+\frac{\log n}{np}
                  +\sqrt{\frac{\log n}{np}}\right\}.
\]

np>=M_q log n 时可吸收到上面的平方根界。通用 Theorem 3.4 本身足够，不需额外使用 Corollary 3.6 的条件。正、负 tails 分开保证每次选择连续 eigenvalue interval；谱 norm 事件保证最终保留的 top-|lambda| K 分量正是这些 tails。所有常数仅依赖固定模型界，不依赖 P/p 或 n_a/n 的极限。

**步骤 4：显式谱解码。** 对节点 i,j，如果

\[
 \|U_i-U_j\|_2\le 1/(2\sqrt n),
\]

就在一个辅助图中连边。若 connected components 恰有 K 个且各组人数至少为 pi_0 n，则返回它们；否则返回一个固定的可行 K 分区。pi_0 是参数类已规定的常数，算法仍不使用未知 P。

选择 M_q 足够大，使逐行误差不超过 1/(8 sqrt(n))。在好事件上，同组的 row distance 至多 1/(4 sqrt(n))；不同组的 row distance 至少 (sqrt(2)-1/4)/sqrt(n)，大于阈值。因此每个真组恰是一个 clique，且没有跨组边；components 恰为 C。上述概率界即给出一致完全恢复。证毕。

解码只用 U，不用 P、真 centers 或 A。它在这里证明谱统计量的信息含量；不宣称前述 likelihood 初始化、refinement 或 MCMC 自动等同于此解码器或具有相同的收敛保证。

### 推论：P 未知的分组近似充分性

结合定理 1 与定理 2，对每个 q>0，在上述参数类及 np>=M_q log n 下，T 对所有有限行动集、有界、标签不变的分组决策具有 n^{-q}-充分性。这个命题对每个固定的 (C,P) 成立，没有 C 的先验，也不要求已知 P。

## 4. 固定已知 P 时的标准 Le Cam 充分性

这里只把 C 作为实验参数，P 是实验中已知的固定常量。需要可识别的标签约定：例如 C 使用上面 canonical block order，P 按该顺序索引；若使用命名 labels，则恢复条件必须包含对 P 的正确标号，不能只恢复未标号的 partition。

设 Psf_C 是原图分布，Qsf_C 是谱统计量分布，且 sup_C Psf_C(Chat!=C)<=epsilon_n。定义不依赖未知 C 的随机重构核

\[
 \mathcal K_P(t,\cdot)=\mathsf P_{\widehat C(t),P}(\cdot).
\]

它根据估计分区和已知 P 独立生成一张新 SBM 图。对每个固定 C，令 epsilon_C=Psf_C(Chat!=C)，则对某概率分布 H_C，

\[
 \mathcal K_P\mathsf Q_C
 =(1-\epsilon_C)\mathsf P_C+\epsilon_C\mathsf H_C.
\]

所以 sup_C ||K_P Qsf_C-Psf_C||_TV<=epsilon_n。原图到 T 的方向是确定映射，deficiency 为零，故

\[
 \Delta(\mathcal E_{n,P},\mathcal F_{n,P})\le\epsilon_n\to0.
\]

这就是固定分组参数实验的标准渐近充分性。C 并未被抽样。P 未知时不能把 K_P 当作允许的重构核；定理 1 的分组决策结论则仍然有效。

## 5. 与 nearly-optimal 错误指数的联系和范围

如果另已独立证明一个只用 T 的算法在某参数类上一致满足 E_(C,P) Mis <= exp(-(1-o(1))I_n)，则

\[
 \mathsf P_{C,P}(\widehat C\ne C)
 \le n\mathbb E_{C,P}\mathrm{Mis}
 \le n\exp\{-(1-o(1))I_n\}.
\]

若参数类还满足 I_n>=(1+tau)log n，tau>0 固定，就得到同样的充分性推论。这个连接不依赖分区随机化；这里仅证明该蕴含，不在本次重新认证旧稿全部最优指数证明。

本次已经闭合的是任意固定 K、统一谱分离和足够大常数的对数密度下的近似分组充分性。不能仅凭 p=Omega(log n/n) 而不说明信号/常数就套用；也没有覆盖仅弱恢复而非完全恢复的区间。完全恢复是本证明的充分条件，不被声称为充分性本身的必要条件。

这套证明不需要让谱工作 loss 等于真实谱数据 density，不证明 reconstructed fractional counts 的精确 binomial 分布，也不证明 likelihood 优化或 MCMC mixing。它证明的是统计量 T 对分组决策的信息保留。

## 原始文献

Lihua Lei. *Unified l_(2->infty) Eigenspace Perturbation Theory for Symmetric Random Matrices*. arXiv:1909.04798v2 (2020), Theorem 3.4；Theorem 5.2 是固定形状/组比例情形的直接 exact-recovery 对照。https://arxiv.org/pdf/1909.04798

Le Cam 距离采用统计实验比较的标准定义。一个可公开核对的定义来源：Yanjun Han, *Statistical decision theory: model distance and equivalence*, Lecture 4 (2021), pp.11-18. https://web.stanford.edu/class/ee378c/lecture4_annotated.pdf

# 论文定位与六节写作方案

2026-10-09。接续已经保存的理论讨论和最新 240 图实验。本文件确定当前稿的结构、标准记号和拟证明的统计目标；不是宣布 nearly-optimal 主定理已经闭合。

## 1. 定位和题目

可以开始写研究稿，采用用户提出的六节结构。初始化及当前有限模型网格上的数值收敛已足够支撑算法与实验章节。论文的主要统计贡献应围绕以下问题展开：

> 只保留邻接矩阵的少量 eigenpairs，能否构造社区间的 likelihood-ratio 分类分数，并在明确的 SBM 条件下保留使用原图 incident edges 的 oracle LRT 的领先错误指数？

“谱方法接近最优”以及“谱初始化后作局部 likelihood 更新”已有先例，不能作为本论文的泛化新颖性声明。拟贡献应具体到 **谱压缩之后的 LLR 信息保留、一般 block matrix 的非欧氏判别，以及相应错误指数**。

工作题目建议：

**Nearly Optimal Community Recovery via Spectral Likelihood Ratios**

用户题目 *Spectral Likelihood-Ratio Tests Are Nearly Optimal for the Stochastic Block Model* 也可作为目标标题；但应在摘要第一段明确是社区标签分类，避免被理解为 ER 对 SBM、社区数 K 或 block matrix 的模型检验。“nearly optimal” 暂按领先错误指数与 oracle 相同来定义，不表示风险比趋于 1，也不表示谱统计量对所有 SBM 参数都是充分统计量。

## 2. 经典 SBM 记号和主情形

考虑已知、固定社区数 K 的无向无自环 SBM：

\[
z_i\in[K],\quad Z_{ia}=\mathbf1\{z_i=a\},\quad
n_a=\sum_i Z_{ia},\quad \pi_a=n_a/n,
\]

\[
B_n=\rho_n B,\qquad
A_{ij}\mid z\sim\operatorname{Bernoulli}((B_n)_{z_i z_j}),\ i<j,
\qquad A_{ji}=A_{ij},\ A_{ii}=0.
\]

主文先取 \(\rho_n=\log n/n\)。B 为固定对称矩阵，正元素有正常数上下界，社区比例有正下界。先处理满秩 B、所有信号方向可分离的情形；允许 B 有负特征值，不强制 assortative。密度尺度记为 \(d_n=n\rho_n\)，各社区期望 degree 与它同阶，不一定恰好等于 d_n。

严格地，均值矩阵是

\[
P=\mathbb E[A\mid z]
=\rho_nZBZ^\top-\operatorname{diag}(\rho_n B_{z_i z_i}).
\]

补回对角后的比较矩阵 \(P^\circ=\rho_nZBZ^\top\) 才精确秩 K。证明要保留这个小对角修正，不能把无自环均值直接称为精确秩 K。

保存 A 的按绝对值排序的前 K 个 eigenpairs：\(AU=U\Lambda\)。主文不做 degree regularization，不需要保存或乘回 degree。Degree 的社区差异仍包含在原始谱特征中；“不另存 degree”不等于假定所有社区期望 degree 相同。

## 3. Section 1：Introduction and related work

先给出统计任务、谱压缩的信息限制以及 oracle LRT benchmark，再按下表总结相关工作。以下均为原始论文入口；文献范围核对至本次日期，尚不是穷尽性的 novelty 排查。

| 研究线 | 原始文献与已有结果 | 本文需要区别的地方 |
|---|---|---|
| General SBM 的信息论界和 likelihood 分类 | [Abbe–Sandon, 2015](https://arxiv.org/abs/1503.00609)：CH divergence 刻画 logarithmic regime 的 exact recovery 阈值，并给出达到阈值的高效算法 | 定义 oracle 与 CH benchmark；本文限制后续只读 eigenpairs |
| 谱方法本身的最优性 | [Abbe–Fan–Wang–Zhong, AoS 2020](https://arxiv.org/pdf/1709.09565)，Theorem 3.2：二等大对称社区、logarithmic density，未 trimming/cleaning 的第二特征向量符号法达到最优阈值及误分指数 | 该特例不是新的；本文应解释 general block matrix 的 LLR contrasts |
| 谱初始化后 likelihood refinement | [Gao–Ma–Zhang–Zhou, JMLR 2017](https://www.jmlr.org/papers/volume18/16-245/16-245.pdf)：弱一致初始化后用局部 penalized likelihood 达到最优误分率 | 其 refinement 使用原始 A；本文的后续分数必须由保存的谱信息计算 |
| 稀疏谱 k-means 的精确性能 | [Anderson Ye Zhang, Fundamental Limits of Spectral Clustering in SBM](https://arxiv.org/html/2301.09289v3)：对 p/q 型多社区 SBM 得到 matching spectral exponent；§3.3 区分此指数与 CH 指数 | 不能声称普通谱 k-means 在所有模型中都已经等同 LRT；本文尝试改变判别准则 |
| 初始分区加 counts 的工作 likelihood | [Amini–Chen–Bickel–Levina, 2013](https://arxiv.org/abs/1207.2340)：pseudo-likelihood 社区检测 | 其 counts 来自 A；本文不能暗中重新读取 A |
| 谱 embedding 的概率分类 | [Suwan et al., Empirical Bayes Estimation for the Stochastic Blockmodel](https://arxiv.org/abs/1405.6070)：ASE 加 Gaussian-mixture 型 empirical Bayes 估计 | 在谱空间拟合概率模型并非新概念；本文关注稀疏大偏差与 LRT 指数 |
| 谱编码与最优 likelihood 几何 | [Dhara–Gaudio–Mossel–Sandon, The Power of Two Matrices](https://arxiv.org/html/2210.05893v3)：censored SBM 中研究两矩阵谱算法的最优性和指定单矩阵算法类的限制 | 模型和算法类不同；用于说明保留哪些 likelihood contrasts 是核心问题，不把其负结论直接套到本文 |

## 4. Section 2：Spectral likelihood-ratio decoding

### 4.1 先给出实际已经运行的算法

定义谱重建与当前计算使用的正 profile：

\[
\widetilde P=U\Lambda U^\top,\qquad
q_{ij}=\max\{\widetilde P_{ij},\varepsilon_n\},
\qquad \varepsilon_n=10^{-4}\log n/n.
\]

最新实验保留重建对角线，论文和程序必须对齐；不能默默改成排除对角后把旧结果当新算法结果。原图 LRT 本身排除 j=i，此差异需要单独控制或通过新的敏感性实验验证。

对正 profile 参数 \(\mu_a\in\mathbb R_{>0}^n\)，取

\[
\ell_i(\mu_a)=\sum_j q_{ij}\log\mu_{aj}-\sum_j\mu_{aj}.
\]

主初始化采用本线程讨论的“首 seed 按 smaller top-subset 选，后续按全局增益 greedy 选”版本。第一轮从每个节点候选 q_c 出发，对最适合的 \(\lfloor2n/(3K)\rfloor\) 个其他节点求 raw score 总和，选最高者。只把这一规则用于首个参数；后续保留所有节点，选

\[
c_t\in\arg\max_c
\sum_i\max\{\ell_i(q_c)-\max_{a<t}\ell_i(\mu_a),0\},
\qquad \mu_t=q_{c_t}.
\]

选齐 K 个参数后，在全体节点上做 soft EM，更新 responsibilities、profile means 和 mixture weights。当前没有强制每组固定人数。

参数替换作为明确声明的稳定化步骤：拟合后逐个尝试替换一个参数，用同一全局增益筛候选，重新 EM，只在最终 observed mixture likelihood 改善超过 1e-5 时接受。当前实验验证的是：原 smaller peeling + 替换，以及全局增益初始化两个独立分支；二者组合成“全局增益初始化 + 替换”的单一最终算法尚未单独跑完整配对验证，不能把两组结果合并冒充组合算法已验证。

初始化规则、profile floor、是否保留对角、EM 类型、停止条件、是否使用替换，都应作为一个明确 Algorithm 1 的定义，不把不同分支混成“我们的算法”。

### 4.2 为什么与 LRT 有关

最终比较两个候选社区时，工作分数差为

\[
\ell_i(\mu_a)-\ell_i(\mu_b)
=\sum_jq_{ij}\log\frac{\mu_{aj}}{\mu_{bj}}
-\sum_j(\mu_{aj}-\mu_{bj}).
\]

它具有 Poisson log-likelihood-ratio 的代数形式。若加入 mixture prior，还要加 \(\log(\omega_a/\omega_b)\)，得到相应工作模型的 posterior/MAP 比较；正文区分 LLR 本身与分类阈值。

但 q 是同一图的相关、非整数谱重建，不是独立 Poisson 原始观测；当前自由 n 维 profile 参数也尚未强制 SBM 结构 \(\mu_{aj}=\rho_nB_{a,z_j}\)。因此这里先称 spectral working likelihood-ratio score。要获得统计上近最优的原图 LRT 结论，必须证明以下对应关系。

定义 oracle Bernoulli contrast

\[
w^{ab}_j=\log\frac{B_{a,z_j}(1-\rho_nB_{b,z_j})}
{B_{b,z_j}(1-\rho_nB_{a,z_j})}.
\]

Oracle nodewise LLR 为

\[
L_i^{ab}=A_iw^{ab}
+\sum_{j\ne i}\log\frac{1-\rho_nB_{a,z_j}}{1-\rho_nB_{b,z_j}}.
\]

谱版本的关键代数是

\[
\widetilde P=A UU^\top,\qquad
\widetilde P_iw^{ab}-A_iw^{ab}
=A_i(UU^\top-I)w^{ab}.
\]

若 contrast 恰在所保留谱空间中，线性项精确保留；满秩人口 SBM 的 block-constant contrasts 位于 \(\operatorname{col}(Z)=\operatorname{col}(P^\circ)\)。这说明谱分解为什么有机会保留 LRT 所需方向，而不要求 SVD 本身计算 log B。

实际算法还要处理经验谱空间误差、clip 项 \((q_i-\widetilde P_i)w^{ab}\)、未知分组/参数的 contrast 误差以及非边项。仅有 \(\widetilde P\approx P\) 的平均重建误差并不够，必须在稀有误分类事件的尺度上控制这些项。秩不足的 B 需要另加 contrast-span 条件，不能直接宣称任意 general B 都适用。

### 4.3 可以立即放进初稿的算法命题

冻结 profile 候选、固定首 seed 后的 greedy 新增收益有次模近似保证；无噪音 profile 模型中 K 步选齐 K 种 profile；精确 EM 使工作目标单调有界。简短证明见下方 Appendix A。它们不等于随机 SBM 的恢复定理。

## 5. Section 3：Main results in the logarithmic regime

建议将主结果按三层组织：谱 LLR 与 oracle LLR 的近似；未知参数与迭代终点的控制；oracle 错误指数的转移。主定理先限定固定 K、正比例社区、正元素有界且满秩 B、信号方向分离，不带 regularization。

令 \(\alpha_a=(\pi_c B_{ac})_{c=1}^K\)，并定义

\[
I_{ab}=D_+(\alpha_a,\alpha_b)
=\max_{t\in[0,1]}\sum_c
[t\alpha_{ac}+(1-t)\alpha_{bc}-\alpha_{ac}^t\alpha_{bc}^{1-t}],
\qquad I_*=\min_{a\ne b}I_{ab}.
\]

拟证明的“nearly optimal”是 oracle leading exponent：例如对合适定义的成对 Bayes 分类风险 \(R_{ab}\)，

\[
-\frac{\log R_{ab}^{\rm spec}}{n\rho_n}\longrightarrow I_{ab},
\]

并希望得到全体标签的置换不变平均误分率 r 满足

\[
\mathbb E r(\widehat z,z)\le n^{-I_*+o(1)}
\]

以及 \(I_*>1\) 时 exact recovery 的推论。边界 \(I_*=1\)、零连接概率、退化 B 和随 n 改变的 K 暂不声称覆盖。对成对风险、全局风险与 minimax 参数空间的 lower bound 要分别定义；不能用一句 oracle optimality 混过去。

**这些是当前稿的主定理目标，不是本轮已经证明的主定理。** 初始化次模保证、目标收敛和 240 图实验不能代替 score 尾部等价。至少需要证明：初值进入合适区域、估计的 log contrasts 足够准确、EM/替换终点满足该准确性，以及经验谱/clip 的误差不会改变领先 Chernoff 指数。LOO 可以作为分析工具，但不能在算法需要只读现存 eigenpairs 时暗中增加逐节点读取原 A 的计算步骤。

## 6. Section 4：Beyond logarithmic density

先取 \(1\ll d_n=n\rho_n\ll\log n\)，讨论 almost exact recovery 与

\[
\mathbb E r(\widehat z,z)\lesssim
\exp\{-(1-o(1))d_nI_*\}.
\]

固定正 B 的这个密度范围通常不可能 exact recovery，不能把 Section 3 的 exact 推论直接延用。低密度还要处理高 degree 导致的噪音谱方向；regularized eigenpairs 和相应的权重补偿应在本节明确引入，并重新证明它们保留哪些 LLR 方向。

目前低密度工作路线只作扩展目标。旧稿曾讨论 degree+regularized spectral counts，但旧稿 README 已说明证明尚未重新审核，而且它不是当前 clipped-profile EM；不能原样挪成新算法定理。\(d_n=O(1)\) 的 bounded-degree detection 有不同目标和阈值，暂不纳入主定理。

## 7. Section 5：Experiments

最新数据是同一协议下的设计集 120 图和预先声明的新种子验证集 120 图；旧 regularized 720 图另列，不合并成同一个实验。最新方法只读 U,Lambda，生成图和评价程序可以使用 A/真标签，但拟合不能使用。

| 方法 | 原 120 图平均错误率 | 新 120 图平均错误率 |
|---|---:|---:|
| Smaller fixed-K peeling + EM | 0.3917% | 0.3725% |
| 原方法 + 一轮参数替换 | 0.0967% | 0.0925% |
| 保留首 seed，后续全局增益 + EM | 0.0967% | 0.0925% |
| 每加参数后 EM 的全局增益 | 0.0967% | 0.0925% |
| Greedy residual++ best10 | 0.0967% | 0.0925% |
| 单次 residual++ | 4.0129% | 3.1675% |
| X k-means，10 starts | 0.3454% | 0.2933% |

三类内容应分开报告：初始化严重失败的修复、最终错分率、收敛/计算成本。每批仅一张严重失败图的错分数改善，其余 119 张不变；exact 图数没有因此增加。比较分支并不具有相同时间预算，尤其全候选 O(n³) score 缓存与十次重启不能当作等价成本。

为了支撑标题，仍需增加两类实验，不在这里冒称已完成：

1. 固定同一 partition/参数，比较原图 oracle LLR、谱 LLR 与当前 clipped-profile score，画分数差和误分类尾部；truth partition/true B 只作明确的诊断组。
2. 随 n 和 d_n 改变的序列实验，检验 \(-\log(\text{error})/d_n\) 是否接近 I_*，并覆盖低于 log n 的密度。不能以当前固定 n=1000/2000 的均值表代替指数或 threshold 验证。

完整结果见 [最新实验报告](../../experiments/20261009_residual_likelihood_seeding/REPORT_20261009.md)。

## 8. Section 6：Discussion and conclusion

总结谱信息对分类 LLR contrasts 的作用、与谱 k-means 和原图 likelihood refinement 的区别，以及密度/谱秩/计算成本的适用范围。结论按已完成的 theorem 和 experiment 写，避免把 future extensions 放成 established result。

## Appendix A：当前可写出的优化命题与证明

### A.1 固定首 seed 的 greedy 增益

令 \(s_{ic}=\ell_i(q_c)\)，首 seed 集合 \(S_0=\{c_1\}\) 固定。定义

\[
F(S)=\sum_i\max_{c\in S}s_{ic},\qquad
G(T)=F(S_0\cup T)-F(S_0).
\]

则 G(∅)=0、单调且次模。证明：如果 S⊆T，当前逐节点最大 score 对 T 不小于对 S；因此加入 c 的边际收益

\[
\Delta(c\mid S)=\sum_i(s_{ic}-\max_{b\in S_0\cup S}s_{ib})_+
\]

对 S 不小于对 T，得到 diminishing returns。令 m=K−1≥1，T* 为 m 个新增候选的最优集合，T_t 为 t 步 greedy 集合。由次模性，

\[
G(T^*)-G(T_t)\le
\sum_{c\in T^*\setminus T_t}\Delta(c\mid T_t)
\le m\,[G(T_{t+1})-G(T_t)].
\]

递推得到

\[
G(T_m)\ge[1-(1-1/m)^m]G(T^*)\ge(1-1/e)G(T^*).
\]

K=1 时没有新增步骤。此处是相对于首 seed 的新增收益保证，不是负 log score 的乘法近似或恢复率保证；也不适用于每次增加后改变 profile centers 的 EM 版本。这是经典次模 greedy 定理的直接应用，应引用 [Nemhauser–Wolsey–Fisher (1978)](https://thibaut.horel.org/submodularity/papers/nemhauser1978.pdf)，不能作为新的通用次模定理。

### A.2 无噪音正 profile 模型

假定 \(q_i=\theta_{z_i}\)，K 种正向量 θ_a 两两不同，每类非空。若 profile θ_a 已选，则其自身类别每个节点的饱和 score 已被达到。重复 θ_a 的 gain 为零。未选 θ_b 对其自身类的每个节点给出

\[
\ell_i(\theta_b)-\max_{a\in S}\ell_i(\theta_a)
=\min_{a\in S}D_{\rm Pois}(\theta_b\Vert\theta_a)>0.
\]

所以 greedy 必选尚未出现的 profile，K 步选齐所有 profile，随后 hard score 分类恢复分组。此为抽象无噪音 profile 命题；它不证明经验随机谱 profile 的稳定性，也不忽略无自环 SBM 的对角差异。

### A.3 工作 EM 的 objective 收敛

工作 objective 为

\[
\mathcal L(\omega,\mu)
=\sum_i\log\sum_a\omega_a e^{\ell_i(\mu_a)}.
\]

精确 E-step 和 M-step（包含声明的 simplex floor）通过标准 variational lower bound 使它不下降。每个正 profile 的饱和 score \(\ell_i(q_i)\) 上界所有 \(\ell_i(\mu_a)\)，且 weights 总和为 1，因此

\[
\mathcal L(\omega,\mu)\le\sum_i\ell_i(q_i)<\infty.
\]

故 objective 数值收敛；µ 更新是正 profile 的加权均值。此结论不证明迭代达到全局最大值或真分区，也不直接证明整个参数序列收敛。替换接受准则只保证同一工作 objective 不下降。

## Appendix B 及后续证明布局

B：人口 signal span 与 LLR contrasts；无自环对角修正；Bernoulli-to-Poisson 稀疏修正。

C：经验 eigenspace 的 rowwise/LOO 分析；谱投影、clip 和参数误差在相关尾部尺度上的控制。

D：初始化进入合适区域、交替更新的统计稳定性；若只能得到 conditional theorem，明确条件而不藏在证明末尾。

E：Chernoff 指数转移、成对风险和全局风险、exact recovery 推论。

F：低密度 regularization、权重补偿和适用的恢复目标。

G：实验协议、模型矩阵、全部分支、负面结果、时间/内存成本。

这些 appendix 的主统计证明仍需补齐。先写清楚 theorem 的目标与条件，再逐个补证明；不把优化问题的有限样本修复等同于 nearly-optimal statistical theorem。

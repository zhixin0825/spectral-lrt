# Spectral likelihood community detection for the SBM

**当前论文（2026-10-18）：** [编译 PDF](paper/current/main.pdf) · [完整 TeX（含证明）](paper/current/main.tex) · [模型与算法](paper/current/README.md) · [可编译源文件包](paper/current/latex-source.zip)。

全文以 oracle likelihood-ratio test 为起点，保留 **degree 和 spectral information**。先介绍未正则化、`p = Ω(log n/n)` 的容易情形并证明 sharp oracle rate；随后在 Section 4 介绍保留 `(D,U,Lambda)` 的一般算法，包括低于 logarithmic degree 的情形。该一般算法的精确代数恒等式和 sharp-rate 转移命题已写出；对完整正则化算法逐项验证稀疏统计条件仍待完成。重构 counts 不必恢复原始整数，也不声称 risk ratio 趋于一。

容易情形的无条件 sharp 主定理覆盖**任意固定 K**、密度 **`p = Ω(log n/n)`**，其中 `p = max_ab P_ab`，`P` 是实际 Bernoulli block probability matrix。允许社区大小和期望 degree 不同，也允许中间密度和 dense probabilities。不再固定 `(n/log n)P`，`P/p` 和社区比例也不必收敛。

常数倍 oracle 保证的统一条件为：`n_a >= n/(beta*K)`，其中 `beta >= 1` 是通常的社区不平衡参数，所有 `a,b` 均有 `P_ab >= kappa*p`，`sigma_min(P) >= kappa*p`，`p <= 1-eta`，以及 `p = Ω(log n/n)`。这里 `beta`、`kappa`、`eta` 是固定常数。这些条件使每个节点的期望 degree 均为 `Θ(np)`，因而至少为 `Ω(log n)`；期望 degree 可以因社区而异。主算法的固定 floor 系数还需满足 `0 < c0 < min(kappa,eta,1)/8`。信号条件不能由 degree 下界单独替代。

## 容易情形：未正则化算法与 sharp 定理

Algorithm 1 保留原始 adjacency 的 top-|lambda| K 个 eigenpairs，然后只使用这些谱信息：

1. 用谱估计的尺度设置双侧 clipping，形成连接 profiles。
2. 以完整 Bernoulli working score 做 global-mean growing。
3. 至多 `ceil(log n)` 轮 component replacement：每轮检查全部 K 个删除位置，对每个位置按 all-node likelihood gain 选新节点 profile，以均匀权重启动有限 EM；用同一个 working likelihood 与旧 fit 比较。
4. 强制最后一次 E/M 更新，再做 likelihood classification。

**主定理是这套明确修订后的完整算法的定理。** 任意有限 EM 预算均可，每次 fit 至少一次 M-step；在一整轮无改进时可提前停止。未声称 unchanged growing-only 在任意 K 下已证明，也没有要求随机 restart 或 K-means warm start。

全文用 $\mathcal R_{\mathrm{oracle}}$ 表示最大 equal-prior pairwise oracle error，写全 `oracle`。Algorithm 1 的无条件保证为 **`E Mis <= C R_oracle`**。定义是 `max_(i,a!=b) (1/2) sum_x min(phi_ia(x),phi_ib(x))`；Zhou–Li 使用的 total variation affinity 是该 binary error 的两倍。在上述可比概率条件下，`R_oracle` 等价于 `exp(-I_n)/sqrt(np)`，其中 `I_n` 是精确 incident-edge Bernoulli Chernoff information 的最小值。因此结论保留 Chernoff exponent 和 `1/sqrt(np)` 因子，允许常数倍损失；不声称 risk ratio 趋于一或新的全局 minimax 下界。Exact recovery 写为 `Pr(Mis > 0) <= C n R_oracle`，条件为 **`n R_oracle -> 0`**。

证明首先用确定性的 `1-1/K` replacement contraction 得到目标函数证书。该证书使算法已有的最后 E-step 具有 `O(1/(np))` 的 responsibility error，随后已有的 M-step 即达到 sharp-rate 所需精度。Leave-one-out arrowhead 展开把谱统计量化为独立 incident edges 的线性统计量加有界余项；投影权重的 conditional log-MGF 仅改变常数量级，clipping 的有限分支和 Chernoff tilt 下的局部集中保留 `1/sqrt(np)` 因子。谱事件失败概率为任意固定 `H` 下的 `C_H exp(-H np)`。算法不需要新增 refinement step，也不假设 clipped counts 独立。

节点到社区的 edge counts 统一记为 `b_ik(e)`，与 Amini–Chen–Bickel–Levina (2013) 一致。Planted-partition 特例也直接使用上述 sharp oracle error 保证，保留相应的 polynomial factor。

## 一般情形：degree 与 regularized spectrum

Section 4 恢复一般算法。令 `D=A*1`，`Dbar=mean(D)`，`W_ii=min(1,2*Dbar/D_i)`（`D_i=0` 时取 1），保留 `W A W` 的 top-|lambda| K 个 eigenpairs 和原始 degree 向量。压缩后只使用 **`(D,U,Lambda)`**。

精确恒等式为 `A(WU)=W^-1 U Lambda` 和 `A*1=D`。令

`X=[1,WU], Y=[D,W^-1 U Lambda], R=Y pinv(X)=A Proj(col(X))`。

则 **`R*1=D`**，对任意当前分组 `e`，重构 block counts 为 `R Z(e)`，总和恰好是每个节点的原始 degree。正文也恢复此前按社区中心反解、再按 degree 校准的 block-count 公式。真实社区空间若完全包含于保留空间，则 counts 恰好恢复；经验近似不作 rounding 保证。

Algorithm 2 对固定的实数重构 `R` 使用完整 Bernoulli row criterion。初始化、replacement、E/M 与分类均使用同一个 criterion；不引入 K-means initializer。重构数据不 clipping、不对称化，避免破坏精确 degree identity；候选和拟合概率才 clipping，M-step 为 clipped weighted mean。

- **实现：** [DegreeSpectralLikelihood.decode()](experiments/20261018_degree_spectral/degree_spectral_likelihood.py)。
- **验证：** [可复现检查](experiments/20261018_degree_spectral/verify_degree_spectral.py) 与 [检查结果](experiments/20261018_degree_spectral/CHECKS.json)。只验证保留信息的代数恒等式和 EM 更新，不作为稀疏统计 rate 的实验结论。
- **理论范围：** 一般算法本身不要求 `np >= c log n`。正文的转移命题在明确的 leave-one-out affine-score、conditional cumulant 和 tilted local concentration 条件下给出同样的 **`E Mis <= C R_oracle`**；完整正则化算法在任意 `np -> infinity` 下满足这些 sharp 条件尚未证明。未把容易情形的定理直接套到此算法，也未恢复较弱的 leading-exponent rate 声明。

## 实现与证据

- **Algorithm 1 入口：** [SpectralLikelihood.decode()](experiments/20261010_general_density/spectral_likelihood.py)，默认完整 Bernoulli score；原始 adjacency 不做 degree regularization。
- **60 图完整审计：** [报告、代码和数据](experiments/20261010_general_density/)。覆盖 K=2–6、两档密度、三类矩阵、两颗 seed。原 A、P、U、Lambda、全部 candidate scores、EM traces 和输出均保留。
- **结果边界：** 此 60 图中 replacement 没有改变任何一张的误分，weak general 模型平均误分约 0.66。不能把实现检查写成普遍的 accuracy 优势。
- **严格 population 反例：** [K=5 有限步 growing 失败及 replacement 修复](experiments/20261010_general_density/population_audit/)。每阶段一次 M-step 时 limiting error 为 13/200；一轮 replacement 修复。它不证明 fully converged growing 或 sampled graph 必然失败。
- **历史 240 图：** [原 sparse-score 实验](experiments/20261009_residual_likelihood_seeding/)。其 growing 平均误分 0.0946%、exact recovery 163/240，对照谱 K-means 为 0.3194%、139/240。它与当前 full-score 60 图不是同一实验。

## 已合并的稀疏版一般 K 修订

远端 `452ca1d` 的一般 K sparse-score 证明、threshold-certificate 实现及 [24图检查](experiments/20261009_residual_likelihood_seeding/general_k_fresh_checks.json) 已一并保留。该版本已经覆盖 expected degree Ω(log n)，但要求 `p -> 0`。它在 `residual_seed.py` 中的默认入口为 `global_gain_certified`；随机 residual starts 的入口为 `global_gain_restarts`。本页的完整 Bernoulli 算法使用独立的 `SpectralLikelihood.decode()` 入口，进一步覆盖 dense probabilities，并采用直接的软损失几何收缩。

2026-10-18 修订将 standalone sharp-rate 证明整合进 `paper/current/main.tex`，并同步重写摘要、引言、LRT/谱投影解释、主定理、证明和结论。先前的单独证明笔记保留在 [paper/notes/spectral_sharp_rate_20261018.pdf](paper/notes/spectral_sharp_rate_20261018.pdf) 作为推导记录；当前论文以 `paper/current/main.tex` 和同目录的 `main.pdf` 为准。

## 归档入口

| 内容 | 入口 |
| --- | --- |
| 当前论文、全密度一般 K 证明及构建信息 | [paper/current](paper/current/) |
| 较早 random-residual safeguard 实现检查 | [20261010_certified_likelihood](experiments/20261010_certified_likelihood/) |
| 原 growing / residual likelihood 初始化 | [20261009_residual_likelihood_seeding](experiments/20261009_residual_likelihood_seeding/) |
| 历史 partial-peeling 和失败分析 | [20261009_partial_likelihood_peeling](experiments/20261009_partial_likelihood_peeling/) |
| 历史固定谱特征、随机初始化与 ++ 比较 | [20261009_spectral_em](experiments/20261009_spectral_em/) |
| 更早的 720 图 degree 与 normalized-spectrum 实验 | [20261009_likelihood_initialization](experiments/20261009_likelihood_initialization/) |
| 较早理论草稿 | [paper/previous_draft](paper/previous_draft/) |

历史报告反映各自阶段的算法和理论状态。接续研究以本页和当前论文为准。






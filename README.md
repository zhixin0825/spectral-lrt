# Spectral likelihood community detection for the SBM

**当前论文（2026-10-18）：** [编译 PDF](paper/current/main.pdf) · [完整 TeX（含证明）](paper/current/main.tex) · [模型与算法](paper/current/README.md) · [可编译源文件包](paper/current/latex-source.zip)。

全文以 oracle likelihood-ratio test 为起点：LRT 需要节点到各社区的 edge counts；经验谱投影得到的实数 counts 不必精确还原原始整数，但对相关 likelihood tails 的扰动足够小，使误分率保持在 oracle rate 的常数倍内。这里不把 rate preservation 称为精确充分性，也不声称两个错误概率的比值趋于一。

当前主定理覆盖**任意固定 K**、密度 **`p = Ω(log n/n)`**，其中 `p = max_ab P_ab`，`P` 是实际 Bernoulli block probability matrix。允许社区大小和期望 degree 不同，也允许中间密度和 dense probabilities。不再固定 `(n/log n)P`，`P/p` 和社区比例也不必收敛。

常数倍 oracle 保证的统一条件为：`n_a >= n/(beta*K)`，其中 `beta >= 1` 是通常的社区不平衡参数，所有 `a,b` 均有 `P_ab >= kappa*p`，`sigma_min(P) >= kappa*p`，`p <= 1-eta`，以及 `p = Ω(log n/n)`。这里 `beta`、`kappa`、`eta` 是固定常数。这些条件使每个节点的期望 degree 均为 `Θ(np)`，因而至少为 `Ω(log n)`；期望 degree 可以因社区而异。主算法的固定 floor 系数还需满足 `0 < c0 < min(kappa,eta,1)/8`。信号条件不能由 degree 下界单独替代。

## 已完成的算法与定理

Algorithm 1 保留原始 adjacency 的 top-|lambda| K 个 eigenpairs，然后只使用这些谱信息：

1. 用谱估计的尺度设置双侧 clipping，形成连接 profiles。
2. 以完整 Bernoulli working score 做 global-mean growing。
3. 至多 `ceil(log n)` 轮 component replacement：每轮检查全部 K 个删除位置，对每个位置按 all-node likelihood gain 选新节点 profile，以均匀权重启动有限 EM；用同一个 working likelihood 与旧 fit 比较。
4. 强制最后一次 E/M 更新，再做 likelihood classification。

**主定理是这套明确修订后的完整算法的定理。** 任意有限 EM 预算均可，每次 fit 至少一次 M-step；在一整轮无改进时可提前停止。未声称 unchanged growing-only 在任意 K 下已证明，也没有要求随机 restart 或 K-means warm start。

主定理采用 Zhou–Li 的 affinity 记号：**`E Mis <= C eta*`**，其中 `eta*` 是各 binary product-Bernoulli laws 的最大 total variation affinity（是 equal-prior oracle risk 的两倍）。在可比概率条件下，它等价于 **`C exp(-I_n)/sqrt(np)`**。其中 `I_n` 是精确 product-Bernoulli oracle Chernoff information 的最小值。在上述统一条件下，Zhou–Li 的 binary oracle scale 为 `exp(-I_n)/sqrt(np)`；因此结论保留 Chernoff exponent 和 `1/sqrt(np)` 因子，允许常数倍损失。它强于旧版的 leading-exponent bound，但不声称 risk ratio 趋于一或新的全局 minimax 下界。`n exp(-I_n)/sqrt(np) -> 0` 时得到 exact recovery，特别包含 `liminf I_n/log n > 1`。

证明首先用确定性的 `1-1/K` replacement contraction 得到目标函数证书。该证书使算法已有的最后 E-step 具有 `O(1/(np))` 的 responsibility error，随后已有的 M-step 即达到 sharp-rate 所需精度。Leave-one-out arrowhead 展开把谱统计量化为独立 incident edges 的线性统计量加有界余项；投影权重的 conditional log-MGF 仅改变常数量级，clipping 的有限分支和 Chernoff tilt 下的局部集中保留 `1/sqrt(np)` 因子。谱事件失败概率为任意固定 `H` 下的 `C_H exp(-H np)`。算法不需要新增 refinement step，也不假设 clipped counts 独立。

## 可为零的 block probabilities

新增 Theorem **Leading-exponent guarantee without a minimum block probability** 去掉 `P_ab >= kappa*p`，允许任意小的概率乃至 `P_ab=0`。保留固定 K、`n_a >= n/(beta*K)`、`sigma_min(P)>=kappa*p`、`p<=1-eta` 与 `np>=c log n`。只将谱尺度 `s` 决定的 clipping floor 改为

`epsilon = c0*(s/n)*exp(-sqrt(log(e+s)))`.

相同的 growing、replacement 与最后 E/M 步骤给出

**`E Mis <= exp(-I_n + o(np)) = exp(-(1-o(1))*I_n)`**.

这是常见的 leading Chernoff exponent，不把原有 `1/sqrt(np)` 因子搬到边界概率情形。当前 `SpectralLikelihood` 已有的 `floor` 参数接收乘在 `s/n` 上的系数，因而传入 `c0*exp(-sqrt(log(e+s)))` 即可；不改变核心 fitting engine。旧实验仍使用固定 floor，不能据此声称新选项已经过同一实验评估。

正文还加入了 homogeneous planted-partition 模型中常用的 Rényi exponent `n*I*/K`（平衡）以及相应不平衡情形。节点到社区的 edge counts 统一记为 `b_ik(e)`，与 Amini–Chen–Bickel–Levina (2013) 一致。

## 实现与证据

- **当前入口：** [SpectralLikelihood.decode()](experiments/20261010_general_density/spectral_likelihood.py)，默认完整 Bernoulli score；原始 adjacency 不做 degree regularization。
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



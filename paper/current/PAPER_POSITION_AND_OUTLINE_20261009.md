# 论文定位与六节写作方案

2026-10-09 修订。本文件对应当前 \`paper/current/manuscript.tex\`，取代同日较早的写作提纲。较早提纲中的 subset-first 初始化、1.5K 筛选和尚未建立的谱误差条件，不再定义当前正文算法或理论状态；历史内容保留在 Git 提交历史和实验原始记录中。

## 1. 定位和题目

**Spectral Likelihood-Ratio Decoding for Community Recovery in the Stochastic Block Model**

文章研究只保存邻接矩阵前 K 个 eigenpairs 后的社区 likelihood-ratio 解码。重点是实际全图谱重建如何保留 oracle 的领先错误指数，以及依赖同一张图的 profile 拟合和 EM 更新如何保持该指数。

Zhou–Li 已研究 optimal rates；正文引用其工作并参考其符号，不重述其更精细的 rate 或前因子。本文统计目标为 \(\exp\{-(1+o(1))I_n\}\)。不把领先指数的结论描述为更精细的风险等价。

## 2. 统一符号

采用已知、固定 K 的无向无自环 SBM。社区标签为 \(z_i\)，成员矩阵为 Z，社区比例为 \(\pi_a=n_a/n\)，\(P=\rho_n B\) 是 K × K block probability matrix。真正的 n × n 图均值为 \(\Omega\)，补对角后的比较矩阵为 \(\Omega^\circ=ZPZ^\top\)。

参考 Zhou–Li，真实 connection profile 为 \(p_{kj}=P_{k,z_j}\)，拟合 profile 为 \(\widehat p_{k*}\)。谱重建为 \(\widetilde A=U\Lambda U^\top\)，投影为 \(\Pi=UU^\top\)，输入工作观测为 \(q_{ij}=\max\{\widetilde A_{ij},c_0\rho_n\}\)。主理论要求 \(0<c_0<\min_{ab}B_{ab}\)。

\(D_\alpha(p_{a,-i}\Vert p_{b,-i})\) 是精确 Bernoulli Chernoff quantity，\(\alpha^\star\) 为最优参数，\(D^\star_{i,ab}=\max_{\alpha\in[0,1]}D_\alpha\)，\(I_n=\min_{i,a\ne b}D^\star_{i,ab}\)。主情形 \(\rho_n=\log n/n\)，\(d_n=n\rho_n\)，固定 B 满秩且元素与社区比例有正的常数上下界。允许负的信号特征值。

## 3. 六节正文结构

1. Introduction：说明问题、相关工作、指数目标、已证结论和准确的初始化边界。
2. Spectral likelihood-ratio decoding：定义谱 profile、工作 likelihood、growing gain 初始化、逐阶段 EM 和解码规则。
3. Main results：先给谱分数与 oracle 的关系，再给实际全图谱误差界、局部 EM 指数定理及有保证 warm-start 推论。
4. Sparse extensions：讨论常数平均度、正则化、degree correction 等扩展，明确哪些不属于主定理。
5. Experiments：用现有配对记录比较实际 growing 分支与谱 k-means baseline；历史冻结参数控制独立标明。
6. Conclusion：总结已证明的谱解码与 EM 保证，指出主 growing 初始化的剩余问题。

附录给出全部证明及实验实现对应关系。

## 4. 当前 Algorithm 1

第一组件由所有 q 行的经验均值初始化，不选首个节点 seed，也不筛 top subset。随后从全部尚未选过的节点中，按全局正 likelihood gain 增添一个组件。

每增加一个组件，都将该阶段 mixture weights 重置为均匀值，并在全体节点上运行联合 soft EM；拟合后的 means 用于下一次 gain 选择。直到达到 K 个组件后，再以最大 posterior score 解码。工作 likelihood 在每一固定组件数的 EM 阶段内单调；不宣称增加组件并重置权重后的跨阶段单调性。

本算法对应实际测试的 \`growing_global_gain_EM\` / \`Work.growing()\` / \`method='global_gain'\`。历史 \`global_gain_no_update\` 是 subset-first、冻结候选参数的控制，不能与当前主算法混写。冻结目标的 submodularity 或无噪声覆盖性质，也不能当成 growing EM 初始化的证明。

## 5. 当前统计结论及证明

### 5.1 实际全图谱重建

对每个固定多项式置信指数 H，谱事件的失败概率可控制为 \(n^{-H}\)。在该事件上，
\[
\max_i\|q_i-q_i^0\|_1=o(d_n),\qquad
\frac1{nd_n}\sum_i\|q_i-p_{z_i,*}\|_1=o(1),
\]
其中 \(q^0=\max\{A\Pi^\star,c_0\rho_n\}\) 仅为证明中的 oracle 比较量，主算法不使用真实标签。

证明检查 Lei–Rinaldo 的谱范数界及 Abbe–Fan–Wang–Zhong 的 entrywise 特征向量线性化条件，并处理无自环对角修正、正负信号子空间和 clipping。clipped oracle 分数通过独立块计数的 Chernoff moment 控制，不假设所有稀有错误节点的 clipping 误差逐点为 \(o(d_n)\)。

### 5.2 局部谱 EM

若初始 responsibilities 在共同标签置换下弱一致，且初始化失败概率 \(\delta_n\) 满足
\[
\liminf_n\frac{-\log\delta_n}{I_n}\ge1,
\]
则首次更新 profile 和 weights 后的解码达到
\[
\mathbb E\,\operatorname{Mis}(\widehat z,z)
\le \exp\{-(1+o(1))I_n\}.
\]
证明对弱一致盆地中的全部 responsibility matrices 统一成立，因此允许它们依赖同一张图。盆地不变性保证后续精确 EM 更新及有限自适应停止保持该指数。若 \(I_n/\log n\to J_\star>1\)，则得到 exact recovery。

### 5.3 有保证的 warm-start 变体

谱特征向量上的固定因子 k-means objective approximation 给出 \(O(1/d_n)\) 的弱误分率。以 \(O(\log n)\) 次独立 k-means++ 重启选最小 objective，可把初始化失败概率压至任意所需的固定多项式尺度；结合局部谱 EM 定理得到端到端指数推论。

该推论是一个独立、完全指定的算法变体。它不是主 growing 初始化的保证，也不是实验中普通十次 Lloyd baseline 的理论保证。

### 5.4 唯一尚未闭合的主流程环节

尚未证明主 Algorithm 1 的 growing likelihood 初始化以 oracle 指数尺度的可靠性进入上述弱一致盆地。因此，不能宣称主 growing 分支已有无条件 nearly-optimal theorem。谱重建、profile 拟合与后续 EM 的理论环节已有证明；主 growing 初始化需另行分析 population dynamics 及高概率扰动。

## 6. 实验对应关系

保留现有 240 图的两批配对记录，不将当前修订说成新增仿真。实际 growing 分支的平均错误率约 0.0946%，exact recovery 为 163/240；配对谱 k-means baseline 约 0.3194%，exact recovery 为 139/240。分模型表和机器可读 CSV 都使用实际 growing 方法名。

历史控制、替换组件实验、其他 profile normalization 与 degree correction 结果不冒充当前主算法结果。有限模型网格上的表现不能证明一般 SBM 的初始化一致性或最优指数。

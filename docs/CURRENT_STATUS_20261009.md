# 当前研究状态（更新至 2026-10-10）

此文件保留原日期的入口名，接续工作以 [当前论文](../paper/current/README.md)、[全文审计](../paper/current/REVIEW_20261010.md) 和 [完整证明说明](../paper/current/END_TO_END_20261010.md) 为准。

当前主定理已经覆盖任意固定 K、密度 `p = Ω(log n/n)`，其中 `p = max_ab P_ab`，`P` 是实际对称 Bernoulli 概率矩阵。统一条件为 `n_a >= pi0*n`、`P_ab >= kappa*p`、`sigma_min(P) >= kappa*p`、`p <= 1-eta` 和上述密度下界，常数固定且为正。每个节点的期望 degree 为 `Θ(np)`，允许不等 degree 和一般密度；`P/p` 与社区比例不必收敛。

算法使用原始 A 的 top-|lambda| K 个 eigenpairs，之后只依赖谱重构。完整 Bernoulli 工作评分加上 growing、重复 component replacement 和最后一次强制 E/M，达到 `E Mis <= exp(-(1+o(1)) I_n)`。确定性 replacement 的 `1-1/K` 收缩补上 general-K 初始化证明；概率部分已使用 `exp(-H np)` 的谱置信度。

原 growing-only 的 unrestricted general-K 保证仍未证明。K=5 每阶段一次 M-step 的 population 路径已有严格反例，一轮 replacement 修复；不能把它外推为 fully converged growing 或 sampled-graph 必然失败。

新 [60 图审计](../experiments/20261010_general_density/) 完整保留原图、谱信息、候选损失和轨迹。Replacement 在该集合没有改变误分，weak general 模型有明显失败。旧 240 图 sparse-score 结果、720 图 degree/normalized-spectrum 结果属于各自的历史实验，不与新主算法混用。

下一阶段可研究 growing-only 的充分迭代行为、更大规模的有限样本表现、候选扫描的计算优化和更弱信号条件。当前完整谱算法保证以 `p = Ω(log n/n)` 为密度范围。

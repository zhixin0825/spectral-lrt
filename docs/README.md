# 研究状态与历史记录

从 [当前状态](CURRENT_STATUS_20261009.md) 或 [当前论文](../paper/current/README.md) 开始。当前结果为任意固定 K、密度 `p = Ω(log n/n)` 的完整 Bernoulli 谱 likelihood 算法，其中 `p = max_ab P_ab`。它包含确定性的 repeated component replacement 和 leading Chernoff exponent 证明。

模型条件为 `n_a >= pi0*n`、`P_ab >= kappa*p`、`sigma_min(P) >= kappa*p`、`p <= 1-eta` 和上述密度下界，常数固定且为正。每个节点的期望 degree 为 `Θ(np)`，可以因社区而异；不要求 `P/p` 或社区比例收敛。

[本轮完整证明说明](../paper/current/END_TO_END_20261010.md) 解释 general-K 收缩、全密度谱置信度和最终 EM 更新。[60 图验证与严格 population 反例](../experiments/20261010_general_density/) 给出当前实现的完整证据与失败边界。

本目录其他日期文件是早期 degree 加谱信息、partition likelihood 和初始化实验的历史记录。它们保留当时的想法与未完成状态，不代表当前定理的范围。

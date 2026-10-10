> Historical record. Superseded for current model, algorithm and theorem scope by [README.md](README.md) and [REVIEW_20261010.md](REVIEW_20261010.md). The original content below is retained as a dated record.

# 当前论文结构与证明状态

2026-10-10 更新，取代此前提纲。论文题目为 Nearly Optimal Community Recovery by Spectral Likelihood Decoding。

六节结构保持：Introduction；算法；Main results；低密度扩展；Experiments；Conclusion。附录给出完整证明和实现对应关系。

## 当前主算法

仅用 U、Lambda 构造正谱 profile q。保留全局均值起步、最大 all-node gain、逐阶段 EM 的 growing fit。一般固定 K 下，另外做 ceil(log n) 次独立 likelihood-residual++ 初始化：首节点均匀选，其余按当前最小 Poisson deviance 抽取；全部 K seeds 选齐后才运行 joint EM。按相同 working mixture likelihood 选最好 fit，强制最后一次 E/M，再解码。

这段 safeguard 是明确的新算法步骤。没有 top-subset 或 1.5K 参数，没有 k-means warm start。原 growing 分支仍可独立调用。

## 已证明的最终结果

固定已知 K、固定正对称满秩 (n/log n)P、正社区比例和 admissible floor 下：

- 原 growing 分支在 K=2 时端到端达到 E Mis <= exp(-(1+o(1)) I_n)，包含实际有限停止规则。
- 上述带 likelihood safeguard 的完整算法对每个固定 K 达到同一指数。
- liminf I_n/log n>1 时 exact recovery；预设多项式更新次数时算法为多项式时间。
- 完整算法的 theorem 不假设 weak initialization；其可靠性由 residual sampling、重复及相同 likelihood 选优建立。

已有全图经验谱近似、clipped Chernoff、局部谱 EM 与盆地不变性均作为完整 theorem 的证明工具。

一般 K、去掉 safeguard 的原 growing 分支仍未证明；这不作为当前完整算法 theorem 的假设。没有得到该分支严格反例，不能声称它不可能。

## 符号与实验

参考 Zhou–Li：P为K×K block probability，p_{kj}=P_{k,z_j}，Mis错分率，D_alpha Chernoff quantity；graph mean 直接写 E[A|z]。I_n定义为最难对的精确信息量，只写 leading exponent，不重述 refined optimal rate。

原240图只评价原growing和历史control/repair。新24例仅验证完整组合的objective筛选和final M-step，不能冒充完整240图重跑或finite-sample性能保证。

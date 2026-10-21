# Spectral likelihood community detection for the SBM

## 当前文章：普通 SBM 的 adjacency 版本（2026-10-21）

| 文稿 | 内容 | 文件 |
| --- | --- | --- |
| **第一篇：普通 SBM，22 页** | Adjacency spectrum + 原始 degree；独立的算法、主定理与完整证明 | [PDF](paper/sbm/sbm_spectral_likelihood.pdf) · [源文件包](paper/sbm/sbm_spectral_likelihood_source.zip) · [说明](paper/sbm/README.md) |
| **第二篇：补充工作笔记，59 页** | DC-SBM 的 normalized 版本、其他普通 SBM 算法、推导、反例与历史审查记录 | [PDF](paper/companion/spectral_likelihood_companion.pdf) · [源文件包](paper/companion/spectral_likelihood_companion_source.zip) · [材料索引](paper/companion/README.md) |

当前正则化主定理保持 **np → ∞、np² ≤ C₁**，没有 **np ≥ c log n** 的下限。第一篇覆盖任意固定 K、可比社区大小与一般正、有界、满秩 block profiles，证明实际平均误分风险 **E loss ≤ C R_SBM,***，保留精确 Bernoulli Chernoff information 和 **d^(-1/2)** 因子。

第一篇的算法保留原始 degree 和高程度截断 adjacency 矩阵 **A_tilde = W A W** 的一个 eigensystem。阈值 **tau = b0 × 平均度数**，其中 b0 是足够大的固定常数；D_i ≤ tau 时 W_i = 1，超出时 W_i = sqrt(tau / D_i)。截断未触发时，保留的就是原始 A 的特征对。算法使用精确 degree correction、hollowing、radius-count initializer 与普通 Bernoulli block fitting/LRT；完整非边截距保留，目标 degree 不 profiling。

第一篇正文恢复 U、Lambda、F 的谱记号，用一条逐行公式定义保留原始度数的 A_hat；G、H、Q 等辅助量仅用于证明附录。算法和定理范围不变。

第二篇保留当前无 log 下限的 normalized DC-SBM 结果，并在本篇内完整证明其 small-floor geometry 与高程度范围的概率闭合；共享工具的引用已同步更新。此前普通 SBM 的 normalized 源稿保存在 [源文件快照](paper/companion/archive/ordinary_normalized_20261020/)。第二篇仍为补充工作材料集。

原来的 [68 页合并证明（revision 3）](paper/notes/normalized_oracle_risk_20261020/normalized_spectral_oracle_risk_20261020.pdf) 及其 [源文件包](paper/notes/normalized_oracle_risk_20261020/normalized_spectral_oracle_risk_20261020_source.zip) 保留归档。10 月 18 日正文和实验说明已移至补充材料中的 [历史记录](paper/companion/archive/repository_history_20261018.md)；其中关于旧 Algorithm 2 的未完成项不描述当前主定理。

## 历史实现与实验入口

- [10 月 18 日正文、原算法与实验说明](paper/companion/archive/repository_history_20261018.md)
- [原始 degree-spectrum 实现及其代数验证](experiments/20261018_degree_spectral/)
- [完整 Bernoulli score 的 60 图审计](experiments/20261010_general_density/)
- [较早稀疏 score 的 240 图实验](experiments/20261009_residual_likelihood_seeding/)

历史实验按各自算法解释；当前两篇的定理范围以对应正文为准。

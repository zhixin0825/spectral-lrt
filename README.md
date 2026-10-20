# Spectral likelihood community detection for the SBM

## 当前文章：普通 SBM 与补充材料分开（2026-10-20）

| 文稿 | 内容 | 文件 |
| --- | --- | --- |
| **第一篇：普通 SBM，21 页** | 独立的模型、算法、主定理与完整证明；不依赖第二篇 | [PDF](paper/sbm/sbm_spectral_likelihood.pdf) · [源文件包](paper/sbm/sbm_spectral_likelihood_source.zip) · [说明](paper/sbm/README.md) |
| **第二篇：补充工作笔记，57 页** | DC-SBM、第一篇未纳入的其他普通 SBM 算法、推导、反例与历史审查记录 | [PDF](paper/companion/spectral_likelihood_companion.pdf) · [源文件包](paper/companion/spectral_likelihood_companion_source.zip) · [材料索引](paper/companion/README.md) |

当前正则化主定理保持 **np → ∞、np² ≤ C₁**，没有 **np ≥ c log n** 的下限。第一篇覆盖任意固定 K、可比社区大小与一般正、有界、满秩 block profiles，证明实际平均误分风险 **E loss ≤ C R_SBM,***，保留精确 Bernoulli Chernoff information 和 **d^(-1/2)** 因子。

第一篇的算法保留原始 degree 和一个 degree-floor normalized eigensystem，使用精确 degree correction、hollowing、radius-count initializer 与普通 Bernoulli block fitting/LRT；完整非边截距保留，目标 degree 不 profiling。第二篇先列当前无 log 下限的 DC-SBM 结果，再收纳较早算法及其各自适用条件。第二篇暂为工作材料集，不要求统一成篇。

原来的 [68 页合并证明（revision 3）](paper/notes/normalized_oracle_risk_20261020/normalized_spectral_oracle_risk_20261020.pdf) 及其 [源文件包](paper/notes/normalized_oracle_risk_20261020/normalized_spectral_oracle_risk_20261020_source.zip) 保留归档。10 月 18 日正文和实验说明已移至补充材料中的 [历史记录](paper/companion/archive/repository_history_20261018.md)；其中关于旧 Algorithm 2 的未完成项不描述当前主定理。

## 历史实现与实验入口

- [10 月 18 日正文、原算法与实验说明](paper/companion/archive/repository_history_20261018.md)
- [原始 degree-spectrum 实现及其代数验证](experiments/20261018_degree_spectral/)
- [完整 Bernoulli score 的 60 图审计](experiments/20261010_general_density/)
- [较早稀疏 score 的 240 图实验](experiments/20261009_residual_likelihood_seeding/)

历史实验按各自算法解释；当前两篇的定理范围以对应正文为准。

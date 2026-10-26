# Spectral likelihood community detection for the SBM

## 当前源稿：普通 SBM，先简单后一般，Zhou–Li 符号统一（2026-10-26）

[当前正文源稿](paper/sbm/article.tex) · [完整源稿说明](paper/sbm/README.md) · [本次符号与完整性核验](paper/sbm/NOTATION_AUDIT_20261026.md)

本次只统一符号及规范化固定参数尺度。第一篇保持先描述不额外使用 degree 的 spectrum-only 方法，再完整描述 degree-plus-spectrum 的一般算法。模型改用 Zhou–Li 的 beta、omega、p*=max P_ab，以及作者指定的 sigma_K(P)。pi_-、c_-、c_+ 和单独命名的 kappa 不再出现在当前 TeX 中。两个算法整节、oracle 定义和13项关键公式不变，七份证明全部保留。

一般版主定理仍为 **n p* → ∞、n(p*)² ≤ C₁** 下的 **E Mis(z_hat,z) ≤ C R_oracle**，没有 **n p* ≥ c log n** 下限。归一化到最大概率 p* 会重标固定常数，不删除正概率可比性和信号非退化假设。R_oracle 仍为真实参数、原始 incident row 的实际平均分类错误概率，不换成 affinity 或仅指数率。

**编译文件区别：**26日的25页PDF和配套源文件包已作为对话附件生成。本次更新仓库中的源稿和核验记录，尚未替换下表中的历史PDF/ZIP二进制文件；不能把它们当作26日编译版。

| 文稿 | 历史编译快照 | 当前源稿 |
| --- | --- | --- |
| 第一篇：普通 SBM | [21日22页PDF](paper/sbm/sbm_spectral_likelihood.pdf) · [旧源文件包](paper/sbm/sbm_spectral_likelihood_source.zip) | [正文](paper/sbm/article.tex) · [说明](paper/sbm/README.md) |
| 第二篇：补充工作笔记 | [59页PDF](paper/companion/spectral_likelihood_companion.pdf) · [源文件包](paper/companion/spectral_likelihood_companion_source.zip) | [材料索引](paper/companion/README.md) |

第一篇完整算法保留原始 degree 和高程度截断 adjacency 矩阵 A_tilde=WAW 的一个 eigensystem。tau=b0×平均度数；D_i≤tau 时 W_i=1，否则 W_i=sqrt(tau/D_i)。后续保留原尺度重构、精确 degree correction、hollowing、radius-count 初始化、对称 Bernoulli block fitting 和完整非边 LRT。前面的简单版不自动继承一般版的定理。普通 SBM 的 degree correction 及所需证明仍留在第一篇。

第二篇的 DC-SBM normalized 版本、其他算法、反例、推导和历史审查材料本轮未修改。此前普通 SBM normalized 源稿保存在[源文件快照](paper/companion/archive/ordinary_normalized_20261020/)。原[68页合并证明](paper/notes/normalized_oracle_risk_20261020/normalized_spectral_oracle_risk_20261020.pdf)及[源文件包](paper/notes/normalized_oracle_risk_20261020/normalized_spectral_oracle_risk_20261020_source.zip)继续归档。

## 历史实现与实验入口

- [10月18日正文、原算法与实验说明](paper/companion/archive/repository_history_20261018.md)
- [原始 degree-spectrum 实现及代数验证](experiments/20261018_degree_spectral/)
- [完整 Bernoulli score 的60图审计](experiments/20261010_general_density/)
- [较早稀疏 score 的240图实验](experiments/20261009_residual_likelihood_seeding/)

历史实验按各自算法解释；各篇定理范围以其正文为准。符号与编译核验不等于重新独立审定全部数学证明。

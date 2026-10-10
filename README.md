# Spectral likelihood decoding for the SBM

**当前论文（2026-10-10）：** [编译 PDF](paper/current/manuscript.pdf) · [源码与模型条件](paper/current/README.md) · [完整证明](paper/current/theory.tex) · [全文检查和修正](paper/current/REVIEW_20261010.md)。

主模型直接使用 Bernoulli block matrix `P`，假定 `(n/log n)P` 是固定正满秩对称矩阵。允许社区大小和期望 degree 不同。谱信息来自原始 adjacency 的 top-|lambda| K 个 eigenpairs。拟合和分类只使用这些 eigenpairs。

**已证明的范围：** 原 global-mean growing 方法在 K=2 达到 oracle Chernoff leading exponent。一般固定 K 的定理对应完整算法：保留 growing fit，加入 `ceil(log n)` 次独立 likelihood-residual 初始化，按同一个谱工作似然选优，最后强制一次 E/M 更新。保证为 `E Mis <= exp(-(1+o(1)) I_n)`；当 `liminf I_n/log n > 1` 时 exact recovery。原 growing 方法的一般 K 保证仍待证明。

**必须保留的条件：** 数值实现使用 `q=max(U Lambda U^T,1e-4 log n/n)` 以定义候选评分中的对数。现证明要求 `min_ab nP_ab/log n > 1e-4`。正值截断发生在谱重建之后，原始 adjacency 没有被修改。这里不声称精确图似然最大化或新的全局 minimax 下界。

**算法入口：** `method='global_gain_certified'` / `Work.certified_growing()` 是上述完整算法；`method='global_gain'` 是原 growing 分支；默认 `method='repair'` 是历史替换分支。当前算法没有 top-fraction 或 1.5K 参数。

**原 240 图实验：** growing 平均误分率 0.0946%，exact recovery 163/240；谱 K-means baseline 为 0.3194%、139/240。这批结果未评估新完整算法。[24 个新实现检查](experiments/20261010_certified_likelihood/) 验证选优和最后更新，并非误分率优势证明。

## 归档与复现入口

| 内容 | 入口 |
| --- | --- |
| 原 growing 与 residual likelihood 初始化的 240 图比较 | [实验目录](experiments/20261009_residual_likelihood_seeding/) |
| 历史 partial-peeling 控制和失败分析 | [实验目录](experiments/20261009_partial_likelihood_peeling/) |
| 历史固定谱特征拟合、随机初始化与 ++ 比较 | [实验目录](experiments/20261009_spectral_em/) |
| 更早的 720 图 degree 与 normalized-spectrum 实验 | [实验目录](experiments/20261009_likelihood_initialization/) |
| 原始逐图输出与校验 | [正式记录](experiments/20261009_likelihood_initialization/final/archives/) |
| 较早理论草稿，仅供历史参考 | [旧稿目录](paper/previous_draft/) |
| 当前新增端到端证明 | [说明](paper/current/END_TO_END_20261010.md) |

历史报告中的算法、输入信息和未完成状态属于各自阶段，不能代替当前稿件的条件和结论。原实验数据、代码和归档保持原样。后续工作从当前论文与 REVIEW_20261010.md 接续。

# 固定谱特征上的 likelihood 交替更新：已有 240 图与独立 120 图复验

更新：2026-10-09。当前问题是：在 $B=C\log(n)/n$、已知 $K$ 的 SBM 中，只给谱信息和初始分区，像 k-means 一样交替更新参数与分组，能否收敛并改善恢复？

**实验答案是：合适的固定特征工作 likelihood 可以稳定收敛，也能在部分 SBM 中明显改善合理的 k-means 起点；直接换成任意 likelihood 并不成立。精确 graph likelihood、精确逐节点 LOO 和全局最优性尚未得到验证。**

## 实际实现了什么

谱输入是原始邻接矩阵按特征值绝对值取前 $K$ 个 eigenpairs $(U,\Lambda)$。生成器只负责模拟图和计算谱；拟合函数不接收原始 $A$、degree 或真实分组。

先计算 $X=\sqrt n U\Lambda$，分别做十次重启的 U-k-means 和 X-k-means，按各自 Euclidean objective 选择起点。用 U-k-means 的每组谱均值构成行矩阵

$$
H_U[b,:]=\operatorname{mean}_{i:\hat z^U_i=b}(\sqrt n U_i),\qquad
Y=XH_U^{-1}.
$$

**将 $H_U$ 和 $Y$ 固定。** 实际分组的初值取 X-k-means。这里把“特征坐标的建立”与“分组的初值”分开了：前者用 U 分组，后者用 X 分组；两者都从相同谱信息得到，均不使用真值。

对实数 $Y$ 使用逐坐标 Poisson 加 Gaussian 噪声的模型：

$$
Y_{ib}=N_{ib}+\epsilon_{ib},\quad
N_{ib}\mid z_i=a\sim\operatorname{Pois}(\mu_{ab}),\quad
\epsilon_{ib}\sim N(0,1).
$$

Gaussian 噪声使负的谱特征也有合法密度。优化固定的 observed mixture likelihood

$$
\ell(\pi,\mu)=\sum_i\log\left[
\sum_a\pi_a\prod_b\sum_{m=0}^{\infty}
e^{-\mu_{ab}}\frac{\mu_{ab}^m}{m!}\phi(Y_{ib}-m)
\right].
$$

交替更新是标准 soft EM：E 步计算组别后验概率和潜在 counts 的后验均值，M 步更新组别权重和 rates，最后用最大组别后验输出分组。rates 和权重有固定下界；数值求和的支持、误差界和每轮 objective 都保留。算法没有逐轮重算 $H_U$，没有强制 SBM 的对称连接参数结构，也没有把上述工作模型当成谱特征的精确分布。

## 上一轮确实已经完成的实验

240 张独立模拟图：6 个 SBM、$n\in\{1000,2000\}$、最小 pairwise Chernoff-Hellinger divergence $\in\{0.8,1.4\}$，每条件 10 个 seeds 5000--5009，共 24 条件。$C$ 经过缩放，使上述 divergence 为指定值；这里的 0.8/1.4 是信号强度，非平均 degree。

原始比较及补充比较合计 16 方法、3840 条拟合/评估记录。最佳固定组合 `soft_Ucal_Xinit_tau1` 是原始实验诊断后添加的 follow-up，不能把它声称为在这批 240 张图前完全预指定的方法。固定 $\tau=1$，无真值调参。

| 方法 | 平均误分率 | exact recovery 图数 |
|---|---:|---:|
| X-k-means | 0.309375% | 132/240 |
| 参数从起点估计后第一次 likelihood 重新分组 | 0.121042% | 154/240 |
| 做一次 EM 参数更新 | 0.105417% | 157/240 |
| 交替更新至收敛 | 0.099583% | 158/240 |
| 直接 X full-Gaussian EM | 0.754583% | 139/240 |

优选组合 240/240 收敛，平均 8.68 轮、最多 60 轮。相对 X-k-means，61 图改善、173 图相同、6 图恶化。优势主要出现在 heterogeneous3 与 unequal3；很多其它条件已经接近或达到零错误，收益很小。

失败版本也保留了：在 heterogeneous3 的 $n=1000,CH=0.8$ 条件，U-k-means 起点的 Poisson EM 误分率为 47.97%；把 decoder 也改成 X 分组的中心后为 51.93%；固定 U decoder 加 X 分组起点则为 0.42%。可见 feature decoder 与初始盆地都重要。部分 X decoder 近乎奇异，是先前诊断中的具体失效机制。

## 本轮独立复验

看过上一批结果后，先冻结上述 recipe，再生成 seeds 20000--20004 的新图。模型、$n$ 和 CH 网格相同，每条件 5 图，总计 120 图；所有六个比较臂都报告。没有根据新图结果调参、选组或选重启。

| 方法 | 平均误分率 | exact recovery 图数 |
|---|---:|---:|
| X-k-means | 0.345417% | 73/120 |
| 第一次 likelihood 重新分组 | 0.117083% | 85/120 |
| 一次 EM 参数更新 | 0.105833% | 85/120 |
| 交替更新至收敛 | **0.099167%** | **86/120** |
| 直接 X full-Gaussian EM | 0.535833% | 76/120 |

收敛 EM 为 120/120，平均 8.125 轮、最多 40 轮；没有 objective 下降。相对 X-k-means，30 图改善、87 图相同、3 图恶化。平均误分率降低 0.24625 个百分点；在固定 24 条件内逐图配对 bootstrap 的描述性 95% 区间是 [0.2025,0.295427] 个百分点。这不是对所有 SBM 的统一保证。

| 新种子代表条件 | X-k-means | 固定特征 Poisson EM |
|---|---:|---:|
| heterogeneous3, n=1000, CH=0.8 | 2.88% | 0.42% |
| heterogeneous3, n=2000, CH=0.8 | 2.26% | 0.35% |
| heterogeneous3, n=1000, CH=1.4 | 0.52% | 0.00% |
| unequal3, n=1000, CH=0.8 | 0.72% | 0.18% |
| symmetric2, n=1000, CH=0.8 | 0.04% | 0.06% |

最后一行给出本轮的负面条件，避免把平均改善写成逐条件优势。表内每行平均 5 张图。

**来回 update 的额外作用有限但可测量。** 从 0.345417% 到 0.117083% 的第一次分组贡献了最终净改善的 92.724%。继续迭代再降到 0.099167%，相对第一次分组，10 图改善、109 图相同、1 图恶化；相对一次 EM 更新，7 图改善、112 图相同、1 图恶化。大部分收益来自 likelihood 分类几何，后续更新主要微调参数和少量边界节点。

一个保留的具体案例：`heterogeneous3_n1000_ch0.8_s20000`，X-k-means 错 28 点，第一次重新分组错 4 点，一次 EM 更新和收敛 EM 均错 5 点。objective 照样上升，因此不能用单调性代替恢复正确性。

## 收敛和恢复应当如何表述

固定 $Y$ 后，每轮 EM 比较同一个 observed working likelihood；我们的数值实验确认它上升并达到规定的 objective、参数、标签停止条件。精确 E/M 步的通常 EM 上升论证只说明此固定目标不下降；它不推出参数收敛到唯一解，也不推出标签到真值、全局最大值或有限轮多项式复杂度。

若每轮依据当前 labels 重算 $H$，$Y$ 本身随之改变，前后 likelihood 数值就不再自动可比。必须另建统一目标、包含必要变换项，或给出相应上升证明；不能把本实验的固定特征 EM 单调性套过去。

当前 top-K eigenpairs 是从完整图计算的，节点的 incident edges 参与过谱分解。代码没有逐节点删除信息、重算谱后再做分类，因此**精确 LOO 版本尚未实验**。没有证明仅由保留的 top-K 能恢复精确 LOO eigensystem 或原始 SBM row likelihood，也没有验证最优错误指数。此次可支持的是“谱信息上的工作 likelihood refinement 有明确实验价值”。

## 文件与重现

本目录代码固定原始算法；`final/` 是此前 240 图记录，`independent_20000/` 是本次 120 图。完整下载包还保存了 exact float64 谱输入、预测标签、模型参数及逐轮轨迹。原始 A 未单独保存，可由冻结模型和种子重生成。

```bash
python -m pip install -r requirements.txt
python restore_records.py
python replay.py final/inputs/heterogeneous3_n1000_ch0.8_s5000.npz
python independent_validation.py --out independent_new_run --workers 8
```

重新跑会使用同一冻结新种子网格；不同的数值库或线性代数实现可能有小量差异。若要再增加新种子，先另存协议与输出目录，保留本批 120 图基准。

下一步应针对精确或可控制的 LOO 近似、参数对称约束和 decoder 稳定性建立统一目标；随机分区的全局收敛与所有 general SBM 的优势仍不在已证范围内。

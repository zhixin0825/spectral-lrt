# 节点候选参数 → top-fraction总likelihood贪心初始化 → 迭代

2026-10-09。用户提出的初始化已按原规则实现，完成同一批120张谱图上的比较。它能作为EM初值；其效果依赖likelihood表示和组比例，不能保证候选覆盖每个社区。

## 1. 算法的精确定义

固定每个节点的谱观测 $Y_i$ 和候选连接参数 $\theta_i$。设本轮剩余节点为 $R$，剩余组数为 $r$，$M=|R|$。用户说的是“其他node”，因此候选自己不参与评分。

$$m=\begin{cases}\min(\lfloor M/r\rfloor,M-r),&r>1,\\M-1,&r=1.\end{cases}$$

对每个 $c\in R$，取使 $\ell_i(\theta_c)$ 最大的 $m$ 个其他节点，记为 $S_c\subset R\setminus\{c\}$，并计算

$$T_c=\sum_{i\in S_c}\ell_i(\theta_c),\qquad c^*=\arg\max_{c\in R}T_c.$$

记录 $\theta_{c^*}$ 作为一个初始参数，移除 $S_{c^*}\cup\{c^*\}$，令 $r\leftarrow r-1$，重复到K个参数。$M-r$保护每个后续组至少有一个节点，最后一轮取完；并列选节点编号最小者。例如n=1000,K=3的临时组容量为334、334、332。

**初始参数直接保留候选节点的参数，没有先改成该临时组的均值。移除仅用于挑初值，不删除谱坐标的参考列、不重算谱特征。** 随后把全部n个节点交给EM，初始mixture权重1/K，允许自由重新分组及更新组比例。

EM使用

$$r_{ia}\propto\pi_a\exp(\ell_i(\theta_a)),\quad
\theta_a\leftarrow\arg\max_\theta\sum_i r_{ia}\ell_i(\theta),\quad
\pi_a\leftarrow n^{-1}\sum_i r_{ia}.$$

当前实现沿用精确floored-simplex权重更新。固定同一工作likelihood时，精确EM步骤保证目标不下降；这保证有界目标值的收敛，不保证达到全局最优或真实分区。“实验收敛”指达到预设数值停止准则。

## 2. 固定PG坐标：一次贪心初值的主比较

同一120谱输入：6种SBM，n=1000/2000，最小CH=0.8/1.4，每条件5个种子20000–20004，$B=C\log n/n$。这些图已经用于前面的初始化比较，本轮不是新的独立测试集。truth仅在所有拟合后用于评价。

固定此前Y，$Y_{ib}=N_{ib}+\mathcal N(0,1)$，$N_{ib}\sim\mathrm{Poisson}(\lambda_{ab})$。每个节点的候选是其有界单点MLE，按真实Poisson–Gaussian卷积密度的raw logpdf评分；之后直接以K个MLE启动相同soft latent-count EM。max300、tol1e-6。

| 初值/拟合 | 平均误分率 | 完全恢复 |
|---|---:|---:|
| X k-means（既有10次起点） | 0.3454% | 73/120 |
| k-means标签初值 + PG EM | 0.0992% | 86/120 |
| 用户raw-total贪心初值 + PG EM（1次确定性） | 0.1004% | 86/120 |
| 随机标签10次、按终点PG likelihood选最好 | 0.1000% | 86/120 |
| Greedy Euclidean++ 10次择优 | 0.0996% | 86/120 |
| Greedy likelihood++ 10次择优 | 0.1000% | 86/120 |

用户方案收敛120/120，平均13.05轮，最多99轮；目标无实质下降。相对于k-means标签PG起点的逐图变化见`pg_results/analysis.json`。其平均初始化耗时49.43秒，平均EM耗时1.18秒；全候选评分是主要成本，不能把一次确定性运行说成与++相同计算预算。

**这个实验只替换了初始化。Y的上游H仍由U上的k-means建立，因此不能据此宣称整个流程已经不依赖k-means。** 初始化和EM本身不读取原A、不读取degree或truth。

## 3. 只有U/Lambda的连接profile：解除隐藏decoder依赖

固定 $q_{ij}=\max((U\Lambda U^T)_{ij},10^{-4}\log n/n)$。candidate参数是完整q_c行，包含重建对角项；移除只改变候选行，不删参考列。以下两个更直接的谱工作score均已实测：

- Poisson稀疏近似：$\ell_i(p)=\sum_j[q_{ij}\log p_j-p_j]$，没有$\log\Gamma(q_{ij}+1)$项。
- Bernoulli期望行score：$\ell_i(p)=\sum_j[q_{ij}\log p_j+(1-q_{ij})\log(1-p_j)]$；q截到小正下界与$1-10^{-8}$之间。

两者都直接取选中的节点参数并做全部节点soft EM。它们是谱重建profile的工作score，**不是给定原A时的精确row likelihood，也不是精确逐节点LOO**。

| 同一120图；1次确定性初始化 | 平均误分率 | 完全恢复 | 数值收敛 |
|---|---:|---:|---:|
| 用户raw-total，Poisson期望score | 4.4154% | 71/120 | 120/120 |
| 用户raw-total，Bernoulli期望score | 4.9242% | 69/120 | 120/120 |
| 用户raw-total，fractional Poisson带Gamma基准 | 4.6642% | 70/120 | 120/120 |
| 既有profile greedy likelihood++，1个固定随机起点 | 1.2692% | 85/120 | 120/120 |
| 既有profile greedy likelihood++，10次按目标择优 | 0.0967% | 87/120 | 120/120 |

Poisson期望score的五种平衡模型（100图）平均误分0.1070%；不均衡模型$\pi=(.5,.3,.2)$（20图）为25.9575%。这说明平衡情况下其参数起点可以很好，不均衡时固定top比例会混入其他社区并移除仍需保留的候选。

例如`unequal3_n1000_ch0.8_s20000`，初始参数真实类别序列[1,0,0]，只覆盖2个社区；Poisson期望score最终仍有35.5%误分，Bernoulli为35.4%，原k-means0.1%。这些truth数字是事后诊断，不参与拟合。

## 4. 为什么raw总分与likelihood regret不同

若单点饱和参数为s_i，则$\ell_i(\theta)=\ell_i(s_i)-D_i(\theta)$。不同节点的$\ell_i(s_i)$可以不同，因此跨不同子集比较raw总分，会同时偏好“本来就容易解释”的节点。只要所有n节点都固定参与拟合，这些逐节点常数不影响EM；在top子集选择时，它们会影响选组和候选参数。这不是禁止比较raw likelihood，而是它实际优化的目标包含这种选择偏向。

明确的deviance-peeling诊断对照见`profile_results/`，它平均误分约0.44%，仍有不均衡模型失败，不能把它当作已解决全部问题。无Gamma和Bernoulli对照也失败，排除了仅由fractional Gamma基准造成的解释。

## 5. 无噪声总体profile的反例

令n=6000，组大小(3000,1800,1200)，

$$B=\frac{\log n}{n}\begin{pmatrix}10&1&1\\1&10&1\\1&1&10\end{pmatrix},\qquad Q=ZBZ^T.$$

这里含对角项，是精确rank-3总体profile，用来考察理想谱输入，不声称它本身是一张随机邻接矩阵。按无Gamma Poisson rawscore及本轮other-only规则，三个候选参数对应真实组 **[3, 1, 1]**，缺少第2组。移除组的真实构成为[[0, 801, 1200], [1001, 999, 0], [1999, 0, 0]]。

随后以均匀mixture权重启动标准soft EM，其中两份初始参数完全相同。它们的责任始终相同，参数与权重M步也始终相同，归纳可知这两份组件永远不能自行分开；100轮总体EM数值核对的重复参数差恰为0。因此这个初始化**没有普适的社区覆盖或恢复保证，即使输入是无噪声总体谱信息**。用剥离组人数初始化权重也不能自动解决：两个相同参数的责任比例对所有节点相同，其加权均值更新仍相同。加入打破对称、重置重复组件或自适应组容量，是对原算法的进一步修改，需要单独检验。

## 6. 可复现材料和本轮核验

`pg_peeling.py`、`profile_peeling.py`、`profile_score_controls.py`保存算法；各results目录保存逐图表、每轮候选总分、临时分组、种子编号、初始/最终参数及EM轨迹。穷举小子集独立验证top选择，120图NPZ/CSV误分数与轨迹长度已逐项核对，目标单调检查通过。完整压缩包含同一120份U/Lambda、Y解码矩阵与truth-evaluation-only输入，可离线重跑。

原++主实验此前因路径错误标为待核验；本轮已恢复读取6000次固定PG起点的完整记录，其统计已用于上面的同图对照。原记录没有被改写，验证见`pg_results/verification.json`。

当前没有新增全局最优、exact recovery或k-means++近似定理；这些有限条件实验也不代表未知K、极不均衡、其它B或更稀疏图的一般结论。

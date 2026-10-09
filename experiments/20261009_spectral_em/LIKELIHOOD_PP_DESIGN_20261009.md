# Likelihood++：目标贡献驱动的逐中心初始化

日期：2026-10-09。已实现当前 Poisson–Gaussian (PG) working likelihood 的 exact singleton-regret initializer，并完成一个无 k-means decoder 的独立谱 profile 控制实验。

## 1. 从 k-means++ 原理构造

令固定观测为 Y_i，固定模型的 log density 为 ell_i(theta)。先求单点最优参数：

s_i = argmax_theta ell_i(theta).

已有中心集合 C 的逐点损失为：

R_i(C) = ell_i(s_i) - max_{c in C} ell_i(c) >= 0,
Phi(C) = sum_i R_i(C).

第一个节点均匀抽取；以后按 R_i(C)/sum_j R_j(C) 抽取节点，并加入其单点最优参数 s_i。总损失为零时在未选节点中均匀抽取。**权重是 R_i，不再平方。**

在均值无约束、固定方差高斯模型中，R_i(c)=||Y_i-c||^2/(2 tau^2)，因此该分布恰好还原原始 k-means++ 的 D² sampling。纯 Poisson、非负观测且允许零 rate 时，则成为 I-divergence：

D(Y_i||c)=sum_b [Y_ib log(Y_ib/c_b) - Y_ib + c_b].

有 rate floor 时应使用有约束单点 MLE 的 regret，边界不能无条件套用无约束恒等式。用单点饱和项去掉各观测自身的基准拟合难度；正确变换模型时，Jacobian 在 regret 中抵消。该方法未声称自动抗离群。

Greedy 版每轮从同一残差分布抽 L=2+floor(log K) 个候选，选择加入后总 Phi 最小者。普通版 L=1。必须与具有相同候选数的 Euclidean++ 比较，不能把更多候选的效果归为 likelihood 几何。

## 2. 当前 PG 密度的精确版本

f_tau(y;lambda)=sum_{m>=0} Pois(m;lambda) Normal(y-m;0,tau^2)，各坐标独立相乘。本轮 tau=1、rate floor=1e-6、rate cap=max(1,max Y)+100。

单点参数通过 score 二分求解，而不是把 y 截成非负数。若后验 latent count 的均值/方差为 m,v，则

d log f / d lambda = m/lambda - 1,
d² log f / d lambda² = (v-m)/lambda² <= 0.

后验 g(n)=(n+1)w(n+1)/w(n)=lambda exp((y-n-1/2)/tau²) 单调下降，有 v=m+Cov(N,g(N))<=m。先检查 floor/cap 的 KKT 条件，内点二分 42 次。零 floor 极限下，y<=1/2 的 MLE 为0，不是 max(y,0)。

选出的 K 个单点 MLE 直接作为 mixture rates，初始权重1/K，接现有 soft latent-count EM，同时更新 rates/weights，max300、tol1e-6；代码见 [likelihood_pp.py](likelihood_pp.py)。上游固定 Y 的 provenance 由 caller 决定，initializer 本身不调用 k-means。

数值验证已通过：
- aligned evaluator 与旧 cross evaluator 的对角线一致；
- 有限差分 score/curvature、floor/cap、饱和密度支配、seed 唯一及 potential 单调；
- 参数起点 EM 与同参数的旧标签起点 EM 得到相同预测、迭代数和 objective；
- 独立 count 0..2000 直接求和，y 从 -20 到300，cap 0.2/600，各21个rate网格：最大 logpdf 差4.44e-16，posterior mean差5.68e-14；独立标量优化器最大 singleton gap8.88e-16。

## 3. 完整谱 profile 实验：4800 个 starts

为检查隐藏 k-means decoder 依赖，另用只有 signed U/Lambda 的固定 profile：

q_ij=max((U diag(Lambda) U^T)_ij,1e-4 log(n)/n).

q 是非负连接 intensity 工作 profile，含重建对角项，只做下界截断、可能超过1，不是逐项合法的 Bernoulli probability matrix。谱行相互依赖；这里是 Poisson-shaped working deviance，非真实 graph likelihood 或 exact LOO。

D(q_i||mu)=sum_j[q_ij log(q_ij/mu_j)-q_ij+mu_j].

用最近 seed 的 D 作为 ++ 权重，再按同一个 D 更新分组和 prototype：prototype 是所分配 profile 的 arithmetic mean。Euclidean 控制仅更改 seed 几何，之后也使用相同 Poisson 更新。

120 张固定图：6个SBM、n=1000/2000、最小CH=0.8/1.4、seed20000–20004；每种方案10个starts。所有拟合和选择先完成，之后才读取truth。best10按最小 final profile deviance选择，不用truth。该批图之前已用于其他算法比较，不能把本次称为新独立验证图。

匹配 greedy 候选数和 RNG（含相同第一节点）的1200次比较：

| 方法 | 全部1200 starts平均误分率 | restart0，120图 | best10，120图 | best10 exact |
|---|---:|---:|---:|---:|
| Greedy Euclidean++ | 2.869750% | 2.895417% | 0.096667% | 87/120 |
| Greedy likelihood++ | 1.565208% | 1.269167% | 0.096667% | 87/120 |

全部1200对：likelihood更好57次、相同1114次、更差29次；restart0更好7图、相同110图、更差3图。seeds覆盖全部真实类的比例为89.92%与93.25%，truth仅作为完成后的诊断。两种best10以及普通版两种best10在全部120图的分区完全相同，已直接检查。

普通 Euclidean++/Poisson++ 的全部1200starts平均误分率分别5.834958%/5.014292%，restart0分别7.295417%/5.975000%。这两个普通版采用独立固定RNG schedule，不是严格共享RNG的pair；都是同图、同重启数、同更新目标。

全部4800个profile starts收敛，deviance未上升，无empty-cluster repairs。best10比既有X-kmeans平均误分率0.345417%低，28图改善、1图变差、91图相同。**主要支持减少坏起点，十次重启后未体现更好的终点。**

独立核对：主实验1440个profile/PG选择结果、3600个profile restarts，控制240个选择结果、1200个restarts；NPZ与CSV errors全部一致，best10分区等价全部通过。

重要负面：把profile分区转成H再接PG EM，best10平均误分率升到9.499167%，exact78/120；720次PG桥接拟合全部数值收敛。全部20个heterogeneous3图的H condition number30–7176，PG平均误分率约53.63%。代表图heterogeneous3_n1000_ch0.8_s20000：保留谱值25.218,13.675,-9.835，人口信号6.074,12.275,24.624；弱方向缺失，噪声方向被保留，H条件数316.7。直接profile误分5/1000，PG桥接537/1000。没有事后过滤这些图。不同H的raw-X loglikelihood需减去n log|det H|；此次选择始终按共同profile objective。

该profile方法每点n个坐标、n²内存、主要迭代约n²K计算，与K维固定Y PG模型不同，不能将其收益当作“只换PG初始化”的收益。

## 4. 固定 PG 初始化主比较：完整结果待核验

另已启动同一120谱输入的5种point-seed方案，各10个starts，共6000次计划拟合：uniform singleton centers、普通Euclidean++/likelihood++、matched greedy Euclidean++/likelihood++。均用相同singleton-rate映射、uniform weights、max300 EM、同图/restart第一节点；fixed Y与上一轮相同，H_U仍来自U-kmeans。随机标签1200starts与km-label PG基线复用上一轮完全相同的已保存模型。

最后可读取的checkpoint为78/120图。运行环境随后切换，normal exec/apply_patch在运行前被“file:///workspace is invalid on windows”配置错误阻断，既有runner session也不可读取。当前不能核验完整120图终点、生成总表或打包完整归档，**不推断最终完成状态、不报告全量PG优势**。已读的部分结果没有显示PG likelihood++稳定超过matched Euclidean++，尚待全量核对。

本地脚本、逐图NPZ/JSON已按checkpoint保存；根目录sources/未改动。完整数据包和主报告的生成等待本地运行环境恢复。这里只保存已核验设计和完成的profile结果，未把partial PG统计写成final benchmark。

## 5. 理论边界与出处

Phi=sum saturated logpdf - sum max center logpdf是等权hard classification likelihood损失。等权soft mixture的饱和损失在Phi与Phi+n log K之间。

当前PG卷积一般不能直接套普通指数族/Bregman correspondence；原始k-means++ O(log K)保证不自动适用。全部singleton位于内点且统一负loglikelihood Hessian处于aI与bI之间时，可用二次夹逼研究后续证明；本批边界MLE不满足此假设。这里不主张全局soft optimum、SBM exact recovery或新的通用++定理。

- [Arthur–Vassilvitskii, SODA2007](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)：按目标D²贡献采样及greedy candidates。
- [Banerjee et al., JMLR2005](https://jmlr.csail.mit.edu/papers/volume6/banerjee05b/banerjee05b.pdf)：指数族与Bregman clustering。
- [Blömer–Bujna, Adaptive Seeding for Gaussian Mixture Models](https://arxiv.org/abs/1312.5946)：已有likelihood-aware adaptive seeding，不能把通用原理本身声称为新贡献。

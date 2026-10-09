# 随机分组能否作为 EM 初值：120 图、每图 10 次随机重启

更新：2026-10-09。使用上一轮独立种子 20000--20004 的同一批 120 图，保持 $B=C\log(n)/n$、已知 K、6 模型、两种 n、两种 CH 信号强度。这里有 1200 次随机 Poisson EM 拟合，**不是 1200 张独立图**。

## 当前结论

随机分组可以成为 EM 初值，但单次随机起点可能进入较差盆地；多个随机重启可以按相同 observed likelihood 选取，不需要真实标签。具体性能见下表。

| 初值/模型 | 平均误分率 | 完全恢复图数 | 收敛图数 | 平均轮数 |
|---|---:|---:|---:|---:|
| k-means 分组初值 | 0.099167% | 86/120 | 120/120 | 8.12 |
| 单次随机分组 | 2.293750% | 80/120 | 120/120 | 18.72 |
| 10 次随机分组，按 likelihood 选最好 | 0.100000% | 86/120 | 120/120 | 15.94 |
| 固定原始 X 的 Gaussian EM，单次随机分组 | 7.003333% | 71/120 | 120/120 | 30.50 |
| 随机 decoder 与随机分组，单次 Poisson EM | 8.509583% | 35/120 | 120/120 | 9.15 |


1200 个随机 Poisson starts 中，1199/1200 达到停止准则，平均 19.04 轮，最多 300 轮。全部 starts 的平均误分率为 3.059583%，exact recovery 为 774/1200；这是比单独挑第一个 restart 更完整的随机初值表现。完整停止条件为 objective、参数和 labels 同时稳定，最大 300 轮。表中的 Gaussian 与随机 decoder 只各跑了一个随机起点，是诊断组，不能拿它们代表相应模型的多次重启性能。

按十个随机重启计算，平均每图累计执行 190.42 个 EM 轮，而 k-means 分组起点为 8.12 轮，约 23.4 倍 EM 迭代数。选中的随机重启平均 15.94 轮只描述被选中那一次，不能当成所有重启的总计算成本；这里没有把比值写成实际 wall-clock 速度比。

## 随机化了哪一部分

对每图、每次 restart，将 n 个节点随机分成尽量等大的 K 组，不采用真实组比例。由随机分组估计初始 mixture 权重和 rates，然后执行与此前一致的 soft EM。随机种子由图 seed、n、K 与固定 offset 确定；每图最后按 observed mixture likelihood 选择 10 个候选之一，完成选择后才读取真实标签评估恢复。

主 Poisson 比较的特征为固定 $Y=\sqrt n U\Lambda H_U^{-1}$，其中 $H_U$ 沿用 U-k-means 的谱中心。这使随机标签初值与 k-means 标签初值优化完全相同的工作 likelihood，但**仍保留了 k-means 建立的 decoder**。不能将这组实验写成整个 pipeline 完全没有 k-means。

因此另做两个不以 k-means 建立模型的诊断：直接在固定、全局标准化的 $X=\sqrt n U\Lambda$ 上跑 full Gaussian EM，以及以随机分组的 U 均值构建 $H_R$ 后再跑 Poisson EM。后者改变了特征变换，不能与 $H_U$ 版本比较 raw likelihood 数值。随机 decoder 的 condition number 范围为 [17.16,3529.44]；必须结合其误分率与数值尺度评估，不能只因优化收敛便认为坐标合理。

## 相同目标下的盆地比较

同一固定 Y、tau=1、max_iter=300 的终点比较：

- `random_PG_0` 相对 k-means 起点：0 图误分更少、112 图相同、8 图更多；19 图终点 objective 高出超过 0.001，19 图低超过 0.001。
- `random_PG_best` 相对 k-means 起点：0 图误分更少、119 图相同、1 图更多；30 图终点 objective 高出超过 0.001，9 图低超过 0.001。

10 次随机 starts 中至少一次 exact recovery 的图有 86/120，十次全部 exact 的图有 50/120。前者是事后诊断，**不是可部署的 oracle 选重启规则**；实际选择始终按 likelihood。

| SBM | n | CH | k-means 初值 | 单次随机 | 随机 10 次选最好 |
|---|---:|---:|---:|---:|---:|
| disassortative3 | 1000 | 0.8 | 0.6200% | 0.6400% | 0.6400% |
| disassortative3 | 1000 | 1.4 | 0.0000% | 0.0000% | 0.0000% |
| disassortative3 | 2000 | 0.8 | 0.5600% | 0.5600% | 0.5600% |
| disassortative3 | 2000 | 1.4 | 0.0000% | 0.0000% | 0.0000% |
| equal3 | 1000 | 0.8 | 0.0400% | 0.0400% | 0.0400% |
| equal3 | 1000 | 1.4 | 0.0000% | 0.0000% | 0.0000% |
| equal3 | 2000 | 0.8 | 0.0000% | 0.0000% | 0.0000% |
| equal3 | 2000 | 1.4 | 0.0000% | 0.0000% | 0.0000% |
| heterogeneous3 | 1000 | 0.8 | 0.4200% | 0.4200% | 0.4200% |
| heterogeneous3 | 1000 | 1.4 | 0.0000% | 0.0000% | 0.0000% |
| heterogeneous3 | 2000 | 0.8 | 0.3500% | 9.5000% | 0.3500% |
| heterogeneous3 | 2000 | 1.4 | 0.0200% | 0.0200% | 0.0200% |
| hierarchical4 | 1000 | 0.8 | 0.1000% | 0.1000% | 0.1000% |
| hierarchical4 | 1000 | 1.4 | 0.0000% | 5.0000% | 0.0000% |
| hierarchical4 | 2000 | 0.8 | 0.0200% | 0.0200% | 0.0200% |
| hierarchical4 | 2000 | 1.4 | 0.0000% | 5.8400% | 0.0000% |
| symmetric2 | 1000 | 0.8 | 0.0600% | 0.0600% | 0.0600% |
| symmetric2 | 1000 | 1.4 | 0.0000% | 0.0000% | 0.0000% |
| symmetric2 | 2000 | 0.8 | 0.0100% | 0.0100% | 0.0100% |
| symmetric2 | 2000 | 1.4 | 0.0000% | 0.0000% | 0.0000% |
| unequal3 | 1000 | 0.8 | 0.1800% | 0.1800% | 0.1800% |
| unequal3 | 1000 | 1.4 | 0.0000% | 8.0000% | 0.0000% |
| unequal3 | 2000 | 0.8 | 0.0000% | 16.2900% | 0.0000% |
| unequal3 | 2000 | 1.4 | 0.0000% | 8.3700% | 0.0000% |


## 为什么完全对称的软初值会失败

随机 hard labels 会造成各组样本均值与参数间的小扰动。若把每个节点的初始 responsibility 都精确设为 1/K，并且每个 component 的参数完全一样，则各 component 密度相同；E 步保持相同的 responsibilities，M 步保持相同的参数。EM 本身不会打破这个对称性。因此“随机分组”与“所有组概率精确相等”是不同初值。

此次实验证明的是已知 K、固定谱工作 likelihood 的数值表现。单调性不代表全局最优或真实标签正确；没有验证精确逐节点 LOO、最优错误指数、所有 SBM 上的成功率或随机初值的多项式理论保证。

## 重现与证据

`random_initialization.py` 是冻结的实验代码。`random_initialization_20000/protocol.json` 保存协议，`per_run.csv` 保存所有 start 及实际选出的结果，`inputs_and_trajectories.zip` 保存 exact float64 谱输入、随机初值、所有输出标签、模型参数和逐轮轨迹。代码不接收 A 或真值作为拟合参数。原始 A 不另存。

完整下载包解压后，先展开 `random_initialization_20000/inputs_and_trajectories.zip` 到该同名目录，随后可运行：

```bash
python -m pip install -r requirements.txt
python random_initialization.py --input random_initialization_20000/inputs --out random_replay --starts 10 --max-iter 300 --workers 8
```

不同模型、不同特征变换的 raw likelihood 不直接比较。保留全部失败盆地和负面条件，没有按误分率挑选随机起点。

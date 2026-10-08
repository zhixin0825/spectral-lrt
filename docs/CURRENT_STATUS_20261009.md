# Spectral–LRT：當前結果、實驗與未完成部分

更新日期：2026-10-09。這份文件記錄目前可以支持的結論，供後續研究接續使用；不是一份宣稱已完成最優性證明的論文。

**目前最具體的成果是：在只使用 degree 與 regularized top-K 譜資訊的限制下，已完成 720 張 SBM 圖的 likelihood 初始化比較。在部分 general SBM 中，Gaussian mixture 或 degree-aware 的 Poisson／conditional-Gaussian 初始化優於合理的 `k-means` 基線；但沒有全面優勢。之前提出的直接反解 block counts 再做 LLR 的實作，在一些初始化已很準的條件下嚴重失敗。**

> 上述 `top-K` 指依特徵值絕對值取前 K 個方向。本文件以下統一使用數學記號 $K$。

## 1. 研究問題與資訊限制

考慮無向、無自環、已知社區數 $K$ 的 SBM。給定 labels $z$，不同無序點對的邊獨立，且

$$
A_{ij}\mid z\sim\operatorname{Bernoulli}(B_{z_i z_j}),\qquad i<j.
$$

我們關心：經過度數 regularization 後，是否只保留少量譜資訊與原始 degree，就能初始化、估計 block connection 資訊，最後接近使用原始 incident edges 的 LRT？更強的問題是，這些統計量是否保留關於 SBM 參數的全部漸近資訊。

本輪採用

$$
D_i=\sum_j A_{ij},\qquad
w_i=\min\left\{1,\frac{2\bar D}{D_i}\right\},\qquad
W=\operatorname{diag}(w_i),\qquad M=WAW.
$$

孤立點取 $w_i=1$。保留的統計量是

$$
\mathcal T(A)=(D,U,\Lambda),\qquad MU=U\Lambda,
$$

其中 $U$ 有 $K$ 欄，對應按 $|\lambda|$ 排序的前 $K$ 個 eigenpairs。採用絕對值排序是為了容許非正定的 general block matrix；本輪沒有利用真實參數來篩掉噪音方向。

所有 compressed-only 初始化及後續 decoder 都只讀取 $\mathcal T(A)$。`raw_bernoulli_vem` 是另外讀取原始 $A$ 的比較組，不能混入「只使用譜資訊與 degree」的結果。

實作來源：[experiment.py](../experiments/20261009_likelihood_initialization/experiment.py)。模型與設定來源：[protocol.json](../experiments/20261009_likelihood_initialization/final/protocol.json)。

## 2. 已成立的代數關係，與尚未成立的統計推論

### 2.1 譜特徵是鄰居特徵的線性和

由 eigenvector equation 直接得到

$$
\boxed{Y:=W^{-1}U\Lambda=AWU.}
$$

這是精確代數關係，不需要 Gaussian approximation，也不要求 SVD 的優化目標與 LLR 相同。因此譜分解可以先提供線性統計量，後續分類再使用非 Euclidean 的機率分數。

若已知其他節點的 labels 與 $B$，令

$$
N_{ic}=\sum_{j\ne i}A_{ij}\mathbf 1\{z_j=c\},\qquad
m_c^{(-i)}=\sum_{j\ne i}\mathbf 1\{z_j=c\}.
$$

候選社區 $a$ 的 nodewise Bernoulli 分數是

$$
s_i(a)=\log\pi_a+
\sum_c\left[N_{ic}\log B_{ac}+
(m_c^{(-i)}-N_{ic})\log(1-B_{ac})\right].
$$

任意兩候選分數之差就是相應 LLR。對數權重由這個後續評分引入，並不需要 SVD 自己算出 $\log B$。

### 2.2 反解 block counts 還需要控制殘差與條件數

對一個給定初始分區 $\widehat Z\in\{0,1\}^{n\times K}$，定義

$$
\widehat n=\widehat Z^\top\mathbf 1,\qquad
\widehat H=\operatorname{diag}(\widehat n)^{-1}\widehat Z^\top WU,
\qquad
R=WU-\widehat Z\widehat H.
$$

若每群非空且 $\widehat H$ 可逆，之前提出的解碼為

$$
\widehat N^{(0)}=Y\widehat H^{-1},\qquad
\widehat N=\widehat N^{(0)}+
\left(D-\widehat N^{(0)}\mathbf 1\right)\widehat\pi^\top,
\qquad \widehat\pi=\widehat n/n.
$$

將 $WU=\widehat Z\widehat H+R$ 代回，可清楚看到其限制：

$$
\widehat N^{(0)}=A\widehat Z+AR\widehat H^{-1},
$$

$$
\widehat N-A\widehat Z
=AR\widehat H^{-1}(I-\mathbf 1\widehat\pi^\top).
$$

Degree 校正保證 $\widehat N\mathbf 1=D$，但不會自動消除其餘方向的誤差。另有 $A\widehat Z$ 與真實 block counts $AZ$ 之差需要控制。即使初始分群很好，弱方向或噪音方向也可能使 $\widehat H^{-1}$ 放大殘差。

**因此，目前可以保存的是精確代數與一個需要穩定性條件的估計構想。它們不構成 $\mathcal T(A)$ 的充分性證明，也不構成 LRT error exponent 的證明。**

先前提出的 leave-one-out 思路旨在處理節點自身的邊與譜特徵的相依性。本輪程式使用全局 eigenvectors，沒有實作逐節點 LOO；不能把 LOO 的條件獨立性當作本實驗已具備的性質。

## 3. 正式實驗：固定設定與可核對資料

| 項目 | 正式設定 |
|---|---|
| 圖數 | 720 = 6 種模型 × 6 個目標 degree × 20 seeds |
| 節點數 | 每圖 $n=1800$ |
| 社區數 | $K=2,3,4$，擬合時已知 |
| 社區比例 | 生成時精確等分；擬合時未強制每群恰為 $n/K$ |
| 目標 degree | $4,6,12,24,64,96$ |
| 正式 seeds | 100–119；與 pilot seeds 分開 |
| 初始化方法 | 13 種，其中 12 種只使用壓縮統計量 |
| 重啟／迭代 | 每法 6 次重啟，最多 150 次迭代 |
| 正式記錄 | 18,720 = 720 ×（13 initial + 12 decoded + 1 reference） |
| 程式失敗 | 0 個 failed records；不代表所有方法都收斂或準確 |

真實 labels 與真實 $B$ 不用於正式 fitting、重啟選擇或正式 seeds 上的超參數調整。每法按自己的 objective 選擇重啟。誤分率經 Hungarian label alignment 計算。

同特徵的 `k-means`／GMM 使用相同的每次起始中心；GMM 的中心抽樣不等於先執行完整 Lloyd clustering。PG 使用不同的混合播種方案，因此 PG 與 `km_neighbor_degree` 的比較同時改變機率模型及播種，不能完全歸因於只替換 objective。

**這不是一張圖內的 train/test edge split，而是獨立 graph seeds 的固定設定比較。** 配對信賴區間為逐條件的描述性 $t(19)$ 區間，未做多重比較校正。

由於 $\log 1800\approx7.50$，只有 $d=4,6$ 在數值上低於 $\log n$。本次固定 $n$ 的網格不能驗證 $d_n\to\infty$ 且 $d_n=o(\log n)$ 的漸近定理。

完整模型包括 `k2_symmetric`、`k3_equal_degree`、`k3_heterogeneous`、`k4_equal_degree`、`k4_hierarchical`、`k3_disassortative`。每個 $B_0$、$\pi$ 與縮放規則 $B=dB_0/(n\pi^\top B_0\pi)$ 均保存在 [protocol.json](../experiments/20261009_likelihood_initialization/final/protocol.json)。三社區「等 degree 型」指 $B_0$ 各列和相同；排除自環後仍有 $O(d/n)$ 的有限樣本期望 degree 差異。

完成度來源：[execution_summary.json](../experiments/20261009_likelihood_initialization/deliverables/execution_summary.json)、[獨立 audit](../experiments/20261009_likelihood_initialization/final_audit/audit_summary.json)。

## 4. 比較了哪些目標？

| 方法 | 輸入／優化目標 | 可支持的解讀 |
|---|---|---|
| `km_classic_U`、`km_ASE`、`km_RRE` | 不同譜幾何上的平方距離 | 合理的傳統譜初始化基線 |
| `km_spec`、`km_spec_degree`、`km_neighbor_degree` | 標準化特徵上的平方距離 | 控制特徵幾何與 degree 的加入 |
| tied／full GMM | 譜特徵 Gaussian mixture likelihood | 工作 likelihood；不是完整 graph likelihood |
| `poisson_gaussian`（PG） | Degree 的 Poisson 加上條件 Gaussian | Degree-aware 工作／composite likelihood |
| `spectral_bernoulli_surrogate` | Degree-matched signed 重建矩陣上的 Bernoulli 形式目標 | 重建值可為負或大於 1，不能稱為真正 Bernoulli likelihood |
| `raw_bernoulli_vem` | 原始 $A$ 的 Bernoulli SBM mean-field ELBO | 額外讀取原始邊；不是精確全局 MLE |

精確特徵定義見 [experiment.py](../experiments/20261009_likelihood_initialization/experiment.py)：令 $X=\sqrt n W^{-1}U$，classic 使用 $X$，ASE 使用 $X|\Lambda|^{1/2}$，RRE 使用 $X\Lambda$。`spec` 是 $X$ 的逐欄標準化。Degree-augmented 版本再加入 $D$ 並標準化。

PG 以 $\sqrt nY_i/D_i$ 的共同 whitening 結果 $T_i$ 為鄰居特徵，工作模型為

$$
D_i\mid z_i=a\sim\operatorname{Poisson}(\delta_a),\qquad
T_i\mid D_i,z_i=a\sim N(\mu_a,\Sigma_a/D_i),\quad D_i>0.
$$

忽略候選社區無關的常數，分類分數為

$$
\log\pi_a+D_i\log\delta_a-\delta_a
-\frac12\log\det\Sigma_a
-\frac{D_i}{2}(T_i-\mu_a)^\top\Sigma_a^{-1}(T_i-\mu_a).
$$

孤立點只用 Poisson／prior 項。PG 以此分數的 mixture likelihood 做 EM；完整實作在 [poisson_gaussian.py](../experiments/20261009_likelihood_initialization/poisson_gaussian.py)。GMM、PG 的 covariance 都有固定下限。它們沒有從圖的真實聯合 likelihood 推導出譜 rows 的獨立性。

## 5. 初始化的正面與負面結果

以下是 20 張獨立圖的平均誤分率，單位為百分比，越低越好。`neighbor+D` 指同一組標準化鄰居特徵與 degree。

| 模型、degree | KM：classic $U$ | KM：ASE | KM：RRE | KM：neighbor+D | GMM：neighbor+D | PG |
|---|---:|---:|---:|---:|---:|---:|
| $K=2$ 對稱，6 | **15.5528** | 15.9306 | 17.4083 | 47.3556 | 20.0611 | 20.6694 |
| $K=3$ degree 異質，6 | 56.7833 | 55.7083 | 54.4167 | 43.7222 | 37.6361 | **34.5833** |
| $K=3$ degree 異質，12 | 51.0000 | 50.5028 | 50.3361 | 19.9278 | 28.6806 | **18.7500** |
| $K=3$ degree 異質，24 | 49.4806 | 49.2833 | 49.2861 | 10.0500 | 14.9639 | **7.5194** |
| $K=3$ 等 degree 型，24 | 3.4583 | 3.2000 | 2.9556 | 3.4750 | **2.4111** | 2.4250 |
| $K=4$ 等 degree，24 | 10.1917 | 6.8111 | **6.8028** | 33.7250 | 8.6528 | 12.8417 |
| $K=4$ 階層／異質，96 | 37.4944 | 36.9306 | 35.1167 | 3.1306 | 2.3306 | **2.1972** |

數值直接取自 [condition_summary.csv](../experiments/20261009_likelihood_initialization/deliverables/condition_summary.csv) 的 `stage=initial`；全 36 個條件與全部方法均保留，沒有只保存有利條件。

### 5.1 兩個可重現的改善

| 比較 | 方法誤分率 | 基線誤分率 | 配對差（百分點） | 逐條件 95% CI | 勝出圖數 |
|---|---:|---:|---:|---:|---:|
| $K=3$ 異質，$d=24$：PG 對 KM neighbor+D | 7.5194% | 10.0500% | −2.5306 | [−2.8793, −2.1818] | 20/20 |
| $K=3$ 等 degree 型，$d=24$：full GMM 對同特徵 KM | 2.4111% | 3.4750% | −1.0639 | [−1.2447, −0.8831] | 20/20 |

第二項同時匹配特徵與起始中心，可較乾淨地觀察 covariance／objective 的作用。第一項屬於相同可用資訊下整套方法的比較。來源：[paired_comparisons.csv](../experiments/20261009_likelihood_initialization/deliverables/paired_comparisons.csv)，其中 `delta_mean=method_error-baseline_error`。

### 5.2 不能宣稱普遍優於 k-means

在 $K=2,d=6$，classic $U$ 的 KM 優於 PG 與 full GMM。對 $K=4$ 等 degree、$d=24$，PG 相比 neighbor+D 的 KM 看似改善很大，但 ASE／RRE 的 KM 仍更準。因此不能把不合適的特徵縮放造成的巨大 KM 誤差，當作 likelihood 方法全面勝出的證據。

Pilot 還觀察到，PG 的較高工作 likelihood 可能選到較差的分區，偏向切出高 degree 與低 degree 群。這是 objective 排序錯配，不能只靠增加重啟修復。原始診斷、各 restart 數值及明確標為事後分析的 truth-start 軌跡見 [pilot_diagnostics.json](../experiments/20261009_likelihood_initialization/pilot_diagnostics.json)；它們不參與正式 fitting。

### 5.3 直接將低秩重建代入 Bernoulli 形式，尚未顯示穩定優勢

例如 $K=3$ 異質、$d=24$：signed spectral surrogate 的初始化誤分率為 **12.5028%**，PG 為 **7.5194%**，原始 $A$ 的 VEM 為 **4.9028%**。來源仍為上述 `condition_summary.csv`。

Signed 重建矩陣的元素可能不在 $[0,1]$，而 raw VEM 讀取了額外資訊，兩者必須分開解讀。另因 fitting 未強制 exact balance，VEM 可能出現 hard-assignment 空群；這些失敗不表示所有 likelihood 初始化都會失敗。

## 6. 共同 LLR decoder 的有限樣本失敗

本輪把 12 種 compressed-only 初始化都接上第 2 節的共同 decoder。若群過小，或 $\operatorname{cond}(\widehat H)>10^6$，程式保留初始分區；其餘情況直接反解，再估計 $B$ 與計算 plug-in LLR。正式設定凍結後沒有事後調整門檻，也沒有裁掉負的 inferred counts。

以下皆以 PG 初始化為例：

| 條件 | 初始誤分率 | 解碼後誤分率 | 平均 $\operatorname{cond}(\widehat H)$ | 平均負 counts 比例 |
|---|---:|---:|---:|---:|
| $K=3$ 異質，$d=64$ | 0.6778% | **58.1639%** | 572.5250 | 46.1565% |
| $K=4$ 等 degree，$d=64$ | 0.1694% | **45.5694%** | 365.7502 | 41.6111% |
| $K=4$ 階層，$d=96$ | 2.1972% | **56.3889%** | 397.3814 | 42.0660% |

誤分率來源：[condition_summary.csv](../experiments/20261009_likelihood_initialization/deliverables/condition_summary.csv)，`method=poisson_gaussian` 的 `initial`／`decoded` 兩個 stage。條件數與負 counts 來源：[audit_decoder_diagnostics.csv](../experiments/20261009_likelihood_initialization/final_audit/audit_decoder_diagnostics.csv)。上述三個條件皆未觸發 fallback。

第一個條件的 20 張圖中，有 19 張所選 top-3 含負向特徵值；而其 $B_0$ 為正定，這個負向方向由噪音主導。Degree 與其他方向仍可能足以做好分群，卻不足以保證全部 $K$ 個中心方向都能穩定反解。逐圖 eigenvalues 與條件數完整保存在 [正式紀錄歸檔](../experiments/20261009_likelihood_initialization/final/archives)；執行 `restore_records.py` 後可在 `final/jobs/` 核對。

目前支持的結論是：**這個未充分穩定化的直接反解 decoder 並不是可靠的有限樣本算法，初始化成功不能視作 LRT 成功。** 單一 decoder 的失敗不證明整份 $\mathcal T(A)$ 在資訊論上必然不足；也不單獨否定某個另有信號分離假設的漸近定理。此前較強的解碼／最優性說法必須受這些限制約束。

## 7. 與已有方法的關係

以下為本輪核對的一手文獻，並非宣稱已完整重現它們的算法。

1. **Le, Levina, Vershynin (2016), _Optimization via Low-rank Approximation for Community Detection in Networks_.** [論文](https://arxiv.org/abs/1406.0067)、[正文](https://arxiv.org/html/1406.0067v2)。以譜子空間縮小候選分區，再用原始 $A$ 上的目標評分，已是直接替代 KM 的先例。其 SBM 準則 $\sum_{ab}O_{ab}\log\{O_{ab}/(n_an_b)\}$ 為 Poisson／sparse profile-likelihood 形式；不應混同精確 finite Bernoulli likelihood。兩社區時有 $O(n)$ 個候選；一般固定 $K$ 的枚舉也是多項式，但次數隨 $K(K-1)$ 增長。它需要原始邊來評分，與本專案的壓縮資訊限制不同。參見原文第 2、2.2、2.3 節與式 (3)。

2. **Pisano, Agterberg, Priebe, Naiman (2022), _Spectral graph clustering via the Expectation-Solution algorithm_.** [論文](https://arxiv.org/abs/2003.13462)、[正文](https://arxiv.org/html/2003.13462v2)。研究譜嵌入上的 Gaussian mixture 及其均值／covariance 曲率。原文第 4 節明示 EM／ES 模擬從真實參數開始，第 5 節因此不討論從一般起始點的局部／全局收斂；ES 也可能降低 likelihood。這些結果不能直接用作未知參數下 initialization 成功的證據。本輪實作的是一般 GMM EM，沒有重現 ES。

3. **Amini, Chen, Bickel, Levina (2013), _Pseudo-likelihood methods for community detection in large sparse networks_.** [論文](https://arxiv.org/abs/1207.2340)、[正文](https://arxiv.org/html/1207.2340v3)。將原始 $A$ 對初始分區加總為 block counts，再做 pseudo-likelihood；有保留 degree 強度與條件於 degree 的版本，並使用 regularized spectral clustering 提供初值。這是 block-count likelihood 思路的重要先例，但其 block sums 直接讀取原始邊，與只由 $(D,U,\Lambda)$ 解碼不同。

本專案的待解問題不能僅以「把 KM 換成 GMM」表述；更具體的焦點是壓縮後的統計資訊、稀疏條件下可靠的初始化、以及能否穩定達到 LRT 的風險或錯誤指數。

## 8. 哪些已驗證，哪些尚未完成？

| 問題 | 本輪狀態 |
|---|---|
| $Y=W^{-1}U\Lambda=AWU$ | 精確代數關係 |
| 初始分區下的 counts 反解與 degree 校正 | 已實作；誤差依賴譜殘差及反解穩定性 |
| Likelihood 型初始化能否改善部分 SBM | 720 圖實驗支持；包含可重現配對改善與負面條件 |
| 一般 likelihood 初始化全面優於合理 KM | 不支持；已觀察到反例 |
| Signed low-rank 重建直接代入 Bernoulli 形式是否足夠 | 不支持穩定優勢；且不是精確 Bernoulli likelihood |
| 目前直接反解 decoder 是否可靠 | 不可靠；存在嚴重有限樣本失敗 |
| 全局 MLE／最優分區能否由此在多項式時間求得 | 未證明；本輪僅運行有限重啟與迭代的算法 |
| $1\ll d_n\ll\log n$ 下的最優 LRT 錯誤指數 | 未完成證明，固定 $n$ 實驗也未驗證該極限 |
| 與 oracle 的 risk ratio 趨近 1 | 未證明、未實驗建立 |
| $(D,U,\Lambda)$ 對 labels 或 SBM 參數的漸近資訊充分性 | 未證明；參數資訊損失未做專門實驗 |
| 原先提出的 LOO 推導能否嚴格支撐 decoder | 本輪未完成，也未實作 LOO benchmark |
| RDPG 的一般推廣 | 本輪回到 SBM，沒有新增 RDPG 定理或實驗 |

下一階段應在保留此固定 benchmark 的前提下，先處理已觀察到的失敗：辨認弱／噪音方向、控制中心反解穩定性、檢查稀疏譜特徵的條件分布，並把 LOO 分析與可由壓縮統計量實作的算法連接。新的穩定化版本需使用新的 seeds，並繼續保留 classic $U$、ASE、RRE 的 KM 基線。

## 9. 重現、診斷與資料範圍

- [實驗使用說明與重跑命令](../experiments/20261009_likelihood_initialization/README.txt)
- [完整解讀限制](../experiments/20261009_likelihood_initialization/EXPERIMENT_LIMITATIONS.txt)
- [繁體中文完整 HTML 報告](../experiments/20261009_likelihood_initialization/deliverables/sbm_likelihood_initialization_report.html)
- [全條件彙整 CSV](../experiments/20261009_likelihood_initialization/deliverables/condition_summary.csv)
- [逐圖方法／stage 結果 CSV](../experiments/20261009_likelihood_initialization/deliverables/per_run_results.csv)
- [配對差與信賴區間](../experiments/20261009_likelihood_initialization/deliverables/paired_comparisons.csv)
- [科學圖 PNG](../experiments/20261009_likelihood_initialization/deliverables/initialization_comparison.png)／[PDF](../experiments/20261009_likelihood_initialization/deliverables/initialization_comparison.pdf)
- [逐圖正式 JSON／NPZ 歸檔](../experiments/20261009_likelihood_initialization/final/archives)：metadata、保留的 $D,U,\Lambda$、評估用 truth、各法預測。以 [restore_records.py](../experiments/20261009_likelihood_initialization/restore_records.py) 校驗及還原，歸檔格式不刪除記錄。
- [獨立 audit](../experiments/20261009_likelihood_initialization/final_audit)：完成度、配對核對、空群／收斂、decoder 診斷。
- [Pilot 診斷程式](../experiments/20261009_likelihood_initialization/pilot_diagnostics.py)與 [pilot 記錄](../experiments/20261009_likelihood_initialization/pilot_v2)。

原始完整 $A$ 未逐圖保存，可用同版本 `experiment.generate(model,n,degree,seed)` 重建。正式保存的壓縮資訊及預測可以直接核對；不同數值平臺下 eigensolver 的近重根方向與局部解可能略有不同。

`edge_only_oracle` 只作 incident-edge／非邊 score 參照：它使用真實 $B,z_{-i}$，但刻意不利用「精確等分＋已知其他全部 labels」可以由缺額直接推知 $z_i$ 的資訊。因此不是 exact-balance 模型的完整 Bayes oracle，也不是逐圖嚴格下界；細節見上述 `EXPERIMENT_LIMITATIONS.txt`。

![六個模型的初始化比較](../experiments/20261009_likelihood_initialization/deliverables/initialization_comparison.png)

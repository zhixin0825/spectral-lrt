# 初始分區、譜資訊與 SBM likelihood

日期：2026-10-09。接續 [當前狀態](CURRENT_STATUS_20261009.md) 的討論。本文件補充精確 likelihood 的定義與代數關係，不把歷史稿或新的解碼構想標記為已完成的最優性定理。

## 1. 可以先分區，再用真正的 likelihood 評分

考慮固定、已知社區數 K 的無向無自環 SBM。B 是對稱的 block probability matrix。給定一個候選分區 g，其 indicator matrix 記為 Z，各組大小為 n_b。

對節點 i，定義它連向每個候選組的邊數

$$
t_{ib}=\sum_{j\ne i}A_{ij}\mathbf 1\{g_j=b\},\qquad T=AZ.
$$

總 degree 是 $D_i=\sum_b t_{ib}$。固定其他節點的候選 labels 與 B，把 i 放到候選組 a 的精確 incident-edge Bernoulli log-likelihood 為

$$
s_i(a)=\sum_b\left[t_{ib}\log B_{ab}+
(m_{ib}-t_{ib})\log(1-B_{ab})\right],
\qquad m_{ib}=n_b-\mathbf 1\{g_i=b\}.
$$

若計算 posterior classification score，再加 $\log\pi_a$；這是 MAP score，與無先驗的 likelihood 應分開。固定 exact group sizes 的搜尋還需遵守可行分區約束，不能任意單點改組。

這個分數即使 g 有錯，仍是以 g 的其他 labels 固定時，原圖 likelihood 的正確評分。初始分區有錯會影響恢復真實 labels 的效果，卻不會使候選 graph likelihood 的公式失效。若同時重估 B，單點 score 不再等於兩個 profile likelihood 的完整差，應更新 block statistics 後重新評分。

**譜方法可以負責提供初始分區，後續評分直接使用上述對數權重；不要求 SVD 本身優化 likelihood，也不要求繼續使用 k-means 的平方距離目標。**

Amini–Chen–Bickel–Levina 的 pseudo-likelihood 路線已有「按初始分區加總邊數，再做 likelihood 類更新」的先例，但其 counts 直接使用原圖 A。本專案的特別問題是只保留 degree 與少量譜資訊時，是否仍可達到同樣的統計效果。

## 2. 分區確定後，全圖與 profile likelihood 都可以明確寫出

令 E_ab 為兩組之間的無序邊數：a<b 時每條邊只計一次，a=b 時只計同組的無序點對。潛在點對數為

$$
M_{ab}=n_an_b\ (a<b),\qquad M_{aa}=\binom{n_a}{2}.
$$

原圖的 conditional log-likelihood 為

$$
\ell(g,B)=\sum_{a\le b}
\left[E_{ab}\log B_{ab}+(M_{ab}-E_{ab})\log(1-B_{ab})\right].
$$

未知 B 時，對 M_ab>0 的組對可取 $\widehat B_{ab}(g)=E_{ab}/M_{ab}$，並得到

$$
\ell_{\rm prof}(g)=\sum_{a\le b}M_{ab}
\left[
\widehat B_{ab}(g)\log\widehat B_{ab}(g)
+(1-\widehat B_{ab}(g))\log(1-\widehat B_{ab}(g))
\right].
$$

邊界採 $0\log0=0$；沒有潛在點對的組對貢獻為零。若限制 B 的參數空間，上述無約束 MLE 要相應調整。

因此，「能寫出與計算 likelihood」可以直接做到；「找到所有分區中的全局最大值」是另一個優化問題。良好的初值並不自動保證局部更新能到達全局最大值。

## 3. 如果只能使用 degree、譜資訊與初始分區

沿用現有程式的 regularization：

$$
M=WAW,\quad MU=U\Lambda,\quad F=WU,\quad
Y=W^{-1}U\Lambda=AF.
$$

W 是由 degree 決定的正對角矩陣。最後一個等式是精確 eigen-equation，因此 Y 可由保留的資料計算。

如果譜 loadings 在每個候選 block 內完全相同，即 $F=ZH$，那麼

$$
[D,Y]=AZ[\mathbf1_K,H].
$$

只要 $\operatorname{rank}[\mathbf1_K,H]=K$，就能從 degree 與譜資訊精確解出 T=AZ，進而計算上一節的 likelihood。這就是「初始分區標定 block 與譜中心，degree 提供總量，譜坐標提供分向資訊」的代數依據。

但 empirical eigenvectors 一般不是組內常數。即使 g 是完美分區，仍應寫為

$$
F=ZH+R,\qquad Y=AZH+AR.
$$

完美分區消除的是 label 誤差，不能自動消除譜殘差 R。小 singular values 還會使反解放大 AR；現有 benchmark 的嚴重 decoder 失敗應在這個框架下分析。

所以必須分清：

- 保留 A：T=AZ 可以精確計算。
- 只保留 D、U、Lambda：一般只能構造近似的 graph-likelihood score；是否保持 LRT 風險或錯誤指數仍需要證明。
- 擬合 Gaussian/Poisson feature likelihood：是另一個明確可定義的工作 likelihood，不能因為分區很好就當成精確 graph likelihood。

## 4. 更直接的構想：只重建 LLR 權重，不先反解所有 counts

對兩個候選組 a、c，定義

$$
w_{ac,j}=\log\frac{B_{a,g_j}(1-B_{c,g_j})}
{B_{c,g_j}(1-B_{a,g_j})}.
$$

固定其他 labels 時，精確 likelihood 差為

$$
s_i(a)-s_i(c)
=A_iw_{ac}+\sum_{j\ne i}
\log\frac{1-B_{a,g_j}}{1-B_{c,g_j}}.
$$

這裡 A_i 是 adjacency 的第 i 行；A_ii=0。若使用先驗，另加 $\log(\pi_a/\pi_c)$。

以初始分區標定 w_ac，再擬合

$$
w_{ac}=\beta_{0,ac}\mathbf1+F\beta_{ac}+r_{ac},
$$

便得到精確的分解

$$
s_i(a)-s_i(c)
=\beta_{0,ac}D_i+Y_i\beta_{ac}
+\sum_{j\ne i}\log\frac{1-B_{a,g_j}}{1-B_{c,g_j}}
+A_ir_{ac}.
$$

前面各項只需要 degree、譜資訊、初始分區及 B。這個公式直接引入 likelihood 的對數權重，不需要 SVD 產生 log B，也不必先反解整個 K 維 counts vector。

當殘差不為零，這是原圖 LLR 的近似，並不自動是壓縮特徵的精確 likelihood ratio。對固定、獨立的訓練 loadings、固定其他 labels，以及 $T_i=(D_i,Y_i)$，真正的壓縮 likelihood ratio 滿足

$
\frac{p_a(T_i)}{p_c(T_i)}
=\mathbb E_c\left[\exp\{s_i(a)-s_i(c)\}\mid T_i\right].
$

若全部所需 contrasts 在所保留的 span 中，原圖 LLR 已是 T_i 的函數，兩者才直接重合。

**要保持 LLR，只需保留相關 likelihood contrasts；不一定需要保留所有 block-count directions。** 最小平方或 ridge 可用於擬合權重，但此處只是代數構想，尚未實驗驗證或證明其最優性。必須分別控制：

1. 初始 labels 與 B 估計誤差；
2. 投影殘差 $A_ir_{ac}$，尤其其尾部而非僅平均平方誤差；
3. 使用全圖 eigenvectors 造成的相依性；
4. 多個 pairwise scores 的一致性；可固定一個 reference class 再构造全部 class scores。

LOO 分析可讓訓練 loadings 與 incident edges 分開；degree weights W 本身也須在 LOO 訓練圖上計算，避免把 A_ij 暗中帶回 F。本輪 benchmark 沒有實作逐節點 LOO，不能把此條件當作已有。

## 5. 也能寫出 LOO 投影特徵的精確條件分布

假設 F_j 在不使用 i 的 incident edges 的訓練圖上算好，並條件於訓練資訊與其他節點的真 labels。對候選 z_i=a，

$$
(D_i,Y_i)=\sum_{j\ne i}A_{ij}(1,F_j).
$$

其條件 moment-generating function 是

$$
\mathbb E_a e^{sD_i+t^\top Y_i}
=\prod_{j\ne i}
\left[1-B_{a,z_j}
+B_{a,z_j}e^{s+t^\top F_j}\right].
$$

這明確定義壓縮特徵的精確條件 likelihood；未知 labels/B 時可以 plug in 估計值，但估計後仍需處理誤差。一般 weighted Bernoulli sum 的 exact density/mass 計算可能很昂貴，也不能把不同節點的 score 相乘就自動稱為全圖精確 joint likelihood。

這一 LOO 條件分布不能直接套到目前由同一全圖 eigenvectors 算出的 Y_i；後者與 incident edges 有相依性。此外，LOO loadings 的 incidence projection 不是本專案目前只保存的全圖 T(A) 中自動可得的資料。

## 6. 完美分區是不是 likelihood 最大？

### 6.1 已知 B：期望 likelihood gap 非負

對真實 labels z* 與任意候選 z，在固定正確 B、0<B_ab<1 的條件下，

$$
\mathbb E_{z^*,B}
\left[\ell(z^*,B)-\ell(z,B)\right]
=
\sum_{i<j}
D_{\rm KL}
\left(
\operatorname{Ber}(B_{z_i^*z_j^*})
\,\Vert\,
\operatorname{Ber}(B_{z_iz_j})
\right)\ge0.
$$

這是期望中的比較。要得到真分區勝過所有錯分區，還需控制抽樣波動及所有候選的數目。比較 labels 時，label permutations 必須與 B 的標號一起處理；非均勻先驗的 MAP 另有 prior 項。

### 6.2 Exact recovery 條件下：以趨近 1 的概率全局最大

對固定正確 K、正確模型與適當 exact-recovery 條件，真分區以高概率是相關全局 MLE 的解；有限 n 並不保證每一張圖都如此。經典 symmetric two-block 模型的已知參數 MLE 達到 exact recovery threshold。

未知 B 的 profile likelihood 也有直接結果：Cerqueira–Leonardi (2026 version), Theorem 3.2，考慮固定 K、正比例社區、$B=\rho_nS$、S 正元素對稱且沒有相同 columns，候選群大小限制 $n_a\ge\alpha n$、$\alpha<\min_a\pi_a$。當 $\rho_n\gg\log n/n$，或 $\rho_n=\log n/n$ 且最小 CH divergence 大於 1 時，profile MLE exact recovery 的概率趨於 1。該文未解決其 profile MLE 在 CH divergence 恰等於 1 的邊界。

注意：未知 B 時，真分區可以最大化 profile likelihood，但真參數 B* 通常不等於有限樣本的 $\widehat B(z^*)$，所以不能把「真分區最大」寫成「真 labels 與真參數這一對永遠是 joint MLE」。

### 6.3 此結論不會自動傳給 feature likelihood

GMM、Poisson/conditional-Gaussian 或 signed spectral surrogate 可以把另一個分區排在真分區前面。原圖 exact recovery 條件也不保證任意初值的局部更新都成功。

若一個近似 likelihood 在全部可行分區上有統一誤差至多 epsilon，且原圖真分區對所有非等價候選的 likelihood gap 大於 2 epsilon，則它保持真分區的全局最大值。這是簡單充分條件，現有結果沒有驗證此種統一界。

如果 K 也自由增加，無懲罰的 profile likelihood 可過度擬合：K=n、每點一組，允許任意 B_ab 時，可以逐邊擬合而把 log-likelihood 提高到 0。此時需要 model penalty 或相應 posterior，不能宣稱真 K 分區最大。

## 7. 下一步應驗證的具體問題

先固定同一初始分區，比較原始 A 的精確 nodewise LLR、現有 counts 反解 LLR、與直接投影 likelihood contrasts 的 compressed score。另設 truth partition 與 true B 的診斷組，可把初始化、參數估計與譜壓縮三種誤差分開。

這是後續實驗建議；本次沒有新增實驗，也沒有把原圖 likelihood 的 exact recovery 定理視為 compressed-only decoder 的定理。

## 一手來源

- Amini, Chen, Bickel, Levina, *Pseudo-likelihood methods for community detection in large sparse networks* (2013): https://arxiv.org/abs/1207.2340
- Abbe, Bandeira, Hall, *Exact Recovery in the Stochastic Block Model* (2016): https://arxiv.org/abs/1405.3267
- Cerqueira, Leonardi, *Optimal recovery by maximum and integrated conditional likelihood in the general Stochastic Block Model*, v3 (2026): https://arxiv.org/html/2311.10153v3

# 歷史推導草稿（v0.1）

本目錄保留先前對話中產生的 TeX 原稿，以便後續研究可以追溯當時的模型、算法與證明路線。**這是歷史草稿，不是本次重新審核並確認成立的定理版本。** 本專案目前的結論與限制，請以 [當前狀態](../../docs/CURRENT_STATUS_20261009.md) 為準。

## 已取得的原件

- [sbm_spectral_lrt_20261009.tex](sbm_spectral_lrt_20261009.tex)：題為「稀疏 SBM 的譜信息與局部似然比檢驗」，文內標示研究稿 v0.1，日期為 2026 年 10 月 9 日。
- 原稿正文未作修改。官方文字讀取服務分兩段返回全部 1,209 行；按行界接合並保留檔尾換行後，UTF-8 大小為 **44,355 bytes**，與原件 metadata 一致。
- 此次取得之原始碼的 SHA-256：`5c5c381e68aec6ba28bc76a92f4588dbc50b2b5e9b073a94c112fcca790a002f`。
- 同名 `sbm_spectral_lrt_20261009.pdf` 的 metadata 存在（441,527 bytes），但原 PDF 的下載回應為 HTTP 403，故 **原 PDF 尚未納入本目錄**；此處不放置指向缺失 PDF 的連結，也不以重新編譯的檔案冒充歷史原件。

大小核對確認取回文字的完整長度；本次沒有取得原件的獨立服務端雜湊值。上述 SHA-256 是對本次保存的完整 UTF-8 原始碼計算，供之後驗證檔案未被更動。

## 可下載的重新編譯 PDF

- [sbm_spectral_lrt_20261009_recompiled.pdf](sbm_spectral_lrt_20261009_recompiled.pdf)：2026-10-09 在 GitHub Actions 以 XeLaTeX 編譯兩次，17 頁、371,163 bytes。這是從現存原始碼生成的重新編譯版，並非此前 441,527 bytes 原 PDF 的恢復。
- 編譯使用的 TeX SHA-256 仍為 `5c5c381e68aec6ba28bc76a92f4588dbc50b2b5e9b073a94c112fcca790a002f`，原稿內容未修改。
- PDF SHA-256 為 `e5049c72da58d7616eeb8b71dbd3c1ef946538dcd7d8317237363c2e1706a891`。已核對全部頁面概覽與首頁、公式頁及末頁的清晰渲染。
- [雜湊校驗](RECOMPILED_SHA256SUMS.txt)、[編譯來源與執行紀錄](RECOMPILED_BUILD.txt)、[可重現 workflow](../../.github/workflows/compile-spectral-draft.yml) 一併保存。後續重新執行的 PDF bytes 可能因 metadata 改變，應以該次生成的校驗檔為準。
- **成功編譯只確認文件可以生成，不代表已審核歷史稿的 theorem／proof。** 目前結論仍以當前狀態為準；初始分區與 likelihood 的新補充見 [後續推導](../../docs/PARTITION_SPECTRAL_LIKELIHOOD_20261009.md)。

## 原稿提出了甚麼

以下是原稿主張的摘要，**不表示本次已驗證其證明**：

1. 模型為固定社區數、固定對稱滿秩正元素連接矩陣，且平均 degree 與 `d_n` 同階，`d_n → ∞`、`d_n = o(log n)`。
2. 保存原始 degree 與一次全圖 soft-capped 矩陣 `W A W` 的 top-`k` 特徵對；特徵值按絕對值排序。
3. 利用精確的 eigen-equation `A W U = W⁻¹ U Λ`，再以估計社區譜中心的逆矩陣反解 block counts，加上 degree 校正及 plug-in LLR。
4. 證明路線先處理獨立 leave-one-out 投影，再使用截斷譜行能量及全圖／LOO 橋接。
5. 原稿宣稱局部檢驗達到 oracle LRT 的領先錯誤指數；原稿自身已區分「相同領先指數」、風險比趨於一，以及整個參數實驗的信息無損，未宣稱後兩者。

原稿初始聚類使用 `√n WU` 的行。本輪初始化實驗另比較多種譜特徵、GMM 與 degree-aware 工作 likelihood；歷史稿與本輪各實作不應不加區別地視為同一算法。

## 為甚麼必須保留歷史狀態標記

本輪實驗發現，初始化誤分率很低時，直接反解完整社區譜中心矩陣後的 LLR 仍可嚴重惡化。當 top-`|λ|` 包含噪音方向、弱信號未分離或中心矩陣反解不穩定時，這一點尤其明顯。

因此：

- 歷史草稿的漸近定理與相關引理需要獨立逐步審核。
- 不得把當前初始化改善視為 LRT 錯誤指數等價的實驗驗證。
- 不得把「好的初始化 + 直接反解」當成已普遍成立的有限樣本結論。
- 目前有限樣本失敗也不自動構成原稿固定參數、degree 趨於無窮之漸近主張的反例；這兩個問題應分開判斷。

本次僅恢復並留檔原始碼，沒有修訂原稿 theorem／proof，也沒有宣稱已完成其證明審核。

## 原稿的編譯方式

檔案使用 XeLaTeX、`xeCJK`，優先使用 Noto Serif CJK TC，並含 Fandol 字體後備設定。參考文獻已內嵌，不需另找 BibTeX 檔。

```bash
xelatex sbm_spectral_lrt_20261009.tex
xelatex sbm_spectral_lrt_20261009.tex
```

編譯會產生一份新的 PDF；它應標記為重新編譯輸出，不能據此聲稱原歷史 PDF 已恢復。

最初歸檔時的執行環境缺少 `xeCJK.sty`，未成功生成 PDF。2026-10-09 已改用倉庫中的 workflow，安裝中文 TeX 支援與 Noto／Latin Modern 字體後成功重新編譯；新輸出採用明確的 `_recompiled.pdf` 檔名，原始碼仍未更動。

SBM likelihood initialization benchmark / SBM 直接 likelihood 初始化實驗
=====================================================================
實驗識別：sbm_likelihood_init_20261009

主要結論
--------
Likelihood 可直接作為初始化的優化目標；在部分 degree 異質的 SBM，
Poisson / conditional-Gaussian 工作模型比 degree-aware KMeans 表現更好。
但它不全面優於合適的 U/ASE/RRE 幾何 KMeans。Gaussian likelihood、
原始 SBM likelihood、Bernoulli mean-field ELBO 與 signed spectral
surrogate 是不同目標，程式與報告中分別標示。

初始化改善不等於 LRT 效果。直接反解中心矩陣的 experimental decoder
在某些已準確初始化的條件嚴重失敗。例如 PG：
  K3 異質，d=64：0.6778% -> 58.1639%
  K4 等 degree，d=64：0.1694% -> 45.5694%
  K4 階層，d=96：2.1972% -> 56.3889%
本資料包不主張精確全局 MLE、最優錯誤指數或 degree+譜資訊無損。
完整限制另見 EXPERIMENT_LIMITATIONS.txt。

先開啟
------
deliverables/sbm_likelihood_initialization_report.html
  繁體中文單檔 HTML，圖像內嵌，不依賴外網字型或數學 renderer。
  含代表結果、正負配對比較、實驗方法、失敗機制、文獻、全部36條件。
deliverables/initialization_comparison.png / .pdf
  可獨立匯出的六模型科學圖。陰影：d<log(1800)；bands：pointwise
  95% Monte Carlo t(19) intervals，不是 simultaneously corrected bands。

正式固定實驗
------------
720 graphs = 6 model families x 6 target degrees x 20 independent seeds。
n=1800；degrees=4,6,12,24,64,96；graph seeds=100--119。
每法6starts、最多150iterations；13個initialization methods。
K已知；生成時精確等分；fitting的比例由資料估計，未强制每群n/K。
K3「等degree型」指C的row sums相同；其對角元素不同，排除自環造成
有限n期望degrees有O(d/n)差異。K4等degree模型則精確相等。
Pilot使用較小編號的獨立seeds，與正式720張圖分開。
Truth不參與fitting、重啟選擇或正式種子的超參數調整。
這是獨立graph-seed protocol，不是把同一張圖的邊分成train/test。

原始A只由raw_bernoulli_vem讀取；其餘初始化與common decoder僅讀
degree D與M=WAW的top-|lambda| K組eigenpairs。W由D決定：
  w_i=min(1,2*mean(D)/D_i)，D_i=0時w_i=1。
Classic U: X=sqrt(n)*W^{-1}U；ASE:X*sqrt(abs(Lambda))；RRE:X*Lambda。
S為X逐欄標準化；T為sqrt(n)*W^{-1}U Lambda / degree的全局whitening。
本輪沒有LOO eigenvectors，也未按真實參數篩掉noise eigenvectors。

同特徵KMeans/GMM用相同k-means++中心；GMM沒有先執行完整Lloyd分群。
PG播種混用degree-weighted k-means++及random-data，且採獨立child seeds，
所以PG與neighbor-degree KMeans同時改變工作模型和播種，不是純objective
替換的配對。Gaussian rows在圖上也非獨立，因此都是工作likelihood。
Degree為0的PG點只貢獻Poisson/prior項，不加入Gaussian密度。

edge_only_oracle 是equal-prior的nodewise incident-edge/非邊score參照。
它使用真實B與其他labels，但故意忽略exact-balance缺額推論。全方法
error均Hungarian對齊，不能把此oracle當成精確等分模型完整Bayes oracle
或逐圖嚴格下界。CSV reference stage保留其定義下的數值。

環境
----
CPython 3.12.14
numpy 2.3.5; scipy 1.17.0; pandas 2.2.3; scikit-learn 1.8.0;
matplotlib 3.10.8; threadpoolctl 3.6.0。
以requirements.txt安裝。程式把每個graph worker內BLAS threads限制為1。
不同數值平臺可能影響近重根方向、播種與局部解；實际labels與壓縮資訊
均保存，方便直接核對，不需將跨平臺bitwise一致視為統計重現的前提。

先還原逐圖記錄（GitHub 版新增）
------------------------------
為減少 GitHub 的零散二進位檔，720 對正式 JSON/NPZ 及 27 對 pilot
JSON/NPZ 按條件收在 45 個 ZIP，每個小於 2 MB。所有原始記錄的相對
路徑、位元組數及 SHA256，以及 ZIP 自身的 SHA256，都記在
record_archives_manifest.json。原始 benchmark 算法未因封裝而修改。

在本目錄先執行（只需 Python 標準函式庫）：
  python restore_records.py --verify-only
  python restore_records.py

第一個命令只核對；第二個核對所有 ZIP 及全部記錄後，還原至
final/jobs/ 和 pilot_v2/jobs/。現有且相同的檔案會保留；若內容不同或
路徑含 symlink，會停止而不覆寫。所有目的地衝突都在開始寫入前檢查。
可用 --split final 或 --split pilot_v2 僅還原一個 split；
--destination /another/directory 可指定其他根目錄。

restore_records.py 同時檢查：協定所要求的完整記錄集合、JSON/NPZ
成對數量、ZIP 成員集合、CRC、大小及 SHA256。它不使用 extractall，
不接受 traversal、絕對路徑或額外成員。SHA256 是對版本內 manifest
的完整性核對，不是數位簽章。

若只閱讀已生成的 HTML、CSV、PDF 或 PNG，不需要解壓或安裝套件。
若要重建分析、跑 audit/pilot 診斷或 package_results.py，需先還原。
.gitignore 排除還原的 jobs/ 及再次生成的大 ZIP，避免重複上傳資料。

重建報告（先完成上面的還原，在本目錄執行，不重跑模型）
--------------------------------------------------
python -m pip install -r requirements.txt
python analyze_results.py --input final --out deliverables
python build_report.py

完整獨立重跑（使用新目錄避免讀取既有cache）
----------------------------------------
python experiment.py --out reproduced_final --n 1800 --degrees 4,6,12,24,64,96 --models k2_symmetric,k3_equal_degree,k3_heterogeneous,k4_equal_degree,k4_hierarchical,k3_disassortative --seeds 100:120 --n-init 6 --max-iter 150 --workers 5
python analyze_results.py --input reproduced_final --out reproduced_deliverables
python build_report.py --data reproduced_deliverables --protocol reproduced_final/protocol.json --output reproduced_deliverables/sbm_likelihood_initialization_report.html
python audit_final.py --results reproduced_final --output reproduced_audit

seed參數100:120包含100、排除120。experiment.py見到已存在的job JSON會
使用cache；若要重新fitting，請選擇新的--out目錄。完整已存設定見
final/protocol.json；所有模型的B0及比例也在其中。

Pilot診斷
---------
python pilot_diagnostics.py
僅讀pilot_v2/jobs；分析既有PG重啟的目標與錯誤率排序、signed矩陣的
block contrast，以及明確標為DIAGNOSTIC_ONLY的truth-start trajectories。
這些真值診斷不參與正式fitting，亦不選擇正式測試集上的模型。
重要觀察：pilot已找到較準的PG分群，但較高的工作likelihood選了
高/低degree的錯誤切分，故單純增加重啟無法修正目標錯配。

檔案範圍
--------
experiment.py           資料、壓縮、所有比較方法、decoder、評估及runner。
poisson_gaussian.py      只接受T,D的PG多次EM；不讀A或truth。
analyze_results.py      由正式job記錄建立表格、配對t區間、圖。
build_report.py         讀CSV與PNG建立自包含繁體中文HTML。
package_results.py      彙整可下載的完整程式與結果包。
audit_final.py          完成度、配對、迭代、decoder的獨立核對。
pilot_diagnostics.py    Pilot事後機制診斷。
pilot_diagnostics.json  Pilot診斷輸出。
restore_records.py      驗證 manifest、ZIP 和記錄後安全還原；標準庫即可。
record_archives_manifest.json  所有原始記錄與各 ZIP 的大小、SHA256、數量。
archive_verification.json  封裝時逐位元组核對原檔與還原／拒絕覆寫的驗證。
final/archives/*.zip    36 個正式條件的逐圖記錄；原始 bytes 不變。
pilot_v2/archives/*.zip 9 個 pilot 條件的逐圖記錄；原始 bytes 不變。
final/protocol.json     正式固定設定。
final/jobs/*.json       每圖metadata、method/stage scores、誤分率與診斷。
final/jobs/*.npz        Truth（僅評估）、D、U、Lambda、各法labels。
final_audit/            獨立審核的CSV與JSON。
pilot_v2/               27張初步獨立圖及完整記錄，與final不可混合平均。
deliverables/           可閱讀圖表與完整CSV彙整。

本 GitHub 封裝以 45 個 ZIP 保存720對正式JSON/NPZ、27對pilot_v2 JSON/NPZ，另含程式、
protocol、audit、圖表、HTML和CSV。早期首次整合的pilot目錄不納入。

未保存完整原始A矩陣；可用experiment.generate(model,n,degree,seed)確定性
重建。壓縮資訊本身和各法預測已逐圖保存。
condition_summary.csv包含initial、decoded、reference各stage；
paired_comparisons.csv是initial stage的9種comparison x36條件=324行。
delta_mean=method_error-baseline_error；負數較好。CI為配對t(19)描述性
區間，未做多重比較校正。完整36條件保留所有負面與無明顯差別結果。

相關一手文獻（本輪非完整重現它們的實作）
--------------------------------------
Le, Levina, Vershynin (2016): Optimization via Low-rank Approximation for
Community Detection in Networks. https://arxiv.org/abs/1406.0067
  以譜子空間縮小標籤搜索並優化社區準則；不同於signed Bernoulli surrogate。
Pisano, Agterberg, Priebe, Naiman (2022): Spectral graph clustering via the
Expectation-Solution algorithm. https://arxiv.org/abs/2003.13462
  主EM/ES simulations用true-parameter starts，不能當unsupervised初始化證據。
Amini, Chen, Bickel, Levina (2013): Pseudo-likelihood methods for community
detection in large sparse networks. https://arxiv.org/abs/1207.2340
  使用block sums的pseudo-likelihood及起始值條件，不等同spectral-only輸入。

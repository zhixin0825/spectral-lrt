#!/usr/bin/env python3
"""Build a standalone Traditional-Chinese report from frozen benchmark CSVs.

No algorithms or experimental settings are changed here. All numerical
benchmark tables come from condition_summary.csv / paired_comparisons.csv;
the figure is embedded as a data URI. No network requests or external math
renderers are needed to read the resulting HTML.
"""
from __future__ import annotations

import argparse
import base64
import html
import importlib.metadata
import json
from pathlib import Path
import platform

import pandas as pd


ROOT = Path(__file__).resolve().parent
MODELS = {
    'k2_symmetric': 'K=2 對稱',
    'k3_equal_degree': 'K=3 等 degree 型',
    'k3_heterogeneous': 'K=3 異質 degree',
    'k4_equal_degree': 'K=4 等期望 degree',
    'k4_hierarchical': 'K=4 階層／異質',
    'k3_disassortative': 'K=3 disassortative',
}
METHODS = {
    'km_classic_U': 'KM U',
    'km_ASE': 'KM ASE',
    'km_RRE': 'KM RRE',
    'km_spec': 'KM S',
    'km_spec_degree': 'KM S+D',
    'km_neighbor_degree': 'KM T+D',
    'gmm_tied_spec': 'GMM tied S',
    'gmm_full_spec': 'GMM full S',
    'gmm_full_spec_degree': 'GMM full S+D',
    'gmm_full_neighbor_degree': 'GMM full T+D',
    'poisson_gaussian': 'PG',
    'spectral_bernoulli_surrogate': 'Signed surrogate',
    'raw_bernoulli_vem': 'Raw Bernoulli VEM',
}
REPRESENTATIVE = [
    ('k2_symmetric', 6),
    ('k3_equal_degree', 24),
    ('k3_heterogeneous', 6),
    ('k3_heterogeneous', 12),
    ('k3_heterogeneous', 24),
    ('k4_equal_degree', 24),
    ('k4_hierarchical', 96),
]
REP_METHODS = ['km_classic_U', 'km_ASE', 'km_RRE', 'km_neighbor_degree',
               'gmm_full_neighbor_degree', 'poisson_gaussian', 'raw_bernoulli_vem']
PAIRED_EXAMPLES = [
    ('k3_equal_degree', 24, 'gmm_full_neighbor_degree', 'km_neighbor_degree'),
    ('k3_heterogeneous', 24, 'poisson_gaussian', 'km_neighbor_degree'),
    ('k4_hierarchical', 96, 'poisson_gaussian', 'km_neighbor_degree'),
    ('k3_disassortative', 24, 'gmm_full_neighbor_degree', 'km_neighbor_degree'),
    ('k2_symmetric', 4, 'gmm_full_spec', 'km_spec'),
    ('k3_heterogeneous', 24, 'gmm_full_neighbor_degree', 'km_neighbor_degree'),
    ('k2_symmetric', 6, 'poisson_gaussian', 'km_classic_U'),
    ('k4_hierarchical', 24, 'spectral_bernoulli_surrogate', 'km_neighbor_degree'),
]


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows, caption=None, css=''):
    title = f'<caption>{esc(caption)}</caption>' if caption else ''
    head = ''.join(f'<th scope="col">{esc(x)}</th>' for x in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows)
    return f'<div class="table-wrap"><table class="{css}">{title}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def get_row(frame, model, degree, method, stage='initial'):
    selected = frame[(frame.model == model) & (frame.degree_target == degree) &
                     (frame.method == method) & (frame.stage == stage)]
    if len(selected) != 1:
        raise ValueError(f'Expected one result for {model}, d={degree}, {method}, {stage}; got {len(selected)}')
    return selected.iloc[0]


def condition_table(summary, conditions, method_names, caption, decimals=4):
    rows = []
    for model, degree in conditions:
        row = [esc(MODELS[model]), str(int(degree))]
        for method in method_names:
            result = get_row(summary, model, degree, method)
            row.append(f'{100 * result.error_mean:.{decimals}f}')
        rows.append(row)
    return table(['模型', 'd'] + [METHODS[m] for m in method_names], rows, caption, 'numeric')


def paired_table(paired):
    rows = []
    for model, degree, method, baseline in PAIRED_EXAMPLES:
        sub = paired[(paired.model == model) & (paired.degree_target == degree) &
                     (paired.method == method) & (paired.baseline == baseline)]
        if len(sub) != 1:
            raise ValueError(f'Missing paired comparison: {model}, {degree}, {method}, {baseline}')
        r = sub.iloc[0]
        if r.ci95_high < 0:
            direction = '<span class="good">區間低於 0</span>'
        elif r.ci95_low > 0:
            direction = '<span class="bad">區間高於 0</span>'
        else:
            direction = '區間包含 0'
        rows.append([
            f'{esc(MODELS[model])}<br>d={int(degree)}',
            esc(METHODS[method]), esc(METHODS[baseline]),
            f'{100*r.method_error:.3f}', f'{100*r.baseline_error:.3f}',
            f'{100*r.delta_mean:+.3f}',
            f'[{100*r.ci95_low:+.3f}, {100*r.ci95_high:+.3f}]',
            f'{int(r.wins)}/{int(r.ties)}/{int(r.losses)}', direction,
        ])
    return table(['條件', '方法', '比較基線', '方法 %', '基線 %', 'Δ 百分點',
                  '配對 95% CI', '勝／平／負', '描述'], rows, css='numeric paired')


def suite_table(paired):
    rows = []
    for (method, baseline), sub in paired.groupby(['method', 'baseline'], sort=False):
        rows.append([
            esc(METHODS[method]), esc(METHODS[baseline]),
            str(int((sub.delta_mean < -1e-12).sum())),
            str(int((sub.delta_mean.abs() <= 1e-12).sum())),
            str(int((sub.delta_mean > 1e-12).sum())),
            str(int((sub.ci95_high < 0).sum())),
            str(int((sub.ci95_low > 0).sum())),
        ])
    return table(['方法', '基線', '均值較低', '均值相同', '均值較高',
                  'CI 全低於 0', 'CI 全高於 0'], rows, css='numeric')


def decoder_table(summary, data_dir):
    conditions = [('k3_heterogeneous', 64), ('k4_equal_degree', 64), ('k4_hierarchical', 96)]
    rows = []
    audit_path = ROOT / 'final_audit' / 'audit_decoder_diagnostics.csv'
    audit = pd.read_csv(audit_path) if audit_path.exists() else None
    # The extra audit columns are used only for the original frozen report.
    use_audit = data_dir.resolve() == (ROOT / 'deliverables').resolve()
    for model, degree in conditions:
        initial = get_row(summary, model, degree, 'poisson_gaussian', 'initial')
        decoded = get_row(summary, model, degree, 'poisson_gaussian', 'decoded')
        row = [esc(MODELS[model]), str(degree), f'{100*initial.error_mean:.4f}',
               f'<strong class="bad">{100*decoded.error_mean:.4f}</strong>']
        if audit is not None and use_audit:
            r = audit[(audit.model == model) & (audit.degree_target == degree) &
                      (audit.method == 'poisson_gaussian')].iloc[0]
            row += [f'{100*r.negative_count_fraction:.2f}', f'{r.centroid_condition_mean:.1f}']
        rows.append(row)
    headings = ['模型', 'd', 'PG 初始化 %', '同一 PG 再解碼 %']
    if audit is not None and use_audit:
        headings += ['負推估 counts %', '平均中心矩陣條件數']
    return table(headings, rows, css='numeric')


def runtime_table(data_dir):
    path = data_dir / 'algorithm_diagnostics.csv'
    if not path.exists():
        return ''
    df = pd.read_csv(path)
    rows = []
    for method in METHODS:
        r = df[df.method == method].iloc[0]
        rows.append([esc(METHODS[method]), str(int(r.fits)), f'{r.median_seconds:.3f}',
                     f'{r.p95_seconds:.3f}', f'{100*r.converged_fraction:.2f}',
                     f'{100*r.empty_cluster_fraction:.2f}'])
    return table(['方法', 'fits', '中位秒數', '95% 分位秒數', '回報收斂 %', 'hard 空群 %'],
                 rows, css='numeric')


def model_table(protocol):
    rows = []
    for model in MODELS:
        spec = protocol['model_definitions'][model]
        matrix = '; '.join(', '.join(str(x) for x in row) for row in spec['B0'])
        rows.append([esc(MODELS[model]), f'<code>[{esc(matrix)}]</code>', f'1/{len(spec["pi"])}'])
    return table(['模型', 'B₀（分號分隔列）', '每群比例'], rows)


CSS = '''
:root { --ink:#202932; --muted:#596774; --blue:#244d63; --line:#d5dde3; --soft:#f2f6f8; --good:#08634b; --bad:#a13a30; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:#eef2f5; color:var(--ink); font:16px/1.8 "Noto Sans CJK TC","Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif; }
main { max-width:1230px; margin:30px auto; background:white; padding:48px 52px 60px; box-shadow:0 5px 25px #25374610; }
header { border-top:7px solid var(--blue); padding-top:20px; }
.eyebrow { font-size:13px; letter-spacing:.08em; color:var(--muted); }
h1 { font-size:34px; line-height:1.4; margin:10px 0 14px; color:var(--blue); }
h2 { font-size:24px; line-height:1.5; margin:38px 0 12px; padding-bottom:7px; border-bottom:2px solid var(--line); }
h3 { font-size:19px; margin:24px 0 8px; }
p { margin:10px 0 14px; }
.lead { font-size:19px; line-height:1.85; }
.muted,.source,figcaption { color:var(--muted); font-size:13px; }
.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:20px 0 22px; }
.stat { background:var(--soft); padding:12px 14px; border-radius:5px; }
.stat strong { display:block; font:700 27px/1.4 sans-serif; color:var(--blue); }
.stat span { color:var(--muted); font-size:13px; }
.callout { background:#edf5f6; border-left:4px solid #388497; padding:14px 18px; margin:20px 0; }
.warning { background:#fcf2ee; border-left-color:#bd654c; }
.callout p:last-child { margin-bottom:0; }
.table-wrap { overflow-x:auto; margin:14px 0 16px; }
table { border-collapse:collapse; width:100%; font-size:13px; line-height:1.6; }
caption { text-align:left; font-weight:600; color:var(--blue); padding:0 0 8px; }
th { background:#eaf0f4; color:#27465a; font-weight:600; border-bottom:2px solid #adbfcd; }
th,td { padding:8px 9px; text-align:left; vertical-align:top; border-bottom:1px solid var(--line); }
tbody tr:nth-child(even) { background:#f8fafb; }
.numeric td { font-variant-numeric:tabular-nums; text-align:right; }
.numeric td:first-child { text-align:left; white-space:nowrap; }
.numeric th { text-align:right; }
.numeric th:first-child { text-align:left; }
.paired td:nth-child(2),.paired td:nth-child(3) { text-align:left; }
.paired td { white-space:nowrap; }
.good { color:var(--good); }.bad { color:var(--bad); }
figure { margin:22px -6px; }
figure img { display:block; width:100%; height:auto; border:1px solid #e3e8eb; }
figcaption { padding:8px 6px; }
.equation { background:#f7f9fa; border:1px solid #e0e6ea; padding:11px 15px; margin:12px 0; font-family:"Cambria Math","STIX Two Math","DejaVu Sans",serif; overflow-wrap:anywhere; }
code { font:12px/1.6 ui-monospace,"DejaVu Sans Mono",monospace; background:#f2f5f7; padding:1px 3px; overflow-wrap:anywhere; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; background:#f3f6f8; padding:15px; border:1px solid #dce4e9; font:12px/1.75 ui-monospace,"DejaVu Sans Mono",monospace; }
nav { font-size:13px; margin:12px 0 22px; display:flex; gap:14px; flex-wrap:wrap; }
a { color:#176180; text-underline-offset:3px; }
ul,ol { padding-left:23px; } li { margin:6px 0; }
.footnote { font-size:13px; color:var(--muted); padding-top:15px; border-top:1px solid var(--line); }
.appendix table { font-size:12px; } .appendix th,.appendix td { padding:7px; }
.refs li { margin-bottom:14px; }
@media(max-width:800px) { main { margin:0; padding:22px 18px 38px; } h1{font-size:27px;} .lead{font-size:17px;} .stats{grid-template-columns:repeat(2,1fr);} th,td{padding:7px;} }
@media print { @page { size:A4 landscape; margin:14mm; } body{background:white;font-size:11px;line-height:1.65;} main{max-width:none;margin:0;padding:0;box-shadow:none;} h1{font-size:25px;} h2{font-size:18px;} h3{font-size:14px;} .lead{font-size:14px;} nav{display:none;} .stats{grid-template-columns:repeat(4,1fr);} .stat strong{font-size:20px;} table,.appendix table{font-size:9px;} th,td,.appendix th,.appendix td{padding:5px;} .table-wrap{overflow:visible;} thead{display:table-header-group;} tr{break-inside:avoid;} h2,h3{break-after:avoid;} figure,.callout,.equation{break-inside:avoid;} .source,figcaption,.muted,.footnote{font-size:10px;} .appendix{break-before:page;} }
'''


def build(data_dir: Path, output: Path, protocol_path: Path):
    summary = pd.read_csv(data_dir / 'condition_summary.csv')
    paired = pd.read_csv(data_dir / 'paired_comparisons.csv')
    protocol = json.loads(protocol_path.read_text())
    execution_path = data_dir / 'execution_summary.json'
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
    initial = summary[summary.stage == 'initial']
    conditions = [(m, d) for m in MODELS for d in sorted(initial[initial.model == m].degree_target.unique())]
    if len(conditions) != 36 or len(initial) != 36*13:
        raise ValueError('This report is designed for the frozen complete 36-condition, 13-method benchmark.')
    if set(initial.repetitions) != {20}:
        raise ValueError('Every reported condition/method must contain 20 graph seeds.')
    png = data_dir / 'initialization_comparison.png'
    figure_uri = 'data:image/png;base64,' + base64.b64encode(png.read_bytes()).decode('ascii')
    representative = condition_table(summary, REPRESENTATIVE, REP_METHODS,
        '七個代表條件：20 張圖的平均初始化誤分率（%；越低越好）', decimals=3)
    km_names = ['km_classic_U','km_ASE','km_RRE','km_spec','km_spec_degree','km_neighbor_degree']
    likelihood_names = ['gmm_tied_spec','gmm_full_spec','gmm_full_spec_degree',
                        'gmm_full_neighbor_degree','poisson_gaussian',
                        'spectral_bernoulli_surrogate','raw_bernoulli_vem']
    complete_a = condition_table(summary, conditions, km_names,
        '全部 36 條件 × 6 個 KMeans 幾何基線：平均初始化誤分率（%）')
    complete_b = condition_table(summary, conditions, likelihood_names,
        '全部 36 條件 × 7 個 likelihood／surrogate 方法：平均初始化誤分率（%）')
    versions = {'Python': platform.python_version()}
    versions.update({name:importlib.metadata.version(name) for name in
                     ('numpy','scipy','pandas','scikit-learn','matplotlib','threadpoolctl')})
    version_html = table(['套件','產生本報告時的版本'], [[esc(k),esc(v)] for k,v in versions.items()])
    methods_rows = [
        ('KM U / ASE / RRE','X / X|Λ|¹ᐟ² / XΛ','平方距離；三種 conventional spectral geometry'),
        ('KM S / KM S+D / KM T+D','S / std[S,D] / std[T,D]','平方距離；作同特徵比較'),
        ('GMM tied S','S','Gaussian mixture，所有群共用 covariance'),
        ('GMM full S / S+D / T+D','與同名 KM 相同','Gaussian mixture，各群 covariance 分別估計'),
        ('PG','T 與原始 D','Poisson degree × conditional Gaussian 工作模型'),
        ('Signed surrogate','degree-matched 的 signed 低秩矩陣','Bernoulli-shaped variational surrogate；不是 likelihood'),
        ('Raw Bernoulli VEM','原始 A；譜中心用於播種','完整邊／非邊 Bernoulli model 的 mean-field ELBO'),
    ]
    method_html = table(['方法縮寫','輸入','實際優化目標'], [[esc(x) for x in row] for row in methods_rows])
    commands = '''python -m pip install -r requirements.txt

# 已有正式結果：重建表格、圖、HTML；不重跑 graph fitting。
python analyze_results.py --input final --out deliverables
python build_report.py

# 獨立重現 720 張圖（100:120 包含 100，排除 120）。
python experiment.py --out reproduced_final --n 1800 \\
  --degrees 4,6,12,24,64,96 \\
  --models k2_symmetric,k3_equal_degree,k3_heterogeneous,k4_equal_degree,k4_hierarchical,k3_disassortative \\
  --seeds 100:120 --n-init 6 --max-iter 150 --workers 5
python analyze_results.py --input reproduced_final --out reproduced_deliverables
python build_report.py --data reproduced_deliverables \\
  --protocol reproduced_final/protocol.json \\
  --output reproduced_deliverables/sbm_likelihood_initialization_report.html
python audit_final.py --results reproduced_final --output reproduced_audit

# 只重現 pilot 的事後失敗機制分析；含明確標記的 truth-start 診斷。
python pilot_diagnostics.py'''
    parts = [f'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SBM 初始化能否直接優化 likelihood？｜720 張圖的配對實驗</title>
<style>{CSS}</style></head><body><main>
<header><div class="eyebrow">研究實驗記錄 · SBM likelihood initialization · 固定設定的獨立圖種子實驗</div>
<h1>SBM 初始化能否直接優化 likelihood？</h1>
<p class="lead"><strong>可以把 likelihood 當成初始化目標，而且在部分 SBM 中有效；但它不會自動比合適的 KMeans 更好。</strong>
本輪完成 720 張圖的配對實驗。Poisson／conditional-Gaussian 方法在 degree 異質的模型有明確改善，
也觀察到工作 likelihood 選錯群、signed surrogate 塌縮，以及後續譜解碼嚴重失敗的情況。</p>
<div class="stats"><div class="stat"><strong>720</strong><span>獨立生成的 SBM 圖</span></div><div class="stat"><strong>36</strong><span>6 模型 × 6 degree 條件</span></div><div class="stat"><strong>1,800</strong><span>每張圖的節點數</span></div><div class="stat"><strong>13 × 6</strong><span>初始化方法 × 每法重啟數</span></div></div>
<nav><a href="#results">主要結果</a><a href="#paired">配對比較</a><a href="#methods">方法與資料</a><a href="#decoder">解碼失敗</a><a href="#pilot">失敗機制</a><a href="#literature">文獻</a><a href="#reproduce">重現</a><a href="#all">全部結果</a></nav>
</header>
<div class="callout warning"><strong>本報告的結論是初始化的有限樣本表現。</strong>
<p>沒有證明精確全局 MLE、與 oracle 相同的風險、LRT 的最優錯誤指數，或 degree 加 top-K 譜資訊在統計上無損。
尤其本輪直接反解中心矩陣的 empirical decoder 在某些初始化已非常準確的條件仍嚴重失敗，必須保留這項負面結果。</p></div>

<section id="results"><h2>1. 主要結果：有用，但改善依賴模型與基線</h2>
<p>以下每格為 20 張圖的 label-aligned 誤分率均值。所有方法使用同一張圖、同一份譜分解；
Raw Bernoulli VEM 額外读取原始 adjacency matrix，與只使用壓縮資訊的方法具有不同資訊預算。</p>
{representative}
<p class="source">數值來源：condition_summary.csv，stage=initial。代表條件為說明正面與負面現象而列；完整 36 條件及 13 方法在附錄。
KM＝KMeans；T+D＝whitened 鄰居平均與 degree；PG＝本輪實作的 Poisson／conditional-Gaussian mixture。</p>
<p><strong>強基線會改變解讀。</strong>對 K=2、d=6，普通 U 幾何的 KMeans 平均誤分率約 15.55%，
優於 PG 的 20.67% 與 neighbor-GMM 的 20.06%。相較之下，將 leading eigenvector 標準化、再加上 degree，
可能重複放大 degree noise，讓某些同特徵 KMeans 基線變差；不能只挑這些較弱基線宣稱大幅優勢。
對 K=3 異質模型，U／ASE／RRE 在弱方向未穩定時都可能很差，此時應與 degree-aware 的 KM T+D 比較。
另一个不能忽略的負面例子是 K=4 等 degree、d=24：ASE／RRE KMeans 為 6.81%／6.80%，優於 PG 的 12.84%；
只比較較弱的 KM T+D（33.73%）會掩蓋這一點。</p>
<figure><img src="{figure_uri}" alt="六種 SBM 的初始化誤分率對平均 degree 圖；包含 U、ASE、neighbor-degree KMeans 與 GMM、PG、raw Bernoulli VEM；720 張圖的結果。">
<figcaption>圖中陰影 degree 區域僅表示 d &lt; log(1800) ≈ 7.50。誤差帶為各條件均值的 pointwise 95% Monte Carlo t 區間（20 張圖，19 自由度）。
固定 n 的六個 degree 值不能驗證 dₙ→∞、dₙ=o(log n) 的漸近結論。</figcaption></figure></section>

<section id="paired"><h2>2. 同一張圖上的配對差異：正面與負面都保留</h2>
<p>定義 Δ＝方法誤分率 − 基線誤分率。負值表示方法較好，單位為<strong>百分點</strong>。
區間以 20 個 graph-seed 的配對差異計算，使用 t(19)；這些是描述性、未做多重比較校正的 95% 區間。
勝／平／負也以同一張圖比較。</p>
{paired_table(paired)}
<p>PG 對 K=3 異質 d=24 的改善約 2.53 個百分點，且 20 張圖全部勝過 KM T+D。
同一條件下，full GMM on T+D 卻比 KM T+D 差約 4.91 個百分點。
這说明「改成 likelihood」不是足夠的設計原則，工作分布與 degree 的處理方式同樣重要。
K=2、d=6 的 PG 相較 U-KMeans 均值較差，但此組配對區間包含 0，應保留這個不確定性。</p>
<h3>整個固定 benchmark 的條件計數</h3>
{suite_table(paired)}
<p class="source">每行涵蓋全部 36 條件；計數直接由 paired_comparisons.csv 聚合。
這是事先固定的 benchmark 套件，並非從所有 SBM 隨機抽樣，條件勝率不能解讀為一般 SBM 的成功機率。</p></section>

<section id="methods"><h2>3. 資料、壓縮資訊與實際優化目標</h2>
<h3>生成與 fitting 的區別</h3>
<p>每圖 n=1800；K 已知；生成時每群恰有 n/K 個節點。令 π=(1/K,…,1/K)，
機率矩陣 B=(d/n)B₀/(πᵀB₀π)，無自環，無向邊獨立生成。
d∈{{4,6,12,24,64,96}}；每條件使用 seeds 100–119，與 pilot 的小編號種子分開。
K=3「等 degree 型」指 C 的各 row sum 相同；因對角元素不同，排除自環後的有限 n 期望 degrees 仍有 O(d/n) 差異。
K=4 等 degree 模型的對角元素也相同，期望 degrees 精確相等。
<strong>Fitting 未強制精確等分：</strong>mixture weights／社區比例由資料估計，hard assignment 可以有空群。
因此這不是充分利用已知 balance constraint 後的最佳算法比較。</p>
{model_table(protocol)}
<h3>Degree soft cap 與 features</h3>
<div class="equation">Dᵢ = ∑ⱼ Aᵢⱼ；　d̄ = n⁻¹∑ᵢDᵢ；　wᵢ = min(1, 2d̄/Dᵢ)（Dᵢ=0 時取 1）；　W=diag(wᵢ)。<br>
M=WAW；　MU=UΛ，按 |λ| 保留 top K；　X=√n W⁻¹U；　S=std(X)。</div>
<p>std 表示按觀测到的每一欄均值與標準差做標準化。Classic U 用 X；ASE 用 X|Λ|¹ᐟ²；RRE 用 XΛ。
為容許 disassortative／indefinite 模型，本輪一律按特徵值絕對值選取 K 個方向，沒有按真實參數篩去噪音方向。</p>
<div class="equation">Y=W⁻¹UΛ=AWU；　Tᵢ⁽raw⁾=√n Yᵢ/Dᵢ（Dᵢ&gt;0）。</div>
<p>T 是在正 degree 節點上對 T⁽raw⁾ 做全局 centering 與 covariance whitening 的結果；
whitening 的特徵值下限為 max(10⁻⁴×最大特徵值,10⁻¹⁰)。孤立節點的 T 設為零。
S+D、T+D 再將連接後的矩陣逐欄標準化。壓縮方法只能讀取 (D,U,Λ)，W 由 D 重建；沒有另讀 A。
這些都是<strong>單次全局譜資訊</strong>，本輪沒有計算 LOO eigenvectors。</p>
{method_html}
<h3>Gaussian mixture：能改變分類邊界，但只是工作 likelihood</h3>
<div class="equation">ℓ<sub>GMM</sub>(π,μ,Σ) = ∑ᵢ log [∑ₐ πₐ φ(xᵢ; μₐ,Σₐ)]。</div>
<p>Full GMM 用各群不同的 covariance，tied GMM 共用 covariance。它們都在 covariance eigenvalues ≥0.01 的約束下做 EM；
M-step 的特徵值下限裁切是該約束下的更新。Gaussian feature density 並不是原始 SBM 的 Bernoulli graph likelihood，
譜向量各行也不獨立。漸近 Gaussian 的中心近似本身不能推出 exp{{−dI+o(d)}} 的 LRT 尾機率。</p>
<h3>PG：將 degree 強度與鄰居構成分開</h3>
<div class="equation">Dᵢ | zᵢ=a ∼ Poisson(δₐ)；　Tᵢ | Dᵢ&gt;0,zᵢ=a ∼ N(μₐ,Σₐ/Dᵢ)。<br>
Dᵢ=0 時，class score 只有 log πₐ − δₐ，不加入 Gaussian 項。</div>
<p>PG 在 covariance eigenvalues ≥0.02 的約束下做多次 EM，按自身 observed working likelihood 選擇重啟。
它反映 degree 較大時鄰居平均通常較精確，卻仍是 Poisson/Gaussian composite 工作模型。
稀疏時鄰居平均的離散性、全局 eigenvectors 的相依性與 degree 回饋可能違反此模型。</p>
<h3>Bernoulli VEM 與 signed surrogate 必須區分</h3>
<div class="equation">ELBO = ∑<sub>i&lt;j,a,b</sub> rᵢₐrⱼᵦ[Aᵢⱼ log Bₐᵦ +(1−Aᵢⱼ)log(1−Bₐᵦ)]<br>
　　　　+ ∑ᵢₐrᵢₐ log πₐ − ∑ᵢₐrᵢₐ log rᵢₐ。</div>
<p>Raw VEM 使用原始 A 的邊與非邊，對 batch E-step 做 backtracking，優化 mean-field lower bound；
ELBO 收斂不等於找到全局 likelihood 最大值。Signed surrogate 把 A 換成零對角、與 D 有相同 row sums 的低秩重建與低秩 degree 修正。
重建元素可以為負或超過 1，故這個 Bernoulli-shaped 目標<strong>不是機率模型的 likelihood</strong>。
本輪沒有事後 clipping 整個矩陣，也沒有把負面結果移除。</p>
<h3>初始化公平性與可重現性</h3>
<p>每法 6 starts、最多 150 iterations。同特徵的 KMeans 與 GMM 使用完全相同的 k-means++ 中心抽樣；
GMM 不先跑出一個 Lloyd partition。PG 則交替使用 degree-weighted k-means++ 與 random-data centers，
並用獨立 child seeds，所以 PG 對 KM T+D 是<strong>相同可用資訊、不同模型也不同播種</strong>的比較，不能歸因為只替換 objective。
中心抽樣仍可能使用平方距離；本輪的意思是最終初始化目標改為 likelihood，而非禁止任何 L² 運算。
真實標籤／B 不用於 fitting、重啟選擇或正式種子上的超參數調整。</p></section>

<section id="decoder"><h2>4. 初始化成功，不代表目前的 LLR 解碼器成功</h2>
<p>本輪同時保留共同的 compressed-information decoder：由初始化標籤估计 weighted centroids，
反解 K×K 中心矩陣得到推估的 block connection counts，再用 D 修正 row sums、以推估 B 計算 LLR。
這是先前討論公式的直接有限樣本 plug-in，不是經過充分穩定化的最終算法。</p>
{decoder_table(summary, data_dir)}
<div class="callout warning"><strong>這三個例子已排除「只要初始化夠準，直接反解就一定可靠」的說法。</strong>
<p>低秩資訊中的弱社區方向可能被 noise eigenvector 取代；某些群仍可由 degree 與其他穩定方向準確分類，
但反解完整 K×K 中心矩陣會放大幾乎無訊號的方向，產生大量負 counts。K=3 異質 d=64 的 20 張圖中，
19 張 top 3 包含負的 noise eigenvalue，而生成的 B₀ 是正定的。</p></div>
<p>程式對極小群與 condition number &gt;10⁶ 有 fallback，仍沒有擋住這些失敗。
目前只能說這個 decoder 實作在上述條件不可靠；不能單由其失敗推論整份 (D,U,Λ) 在資訊論上必然不足。
本輪也未用正式測試結果重新挑 eigenvectors、regularization、門檻或方法。</p>
<h3>Oracle 是有額外資訊的 nodewise 參照，不是這裡的嚴格有限樣本下界</h3>
<p><code>edge_only_oracle</code> 使用真實 B 與其他節點真實標籤，計算目標節點的 Bernoulli 邊／非邊 score；
所有 prior 相等，省略共同的 log π 不影響決策。它特意不利用「精確等分、其餘標籤已知」可由人數缺額推出目標標籤的資訊。
評估仍用所有方法共同的 Hungarian label alignment，故應稱 equal-prior edge-only reference，
不能把它當作精確等分模型的完整 Bayes oracle 或逐張圖必須遵守的誤差下界。相關結果保留在 CSV 的 reference stage。</p></section>

<section id="pilot"><h2>5. Pilot 診斷：為何較高的工作 likelihood 會選錯？</h2>
<p>此節另用 n=1200、小編號 seeds 的 pilot；數值不混入 720 張正式圖。所有 truth-start 或按真實錯誤率排序，
只用於事後診斷，未用於正式方法的選擇。</p>
{table(['Pilot：K=2,d=6','被自身 likelihood 選中的解','同批重啟已有的較好分類解'], [
['seed 1','error 43.83%；LL −6038.74','error 16.08%；LL −6073.99'],
['seed 2','error 47.58%；LL −6037.32','error 21.33%；LL −6095.47']])}
<p>Seed 1 真實兩社區的平均 degrees 是 6.07 與 5.98，PG 所选群卻是 7.47 與 4.96。
錯解的 covariance eigenvalues 遠高於 0.02 的 floor，因此不是 covariance 塌到零。
它利用混合群去擬合 degree 相關的 feature 均值／變異，工作 likelihood 的排序因此偏離社區正確率；單純增加重啟不會修正這個排序。</p>
<p>在 unweighted 情形，若 Ah=λh、Tᵢ=(Ah)ᵢ/Dᵢ，有精確關係</p>
<div class="equation">(1−Dᵢ/λ²)Tᵢ = [∑ⱼ Aᵢⱼ ∑<sub>ℓ≠i</sub>Aⱼℓ hℓ] /(λDᵢ)。</div>
<p>立即折返 i→j→i 造成 degree 回饋，說明全局 spectral neighbor marks 並不是固定獨立樣本。
Pilot 中同一真實群內，第一個 whitened feature 與 degree 的相關約 −0.29、−0.35，
與 PG 所假設的 E[T|D,z]=μ<sub>z</sub> 不變有落差。這是有數據支持的機制，不是對所有錯配原因的完整排除。</p>
<p>Signed surrogate 的稀疏失敗也不只來自負元素。K=2、d=6、seed 1 的真實 block contrast 在譜重建後明顯縮小，
true-label 的 surrogate 含 prior 分數比單群解低 516.08 nats；在 d=24 同 seed 卻比單群高 1916.90 nats。
自由比例與 entropy 加上訊號衰減會偏愛無資訊／塌縮解。在另一些 K=4 pilot 中，確實出現負的 block totals，
那又是 signed surrogate 的獨立問題。完整診斷見 <code>pilot_diagnostics.json</code>。</p></section>

<section id="literature"><h2>6. 已有文獻：方向有根據，但本輪不是新穎性或最優性的證明</h2>
<ol class="refs">
<li><strong>Le, Levina &amp; Vershynin (2016), <em>Optimization via Low-rank Approximation for Community Detection in Networks</em>.</strong>
將標籤集合投影到近似信號子空間，縮小離散搜索，再用<strong>原始 A</strong>對候選分區的目標計分；直接支持「譜方法不必以 KMeans 收尾」。
其 SBM 準則 ∑ₐᵦOₐᵦ log[Oₐᵦ/(nₐnᵦ)] 是 Poisson／sparse profile-likelihood 形式，不能混同為有限樣本精確 Bernoulli likelihood。
K=2 時候選數為 O(n)；一般固定 K 的枚舉雖為多項式，其次數隨 K(K−1) 增大，可能不實用。
該法需要原始邊計分，與本輪把 signed 重建矩陣代入 Bernoulli-shaped ELBO 不同；本輪沒有重現該文算法。
<a href="https://arxiv.org/abs/1406.0067">arXiv:1406.0067</a>。</li>
<li><strong>Pisano, Agterberg, Priebe &amp; Naiman (2022), <em>Spectral graph clustering via the Expectation-Solution algorithm</em>.</strong>
比較譜嵌入上的 Gaussian EM 與利用 covariance 結構的 ES，與此處的 likelihood 初始化問題直接相關。
但原文 Section 4 明示主要 EM／ES simulations 從真實參數開始，Discussion 也指出此限制；
不能將其性能比較當成未知 labels、未知參數下 unsupervised initialization 的證明。
ES 使用 estimating equations，並不保證 likelihood 每次增加；不能將 ES 與真正優化 GMM likelihood 的 EM 混稱。本輪沒有實作其 ES。
<a href="https://arxiv.org/abs/2003.13462">arXiv:2003.13462</a>；
<a href="https://arxiv.org/html/2003.13462v2">全文與模擬設定</a>。</li>
<li><strong>Amini, Chen, Bickel &amp; Levina (2013), <em>Pseudo-likelihood methods for community detection in large sparse networks</em>.</strong>
以 block sums 與 Poisson／條件化 degree 的 pseudo-likelihood 做快速估計，包含稀疏網絡實驗與對起始值的條件。
該文也用 perturbation spectral clustering 提供初始化；不能直接等同於只保留 (D,U,Λ) 的算法。
<a href="https://arxiv.org/abs/1207.2340">arXiv:1207.2340</a>。</li></ol>
<p>可支持的研究方向是：釐清什麼 feature likelihood 能在未知社區下提供穩定初始化，並處理全局譜估計的相依性；
若要追求 LRT 指數，還必須另外證明解碼誤差的大偏差控制。這批實驗支持繼續研究這些具體問題，尚不支持「Gaussian EM 已消除 spectral gap」的結論。</p></section>

<section id="reproduce"><h2>7. 重現、程式與資料範圍</h2>
<p>正式 protocol：<code>final/protocol.json</code>。已完成 {execution.get('graphs',720)} 圖、
{execution.get('records',18720)} 筆 method/stage records；回報例外失敗 {execution.get('failed_records',0)} 筆。
沒有例外不代表每次 optimizer 收斂；以下單独列出選中重啟的迭代診斷。
時間為本環境中每圖 6 starts 的 fitting 時間，不含共用譜分解；譜分解平均约 {execution.get('average_eigendecomposition_seconds',0):.4f} 秒。
各法停止規則不同，且同時有多個 graph jobs，不能把秒數當成跨硬體的複雜度保證。</p>
{runtime_table(data_dir)}
<p class="source">收斂與空群比例來自 algorithm_diagnostics.csv；audit 的單調性診斷主要覆蓋各方法選中的重啟，
不是把全部重啟軌跡都存下。PG 的細小負浮點改善會拒絕更新並標為 numerical plateau；完整診斷保留在程式輸出。</p>
{version_html}
<pre>{esc(commands)}</pre>
<p>執行命令時，工作目錄設為本 project 根目錄。<code>experiment.py</code> 是資料生成、譜壓縮、KMeans/GMM/VEM、decoder 與評估；
<code>poisson_gaussian.py</code> 是 PG 模組；<code>analyze_results.py</code> 重建 CSV 與圖；
<code>audit_final.py</code> 獨立核對完成度、配對與 decoder；本 <code>build_report.py</code> 只讀取現有結果，不修改算法。
<code>package_results.py</code> 彙整程式、720 對正式 JSON/NPZ、27 組 pilot_v2 jobs、audit 與圖表報告。</p>
<p><code>final/jobs/*.json</code> 保存每張圖的 metadata 與 method/stage 結果；
<code>*.npz</code> 保存 truth（僅評估）、D、U、Λ 及各法 labels。原始 A 未另存大型矩陣，可由指定模型與 graph seed 重建。
不同 BLAS／eigensolver 平臺可能改變近重根子空間與局部 optimizer 的細節，故同時附上實際壓縮資訊與預測供直接核對。</p>
<p>資料檔：<a href="condition_summary.csv">condition_summary.csv</a>（全部條件／stage）、
<a href="paired_comparisons.csv">paired_comparisons.csv</a>（324 組配對比較）、
<a href="per_run_results.csv">per_run_results.csv</a>（逐圖）、
<a href="decoder_diagnostics.csv">decoder_diagnostics.csv</a>。
HTML 的圖像與下列表格已自含，單獨下載也可離線閱讀；上述 CSV 連結則需同一資料包中的檔案。</p></section>

<section id="all" class="appendix"><h2>附錄：全部 36 條件，沒有挑掉失敗條件</h2>
<p>為控制表格寬度，13 個初始化方法拆成兩張表。每格是 20 張圖的平均 Hungarian label-aligned 誤分率（%），保留四位小數。
所有數值都由 <code>condition_summary.csv</code> 的 initial stage 讀入；標準差、ARI、完整 decoded/reference stage 在原 CSV。
四位小數只是忠實呈現 Monte Carlo 平均，不代表這個精度已被統計確定。</p>
{complete_a}
{complete_b}
<p class="footnote">本報告以凍結 benchmark 的有限樣本觀測為依據。
工作 likelihood、原始 graph likelihood、ELBO 與 signed surrogate 在全文分別標示；不主張全局最優、最優 LRT 指數或資訊無損。</p>
</section></main></body></html>''']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(parts), encoding='utf-8')
    print(json.dumps({'output':str(output),'bytes':output.stat().st_size,
                      'conditions':len(conditions),'initial_methods':len(METHODS),
                      'paired_comparisons':len(paired),'figure_embedded':True},ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'deliverables')
    parser.add_argument('--output', type=Path, default=ROOT/'deliverables'/'sbm_likelihood_initialization_report.html')
    parser.add_argument('--protocol', type=Path, default=ROOT/'final'/'protocol.json')
    args = parser.parse_args()
    build(args.data, args.output, args.protocol)


if __name__ == '__main__':
    main()

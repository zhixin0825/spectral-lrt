"""Rebuild the v21 figure from its saved whole-graph simulation summaries.

Only the output directory is adapted from the conversation archive layout.
No simulation or statistical fitting is performed here.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
root=Path(__file__).resolve().parent.parent
summary=pd.read_csv(root/'experiments/production/summary.csv')
figures=root/'figures';figures.mkdir(parents=True,exist_ok=True)
labels={'two_blocks':r'$K=2,\ np=4\log n$',
        'three_blocks_log6':r'$K=3,\ np=6\log n$',
        'sublog_two_blocks':r'$K=2,\ np=6\sqrt{\log n}$'}
plt.figure(figsize=(7.4,4.5))
for model,label in labels.items():
    data=summary[summary.model==model].sort_values('n')
    plt.errorbar(data.n,data.risk_ratio,
                 yerr=[data.risk_ratio-data.ratio_ci_lower,data.ratio_ci_upper-data.risk_ratio],
                 marker='o',capsize=4,linewidth=1.6,label=label)
plt.axhline(1,linestyle='--',linewidth=1,label='Oracle reference')
plt.xscale('log',base=2);plt.xticks([1000,2000,4000,8000,16000,32000])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:,.0f}'))
plt.ylim(.8,3.8);plt.xlabel('Number of vertices');plt.ylabel('Mean misclassification / oracle risk')
plt.title('Risk ratios across increasing graph sizes');plt.legend(loc='lower right',fontsize=9)
plt.grid(axis='y',alpha=.25);plt.tight_layout()
plt.savefig(figures/'risk_ratios.pdf',bbox_inches='tight')
plt.savefig(figures/'risk_ratios.png',dpi=180,bbox_inches='tight')
plt.close()

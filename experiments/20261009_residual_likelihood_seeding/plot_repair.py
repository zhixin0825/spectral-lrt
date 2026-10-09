"""Scientific illustration of the saved failure and its non-oracle repair."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
graph='equal3_n2000_ch0.8_s20004'
a=np.load(ROOT/'design_results/arrays'/f'{graph}.npz')
x=np.sqrt(len(a['u']))*a['u']*a['lam']
z=a['truth_evaluation_only']

def aligned(pred):
    table=np.zeros((3,3),int)
    np.add.at(table,(pred,z),1)
    r,c=linear_sum_assignment(-table)
    mapping=np.zeros(3,int);mapping[r]=c
    return mapping[pred]

colors=['#3478B9','#24A078','#E59132']
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
    'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharex=True,sharey=True,layout='constrained')
for ax,labels,title in zip(axes,[z,aligned(a['raw_peel_labels']),aligned(a['raw_peel_repaired_labels'])],
        ['真实社区（仅用于事后评价）','原初始化 + EM：错 709 / 2000','参数替换 + EM：错 1 / 2000']):
    for k,color in enumerate(colors):
        sel=labels==k
        ax.scatter(x[sel,2],x[sel,1],s=9,color=color,alpha=.48,edgecolors='none',label=f'社区 {k+1}')
    ax.set_title(title,pad=12,fontweight='bold')
    ax.set_xlabel('谱方向：区分社区 1 和 3')
    ax.grid(alpha=.12)
axes[0].set_ylabel('谱方向：区分社区 2')
axes[0].legend(frameon=False,fontsize=9,loc='upper right')
fig.suptitle('跨过局部解：把重复参数换成遗漏社区的候选，随后重做全体节点 EM',fontsize=16,fontweight='bold')
target=ROOT/'deliverables/spectral_parameter_repair_20261009.png'
target.parent.mkdir(exist_ok=True)
fig.savefig(target,dpi=160,facecolor='white')
plt.close(fig)
print(target)

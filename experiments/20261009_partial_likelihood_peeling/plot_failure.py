"""Scientific diagnostic figure; truth colors are for post-hoc explanation."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'failure_diagnosis'

def main():
    with np.load(ROOT/'plot_data.npz') as data:
        x,z,centers,ids,pred=(data[key] for key in ('x','truth','centers_x','seeds','bad_initial'))
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
        'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#3478B9','#24A078','#E59132']
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.6),sharex=True,sharey=True,layout='constrained')
    for ax,labels,title,names in [
        (axes[0],z,'真实社区：两组更近，但仍清楚分开',['社区 1','社区 2','社区 3']),
        (axes[1],pred,'初始分配：参数 2 已把社区 1 和 3 合并',['参数 1','参数 2','参数 3'])]:
        for a in range(3):
            sel=labels==a
            ax.scatter(x[sel,2],x[sel,1],s=11,color=colors[a],alpha=.48,edgecolors='none',label=names[a])
        ax.scatter(centers[:,2],centers[:,1],s=95,marker='D',facecolor='white',edgecolor='#263442',linewidth=1.7,zorder=4)
        ax.scatter(x[ids,2],x[ids,1],s=270,marker='*',color='#17252A',edgecolor='white',linewidth=.8,zorder=5)
        for j,i in enumerate(ids):
            offset=[(25,-22),(30,-21),(29,16)][j]
            ax.annotate(f'参数 {j+1}',(x[i,2],x[i,1]),xytext=offset,textcoords='offset points',
                fontsize=11,fontweight='bold',arrowprops={'arrowstyle':'-','color':'#17252A'},zorder=6)
        ax.set_title(title,fontsize=14,pad=15,fontweight='bold')
        ax.set_xlabel('区分社区 1 / 3 的谱坐标',labelpad=10)
        ax.grid(alpha=.12,zorder=0)
        ax.legend(loc='upper right',frameon=False,fontsize=10)
    axes[0].set_ylabel('区分社区 2 的谱坐标',labelpad=10)
    axes[0].text(.02,.98,'菱形：真实社区均值\n黑星：三个候选参数',transform=axes[0].transAxes,
        ha='left',va='top',fontsize=10,color='#33424B')
    fig.suptitle('失败原因：选到两组之间的候选，然后重复选择另一社区',fontsize=17,fontweight='bold')
    fig.savefig(ROOT/'spectral_initialization_failure_20261009.png',dpi=170,facecolor='white')
    plt.close(fig)
    print(ROOT/'spectral_initialization_failure_20261009.png')

if __name__=='__main__':main()

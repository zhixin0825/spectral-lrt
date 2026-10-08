#!/usr/bin/env python3
"""Regenerate all benchmark tables and the scientific summary figure."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

MODEL_NAMES={
 'k2_symmetric':'K=2: symmetric',
 'k3_equal_degree':'K=3: equal expected degrees',
 'k3_heterogeneous':'K=3: heterogeneous degrees',
 'k4_equal_degree':'K=4: equal expected degrees',
 'k4_hierarchical':'K=4: hierarchical / heterogeneous',
 'k3_disassortative':'K=3: disassortative',
}
PAIRS=[
 ('gmm_full_spec','km_spec','same_spectral_features'),
 ('gmm_tied_spec','km_spec','same_spectral_features_tied'),
 ('gmm_full_spec_degree','km_spec_degree','same_spectral_plus_degree_features'),
 ('gmm_full_neighbor_degree','km_neighbor_degree','same_neighbor_plus_degree_features'),
 ('poisson_gaussian','km_neighbor_degree','degree_conditional_model_vs_same_information'),
 ('poisson_gaussian','km_classic_U','new_model_vs_classic_spectral_baseline'),
 ('gmm_full_spec','km_classic_U','gmm_vs_classic_spectral_baseline'),
 ('spectral_bernoulli_surrogate','km_neighbor_degree','low_rank_likelihood_surrogate'),
 ('raw_bernoulli_vem','km_neighbor_degree','raw_adjacency_extra_information'),
]

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='final');p.add_argument('--out',default='deliverables');args=p.parse_args()
 root=Path(args.input);out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
 paths=sorted((root/'jobs').glob('*.json'))
 records=[r for path in paths for r in json.loads(path.read_text())['records']]
 df=pd.DataFrame(records)
 df.to_csv(out/'per_run_results.csv',index=False)
 good=df[df.failed==0].copy()
 summary=good.groupby(['model','n','degree_target','method','stage']).agg(
  repetitions=('error','size'),error_mean=('error','mean'),error_sd=('error','std'),
  ari_mean=('ari','mean'),seconds_mean=('seconds','mean'),
  cap_mean=('fraction_capped','mean'),isolates_mean=('fraction_isolated','mean'),
  min_cluster_mean=('min_cluster','mean')).reset_index()
 summary['error_se']=summary.error_sd/np.sqrt(summary.repetitions)
 summary.to_csv(out/'condition_summary.csv',index=False)
 initial=good[good.stage=='initial']
 indexed=initial.pivot(index=['model','n','degree_target','seed'],columns='method',values='error')
 comparisons=[]
 for a,b,label in PAIRS:
  if a not in indexed or b not in indexed:continue
  for key,g in indexed.groupby(level=['model','n','degree_target']):
   dif=(g[a]-g[b]).dropna();nn=len(dif)
   if nn<2:continue
   se=dif.std(ddof=1)/np.sqrt(nn);margin=t.ppf(.975,nn-1)*se
   comparisons.append(dict(model=key[0],n=key[1],degree_target=key[2],
    method=a,baseline=b,comparison=label,repetitions=nn,
    method_error=g[a].mean(),baseline_error=g[b].mean(),delta_mean=dif.mean(),
    delta_se=se,ci95_low=dif.mean()-margin,ci95_high=dif.mean()+margin,
    wins=int((dif<-1e-12).sum()),ties=int((abs(dif)<=1e-12).sum()),losses=int((dif>1e-12).sum())))
 paired=pd.DataFrame(comparisons);paired.to_csv(out/'paired_comparisons.csv',index=False)
 # Within-case seed repetitions are the sampling units. Aggregated comparisons
 # below summarize a fixed benchmark suite, not a sampled population of SBMs.
 overall=[]
 for a,b,label in PAIRS:
  sub=paired[(paired.method==a)&(paired.baseline==b)]
  overall.append(dict(method=a,baseline=b,conditions=len(sub),
   conditions_better=int((sub.delta_mean<-1e-12).sum()),
   conditions_tied=int((abs(sub.delta_mean)<=1e-12).sum()),
   conditions_worse=int((sub.delta_mean>1e-12).sum()),
   ci_better=int((sub.ci95_high<0).sum()),ci_worse=int((sub.ci95_low>0).sum()),
   mean_delta_percentage_points=100*sub.delta_mean.mean(),
   pooled_wins=int(sub.wins.sum()),pooled_ties=int(sub.ties.sum()),pooled_losses=int(sub.losses.sum())))
 pd.DataFrame(overall).to_csv(out/'benchmark_comparison_summary.csv',index=False)
 diagnostics=initial.groupby('method').agg(
  fits=('error','size'),mean_seconds=('seconds','mean'),median_seconds=('seconds','median'),
  p95_seconds=('seconds',lambda x:np.quantile(x,.95)),
  converged_fraction=('converged','mean'),mean_iterations=('n_iter','mean'),
  empty_cluster_fraction=('min_cluster',lambda x:np.mean(x==0)),
  small_cluster_fraction=('min_cluster',lambda x:np.mean(x<.01*1800))).reset_index()
 diagnostics.to_csv(out/'algorithm_diagnostics.csv',index=False)
 decoded=good[good.stage=='decoded']
 dcompare=[]
 for name,g in initial.groupby('method'):
  h=decoded[decoded.method==name]
  if h.empty:continue
  merged=g.merge(h,on=['model','n','degree_target','seed'],suffixes=('_init','_decoded'))
  dcompare.append(dict(method=name,mean_initial_error=merged.error_init.mean(),
   mean_decoded_error=merged.error_decoded.mean(),
   improved_fraction=float(np.mean(merged.error_decoded<merged.error_init-1e-12)),
   worsened_fraction=float(np.mean(merged.error_decoded>merged.error_init+1e-12)),
   fallback_fraction=float(merged.decoder_fallback_decoded.mean())))
 pd.DataFrame(dcompare).to_csv(out/'decoder_diagnostics.csv',index=False)
 unique=good.drop_duplicates(['model','n','degree_target','seed'])
 (out/'execution_summary.json').write_text(json.dumps(dict(graphs=len(paths),records=len(df),
   failed_records=int(df.failed.sum()),n=int(unique.n.iloc[0]),
   unique_models=unique.model.nunique(),degrees=sorted(unique.degree_target.unique().tolist()),
   repetitions=int(unique.groupby(['model','degree_target']).size().min()),
   mean_cap_fraction=unique.fraction_capped.mean(),max_cap_fraction=unique.fraction_capped.max(),
   average_eigendecomposition_seconds=unique.eig_seconds.mean(),
   total_graph_job_seconds=sum(json.loads(path.read_text())['elapsed'] for path in paths)),indent=2))
 # Main figure intentionally includes a strong conventional baseline and a
 # degree-aware KMeans baseline, not just the matched-feature weak baseline.
 styles=[('km_classic_U','KMeans: eigenvectors','#385170','-', 'o'),
         ('km_ASE','KMeans: adjacency spectral embedding','#926ab0','--','v'),
         ('km_neighbor_degree','KMeans: neighbor mean + degree','#87939e','--','s'),
         ('gmm_full_spec','Gaussian likelihood: spectrum','#007f86','-','^'),
         ('poisson_gaussian','Poisson / conditional Gaussian','#cf572b','-','D'),
         ('raw_bernoulli_vem','Bernoulli VEM: uses raw edges','#2a2a2a',':','x')]
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                      'axes.titleweight':'bold','savefig.facecolor':'white'})
 fig,axes=plt.subplots(2,3,figsize=(14.5,8.5),sharex=True,sharey=True)
 for ax,model in zip(axes.flat,MODEL_NAMES):
  sub=summary[(summary.model==model)&(summary.stage=='initial')]
  ax.axvspan(3.7,np.log(1800),color='#ece9f2',alpha=.7,zorder=-2)
  for name,label,color,line,marker in styles:
   g=sub[sub.method==name].sort_values('degree_target')
   xx=g.degree_target.to_numpy();yy=100*g.error_mean.to_numpy();se=100*g.error_se.to_numpy()
   ax.plot(xx,yy,label=label,color=color,ls=line,marker=marker,lw=1.8,ms=4)
   critical=t.ppf(.975,g.repetitions.to_numpy()-1)
   ax.fill_between(xx,np.maximum(0,yy-critical*se),yy+critical*se,color=color,alpha=.09,lw=0)
  ax.set_title(MODEL_NAMES[model],loc='left',fontsize=11,pad=10)
  ax.set_xscale('log');ax.set_xticks([4,6,12,24,64,96]);ax.xaxis.set_major_formatter(ScalarFormatter())
  ax.set_xlim(3.7,108);ax.set_ylim(-1,75);ax.grid(axis='y',alpha=.16)
 for ax in axes[-1]:ax.set_xlabel('Target mean degree d')
 for ax in axes[:,0]:ax.set_ylabel('Misclassification (%)')
 fig.suptitle('Likelihood initialization is useful, but is not uniformly better than KMeans',
              x=.065,y=.995,ha='left',fontsize=16,fontweight='bold')
 fig.text(.065,.949,'720 graphs · n = 1,800 · 20 paired graph seeds per condition · 6 starts per method',fontsize=11,color='#555555')
 handles,labels=axes[0,0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.015),frameon=False,fontsize=10)
 fig.text(.065,.045,'Shaded x region: d < log(n). Bands: pointwise 95% Monte Carlo intervals. All other curves use only degree + top-K spectrum.',fontsize=9,color='#555555')
 fig.subplots_adjust(left=.065,right=.985,top=.89,bottom=.13,hspace=.29,wspace=.18)
 fig.savefig(out/'initialization_comparison.png',dpi=180,bbox_inches='tight')
 fig.savefig(out/'initialization_comparison.pdf',bbox_inches='tight')
 print(json.dumps({'graphs':len(paths),'failed_records':int(df.failed.sum()),'output':str(out)}))
 print(pd.DataFrame(overall).to_string(index=False))

if __name__=='__main__':main()

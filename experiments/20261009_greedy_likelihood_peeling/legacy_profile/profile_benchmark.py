"""Decoder-free spectral probability-profile Poisson deviance++ experiment.

Fits see only U and Lambda. Truth and saved baseline labels are read only
after all seeding, restart selection, decoders, and PG EM fits finish.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from poisson_em import fit_poisson_em

PROTOCOL={
 'input':'Only K eigenpairs U,Lambda; no adjacency read.',
 'profile':'q_ij=max((U diag(Lambda) U^T)_ij, 1e-4 log(n)/n). No upper clipping for Poisson working deviance.',
 'loss':'D(q_i||mu)=sum_j(q_ij log(q_ij/mu_j)-q_ij+mu_j). Fixed fractional spectral profiles, including diagonal.',
 'first_seed':'Uniform observation; independent fixed RNG seed per graph, method and restart.',
 'methods':['profile_euclidean_pp','profile_poisson_pp','profile_poisson_greedy_pp'],
 'subsequent_seeds':'Sample proportional minimum saturated deviance (Euclidean control uses squared Euclidean), never squared again.',
 'greedy_trials':'2+floor(log K); each candidate sampled from current residual weights; choose smallest total post-addition seeding loss.',
 'refinement':'Poisson Bregman Lloyd: arithmetic mean prototypes then nearest Poisson deviance, maximum 150 iterations; empty cluster repaired by largest unexplained observation.',
 'restarts':10,
 'restart_selection':'Minimum final profile Poisson deviance only. Selection never uses true labels or decoded PG likelihood.',
 'reported_starts':'Restart 0 and best of all 10, including same selections if identical.',
 'decoder':'H_a=mean_{i assigned a}(sqrt(n) U_i); fixed Y=sqrt(n) U Lambda H^{-1}. Reject cond(H)>1e8.',
 'PG':'Fixed tau=1 Poisson-Gaussian EM, max_iter=300, tol=1e-6. Decoder fixed throughout.',
 'PG_cross_decoder_score':'Raw X log likelihood = PG Y log likelihood - n log(abs(det(H))); diagnostic only, not used selection.',
 'evaluation':'Truth and baseline km_X labels consulted only after all fits and selections finish.',
 'caveat':'Working likelihood deviance on dependent fractional spectral profiles, not exact Bernoulli graph likelihood or exact LOO; no global or recovery theorem claimed.',
}

def as_json(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)

def metric(labels,truth,k):
    table=np.zeros((k,k),int)
    np.add.at(table,(truth,labels),1)
    rows,cols=linear_sum_assignment(-table)
    errors=int(len(labels)-table[rows,cols].sum())
    return dict(errors=errors,error_rate=errors/len(labels),exact=int(errors==0))

class Profile:
    def __init__(self,u,lam):
        self.u=u; self.lam=lam
        self.n,self.k=u.shape
        self.q=np.maximum((u*lam)@u.T,1e-4*np.log(self.n)/self.n)
        self.entropy=np.sum(self.q*np.log(self.q)-self.q,axis=1)
        self.sqnorm=np.sum(self.q*self.q,axis=1)
    def deviance(self,mu):
        return np.maximum(self.entropy[:,None]-self.q@np.log(mu).T+mu.sum(axis=1)[None,:],0)
    def squared(self,mu):
        return np.maximum(self.sqnorm[:,None]-2*self.q@mu.T+(mu*mu).sum(axis=1)[None,:],0)
    def seed(self,rng,kind,greedy):
        first=int(rng.integers(self.n)); selected=[first]; events=[{'seed':first}]
        distance=self.squared if kind=='euclidean' else self.deviance
        loss=distance(self.q[[first]]).ravel()
        for _ in range(1,self.k):
            total=float(loss.sum())
            if total<=1e-14:
                available=np.setdiff1d(np.arange(self.n),selected)
                candidate=int(rng.choice(available))
                candidates=np.array([candidate])
            else:
                candidates=rng.choice(self.n,size=2+int(math.log(self.k)) if greedy else 1,p=loss/total)
            candidate_dist=distance(self.q[candidates])
            totals=np.minimum(loss[:,None],candidate_dist).sum(axis=0)
            winner=int(np.argmin(totals)); chosen=int(candidates[winner])
            selected.append(chosen)
            loss=np.minimum(loss,candidate_dist[:,winner])
            events.append(dict(seed=chosen,candidates=candidates.tolist(),candidate_potentials=totals.tolist(),potential=float(loss.sum())))
        # All methods use SAME Poisson likelihood for assignments and refinement.
        labels=self.deviance(self.q[selected]).argmin(axis=1)
        return labels,selected,events
    def lloyd(self,initial,max_iter=150):
        labels=initial.copy(); trace=[]; repairs=0
        for iteration in range(max_iter+1):
            sizes=np.bincount(labels,minlength=self.k)
            if np.any(sizes==0):
                existing=np.flatnonzero(sizes)
                mu=np.stack([self.q[labels==a].mean(0) for a in existing])
                loss=self.deviance(mu).min(axis=1)
                for a in np.flatnonzero(sizes==0):
                    candidates=np.argsort(-loss)
                    ix=next(int(i) for i in candidates if sizes[labels[i]]>1)
                    sizes[labels[ix]]-=1; labels[ix]=a; sizes[a]+=1; loss[ix]=-np.inf; repairs+=1
            mu=np.stack([self.q[labels==a].mean(0) for a in range(self.k)])
            ds=self.deviance(mu)
            objective=float(ds[np.arange(self.n),labels].sum())
            new=ds.argmin(axis=1); changes=int((new!=labels).sum())
            trace.append(dict(iteration=iteration,objective=objective,changes=changes,class_sizes=sizes.tolist()))
            if changes==0:
                return labels,objective,trace,True,repairs
            labels=new
        return labels,objective,trace,False,repairs

def fit_all(u,lam,seed):
    profile=Profile(u,lam); n,k=u.shape
    outputs=[]; arrays=dict(u=u,lam=lam); traces={}; restarts=[]
    for method,kind,greedy in [('profile_euclidean_pp','euclidean',False),('profile_poisson_pp','poisson',False),('profile_poisson_greedy_pp','poisson',True)]:
        runs=[]
        method_tag={'profile_euclidean_pp':101,'profile_poisson_pp':211,'profile_poisson_greedy_pp':307}[method]
        for restart in range(10):
            rng=np.random.default_rng(np.random.SeedSequence([seed,method_tag,restart,92173]))
            initial,seeds,events=profile.seed(rng,kind,greedy)
            labels,objective,trace,converged,repairs=profile.lloyd(initial)
            arrays[f'{method}_r{restart}_seed_labels']=initial
            arrays[f'{method}_r{restart}_labels']=labels
            traces[f'{method}_r{restart}']=dict(seed_events=events,lloyd=trace)
            record=dict(method=method,restart=restart,profile_deviance=objective,lloyd_iterations=len(trace)-1,lloyd_converged=converged,empty_repairs=repairs,seeds=seeds)
            restarts.append(record); runs.append((labels,record))
        best_index=min(range(len(runs)),key=lambda r:runs[r][1]['profile_deviance'])
        for selection,index in [('single',0),('best10',best_index)]:
            labels,record=runs[index]
            key=f'{method}_{selection}'
            stats=record.copy(); stats.update(selection=selection,selected_restart=index)
            outputs.append((key+'_profile',labels,stats|{'stage':'profile'}))
            arrays[key+'_profile_labels']=labels
            h=np.stack([np.sqrt(n)*u[labels==a].mean(0) for a in range(k)])
            cond=float(np.linalg.cond(h)); stats['cond_H']=cond
            if not np.isfinite(cond) or cond>1e8:
                outputs.append((key+'_pg',labels,stats|{'stage':'pg','decoder_failed':True}))
                continue
            y=np.linalg.solve(h.T,(np.sqrt(n)*u*lam).T).T
            result=fit_poisson_em(y,labels,tau=1.,max_iter=300,tol=1e-6)
            prediction=result.pop('labels'); trace=result.pop('trace'); result.pop('r')
            arrays[key+'_H']=h; arrays[key+'_Y']=y; arrays[key+'_pg_labels']=prediction
            traces[key+'_pg']=trace
            logabsdet=float(np.linalg.slogdet(h)[1])
            result.update(raw_X_objective=result['objective']-n*logabsdet,logabsdet_H=logabsdet)
            outputs.append((key+'_pg',prediction,stats|result|{'stage':'pg','decoder_failed':False}))
    return outputs,arrays,traces,restarts

def run_graph(task):
    path,out=map(Path,task); started=time.perf_counter()
    matched=re.fullmatch(r'(.+)_n(\d+)_ch([\d.]+)_s(\d+)',path.stem)
    model,n,ch,seed=matched.groups()
    with np.load(path) as source:
        u,lam=source['u'],source['lam']
        with threadpool_limits(limits=1):
            outputs,arrays,traces,restarts=fit_all(u,lam,int(seed))
        # Evaluation starts here, after every fit and selection.
        truth=source['truth_evaluation_only']; baseline=source['labels_km_X'] if 'labels_km_X' in source else source['baseline_km_X']
        arrays['truth_evaluation_only']=truth; arrays['baseline_km_X']=baseline
    rows=[]
    baseline_metric=metric(baseline,truth,u.shape[1])
    for key,labels,stats in outputs:
        rows.append(dict(graph=path.stem,model=model,n=int(n),ch=float(ch),seed=int(seed),key=key,**stats,**metric(labels,truth,u.shape[1]),baseline_errors=baseline_metric['errors'],baseline_error_rate=baseline_metric['error_rate']))
    for restart in restarts:
        labels=arrays[f"{restart['method']}_r{restart['restart']}_labels"]
        restart.update(metric(labels,truth,u.shape[1]))
    np.savez_compressed(out/'inputs'/f'{path.stem}.npz',**arrays)
    payload=dict(graph=path.stem,baseline=baseline_metric,results=rows,restarts=restarts,traces=traces,elapsed=time.perf_counter()-started)
    (out/'runs'/f'{path.stem}.json').write_text(json.dumps(payload,default=as_json),encoding='utf-8')
    return dict(graph=path.stem,rows=rows,restarts=[dict(graph=path.stem,model=model,n=int(n),ch=float(ch),**r) for r in restarts],elapsed=payload['elapsed'])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=4); parser.add_argument('--pilot',action='store_true'); parser.add_argument('--input-dir',type=str)
    args=parser.parse_args()
    own=Path(__file__).resolve().parent; source=Path(args.input_dir) if args.input_dir else own.parent/'spectral_em_20261009'/'independent_20000'/'inputs'
    if not source.exists(): source=own/'full120'/'inputs'
    paths=sorted(source.glob('*.npz'))
    if args.pilot: paths=[p for p in paths if '_n1000_ch0.8_s20000' in p.stem]
    out=own/('pilot' if args.pilot else 'full120'); (out/'inputs').mkdir(parents=True,exist_ok=True); (out/'runs').mkdir(exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(PROTOCOL|dict(graph_count=len(paths),workers=args.workers),indent=2),encoding='utf-8')
    rows=[]; restarts=[]; started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending={executor.submit(run_graph,(str(path),str(out))):path for path in paths}
        for future in concurrent.futures.as_completed(pending):
            result=future.result(); rows+=result['rows']; restarts+=result['restarts']
            print(json.dumps(dict(done=len(rows)//12,total=len(paths),graph=result['graph'],elapsed=time.perf_counter()-started)),flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(out/'per_graph.csv',index=False)
    pd.DataFrame(restarts).to_csv(out/'per_restart.csv',index=False)
    frame['improved']=frame.errors<frame.baseline_errors; frame['worsened']=frame.errors>frame.baseline_errors
    summary=frame.groupby('key').agg(graphs=('graph','size'),mean_error=('error_rate','mean'),exact=('exact','sum'),baseline_mean_error=('baseline_error_rate','mean'),improved=('improved','sum'),worsened=('worsened','sum'))
    summary.to_csv(out/'summary.csv'); print(summary.to_string(),flush=True)
    print(json.dumps(dict(elapsed=time.perf_counter()-started,graphs=len(paths))),flush=True)

if __name__=='__main__':main()

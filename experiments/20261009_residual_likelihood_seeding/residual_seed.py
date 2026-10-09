"""Likelihood-residual seeding and one-component replacement.

Only U,Lambda enter production methods. Poisson spectral profile working score
is inherited; no exact graph/LOO likelihood or k-means++ recovery theorem.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import gammaln,logsumexp
from sklearn.cluster import kmeans_plusplus
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PARTIAL=ROOT.parent/'spectral_partial_peel_20261009'
if not PARTIAL.exists(): PARTIAL=ROOT.parent/'20261009_partial_likelihood_peeling'
sys.path.insert(0,str(PARTIAL))
from partial_peeling import peel
OLD=ROOT.parent/'spectral_greedy_peel_20261009'
if not OLD.exists(): OLD=ROOT.parent/'20261009_greedy_likelihood_peeling'
sys.path.insert(0,str(OLD))
from profile_peeling import metric,as_json,LEGACY
from poisson_em import _floored_weights

PROTOCOL=dict(
 input='Only U,Lambda during fitting; all saved truth/baseline fields after every fit and selection',
 profile='q=max(U diag(Lambda) U.T,1e-4 log(n)/n), retain diagonal',
 score='ell_i(mu)=sum_j q_ij log(mu_j)-mu_j; all proposals accepted by same raw observed mixture objective',
 residual='min_a D(q_i||mu_a), D(q_i||mu)=sum q_i log(q_i/mu)-q_i+mu; not squared again',
 pp='Uniform first node; sample subsequent candidates proportional residual deviance; no node deletion',
 greedy_pp='2+floor(log K) residual sampled candidate trials; minimize full-data post-addition deviance',
 global_gain='Choose all-node candidate maximizing sum_i max(current_min_deviance_i-D(q_i||q_c),0)',
 growing='Begin with global mean q, add candidate by global gain, all-node EM after each addition',
 growing_sampled='Same growing procedure, but 2+floor(log K) residual sampled candidate trials per addition',
 replacement='Try deleting each current component in turn; choose global-gain candidate against retained components; refit with current weights; accept best observed-mixture likelihood improvement >1e-5; one pass',
 EM='max_iter=300,tol=1e-6, floored-simplex weights; same stopping scale including cached observation Gamma sum',
 restarts='10 greedy PP starts fixed before fitting; select highest final observed likelihood, never truth',
 RNG='SeedSequence([graph_seed,method_tag,restart,20261009]); fixed throughout',
 limitation='Same spectral working score; no exact LOO/graph likelihood; design-stage spectra reused; independent seeds tested separately')

class Work:
    def __init__(self,u,lam):
        self.u,self.lam=u,lam;self.n,self.k=u.shape
        self.q=np.maximum((u*lam)@u.T,1e-4*np.log(self.n)/self.n)
        self.entropy=(self.q*np.log(self.q)-self.q).sum(1)
        self.gamma_sum=float(gammaln(self.q+1).sum())
        self.singleton_scores=None
    def score(self,mu):
        return self.q@np.log(mu).T-mu.sum(1)[None,:]
    def distance(self,mu):
        return np.maximum(self.entropy[:,None]-self.score(mu),0.)
    def all_scores(self):
        if self.singleton_scores is None:self.singleton_scores=self.score(self.q)
        return self.singleton_scores
    def fit(self,initial,weights=None,max_iter=300,tol=1e-6):
        mu=np.array(initial,copy=True);k=len(mu)
        weights=np.ones(k)/k if weights is None else np.array(weights,copy=True)
        trace=[];previous=None;converged=False
        for iteration in range(max_iter+1):
            scores=self.score(mu)+np.log(weights)[None,:]
            normal=logsumexp(scores,axis=1)
            responsibility=np.exp(scores-normal[:,None]);labels=scores.argmax(1)
            obj=float(normal.sum());delta=0. if previous is None else obj-previous[0]
            changed=0 if previous is None else int(np.sum(labels!=previous[1]))
            pdiff=0. if previous is None else float(max(np.max(np.abs(mu-previous[2])/(1+np.abs(previous[2]))),np.max(np.abs(weights-previous[3]))))
            if delta < -1e-7: raise FloatingPointError(f'EM decreased: {delta}')
            trace.append(dict(iteration=iteration,objective=obj,objective_delta=delta,n_changed=changed,
                param_delta=pdiff,class_sizes=np.bincount(labels,minlength=k).tolist()))
            if previous is not None and abs(delta)<=tol*(1+abs(obj-self.gamma_sum)) and changed==0 and pdiff<=np.sqrt(tol):
                converged=True;break
            if iteration==max_iter:break
            mass=responsibility.sum(0)
            previous=(obj,labels.copy(),mu.copy(),weights.copy())
            active=mass>0
            mu[active]=responsibility[:,active].T@self.q/mass[active,None]
            weights=_floored_weights(mass)
        return dict(labels=labels,mu=mu,weights=weights,objective=obj,trace=trace,
            n_iter=len(trace)-1,converged=converged,worst_objective_delta=min(t['objective_delta'] for t in trace))
    def pp(self,rng,greedy=False):
        ids=[int(rng.integers(self.n))];events=[dict(seed=ids[0])]
        loss=self.distance(self.q[ids]).ravel()
        for stage in range(1,self.k):
            probs=loss.copy();probs[ids]=0.
            if probs.sum()<=1e-14:
                candidates=np.asarray([rng.choice(np.setdiff1d(np.arange(self.n),ids))])
            else:
                candidates=rng.choice(self.n,size=2+int(math.log(self.k)) if greedy else 1,p=probs/probs.sum())
            distances=self.distance(self.q[candidates])
            totals=np.minimum(loss[:,None],distances).sum(0)
            chosen=int(candidates[np.argmin(totals)])
            loss=np.minimum(loss,distances[:,np.argmin(totals)])
            ids.append(chosen)
            events.append(dict(seed=chosen,candidates=candidates.tolist(),candidate_total_deviance=totals.tolist(),
                residual_potential=float(loss.sum())))
        return np.asarray(ids),events
    def global_candidate(self,mu,excluded=()):
        loss=self.distance(mu).min(1)
        # gain_i(c) = max(ell_i(c)-max_a ell_i(mu_a),0)
        current=self.entropy-loss
        gains=np.maximum(self.all_scores()-current[:,None],0).sum(0)
        if len(excluded):gains[np.asarray(excluded,int)]=-np.inf
        candidate=int(np.argmax(gains))
        return candidate,dict(candidate=candidate,total_gain=float(gains[candidate]),
            current_potential=float(loss.sum()),candidate_gains=gains.copy())
    def greedy_no_update(self,first):
        ids=[int(first)];events=[]
        for _ in range(1,self.k):
            chosen,event=self.global_candidate(self.q[ids],ids)
            ids.append(chosen);events.append(event)
        return np.asarray(ids),events
    def growing(self,rng=None):
        mu=self.q.mean(0)[None,:];stages=[];ids=[]
        if self.k==1:return self.fit(mu),np.asarray(ids,int),stages
        for size in range(2,self.k+1):
            if rng is None:
                candidate,event=self.global_candidate(mu,ids)
            else:
                loss=self.distance(mu).min(1);probs=loss.copy();probs[ids]=0.
                candidates=rng.choice(self.n,size=2+int(math.log(self.k)),p=probs/probs.sum())
                ds=self.distance(self.q[candidates]);totals=np.minimum(loss[:,None],ds).sum(0)
                candidate=int(candidates[np.argmin(totals)])
                event=dict(candidate=candidate,candidates=candidates.tolist(),candidate_potentials=totals.tolist())
            ids.append(candidate)
            initial=np.vstack([mu,self.q[candidate]])
            fitted=self.fit(initial)
            mu=fitted['mu']
            stages.append(dict(size=size,event=event,fit_trace=fitted['trace'],objective=fitted['objective']))
        return fitted,np.asarray(ids),stages
    def repair(self,baseline):
        records=[];best=baseline;accepted=None
        if self.k==1:return best,dict(accepted=accepted,proposals=records)
        # All proposals evaluated against same original endpoint, one pass.
        for removed in range(self.k):
            others=np.delete(baseline['mu'],removed,axis=0)
            candidate,event=self.global_candidate(others)
            initial=baseline['mu'].copy();initial[removed]=self.q[candidate]
            fitted=self.fit(initial,weights=baseline['weights'])
            records.append(dict(removed_component=removed,candidate=candidate,
                proposal_objective=fitted['objective'],gain_vs_baseline=fitted['objective']-baseline['objective'],
                candidate_event=event,fit_trace=fitted['trace']))
            if fitted['objective']>best['objective']+1e-5:
                best=fitted;accepted=dict(removed_component=removed,candidate=candidate)
        return best,dict(accepted=accepted,proposals=records)


def fit_spectral_profiles(u,lam,method='repair'):
    """Fit without labels, graph adjacency, or a k-means decoder.

    method='repair': user's fixed-K smaller peeling followed by one-component
    likelihood replacement; method='global_gain': deterministic growing EM.
    Returns labels, positive profile means, weights, objective, and fit trace.
    """
    with threadpool_limits(limits=1):
        work=Work(np.asarray(u),np.asarray(lam))
        if method=='global_gain':
            fitted,ids,events=work.growing()
        elif method=='repair':
            ids,cores,peel_events,leftover=peel(work.all_scores(),work.k,'scaled_fixed_k')
            fitted=work.fit(work.q[ids])
            fitted,events=work.repair(fitted)
        else:raise ValueError("method must be 'repair' or 'global_gain'")
        return fitted|dict(seed_ids=ids,events=events)

def run_one(task):
    path,out,restarts=task;path,out=Path(path),Path(out);started=time.perf_counter()
    model,n,ch,seed=re.fullmatch(r'(.+)_n(\d+)_ch([\d.]+)_s(\d+)',path.stem).groups()
    seed=int(seed);arrays={};outputs=[];records={};restart_outputs=[]
    def add(method,fit,initial=None,ids=None,events=None):
        outputs.append((method,fit))
        records[method]=dict(trace=fit['trace'],events=events)
        for name in ('labels','mu','weights'):arrays[method+'_'+name]=fit[name]
        if initial is not None:arrays[method+'_initial_mu']=initial
        if ids is not None:arrays[method+'_seed_ids']=ids
    with np.load(path,allow_pickle=False) as source:
        u,lam=source['u'],source['lam'];arrays.update(u=u,lam=lam)
        with threadpool_limits(limits=1):
            work=Work(u,lam)
            scores=work.all_scores()
            ids,cores,events,leftover=peel(scores,work.k,'scaled_fixed_k')
            original=work.fit(work.q[ids])
            add('raw_peel',original,work.q[ids],ids,events)
            repaired,repair_events=work.repair(original)
            add('raw_peel_repaired',repaired,events=repair_events)
            rng=np.random.default_rng(np.random.SeedSequence([seed,701,0,20261009]))
            plain_ids,plain_events=work.pp(rng,False)
            add('residual_pp_single',work.fit(work.q[plain_ids]),work.q[plain_ids],plain_ids,plain_events)
            fits=[]
            for restart in range(restarts):
                rng=np.random.default_rng(np.random.SeedSequence([seed,709,restart,20261009]))
                pp_ids,pp_events=work.pp(rng,True)
                fit=work.fit(work.q[pp_ids]);fits.append((fit,pp_ids,pp_events))
                arrays[f'greedy_restart_{restart}_labels']=fit['labels']
                arrays[f'greedy_restart_{restart}_ids']=pp_ids
                restart_outputs.append((restart,fit))
                records[f'greedy_restart_{restart}']=dict(trace=fit['trace'],events=pp_events)
            add('residual_greedy_pp_single',fits[0][0],work.q[fits[0][1]],fits[0][1],fits[0][2])
            chosen=max(range(len(fits)),key=lambda i:fits[i][0]['objective'])
            add('residual_greedy_pp_best10',fits[chosen][0],work.q[fits[chosen][1]],fits[chosen][1],dict(selected_restart=chosen,events=fits[chosen][2]))
            greedy_ids,greedy_events=work.greedy_no_update(ids[0])
            add('global_gain_no_update',work.fit(work.q[greedy_ids]),work.q[greedy_ids],greedy_ids,greedy_events)
            grown,grown_ids,grown_events=work.growing()
            add('growing_global_gain_EM',grown,ids=grown_ids,events=grown_events)
            rng=np.random.default_rng(np.random.SeedSequence([seed,719,0,20261009]))
            sampled,sampled_ids,sampled_events=work.growing(rng)
            add('growing_sampled_gain_EM',sampled,ids=sampled_ids,events=sampled_events)
            x=np.sqrt(len(u))*u*lam
            _,euc_ids=kmeans_plusplus(x,n_clusters=len(lam),random_state=seed+17389,n_local_trials=1)
            add('Euclidean_X_pp_single_EM',work.fit(work.q[euc_ids]),work.q[euc_ids],euc_ids)
        # Only now access all truth and precomputed baseline fields.
        truth=source['truth_evaluation_only'];base=source['baseline_km_X']
        arrays.update(truth_evaluation_only=truth,baseline_km_X=base)
        rows=[]
        common=dict(graph=path.stem,model=model,n=len(u),k=len(lam),ch=float(ch),seed=seed)
        for method,fit in outputs:
            row=common|dict(method=method,**metric(fit['labels'],truth,len(lam)),
                objective=fit['objective'],n_iter=fit['n_iter'],converged=fit['converged'],
                worst_objective_delta=fit['worst_objective_delta'])
            if method+'_seed_ids' in arrays:
                row['seed_classes']=truth[arrays[method+'_seed_ids']].tolist()
                row['seed_coverage']=len(np.unique(truth[arrays[method+'_seed_ids']]))
            rows.append(row)
        rows.append(common|dict(method='X_kmeans',**metric(base,truth,len(lam))))
        restart_rows=[common|dict(restart=r,objective=fit['objective'],n_iter=fit['n_iter'],converged=fit['converged'],
            **metric(fit['labels'],truth,len(lam))) for r,fit in restart_outputs]
        if model=='equal3' and len(u)==2000 and float(ch)==.8 and seed==20004:
            records['failure_graph_centroid_profiles_evaluation_only']={method:[[float(mu[truth==a].mean()*len(u)/np.log(len(u))) for a in range(len(lam))] for mu in fit['mu']] for method,fit in outputs}
    np.savez_compressed(out/'arrays'/f'{path.stem}.npz',**arrays)
    payload=dict(graph=path.stem,rows=rows,restart_rows=restart_rows,records=records,elapsed=time.perf_counter()-started)
    (out/'runs'/f'{path.stem}.json').write_text(json.dumps(payload,default=as_json),encoding='utf-8')
    return rows,restart_rows,payload['elapsed']

def summarize(rows,restarts,out):
    f=pd.DataFrame(rows).sort_values(['graph','method']);f.to_csv(out/'per_graph.csv',index=False)
    r=pd.DataFrame(restarts).sort_values(['graph','restart']);r.to_csv(out/'per_restart.csv',index=False)
    summary=f.groupby('method').agg(graphs=('graph','size'),mean_error=('error_rate','mean'),exact=('exact','sum'),
        max_error=('error_rate','max'),converged=('converged','sum'),mean_iter=('n_iter','mean'),min_objective_delta=('worst_objective_delta','min'))
    summary.to_csv(out/'summary.csv')
    f.groupby(['model','method']).agg(graphs=('graph','size'),mean_error=('error_rate','mean'),exact=('exact','sum'),max_error=('error_rate','max')).to_csv(out/'by_model.csv')
    f.groupby(['model','n','ch','method']).agg(graphs=('graph','size'),mean_error=('error_rate','mean'),exact=('exact','sum')).to_csv(out/'by_condition.csv')
    baseline=f[f.method=='raw_peel'];pairs=[]
    for method in f.method.unique():
        if method=='raw_peel':continue
        merged=f[f.method==method].merge(baseline[['graph','errors','objective']],on='graph',suffixes=('_new','_old'))
        d=merged.errors_new-merged.errors_old
        pairs.append(dict(method=method,better=int((d<0).sum()),equal=int((d==0).sum()),worse=int((d>0).sum()),
            lower_objective=int((merged.objective_new-merged.objective_old< -1e-5).sum())))
    (out/'paired.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,default=LEGACY/'full120/inputs')
    parser.add_argument('--out',type=Path,default=ROOT/'design_results');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--failure-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--restarts',type=int,default=10)
    args=parser.parse_args();paths=sorted(args.input_dir.glob('*.npz'))
    if args.failure_only:paths=[p for p in paths if p.stem=='equal3_n2000_ch0.8_s20004']
    if not paths:raise FileNotFoundError(args.input_dir)
    out=args.out
    for sub in ('arrays','runs'):(out/sub).mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(PROTOCOL|dict(graphs=len(paths),restarts=args.restarts),indent=2),encoding='utf-8')
    rows,restarts,todo=[],[],[]
    for path in paths:
        saved=out/'runs'/f'{path.stem}.json'
        if args.resume and saved.exists():
            p=json.loads(saved.read_text(encoding='utf-8'));rows.extend(p['rows']);restarts.extend(p['restart_rows'])
        else:todo.append(path)
    started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(run_one,(str(p),str(out),args.restarts)) for p in todo]
        for done,future in enumerate(concurrent.futures.as_completed(futures),1):
            result,runs,seconds=future.result();rows.extend(result);restarts.extend(runs)
            pd.DataFrame(rows).to_csv(out/'partial.csv',index=False)
            print(json.dumps(dict(done=done,total=len(todo),graph=result[0]['graph'],seconds=round(seconds,2),elapsed=round(time.perf_counter()-started,1))),flush=True)
    summarize(rows,restarts,out)

if __name__=='__main__':main()

"""Raw top-fraction peeling under expected Poisson/Bernoulli row scores.

Only eigenpairs are fitting input. These dependent fractional spectral rows
produce a working expected row log score; this is not exact LOO likelihood.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from threadpoolctl import threadpool_limits
from profile_peeling import peel, fit_soft, Profile, metric, as_json, LEGACY
from poisson_em import _floored_weights

PROTOCOL={
 'input':'Only U and Lambda are consulted during fitting. Truth and baseline labels are read after fitting.',
 'poisson_score':'q_i @ log(mu) - sum(mu); no fractional gamma constant in subset selection.',
 'bernoulli_score':'q_i @ log(p/(1-p)) + sum(log(1-p)); expected raw Bernoulli row log score.',
 'poisson_profile':'max(U diag(Lambda) U.T, 1e-4 log(n)/n), without upper clipping.',
 'bernoulli_profile':'clip(U diag(Lambda) U.T, 1e-4 log(n)/n, 1-1e-8). Diagonal entries retained as in base profile experiment.',
 'peeling':'Imported exact peel rule: floor(M/r) other nodes, guarded to preserve future candidates; candidate excluded from total and removed along with its best nodes.',
 'initial_parameters':'Selected spectral profile rows directly. No replacement with group means, labels or k-means decoder.',
 'soft_iteration':'All n observations free after seeding. Full row parameters update as weighted fractional profile means; weights use exact simplex-constrained floored update.',
 'hard_control':'Poisson legacy Profile.lloyd from direct candidate first E assignment.',
 'caveat':'Spectral expected row working log score only; dependent fractional rows, not an exact graph/LOO likelihood, and no recovery theorem.',
 'max_iter':300,'tolerance':1e-6
}

def fit_bernoulli(q, initial_p, max_iter=300, tol=1e-6):
    p=np.array(initial_p,copy=True)
    n=len(q); k=len(p); weights=np.ones(k)/k
    trace=[]; previous_labels=None; previous_obj=None
    previous_p=None; previous_weights=None; converged=False
    for iteration in range(max_iter+1):
        component=q@np.log(p/(1-p)).T + np.log1p(-p).sum(axis=1)[None,:]
        scores=component+np.log(weights)[None,:]
        norms=logsumexp(scores,axis=1)
        responsibilities=np.exp(scores-norms[:,None])
        labels=scores.argmax(axis=1); objective=float(norms.sum())
        changed=0 if previous_labels is None else int(np.sum(labels!=previous_labels))
        delta=0. if previous_obj is None else objective-previous_obj
        param_delta=0. if previous_p is None else float(max(
            np.max(np.abs(p-previous_p)/(1+np.abs(previous_p))),
            np.max(np.abs(weights-previous_weights))))
        if previous_obj is not None and delta < -1e-9*max(1.,abs(previous_obj)):
            raise FloatingPointError('Bernoulli working mixture objective decreased')
        trace.append(dict(iteration=iteration,objective=objective,
            objective_delta=delta,n_changed=changed,param_delta=param_delta,
            class_sizes=np.bincount(labels,minlength=k).tolist(),
            expected_class_min=float(responsibilities.sum(axis=0).min())))
        if previous_obj is not None and abs(delta)<=tol*(1+abs(previous_obj)) and changed==0 and param_delta<=np.sqrt(tol):
            converged=True; break
        if iteration==max_iter:break
        mass=responsibilities.sum(axis=0)
        previous_labels=labels.copy(); previous_obj=objective
        previous_p=p.copy(); previous_weights=weights.copy()
        active=mass>0
        p[active]=(responsibilities[:,active].T@q)/mass[active,None]
        p=np.clip(p,1e-12,1-1e-8)
        weights=_floored_weights(mass)
    return dict(labels=labels,p=p,weights=weights,trace=trace,
        n_iter=len(trace)-1,converged=converged,objective=objective,
        worst_objective_delta=min(t['objective_delta'] for t in trace))

def validate():
    rng=np.random.default_rng(20261009031)
    truths=rng.integers(0,3,size=24)
    centers=np.array([[.07,.15,.75,.2,.8],[.8,.4,.1,.7,.2],[.3,.7,.6,.15,.4]])
    q=np.clip(centers[truths]+rng.normal(0,.025,size=(24,5)),.01,.99)
    p0=q[[1,7,16]]
    fitted=fit_bernoulli(q,p0,max_iter=300,tol=1e-10)
    # Direct scalar Bernoulli log-score oracle, including fractional observations.
    vec=q@np.log(p0/(1-p0)).T+np.log1p(-p0).sum(1)[None,:]
    oracle=np.array([[sum(float(x)*np.log(float(p))+(1-float(x))*np.log1p(-float(p))
                         for x,p in zip(row,center)) for center in p0] for row in q])
    np.testing.assert_allclose(vec,oracle,atol=2e-14)
    assert fitted['worst_objective_delta']>=-1e-10
    assert fitted['converged']
    # Gamma constants do not affect fixed-model responsibilities, only peeling.
    qpois=rng.uniform(.01,.3,size=(24,5))
    mu0=qpois[[1,7,16]]
    comp=qpois@np.log(mu0).T-mu0.sum(1)[None,:]
    gamma=gammaln(qpois+1).sum(1)
    np.testing.assert_allclose(
        np.exp(comp-logsumexp(comp,axis=1)[:,None]),
        np.exp(comp-gamma[:,None]-logsumexp(comp-gamma[:,None],axis=1)[:,None]),atol=1e-14)
    return dict(bernoulli_scalar_oracle_max_error=float(np.abs(vec-oracle).max()),
        bernoulli_monotone=True,bernoulli_iterations=fitted['n_iter'],
        bernoulli_worst_delta=fitted['worst_objective_delta'],
        gamma_responsibilities_identical=True)

def run_one(task):
    path,out=map(Path,task); started=time.perf_counter(); graph=path.stem
    model=graph.split('_n')[0]
    with np.load(path) as source:
        u,lam=source['u'],source['lam']
        arrays=dict(u=u,lam=lam); payload=dict(graph=graph,methods={}); outputs=[]
        with threadpool_limits(limits=1):
            prof=Profile(u,lam); q=prof.q
            score=q@np.log(q).T-q.sum(1)[None,:]
            ids,peeled,events=peel(score,prof.k)
            mu0=q[ids].copy(); initial=prof.deviance(mu0).argmin(axis=1)
            stem='poisson_expected_raw_peel'
            arrays[stem+'_seed_ids']=ids; arrays[stem+'_peeled_labels']=peeled
            arrays[stem+'_initial_mu']=mu0; arrays[stem+'_initial_E_labels']=initial
            hard,obj,hardtrace,conv,repairs=prof.lloyd(initial)
            arrays[stem+'_hard_labels']=hard
            outputs.append((stem+'_hard',hard,dict(n_iter=len(hardtrace)-1,
                converged=conv,profile_deviance=obj,repairs=repairs)))
            soft=fit_soft(q,mu0)
            pred=soft.pop('labels'); arrays[stem+'_soft_labels']=pred
            arrays[stem+'_final_mu']=soft.pop('mu'); arrays[stem+'_final_weights']=soft.pop('weights')
            # Base function includes gamma term, which is constant during EM.
            gamma_sum=float(gammaln(q+1).sum())
            soft['objective']+=gamma_sum
            for t in soft['trace']:t['objective']+=gamma_sum
            outputs.append((stem+'_soft',pred,{key:val for key,val in soft.items() if key!='trace'}))
            payload['methods'][stem]=dict(seed_events=events,hard_trace=hardtrace,soft_trace=soft['trace'])
            # Bernoulli profile is built independently from the same eigenpairs.
            lower=1e-4*np.log(len(u))/len(u)
            qb=np.clip((u*lam)@u.T,lower,1-1e-8)
            scoreb=qb@np.log(qb/(1-qb)).T+np.log1p(-qb).sum(1)[None,:]
            ids,peeled,events=peel(scoreb,prof.k)
            p0=qb[ids].copy(); initial=scoreb[:,ids].argmax(1)
            stem='bernoulli_expected_raw_peel'
            arrays[stem+'_seed_ids']=ids; arrays[stem+'_peeled_labels']=peeled
            arrays[stem+'_initial_p']=p0; arrays[stem+'_initial_E_labels']=initial
            soft=fit_bernoulli(qb,p0)
            pred=soft.pop('labels'); arrays[stem+'_soft_labels']=pred
            arrays[stem+'_final_p']=soft.pop('p'); arrays[stem+'_final_weights']=soft.pop('weights')
            outputs.append((stem+'_soft',pred,{key:val for key,val in soft.items() if key!='trace'}))
            payload['methods'][stem]=dict(seed_events=events,soft_trace=soft['trace'])
        # Ground truth is used only now, for final evaluation.
        truth=source['truth_evaluation_only']
        base=source['baseline_km_X'] if 'baseline_km_X' in source else source['labels_km_X']
        arrays['truth_evaluation_only']=truth; arrays['baseline_km_X']=base
        base_metric=metric(base,truth,len(lam)); rows=[]
        for name,labels,stats in outputs:
            stem=name.rsplit('_',1)[0]
            rows.append(dict(graph=graph,model=model,n=len(u),k=len(lam),method=name,
                **stats,**metric(labels,truth,len(lam)),
                peeled_error_rate=metric(arrays[stem+'_peeled_labels'],truth,len(lam))['error_rate'],
                initial_E_error_rate=metric(arrays[stem+'_initial_E_labels'],truth,len(lam))['error_rate'],
                seed_coverage=len(np.unique(truth[arrays[stem+'_seed_ids']])),
                baseline_errors=base_metric['errors'],baseline_error_rate=base_metric['error_rate']))
    payload['rows']=rows; payload['elapsed']=time.perf_counter()-started
    np.savez_compressed(out/'arrays'/f'{graph}.npz',**arrays)
    (out/'runs'/f'{graph}.json').write_text(json.dumps(payload,default=as_json),encoding='utf-8')
    return rows,payload['elapsed']

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--validate',action='store_true'); parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--input-dir',type=Path,default=LEGACY/'full120'/'inputs')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(); own=Path(__file__).resolve().parent
    out=args.out or own/'profile_score_controls_results'
    out.mkdir(parents=True,exist_ok=True)
    checks=validate(); (out/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    if args.validate:print(json.dumps(checks));return
    paths=sorted(args.input_dir.glob('*.npz'))
    if not paths:raise FileNotFoundError(f'No spectral inputs in {args.input_dir}')
    if args.pilot:paths=[p for p in paths if '_n1000_ch0.8_s20000' in p.stem]
    (out/'arrays').mkdir(exist_ok=True); (out/'runs').mkdir(exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(PROTOCOL|dict(graph_count=len(paths)),indent=2),encoding='utf-8')
    rows=[]; started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending={executor.submit(run_one,(str(p),str(out))):p for p in paths}
        for done,future in enumerate(concurrent.futures.as_completed(pending),1):
            result,seconds=future.result(); rows.extend(result)
            print(json.dumps(dict(done=done,total=len(paths),graph=result[0]['graph'],graph_seconds=seconds,elapsed=time.perf_counter()-started)),flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(out/'per_graph.csv',index=False)
    frame['improved']=frame.errors<frame.baseline_errors
    frame['worsened']=frame.errors>frame.baseline_errors
    summary=frame.groupby('method').agg(graphs=('graph','size'),mean_error=('error_rate','mean'),
        exact=('exact','sum'),converged=('converged','sum'),mean_iter=('n_iter','mean'),max_iter=('n_iter','max'),
        peeled_error=('peeled_error_rate','mean'),initial_E_error=('initial_E_error_rate','mean'),
        baseline_error=('baseline_error_rate','mean'),improved=('improved','sum'),worsened=('worsened','sum'),
        seed_full_coverage=('seed_coverage',lambda x:int(np.sum(x==frame.loc[x.index,'k']))))
    summary.to_csv(out/'summary.csv')
    frame.groupby(['model','method']).agg(graphs=('graph','size'),mean_error=('error_rate','mean'),exact=('exact','sum')).to_csv(out/'by_model.csv')
    print(summary.to_string());print(json.dumps(dict(elapsed=time.perf_counter()-started,graphs=len(paths))),flush=True)

if __name__=='__main__':main()

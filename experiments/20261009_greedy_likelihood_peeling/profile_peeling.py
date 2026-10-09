"""Greedy top-fraction raw likelihood seeding using only saved eigenpairs.

User algorithm: each observation supplies a candidate profile; each candidate
claims the best floor(M/r) other remaining observations; maximize that
subset's total raw log score, remove it, and repeat. Candidate rates themselves
seed unconstrained full-data iterations. This is a spectral working score,
not an exact Bernoulli or leave-one-out graph likelihood.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import concurrent.futures
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from threadpoolctl import threadpool_limits

LEGACY = Path(__file__).resolve().parent.parent / 'spectral_decoder_pp_20261009'
if not LEGACY.exists():
    LEGACY = Path(__file__).resolve().parent / 'legacy_profile'
sys.path.insert(0, str(LEGACY))
from profile_benchmark import Profile, metric, as_json

PROTOCOL = {
    'input': 'Only U and Lambda used for fitting; truth and baseline read after fitting.',
    'candidate': 'q_c=max((U diag(Lambda) U.T)_c,1e-4 log(n)/n); spectral profiles fixed throughout.',
    'score': 'ell_i(mu)=sum_j[q_ij log(mu_j)-mu_j-gammaln(q_ij+1)]. Generalized fractional Poisson working score including observation-specific constants.',
    'seeding': 'At remaining size M and r components, other_count=min(floor(M/r),M-r) if r>1; otherwise M-1. Score highest-scoring other_count OTHER nodes, excluding candidate self score. Remove them and c. M-r guard leaves >=1 node for each future component. Last step scores every remaining candidate on all OTHER remaining nodes.',
    'ties': 'Smallest candidate/node index wins exact ties.',
    'initial_parameters': 'Selected q_c rows, unchanged; no group-mean replacement.',
    'iteration': 'All n nodes free after seeding. Soft EM updates means and mixture weights; comparison also uses legacy unweighted Poisson Bregman Lloyd after a first E assignment.',
    'control': 'Separate deviance peeling arm chooses largest total -D; it is not the user raw-score algorithm.',
    'max_iter_soft': 300,
    'tolerance_soft': 1e-6,
    'caveat': 'No exact LOO, original adjacency, graph-likelihood claim, or recovery/global-optimum theorem.',
}

def peel(score, k):
    """score[i,c] is raw log score of observation i at candidate c."""
    score = np.asarray(score, float)
    n = score.shape[0]
    if score.shape != (n, n) or not np.all(np.isfinite(score)) or not 1 <= k <= n:
        raise ValueError('Need a finite square score matrix and 1<=K<=n')
    remaining = np.arange(n)
    labels = np.full(n, -1, int)
    ids, events = [], []
    for stage in range(k):
        r = k-stage
        others_count = min(len(remaining)//r, len(remaining)-r) if r>1 else len(remaining)-1
        m = others_count+1
        sub = score[np.ix_(remaining, remaining)]
        diagonal = np.diag(sub).copy()
        if others_count == len(remaining)-1:
            totals = sub.sum(axis=0)-diagonal
        elif others_count == 0:
            totals = np.zeros(len(remaining))
        else:
            # Candidate must belong to its removed group.
            np.fill_diagonal(sub, -np.inf)
            top = np.partition(sub, len(remaining)-others_count, axis=0)[-others_count:]
            totals = top.sum(axis=0)
        winner = int(np.argmax(totals))
        candidate = int(remaining[winner])
        others = remaining[remaining != candidate]
        order = np.lexsort((others, -score[others, candidate]))
        members = np.concatenate(([candidate], others[order[:m-1]]))
        labels[members] = stage
        ids.append(candidate)
        events.append(dict(stage=stage, remaining_components=r, group_size=m,
                           candidate=candidate, scored_other_count=others_count, members=members.tolist(),
                           remaining_candidates=remaining.tolist(),
                           candidate_totals=totals.tolist(),
                           winning_total=float(totals[winner])))
        remaining = remaining[~np.isin(remaining, members)]
    if remaining.size or np.any(labels < 0) or len(set(ids)) != k:
        raise AssertionError('Peeling did not partition all observations')
    return np.array(ids), labels, events

def fit_soft(q, initial_mu, max_iter=300, tol=1e-6):
    """EM for fixed-profile generalized Poisson mixture, directly seeded by mu."""
    n = len(q)
    mu = np.array(initial_mu, copy=True)
    k = len(mu)
    weights = np.ones(k)/k
    const = gammaln(q+1).sum(axis=1)
    trace = []
    previous_labels = None
    previous_obj = None
    previous_mu = None
    previous_weights = None
    converged = False
    for iteration in range(max_iter+1):
        component = q@np.log(mu).T - mu.sum(axis=1)[None,:] - const[:,None]
        scores = component + np.log(weights)[None,:]
        norms = logsumexp(scores, axis=1)
        responsibility = np.exp(scores-norms[:,None])
        labels = scores.argmax(axis=1)
        objective = float(norms.sum())
        changed = 0 if previous_labels is None else int((labels != previous_labels).sum())
        delta = 0. if previous_obj is None else objective-previous_obj
        param_delta = 0. if previous_mu is None else float(max(
            np.max(np.abs(mu-previous_mu)/(1+np.abs(previous_mu))),
            np.max(np.abs(weights-previous_weights))))
        if previous_obj is not None and delta < -1e-9*max(1.,abs(previous_obj)):
            raise FloatingPointError('Profile mixture objective decreased')
        trace.append(dict(iteration=iteration, objective=objective,
                          objective_delta=delta, n_changed=changed,
                          param_delta=param_delta,
                          class_sizes=np.bincount(labels,minlength=k).tolist(),
                          expected_class_min=float(responsibility.sum(axis=0).min())))
        if previous_obj is not None and abs(delta)<=tol*(1+abs(previous_obj)) and changed==0 and param_delta<=np.sqrt(tol):
            converged = True
            break
        if iteration == max_iter:
            break
        mass = responsibility.sum(axis=0)
        previous_labels, previous_obj = labels.copy(), objective
        previous_mu, previous_weights = mu.copy(), weights.copy()
        active = mass > 0
        mu[active] = (responsibility[:,active].T@q)/mass[active,None]
        # Use exact simplex-constrained weight update from the PG module.
        from poisson_em import _floored_weights
        weights = _floored_weights(mass)
    return dict(labels=labels, mu=mu, weights=weights, trace=trace,
                n_iter=len(trace)-1, converged=converged, objective=objective,
                worst_objective_delta=min(t['objective_delta'] for t in trace))

def validate():
    rng = np.random.default_rng(20261009)
    score = rng.normal(size=(8,8))
    ids, labels, events = peel(score,3)
    checked = 0
    for event in events:
        rem = event['remaining_candidates']; m = event['group_size']
        brute = []
        for c in rem:
            subsets = itertools.combinations([i for i in rem if i != c],m-1)
            best = max(sum(score[i,c] for i in s) for s in subsets)
            brute.append(best)
        np.testing.assert_allclose(brute,event['candidate_totals'],atol=1e-12)
        assert rem[int(np.argmax(brute))] == event['candidate']
        checked += len(rem)
    # The raw-score rule changes under node-specific constants. Regret does not.
    scores=np.random.default_rng(99).normal(size=(8,8))
    first_raw=int(peel(scores,3)[0][0])
    shifted=scores+np.arange(8)[:,None]*10
    first_shifted=int(peel(shifted,3)[0][0])
    assert first_raw != first_shifted
    q=rng.uniform(.01,.1,size=(12,7))
    result=fit_soft(q,q[[0,5,8]],max_iter=300)
    assert result['worst_objective_delta']>=-1e-8
    return dict(brute_candidate_checks=checked,partition_sizes=np.bincount(labels).tolist(),
                constants_change_raw_winner=True,soft_em_monotone=True)

def run_one(task):
    path, out = map(Path,task)
    started=time.perf_counter()
    graph=path.stem
    model=graph.split('_n')[0]
    with np.load(path) as source:
        u,lam=source['u'],source['lam']
        with threadpool_limits(limits=1):
            profile=Profile(u,lam)
            q=profile.q
            const=gammaln(q+1).sum(axis=1)
            score=q@np.log(q).T-q.sum(axis=1)[None,:]-const[:,None]
            arrays=dict(u=u,lam=lam)
            payload=dict(graph=graph,methods={})
            candidates=[]
            for name,sc in [('raw_likelihood_peel',score),
                            ('deviance_peel_control',score+const[:,None]-profile.entropy[:,None])]:
                # second score = -D exactly; don't conflate it with raw score.
                ids,peeled,events=peel(sc,profile.k)
                mu0=q[ids].copy()
                arrays[name+'_seed_ids']=ids
                arrays[name+'_peeled_labels']=peeled
                arrays[name+'_initial_mu']=mu0
                e_labels=profile.deviance(mu0).argmin(axis=1)
                arrays[name+'_initial_E_labels']=e_labels
                labels,obj,trace,converged,repairs=profile.lloyd(e_labels)
                arrays[name+'_hard_labels']=labels
                candidates.append((name+'_hard',labels,dict(n_iter=len(trace)-1,
                    converged=converged,profile_deviance=obj,repairs=repairs)))
                soft=fit_soft(q,mu0)
                prediction=soft.pop('labels')
                arrays[name+'_soft_labels']=prediction
                arrays[name+'_final_mu']=soft.pop('mu')
                arrays[name+'_final_weights']=soft.pop('weights')
                candidates.append((name+'_soft',prediction,{a:b for a,b in soft.items() if a!='trace'}))
                payload['methods'][name]=dict(seed_events=events,hard_trace=trace,soft_trace=soft['trace'])
        # Evaluation only now; no ground truth or k-means used in above fits.
        truth=source['truth_evaluation_only']
        base=source['baseline_km_X'] if 'baseline_km_X' in source else source['labels_km_X']
        arrays['truth_evaluation_only']=truth
        rows=[]
        for name,labels,stats in candidates:
            initializer=name.rsplit('_',1)[0]
            rows.append(dict(graph=graph,model=model,n=len(u),k=len(lam),method=name,
                **stats,**metric(labels,truth,len(lam)),
                peeled_error_rate=metric(arrays[initializer+'_peeled_labels'],truth,len(lam))['error_rate'],
                initial_E_error_rate=metric(arrays[initializer+'_initial_E_labels'],truth,len(lam))['error_rate'],
                seed_coverage=len(np.unique(truth[arrays[initializer+'_seed_ids']])),
                baseline_error_rate=metric(base,truth,len(lam))['error_rate']))
    payload['rows']=rows;payload['elapsed']=time.perf_counter()-started
    np.savez_compressed(out/'arrays'/f'{graph}.npz',**arrays)
    (out/'runs'/f'{graph}.json').write_text(json.dumps(payload,default=as_json),encoding='utf-8')
    return rows,payload['elapsed']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--validate',action='store_true')
    ap.add_argument('--input-dir',type=Path,default=LEGACY/'full120'/'inputs')
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    own=Path(__file__).resolve().parent
    if args.validate:
        result=validate();(own/'profile_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result);return
    paths=sorted(args.input_dir.glob('*.npz'))
    if not paths:raise FileNotFoundError(f'No spectral inputs in {args.input_dir}')
    if args.pilot:paths=[p for p in paths if '_n1000_ch0.8_s20000' in p.stem]
    out=args.out or own/('profile_pilot' if args.pilot else 'profile_results')
    (out/'arrays').mkdir(parents=True,exist_ok=True);(out/'runs').mkdir(exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(PROTOCOL|dict(graph_count=len(paths)),indent=2),encoding='utf-8')
    rows=[];started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        pending={ex.submit(run_one,(str(p),str(out))):p for p in paths}
        for done,future in enumerate(concurrent.futures.as_completed(pending),1):
            row,elapsed=future.result();rows.extend(row)
            print(json.dumps(dict(done=done,total=len(paths),graph=row[0]['graph'],graph_seconds=elapsed,elapsed=time.perf_counter()-started)),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_graph.csv',index=False)
    summary=frame.groupby('method').agg(graphs=('graph','size'),mean_error=('error_rate','mean'),exact=('exact','sum'),converged=('converged','sum'),mean_iter=('n_iter','mean'),max_iter=('n_iter','max'),initial_E_error=('initial_E_error_rate','mean'))
    summary.to_csv(out/'summary.csv');print(summary.to_string())

if __name__=='__main__':main()

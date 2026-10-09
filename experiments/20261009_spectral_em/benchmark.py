"""Paired spectral-only likelihood refinement at B=(log n/n) C.

The generator sees A and truth. fit_spectral_methods receives only eigenpairs;
truth is consulted after all fitting and model selection have finished.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from poisson_em import fit_poisson_em

MODELS = {
    'symmetric2': {'C0': [[4,1],[1,4]], 'pi': [.5,.5]},
    'equal3': {'C0': [[10,1,2],[1,11,1],[2,1,10]], 'pi': [1/3]*3},
    'heterogeneous3': {'C0': [[6,.5,.5],[.5,12,.5],[.5,.5,24]], 'pi': [1/3]*3},
    'unequal3': {'C0': [[10,1,1],[1,10,1],[1,1,10]], 'pi': [.5,.3,.2]},
    'hierarchical4': {'C0': [[12,2,.5,.5],[2,12,.5,.5],[.5,.5,12,2],[.5,.5,2,12]], 'pi': [.25]*4},
    'disassortative3': {'C0': [[.5,8,8],[8,.5,8],[8,8,.5]], 'pi': [1/3]*3},
}

def minimum_ch(c, pi):
    ds = []
    for a in range(len(pi)):
        for b in range(a):
            def d(s):
                return float(np.sum(pi*((1-s)*c[a]+s*c[b]-c[a]**(1-s)*c[b]**s)))
            ds.append(-minimize_scalar(lambda s: -d(s), bounds=(0.,1.), method='bounded').fun)
    return min(ds)

def normalized_model(name, target):
    model = MODELS[name]
    pi = np.asarray(model['pi'], float)
    c = np.asarray(model['C0'], float)
    return c*(target/minimum_ch(c, pi)), pi

def generate_eigenpairs(name, n, target, seed):
    """Only this function reads adjacency; returns compressed data + evaluator metadata."""
    rng = np.random.default_rng(seed)
    c, pi = normalized_model(name, target)
    sizes = np.floor(n*pi).astype(int)
    sizes[-1] += n-sizes.sum()
    z = rng.permutation(np.repeat(np.arange(len(pi)), sizes))
    rr, cc = np.triu_indices(n, 1)
    probs = np.log(n)/n*c[z[rr], z[cc]]
    if probs.max() >= 1:
        raise ValueError('Bernoulli probability exceeds one')
    keep = rng.random(len(rr)) < probs
    ri, ci = rr[keep], cc[keep]
    a = sparse.csr_matrix((np.ones(2*len(ri)), (np.r_[ri,ci], np.r_[ci,ri])), shape=(n,n))
    k = len(pi)
    vals, vecs = eigsh(a, k=k+1, which='LM', v0=rng.normal(size=n), tol=1e-8)
    ix = np.argsort(-np.abs(vals))
    selected = ix[:k]
    u, lam = vecs[:,selected], vals[selected]
    # Population signal and adjacency diagnostics are evaluator-only.
    q = sizes/n
    population = np.linalg.eigvalsh(np.sqrt(q)[:,None]*c*np.sqrt(q)[None,:])*np.log(n)
    overlap = sum(np.sum(u[z == b].sum(0)**2)/sizes[b] for b in range(k))
    meta = dict(c=c.tolist(), pi=pi.tolist(), sizes=sizes.tolist(),
                p_max=float(probs.max()), eigenvalues=lam.tolist(),
                first_discarded_eigenvalue=float(vals[ix[k]]),
                population_eigenvalues=population.tolist(),
                edge_count=int(len(ri)), subspace_loss=float(k-overlap))
    return u, lam, z, meta

def km(x, k, seed):
    obj = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=1e-5, random_state=seed).fit(x)
    return obj.labels_, {'iterations': int(obj.n_iter_), 'objective': float(-obj.inertia_)}

def fit_spectral_methods(u, lam, seed, max_iter, restarts=False):
    """No adjacency or ground-truth argument. All hyperparameters fixed in protocol."""
    n, k = u.shape
    x = np.sqrt(n)*u*lam
    labels_u, stat_u = km(np.sqrt(n)*u, k, seed)
    labels_x, stat_x = km(x, k, seed)
    fitted = {'km_U': (labels_u, stat_u), 'km_X': (labels_x, stat_x)}
    h0 = np.stack([np.sqrt(n)*u[labels_u == b].mean(0) for b in range(k)])
    condition = float(np.linalg.cond(h0))
    if not np.isfinite(condition) or condition > 1e8:
        raise ValueError(f'Initial decoder singular: cond={condition}')
    y = np.linalg.solve(h0.T, x.T).T
    labels_y, stat_y = km(y, k, seed)
    fitted['km_Y'] = (labels_y, stat_y)
    initial_means = np.stack([y[labels_u == b].mean(0) for b in range(k)])
    continued = KMeans(k,init=initial_means,n_init=1,max_iter=300,tol=1e-5,random_state=seed).fit(y)
    fitted['km_Y_from_U'] = (continued.labels_, {'iterations':int(continued.n_iter_),
                                               'objective':float(-continued.inertia_)})
    diagnostics = {'cond_H0': condition, 'negative_fraction': float((y < 0).mean()),
                   'y_min': float(y.min()), 'y_max': float(y.max()),
                   'h0': h0.tolist()}
    traces = {}
    for name, tau, hard in [('soft_tau1',1.,False), ('soft_tau2',2.,False), ('hard_tau1',1.,True)]:
        result = fit_poisson_em(y, labels_u, tau=tau, max_iter=max_iter, tol=1e-6, hard=hard)
        labels = result.pop('labels')
        traces[name] = result.pop('trace')
        # Store small final model parameters; arrays must stay JSON serializable.
        result.pop('responsibilities', None)
        result.pop('r', None)
        result['iterations'] = result['n_iter']
        for key in ('rates', 'weights'):
            if key in result:
                result[key] = np.asarray(result[key]).tolist()
        fitted[name] = (labels, result)
    # Control: different continuous feature likelihood on the same fixed Y.
    means = np.stack([y[labels_u == b].mean(0) for b in range(k)])
    weights = np.bincount(labels_u, minlength=k)/n
    regularizer = max(1e-6, float(np.var(y,axis=0).mean())*1e-4)
    started = time.perf_counter()
    gm = GaussianMixture(k, covariance_type='full', n_init=1, max_iter=max_iter,
                         tol=1e-6, reg_covar=regularizer, random_state=seed,
                         means_init=means, weights_init=weights).fit(y)
    fitted['gmm_Y'] = (gm.predict(y), {'converged': bool(gm.converged_),
                                     'iterations': int(gm.n_iter_),
                                     'objective': float(gm.score(y)*n),
                                     'elapsed': time.perf_counter()-started})
    # Restarts are selected by the SAME likelihood, never recovery error.
    if restarts:
        alternatives = [fitted['soft_tau1']]
        start_records = [{'start': 'km_U', 'objective': alternatives[0][1]['objective']}]
        for run in range(3):
            centers, _ = kmeans_plusplus(y, n_clusters=k, random_state=seed+104729*(run+1))
            labels0 = ((y[:,None,:]-centers[None,:,:])**2).sum(2).argmin(1)
            if len(np.unique(labels0)) < k:
                continue
            result = fit_poisson_em(y, labels0, tau=1., max_iter=max_iter, tol=1e-6)
            labels = result.pop('labels')
            traces[f'restart_{run+1}'] = result.pop('trace')
            result.pop('responsibilities', None)
            result.pop('r', None)
            result['iterations'] = result['n_iter']
            for key in ('rates','weights'):
                if key in result:
                    result[key] = np.asarray(result[key]).tolist()
            start_records.append({'start': f'plus_plus_{run+1}', 'objective': result['objective']})
            alternatives.append((labels, result))
        best = max(alternatives, key=lambda pair: pair[1]['objective'])
        stat = dict(best[1])
        stat['restart_objectives'] = start_records
        fitted['soft_tau1_best4'] = (best[0], stat)
    # Follow-up fixed before the final seeds: use the stronger X-k-means start.
    # This is a NEW fixed decoder/model, not changing H inside the preceding EM.
    hx = np.stack([np.sqrt(n)*u[labels_x == b].mean(0) for b in range(k)])
    yx = np.linalg.solve(hx.T, x.T).T
    diagnostics['cond_HX'] = float(np.linalg.cond(hx))
    diagnostics['negative_fraction_X'] = float((yx < 0).mean())
    means_x = np.stack([yx[labels_x == b].mean(0) for b in range(k)])
    lloyd_x = KMeans(k,init=means_x,n_init=1,max_iter=300,tol=1e-5,random_state=seed).fit(yx)
    fitted['km_YX_from_X'] = (lloyd_x.labels_, {'iterations':int(lloyd_x.n_iter_),
                                              'objective':float(-lloyd_x.inertia_)})
    for name,tau in [('soft_X_tau1',1.),('soft_X_tau2',2.)]:
        result = fit_poisson_em(yx, labels_x, tau=tau, max_iter=max_iter, tol=1e-6)
        labels = result.pop('labels')
        traces[name] = result.pop('trace')
        result.pop('r',None)
        result['iterations'] = result['n_iter']
        for key in ('rates','weights'):
            result[key] = np.asarray(result[key]).tolist()
        fitted[name] = (labels,result)
    started = time.perf_counter()
    gx = GaussianMixture(k,covariance_type='full',n_init=1,max_iter=max_iter,tol=1e-6,
                         reg_covar=max(1e-6,float(np.var(yx,axis=0).mean())*1e-4),
                         random_state=seed,means_init=means_x,
                         weights_init=np.bincount(labels_x,minlength=k)/n).fit(yx)
    fitted['gmm_YX_from_X'] = (gx.predict(yx),{'converged':bool(gx.converged_),
                                            'iterations':int(gx.n_iter_),
                                            'objective':float(gx.score(yx)*n),
                                            'elapsed':time.perf_counter()-started})
    # Add new diagnostics to all records after fitting, independently of truth.
    compressed_x = {'hx':hx,'yx':yx}
    return fitted, traces, diagnostics, {'u': u, 'lam': lam, 'h0': h0, 'y': y,**compressed_x}

def recovery(labels, truth, k):
    tab = np.zeros((k,k), int)
    np.add.at(tab, (truth,labels), 1)
    rows, cols = linear_sum_assignment(-tab)
    errors = len(truth)-tab[rows,cols].sum()
    return dict(errors=int(errors), error_rate=float(errors/len(truth)),
                exact=int(errors == 0), ari=float(adjusted_rand_score(truth,labels)))

def run_one(task):
    name, n, target, seed, outdir, max_iter, restarts = task
    out = Path(outdir)
    key = f'{name}_n{n}_ch{target:g}_s{seed}'
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        u, lam, truth, meta = generate_eigenpairs(name,n,target,seed)
        fitted, traces, diagnostics, compressed = fit_spectral_methods(u,lam,seed,max_iter,restarts)
    records = []
    for method, (labels, stats) in fitted.items():
        metric = recovery(labels,truth,u.shape[1])
        record = dict(graph=key,model=name,n=n,ch=target,seed=seed,method=method,
                      **metric,**diagnostics,**stats)
        record.pop('h0',None)
        records.append(record)
        compressed['labels_'+method] = labels
    compressed['truth_evaluation_only'] = truth
    np.savez_compressed(out/'inputs'/f'{key}.npz', **compressed)
    payload = dict(graph=key,metadata=meta,diagnostics=diagnostics,results=records,
                   traces=traces, elapsed=time.perf_counter()-started)
    (out/'runs'/f'{key}.json').write_text(json.dumps(payload,ensure_ascii=False,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
    return dict(graph=key,elapsed=payload['elapsed'],records=records)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--ns',nargs='+',type=int,default=[1000,2000])
    p.add_argument('--chs',nargs='+',type=float,default=[.8,1.4])
    p.add_argument('--models',nargs='+',default=list(MODELS))
    p.add_argument('--seeds',type=int,default=10)
    p.add_argument('--seed-base',type=int,default=5000)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--max-iter',type=int,default=150)
    p.add_argument('--restarts',action='store_true')
    args = p.parse_args()
    out = Path(args.out)
    (out/'inputs').mkdir(parents=True,exist_ok=True)
    (out/'runs').mkdir(exist_ok=True)
    protocol = vars(args)|{'models_definition': MODELS,'density':'B=(log n/n) C',
                           'C_normalization':'min pair Chernoff-Hellinger divergence equals target',
                           'spectral_input':'K eigenpairs of raw A with largest absolute eigenvalues',
                           'initialization':'best of ten U k-means runs; same labels for every refinement',
                           'tau_selection':'fixed tau=1 and tau=2 reported separately, never selected by truth',
                           'truth_access':'generator/evaluation only; fit_spectral_methods has no A/truth argument'}
    protocol['pilot_adaptation'] = 'Pilot seeds 1000 only revealed poor U-start calibration; before final seeds 5000..5009, added separate fixed X-kmeans calibration/refinement arms.'
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    tasks = [(model,n,ch,args.seed_base+s,str(out),args.max_iter,args.restarts)
             for model in args.models for n in args.ns for ch in args.chs for s in range(args.seeds)]
    records = []
    failures = []
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        fs = {pool.submit(run_one,t): t for t in tasks}
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):
            try:
                r = f.result()
                records.extend(r['records'])
                rates = {x['method']:round(100*x['error_rate'],3) for x in r['records']}
                print(json.dumps({'done':i,'total':len(tasks),'graph':r['graph'],
                                  'seconds':round(r['elapsed'],2),'errors_pct':rates}),flush=True)
            except Exception as e:
                failures.append({'task':fs[f],'error':repr(e)})
                print(json.dumps({'done':i,'failure':failures[-1]}),flush=True)
            pd.DataFrame(records).to_csv(out/'per_run.csv',index=False)
            (out/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    (out/'elapsed.json').write_text(json.dumps({'seconds':time.perf_counter()-started,
                                               'successful_graphs':len(records)//(13 if args.restarts else 12),
                                               'failed_graphs':len(failures)}),encoding='utf-8')

if __name__ == '__main__':
    main()

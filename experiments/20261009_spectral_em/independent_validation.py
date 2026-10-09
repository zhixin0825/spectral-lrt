"""Frozen independent-seed validation of the fixed-spectral Poisson EM recipe.

The fitting routine accepts eigenpairs only. Ground truth is used afterwards.
The decoder remains fixed; this is a working likelihood, not exact graph LOO.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from benchmark import MODELS, generate_eigenpairs, km, recovery
from poisson_em import fit_poisson_em
from direct_gaussian import fit_direct_gaussian


def spectral_fit(u, lam, seed, max_iter=150):
    n, k = u.shape
    x = np.sqrt(n) * u * lam
    lu, su = km(np.sqrt(n)*u, k, seed)
    lx, sx = km(x, k, seed)
    h = np.vstack([np.sqrt(n)*u[lu==b].mean(axis=0) for b in range(k)])
    condition = float(np.linalg.cond(h))
    if not np.isfinite(condition) or condition > 1e8:
        raise ValueError(f'Decoder singular, condition={condition}')
    y = np.linalg.solve(h.T, x.T).T
    fitted = {'km_U':(lu,su), 'km_X':(lx,sx)}
    traces = {}
    for method, iterations in [('initial_E',0),('one_EM_iteration',1),('converged_EM',max_iter)]:
        result = fit_poisson_em(y,lx,tau=1.,max_iter=iterations,tol=1e-6)
        labels = result.pop('labels')
        traces[method] = result.pop('trace')
        result.pop('r')
        for key in ('rates','weights'):
            result[key] = result[key].tolist()
        fitted[method] = (labels,result)
    labels, stats, trace = fit_direct_gaussian(u,lam,lx,max_iter=max_iter)
    fitted['direct_Gaussian'] = (labels,stats)
    traces['direct_Gaussian'] = trace
    arrays = {'u':u,'lam':lam,'h0':h}
    arrays.update({'labels_'+name:labels for name,(labels,_) in fitted.items()})
    return fitted,traces,arrays,condition


def run_one(task):
    model,n,ch,seed,outpath = task
    key = f'{model}_n{n}_ch{ch:g}_s{seed}'
    out = Path(outpath)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        u,lam,truth,metadata = generate_eigenpairs(model,n,ch,seed)
        fits,traces,arrays,condition = spectral_fit(u,lam,seed)
    rows = []
    for method,(labels,stats) in fits.items():
        rows.append(dict(graph=key,model=model,n=n,ch=ch,seed=seed,
                         cond_H0=condition,**recovery(labels,truth,len(lam)))
                    | stats | {'method':method})
    arrays['truth_evaluation_only'] = truth
    np.savez_compressed(out/'inputs'/f'{key}.npz',**arrays)
    payload = {'graph':key,'metadata_evaluation_only':metadata,'records':rows,
               'traces':traces,'elapsed_seconds':time.perf_counter()-started}
    (out/'runs'/f'{key}.json').write_text(json.dumps(payload),encoding='utf-8')
    return rows


def analyze(out):
    frame = pd.read_csv(out/'per_run.csv')
    frame['n_iter'] = frame['n_iter'].fillna(frame['iterations'])
    piv = frame.pivot(index='graph',columns='method',values='error_rate')
    summary = frame.groupby(['model','n','ch','method']).agg(
        graphs=('graph','size'),mean_error=('error_rate','mean'),exact=('exact','sum'),
        converged=('converged','sum'),mean_iterations=('n_iter','mean'),
        max_iterations=('n_iter','max'))
    summary.to_csv(out/'condition_summary.csv')
    overall = frame.groupby('method').agg(graphs=('graph','size'),
        mean_error=('error_rate','mean'),exact=('exact','sum'),converged=('converged','sum'),
        mean_iterations=('n_iter','mean'),max_iterations=('n_iter','max'))
    overall.to_csv(out/'overall_summary.csv')
    comparisons=[]
    for before,after in [('km_X','initial_E'),('km_X','converged_EM'),
                         ('initial_E','converged_EM'),('one_EM_iteration','converged_EM'),
                         ('km_X','direct_Gaussian')]:
        delta=piv[before]-piv[after]
        comparisons.append(dict(before=before,after=after,
            mean_improvement_percentage_points=100*float(delta.mean()),
            improved=int((delta>1e-12).sum()),tied=int((abs(delta)<1e-12).sum()),
            worsened=int((delta < -1e-12).sum())))
    gains=[]
    for path in (out/'runs').glob('*.json'):
        tr=json.loads(path.read_text())['traces']['converged_EM']
        gains.extend(tr[i]['objective']-tr[i-1]['objective'] for i in range(1,len(tr)))
    result={'graphs':len(piv),'methods':overall.to_dict(orient='index'),
            'comparisons':comparisons,'minimum_EM_gain':min(gains),
            'material_negative_steps':sum(g < -1e-6 for g in gains)}
    delta=(piv['km_X']-piv['converged_EM']).to_numpy().reshape(24,5)
    rng=np.random.default_rng(20261009)
    samples=delta[np.arange(24)[None,:,None],rng.integers(0,5,size=(10000,24,5))]
    result['paired_bootstrap']={'seed':20261009,'replicates':10000,
        'scope':'within each of the 24 fixed conditions, resample 5 paired graphs',
        'ci95_improvement_percentage_points':(100*np.quantile(samples.mean(axis=(1,2)),[.025,.975])).tolist()}
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    archive=out/'spectral_inputs_and_trajectories.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in ('inputs','runs'):
            for path in sorted((out/folder).glob('*')):
                z.write(path,str(path.relative_to(out)).replace('\\','/'))
    manifest={'file':archive.name,'bytes':archive.stat().st_size,
              'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    (out/'archive_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='independent_20000')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'inputs').mkdir(exist_ok=True)
    (out/'runs').mkdir(exist_ok=True)
    protocol={'created_before_fitting':True,'design':'6 SBM x 2 n x 2 CH x 5 seeds = 120 graphs',
              'models':MODELS,'ns':[1000,2000],'chs':[.8,1.4],
              'seeds':[20000,20001,20002,20003,20004],'density':'B=C log(n)/n',
              'C_normalization':'minimum pair Chernoff-Hellinger equals target',
              'spectral_input':'largest absolute K eigenpairs of raw adjacency',
              'feature':'sqrt(n) U Lambda H_U^{-1}; fixed for entire EM',
              'decoder':'cluster means of sqrt(n) U from ten-start U k-means',
              'initialization':'ten-start k-means on sqrt(n) U Lambda',
              'tau':1.,'tol':1e-6,'max_iter':150,
              'selection':'no restarts or hyperparameter selection; all six arms reported',
              'truth_access':'generator and post-fit evaluation only',
              'scope':'row-wise Poisson-Gaussian working likelihood; exact graph LOO is not implemented',
              'purpose':'independent seed validation after prior final-seed diagnostics'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    tasks=[(m,n,ch,s,str(out)) for m in MODELS for n in [1000,2000]
           for ch in [.8,1.4] for s in range(20000,20005)]
    rows=[]
    failures=[]
    started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_one,t):t for t in tasks}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            try: rows.extend(f.result())
            except Exception as e: failures.append({'task':futures[f],'error':repr(e)})
            if i%10==0:
                print(json.dumps({'completed':i,'total':len(tasks),
                                  'elapsed_seconds':round(time.perf_counter()-started,1),
                                  'failures':len(failures)}),flush=True)
    (out/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    if rows:
        pd.DataFrame(rows).sort_values(['model','n','ch','seed','method']).to_csv(out/'per_run.csv',index=False)
    if failures: raise RuntimeError(f'{len(failures)} failed graphs; see failures.json')
    analyze(out)


if __name__=='__main__': main()

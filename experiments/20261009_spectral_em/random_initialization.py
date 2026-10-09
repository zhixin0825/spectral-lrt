"""Random label initialization with frozen features and truth-free restarts.

Separately audit the inherited k-means decoder and a fully random decoder.
Every label/parameter fit is determined before reading evaluator-only truth.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from benchmark import recovery
from poisson_em import fit_poisson_em
from direct_gaussian import fit_direct_gaussian


def small_fit(y,initial,max_iter):
    fit=fit_poisson_em(y,initial,tau=1.,max_iter=max_iter,tol=1e-6)
    labels=fit.pop('labels')
    trace=fit.pop('trace')
    fit.pop('r')
    for key in ('rates','weights'):
        fit[key]=fit[key].tolist()
    return labels,fit,trace


def fit_all(u,lam,h,labels_km,seed,starts,max_iter):
    n,k=u.shape
    x=np.sqrt(n)*u*lam
    y=np.linalg.solve(h.T,x.T).T
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,k,398719]))
    fits=[]
    labels_saved={}
    traces={}
    initials=[]
    for restart in range(starts):
        # Balanced independently random labels; no oracle community proportions.
        initial=rng.permutation(np.arange(n,dtype=np.int64)%k)
        initials.append(initial)
        labels,stats,trace=small_fit(y,initial,max_iter)
        name=f'random_PG_{restart}'
        fits.append((name,restart,labels,stats))
        labels_saved['labels_'+name]=labels
        labels_saved['initial_'+name]=initial
        traces[name]=trace
    selected=max(fits,key=lambda r:r[3]['objective'])
    # These selected results share exactly Y and tau with the k-means start.
    labels,stats,trace=small_fit(y,labels_km,max_iter)
    fits.append(('km_start_PG',-1,labels,stats))
    labels_saved['labels_km_start_PG']=labels
    traces['km_start_PG']=trace
    labels,stats,trace=fit_direct_gaussian(u,lam,initials[0],max_iter=max_iter)
    stats['n_iter']=stats['iterations']
    fits.append(('random_Gaussian_X',0,labels,stats))
    labels_saved['labels_random_Gaussian_X']=labels
    traces['random_Gaussian_X']=trace
    # Diagnostic: eliminate k-means from the decoder too, with a random H.
    # Different Y => raw objective must NEVER be compared with the fixed-H arms.
    initial=initials[0]
    hr=np.vstack([np.sqrt(n)*u[initial==b].mean(axis=0) for b in range(k)])
    condition=float(np.linalg.cond(hr))
    yr=np.linalg.solve(hr.T,x.T).T
    labels,stats,trace=small_fit(yr,initial,max_iter)
    stats['cond_random_H']=condition
    stats['random_Y_min']=float(yr.min())
    stats['random_Y_max']=float(yr.max())
    fits.append(('random_H_and_labels_PG',0,labels,stats))
    labels_saved['labels_random_H_and_labels_PG']=labels
    labels_saved['random_H']=hr
    traces['random_H_and_labels_PG']=trace
    labels_saved['labels_random_PG_best']=selected[2]
    labels_saved['selected_restart']=np.array(selected[1])
    return fits,selected,labels_saved,traces


def run_one(task):
    input_path,outpath,starts,max_iter=task
    path=Path(input_path)
    out=Path(outpath)
    model,n,ch,seed=re.fullmatch(r'(.+)_n(\d+)_ch([\d.]+)_s(\d+)',path.stem).groups()
    common={'graph':path.stem,'model':model,'n':int(n),'ch':float(ch),'seed':int(seed)}
    with np.load(path,allow_pickle=False) as data:
        u,lam,h,labels0=data['u'],data['lam'],data['h0'],data['labels_km_X']
        with threadpool_limits(limits=1):
            fits,selected,arrays,traces=fit_all(u,lam,h,labels0,int(seed),starts,max_iter)
        # Truth is accessed only after fitting ALL arms and selecting the restart.
        truth=data['truth_evaluation_only']
        rows=[]
        for name,restart,labels,stats in fits:
            rows.append(common|stats|recovery(labels,truth,len(lam))|
                        {'method':name,'restart':restart})
        rows.append(common|selected[3]|recovery(selected[2],truth,len(lam))|
                    {'method':'random_PG_best','restart':selected[1]})
        rows.append(common|recovery(labels0,truth,len(lam))|
                    {'method':'km_X','restart':-1})
        arrays.update({'u':u,'lam':lam,'h0':h,'truth_evaluation_only':truth,'labels_km_X':labels0})
    np.savez_compressed(out/'inputs'/f'{path.stem}.npz',**arrays)
    (out/'runs'/f'{path.stem}.json').write_text(json.dumps(
        {'graph':path.stem,'results':rows,'traces':traces,
         'restart_selection':'largest terminal fixed-Y tau=1 mixture likelihood',
         'selected_restart':selected[1]}),encoding='utf-8')
    return rows


def analyze(out):
    df=pd.read_csv(out/'per_run.csv')
    df['n_iter']=df['n_iter'].fillna(0)
    df['method_group']=df.method.where(~df.method.str.match(r'random_PG_\d+$'),'random_PG_all')
    first=df[df.method=='random_PG_0'].copy()
    first['method_group']='random_PG_0'
    summary_df=pd.concat([df,first],ignore_index=True)
    summary=summary_df.groupby(['model','n','ch','method_group']).agg(
        fits=('graph','size'),mean_error=('error_rate','mean'),exact=('exact','sum'),
        converged=('converged','sum'),mean_iter=('n_iter','mean'),max_iter=('n_iter','max'))
    summary.to_csv(out/'condition_summary.csv')
    overall=summary_df.groupby('method_group').agg(fits=('graph','size'),
        mean_error=('error_rate','mean'),exact=('exact','sum'),converged=('converged','sum'),
        mean_iter=('n_iter','mean'),max_iter=('n_iter','max'))
    overall.to_csv(out/'overall_summary.csv')
    comparisons=[]
    subset=df[df.method.isin(['random_PG_0','random_PG_best','km_start_PG','km_X'])]
    errors=subset.pivot(index='graph',columns='method',values='errors')
    objectives=subset.pivot(index='graph',columns='method',values='objective')
    for name in ('random_PG_0','random_PG_best'):
        delta=errors['km_start_PG']-errors[name]
        odelta=objectives[name]-objectives['km_start_PG']
        comparisons.append({'method':name,'baseline':'km_start_PG',
            'better_errors':int((delta>0).sum()),'tied_errors':int((delta==0).sum()),
            'worse_errors':int((delta<0).sum()),
            'material_higher_objective':int((odelta>1e-3).sum()),
            'material_lower_objective':int((odelta < -1e-3).sum())})
    random=df[df.method.str.match(r'random_PG_\d+$')]
    bygraph=random.groupby('graph').agg(exact_starts=('exact','sum'),
        good_starts=('error_rate',lambda x:int((x<.01).sum())),
        worst_error=('error_rate','max'),best_error=('error_rate','min'))
    bygraph.to_csv(out/'random_start_success_by_graph.csv')
    result={'graphs':df.graph.nunique(),'records':len(df),
            'overall':overall.to_dict(orient='index'),'same_fixed_model_comparisons':comparisons,
            'random_starts_any_exact_graphs':int((bygraph.exact_starts>0).sum()),
            'random_starts_all_exact_graphs':int((bygraph.exact_starts==10).sum()),
            'random_starts_any_sub1pct_graphs':int((bygraph.good_starts>0).sum()),
            'random_decoder_cond_min':float(df.cond_random_H.min()),
            'random_decoder_cond_max':float(df.cond_random_H.max())}
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    archive=out/'inputs_and_trajectories.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in ('inputs','runs'):
            for f in sorted((out/folder).glob('*')):
                z.write(f,str(f.relative_to(out)).replace('\\','/'))
    (out/'archive_manifest.json').write_text(json.dumps({'file':archive.name,
        'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='independent_20000/inputs')
    p.add_argument('--out',default='random_initialization_20000')
    p.add_argument('--starts',type=int,default=10)
    p.add_argument('--max-iter',type=int,default=300)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'inputs').mkdir(exist_ok=True)
    (out/'runs').mkdir(exist_ok=True)
    files=sorted(Path(args.input).glob('*.npz'))
    protocol=vars(args)|{'graphs':len(files),'created_before_fitting':True,
        'initialization':'uniformly randomized balanced K labels, not oracle proportions',
        'same_labels_across_arms':True,'tau':1.,'tol':1e-6,
        'preferred_PG_features':'inherited frozen Y from U-kmeans calibration; this retains a k-means decoder',
        'random_decoder':'separate diagnostic with H from random labels; different likelihood scale',
        'Gaussian_X':'fixed globally standardized sqrt(n) U Lambda; no decoder or k-means in this arm',
        'truth_access':'after all fits and likelihood restart selection',
        'selection':'best terminal observed likelihood, irrespective of recovery or convergence flag',
        'scope':'working likelihood only; exact LOO not implemented'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    failures=[]
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs={pool.submit(run_one,(str(f),str(out),args.starts,args.max_iter)):f.stem for f in files}
        for i,future in enumerate(concurrent.futures.as_completed(fs),1):
            try: rows.extend(future.result())
            except Exception as exc: failures.append({'graph':fs[future],'error':repr(exc)})
            if i%5==0:
                print(json.dumps({'completed':i,'total':len(files),'failures':len(failures),
                                  'elapsed':round(time.perf_counter()-start,1)}),flush=True)
            pd.DataFrame(rows).to_csv(out/'per_run.csv',index=False)
            (out/'failures.json').write_text(json.dumps(failures,indent=2))
    if failures: raise RuntimeError(f'{len(failures)} failed graphs, preserved in failures.json')
    analyze(out)


if __name__=='__main__': main()

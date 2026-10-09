"""Paired original/smaller-fraction seeding on saved U,Lambda; same EM.

No fitting access to A, degree, truth, baseline labels, or a k-means decoder.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import gammaln
from threadpoolctl import threadpool_limits
from partial_peeling import peel,validate,RULES

ROOT = Path(__file__).resolve().parent
# Support this workspace and the archived repository layout.
OLD = ROOT.parent/'spectral_greedy_peel_20261009'
if not OLD.exists(): OLD = ROOT.parent/'20261009_greedy_likelihood_peeling'
sys.path.insert(0,str(OLD))
from profile_peeling import fit_soft,metric,as_json,LEGACY
from profile_score_controls import fit_bernoulli

PROTOCOL = dict(
    input='120 inherited eigenpair inputs; exact same inputs as original peeling comparison',
    fitting_input='U,Lambda only; all truth and saved baselines accessed after every fit on graph',
    density='C log(n)/n; n=1000/2000; CH=0.8/1.4; seeds=20000..20004; 6 models',
    score_poisson='q_i @ log(mu) - sum(mu); q=max(U Lambda U.T,1e-4 log(n)/n); diagonal retained',
    score_bernoulli='q_i @ log(p/(1-p)) + sum(log(1-p)); q=clip(U Lambda U.T,lower,1-1e-8)',
    original='floor(M/r) other nodes, cap at M-r; final r=1 takes M-1',
    scaled_remaining='floor(2M/(3r)) other nodes, cap at M-r; r=K,K-1,...,1; no final forced completion',
    scaled_fixed_k='floor(2M/(3K)) other nodes, cap at M-r; K fixed original number; no final forced completion',
    candidate='Every remaining node; raw total on top other nodes; self excluded; ties smallest node ID',
    parameters='Selected node profiles unchanged; no selected-core mean substitution',
    removal='Candidate and its top nodes removed only from seeding pool; all n observations enter EM',
    EM='Inherited soft EM unchanged; uniform initial weights; max_iter=300; tol=1e-6',
    scope='Working expected row scores on dependent spectral profiles, not exact LOO or graph likelihood',
    independent_test_set=False)

def run_one(task):
    path,out = map(Path,task)
    started = time.perf_counter()
    model,n,ch,seed = re.fullmatch(r'(.+)_n(\d+)_ch([\d.]+)_s(\d+)',path.stem).groups()
    arrays,methods,predictions = {},{},[]
    with np.load(path,allow_pickle=False) as source:
        u,lam = source['u'],source['lam']
        n,k = u.shape
        arrays.update(u=u,lam=lam)
        with threadpool_limits(limits=1):
            reconstructed = (u*lam)@u.T
            lower = 1e-4*np.log(n)/n
            for likelihood in ('poisson','bernoulli'):
                q = np.maximum(reconstructed,lower) if likelihood=='poisson' else np.clip(reconstructed,lower,1-1e-8)
                score = (q@np.log(q).T-q.sum(1)[None,:] if likelihood=='poisson' else
                         q@np.log(q/(1-q)).T+np.log1p(-q).sum(1)[None,:])
                for rule in RULES:
                    method = likelihood+'_'+rule
                    seed_started = time.perf_counter()
                    ids,cores,events,leftover = peel(score,k,rule)
                    seed_elapsed = time.perf_counter()-seed_started
                    initial = score[:,ids].argmax(1)
                    fit_started = time.perf_counter()
                    fitted = fit_soft(q,q[ids]) if likelihood=='poisson' else fit_bernoulli(q,q[ids])
                    fit_elapsed = time.perf_counter()-fit_started
                    final = fitted.pop('labels')
                    final_params = fitted.pop('mu' if likelihood=='poisson' else 'p')
                    final_weights = fitted.pop('weights')
                    trace = fitted.pop('trace')
                    if likelihood=='poisson':
                        constant = float(gammaln(q+1).sum())
                        fitted['objective'] += constant
                        for t in trace: t['objective'] += constant
                    predictions.append((method,final,fitted,seed_elapsed,fit_elapsed))
                    for key,value in dict(seed_ids=ids,core_labels=cores,leftover=leftover,
                        initial_parameters=q[ids],initial_E_labels=initial,final_labels=final,
                        final_parameters=final_params,final_weights=final_weights).items():
                        arrays[method+'_'+key] = value
                    methods[method] = dict(events=events,trace=trace)
        # Access labels only once all six fits have finished.
        truth = source['truth_evaluation_only']
        baseline = source['baseline_km_X']
        arrays.update(truth_evaluation_only=truth,baseline_km_X=baseline)
        baseline_metric = metric(baseline,truth,k)
        rows = []
        for method,final,fitted,seed_seconds,fit_seconds in predictions:
            ids = arrays[method+'_seed_ids']
            cores = arrays[method+'_core_labels']
            events = methods[method]['events']
            composition = [np.bincount(truth[e['members']],minlength=k).tolist() for e in events]
            purity = sum(max(counts) for counts in composition)/np.sum(cores>=0)
            rows.append(dict(graph=path.stem,model=model,n=n,k=k,ch=float(ch),seed=int(seed),method=method,
                **fitted,**metric(final,truth,k),seed_coverage=len(np.unique(truth[ids])),
                seed_classes=truth[ids].tolist(),core_purity=purity,
                leftover_count=int(np.sum(cores<0)),initial_E_error=metric(arrays[method+'_initial_E_labels'],truth,k)['error_rate'],
                seed_elapsed=seed_seconds,fit_elapsed=fit_seconds,
                baseline_errors=baseline_metric['errors'],baseline_error=baseline_metric['error_rate']))
            methods[method]['evaluation_only'] = dict(seed_classes=truth[ids],core_compositions=composition)
    np.savez_compressed(out/'arrays'/f'{path.stem}.npz',**arrays)
    payload = dict(graph=path.stem,rows=rows,methods=methods,elapsed=time.perf_counter()-started)
    (out/'runs'/f'{path.stem}.json').write_text(json.dumps(payload,default=as_json),encoding='utf-8')
    return rows,payload['elapsed']

def summarize(frame,out):
    frame = frame.sort_values(['graph','method'])
    frame.to_csv(out/'per_graph.csv',index=False)
    summary = frame.groupby('method').agg(graphs=('graph','size'),mean_error=('error_rate','mean'),
        exact=('exact','sum'),converged=('converged','sum'),mean_iter=('n_iter','mean'),max_iter=('n_iter','max'),
        full_seed_coverage=('seed_coverage',lambda x:int((x==frame.loc[x.index,'k']).sum())),
        mean_core_purity=('core_purity','mean'),mean_leftover=('leftover_count','mean'),
        mean_initial_error=('initial_E_error','mean'),min_objective_delta=('worst_objective_delta','min'))
    summary.to_csv(out/'summary.csv')
    frame.groupby(['model','method']).agg(graphs=('graph','size'),mean_error=('error_rate','mean'),
        exact=('exact','sum'),full_seed_coverage=('seed_coverage',lambda x:int((x==frame.loc[x.index,'k']).sum())),
        mean_core_purity=('core_purity','mean')).to_csv(out/'by_model.csv')
    frame.groupby(['model','n','ch','method']).agg(graphs=('graph','size'),mean_error=('error_rate','mean'),
        exact=('exact','sum')).to_csv(out/'by_condition.csv')
    pairs = []
    for likelihood in ('poisson','bernoulli'):
        original = frame[frame.method==likelihood+'_original']
        for rule in RULES[1:]:
            merged = frame[frame.method==likelihood+'_'+rule].merge(
                original[['graph','errors','objective','seed_coverage']],on='graph',suffixes=('_new','_old'))
            differences = merged.errors_new-merged.errors_old
            pairs.append(dict(method=likelihood+'_'+rule,graphs=len(merged),
                better=int((differences<0).sum()),equal=int((differences==0).sum()),worse=int((differences>0).sum()),
                coverage_improved=int((merged.seed_coverage_new>merged.seed_coverage_old).sum()),
                coverage_worsened=int((merged.seed_coverage_new<merged.seed_coverage_old).sum()),
                higher_objective=int((merged.objective_new-merged.objective_old>1e-3).sum()),
                lower_objective=int((merged.objective_new-merged.objective_old< -1e-3).sum())))
    (out/'paired_comparison.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=LEGACY/'full120'/'inputs')
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    out=args.out
    for sub in ('arrays','runs'): (out/sub).mkdir(parents=True,exist_ok=True)
    (out/'validation.json').write_text(json.dumps(validate(),indent=2),encoding='utf-8')
    paths=sorted(args.input_dir.glob('*.npz'))
    if args.pilot: paths=[p for p in paths if '_n1000_ch0.8_s20000' in p.stem]
    if not paths: raise FileNotFoundError(args.input_dir)
    input_manifest=[dict(graph=p.stem,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    (out/'protocol.json').write_text(json.dumps(PROTOCOL|dict(graph_count=len(paths),workers=args.workers),indent=2),encoding='utf-8')
    (out/'input_manifest.json').write_text(json.dumps(input_manifest,indent=2),encoding='utf-8')
    rows,todo=[],[]
    for p in paths:
        saved=out/'runs'/f'{p.stem}.json'
        if args.resume and saved.exists(): rows.extend(json.loads(saved.read_text())['rows'])
        else: todo.append(p)
    started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(run_one,(str(p),str(out))) for p in todo]
        for done,future in enumerate(concurrent.futures.as_completed(futures),1):
            result,seconds=future.result()
            rows.extend(result)
            pd.DataFrame(rows).to_csv(out/'per_graph_partial.csv',index=False)
            print(json.dumps(dict(done=done,total=len(todo),graph=result[0]['graph'],graph_seconds=round(seconds,2),
                elapsed=round(time.perf_counter()-started,1))),flush=True)
    frame=pd.DataFrame(rows)
    summarize(frame,out)
    print(json.dumps(dict(completed=len(paths),fits=len(frame),elapsed=time.perf_counter()-started)),flush=True)

if __name__=='__main__': main()

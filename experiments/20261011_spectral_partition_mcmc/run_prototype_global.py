"""Paired global-MH supplement on already frozen benchmark graph eigenpairs."""
import argparse
import json
import os
import shutil
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import engine
import prototype_mcmc


def run_graph(job):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        source=Path(job['source'])
        graph=source/job['graph_id']
        output=Path(job['out'])/job['graph_id']
        output.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(graph/'graph.npz',output/'graph.npz')
        data=np.load(graph/'graph.npz')
        Q=engine.reconstruction(data['U'],data['lam'])
        truth=data['truth'];k=int(data['k'])
        rows=[];bestrows=[]
        # Explicit warmup; keep compilation out of recorded timings.
        prototype_mcmc.prototype_chain(np.array([[0.,.2],[.2,0.]]),2,np.array([0,1]),1,3,U=np.eye(2),record_every=1)
        frame=pd.read_csv(source/'final.csv')
        for start in range(job['starts']):
            template=frame[(frame.graph_id==job['graph_id'])&(frame.start==start)&(frame.method=='gibbs')].iloc[0].to_dict()
            initial=np.load(graph/f'initial_labels_start{start}.npy')
            seed=int(np.random.SeedSequence([910531,k,int(data['n']),int(template['graph_seed']),start]).generate_state(1)[0]%2**31)
            r=prototype_mcmc.prototype_chain(Q,k,initial,job['sweeps'],seed,U=data['U'],record_every=5)
            payload=dict(initial_labels=initial,final_labels=r['labels'],best_posterior_labels=r['best_labels'],trace_labels=r['trace_labels'],trace_sweep=r['trace_sweep'],trace_score=r['trace_score'],copy_accepted=r['copy_accepted'],copy_tries=r['copy_tries'],global_accepted_trace=r['global_accepted_trace'])
            np.savez_compressed(output/f'prototype_global_start{start}.npz',**payload)
            for selection,labels,target in [('final',r['labels'],rows),('best_posterior',r['best_labels'],bestrows)]:
                counts=np.bincount(labels,minlength=k)
                score=engine.log_target(Q,labels,k)
                item=dict(template,method='prototype_global',selection=selection,elapsed=r['elapsed'],log_posterior=score,score_minus_truth=score-template['truth_log_posterior_diagnostic'],group_sizes=';'.join(map(str,counts)),empty_groups=int((counts==0).sum()),min_group_size=int(counts.min()),max_group_size=int(counts.max()),global_accepted=int(r['copy_accepted'].sum()),global_tried=int(r['copy_tries'].sum()),sweeps=job['sweeps'])
                item.update(engine.metrics(labels,truth,k))
                rows.append(item) if target is rows else bestrows.append(item)
        pd.DataFrame(rows).to_csv(output/'final.csv',index=False)
        pd.DataFrame(bestrows).to_csv(output/'best_posterior.csv',index=False)
        return rows,bestrows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sweeps',type=int,default=200)
    p.add_argument('--starts',type=int,default=2)
    p.add_argument('--models',nargs='+',default=['balanced2','unequal3','hierarchy4','disassort3'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1])
    p.add_argument('--signal',type=float,default=1.35)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for source in args.inputs:
        frame=pd.read_csv(source/'final.csv')
        for gid,rows in frame.groupby('graph_id'):
            row=rows.iloc[0]
            if row['model'] in args.models and row['graph_seed'] in args.seeds and abs(row.signal_target-args.signal)<1e-8:
                jobs.append(dict(source=str(source.resolve()),graph_id=gid,out=str(args.out.resolve()),starts=args.starts,sweeps=args.sweeps))
    config=vars(args).copy()
    config.update(scale=2,noise=.2,proposal_interval=1,source_hashes={x:hashlib.sha256((Path(__file__).parent/x).read_bytes()).hexdigest() for x in ['engine.py','prototype_mcmc.py','run_prototype_global.py']})
    (args.out/'config.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf-8')
    print(f'Prototype-global MH supplement on {len(jobs)} frozen graphs',flush=True)
    final=[];best=[]
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ[name]='1'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_graph,j):j for j in jobs}
        for i,future in enumerate(as_completed(futures),1):
            r,b=future.result();final.extend(r);best.extend(b)
            pd.DataFrame(final).to_csv(args.out/'final.csv',index=False)
            pd.DataFrame(best).to_csv(args.out/'best_posterior.csv',index=False)
            print(f'[{i}/{len(jobs)}] {futures[future]["graph_id"]}: exact {sum(x["exact"] for x in r)}/{len(r)}',flush=True)


if __name__=='__main__':
    main()

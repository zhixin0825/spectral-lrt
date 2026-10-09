"""Fresh same-model validation spectra; seed range fixed before fitting."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import concurrent.futures
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'spectral_em_20261009'
if not SOURCE.exists():SOURCE=ROOT.parent/'20261009_spectral_em'
sys.path.insert(0,str(SOURCE))
from benchmark import MODELS,generate_eigenpairs,km

def one(task):
    model,n,ch,seed=task
    graph=f'{model}_n{n}_ch{ch}_s{seed}'
    with threadpool_limits(limits=1):
        u,lam,z,meta=generate_eigenpairs(model,n,ch,seed)
        baseline,_=km(np.sqrt(n)*u*lam,len(lam),seed)
    np.savez_compressed(ROOT/'independent_inputs'/f'{graph}.npz',u=u,lam=lam,
        truth_evaluation_only=z,baseline_km_X=baseline)
    (ROOT/'independent_meta'/f'{graph}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return graph

if __name__=='__main__':
    for name in ('independent_inputs','independent_meta'):(ROOT/name).mkdir(exist_ok=True)
    tasks=[(model,n,ch,seed) for model in MODELS for n in (1000,2000) for ch in (.8,1.4) for seed in range(30000,30005)]
    (ROOT/'independent_protocol.json').write_text(json.dumps(dict(
        tasks=tasks,independent_seeds=[30000,30001,30002,30003,30004],
        models_unchanged=True,design_seeds=[20000,20001,20002,20003,20004],
        candidate_algorithms_declared_in='residual_seed.py',created_before_fitting=True),indent=2),encoding='utf-8')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for i,graph in enumerate(pool.map(one,tasks),1):print(json.dumps(dict(done=i,total=len(tasks),graph=graph)),flush=True)

"""Verify the exact production Work class on fresh SBM graphs."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import ast, csv, json, math, time
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
def load_definition(path, name, namespace):
    tree=ast.parse(path.read_text(encoding='utf8'))
    node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
namespace=dict(np=np,math=math,gammaln=gammaln,logsumexp=logsumexp,WEIGHT_FLOOR=1e-8)
floor_source=ROOT.parent/'20261009_spectral_em'/'poisson_em.py'
work_source=ROOT.parent/'20261009_residual_likelihood_seeding'/'residual_seed.py'
if not floor_source.exists(): floor_source=ROOT/'poisson_em.py'
if not work_source.exists(): work_source=ROOT/'residual_seed.py'
load_definition(floor_source,'_floored_weights',namespace)
load_definition(work_source,'Work',namespace)
Work=namespace['Work']
models={
'symmetric2':(np.array([[5.,1.],[1.,5.]]),[.5,.5]),
'equal3':(np.ones((3,3))+5*np.eye(3),[1/3]*3),
'unequal3':(np.ones((3,3))+5*np.eye(3),[.6,.25,.15]),
'heterogeneous3':(np.array([[10.,1.,1.],[1.,3.,1.],[1.,1.,2.]]),[.5,.3,.2]),
'hierarchical4':(np.array([[6.,4.,1.,1.],[4.,6.,1.,1.],[1.,1.,6.,4.],[1.,1.,4.,6.]]),[.25]*4),
'disassortative3':(np.ones((3,3))*6-5*np.eye(3),[1/3]*3)}
def errors(pred,true,k):
    table=np.zeros((k,k),int)
    np.add.at(table,(pred,true),1)
    a,b=linear_sum_assignment(-table)
    return int(len(true)-table[a,b].sum())
rows=[]
started=time.monotonic()
with threadpool_limits(limits=1):
    for model,(B,pi) in models.items():
        k=len(B)
        for n in (400,800):
            for seed in (1001,1002):
                sizes=np.floor(np.array(pi)*n).astype(int)
                sizes[-1]+=n-sizes.sum()
                z=np.repeat(np.arange(k),sizes)
                rng=np.random.default_rng(seed)
                prob=np.log(n)/n*B[z[:,None],z[None,:]]
                upper=np.triu(rng.random((n,n))<prob,1)
                A=(upper|upper.T).astype(float)
                vals,u=np.linalg.eigh(A)
                ids=np.argsort(np.abs(vals))[-k:]
                w=Work(u[:,ids],vals[ids])
                fit,seed_ids,audit=w.certified_growing(seed=seed+90210)
                objectives=[r['objective'] for r in audit['initialization_runs']]
                assert audit['restarts']==math.ceil(math.log(n))
                assert abs(audit['selected_objective']-max(objectives))<1e-7
                assert audit['final_objective']>=max(objectives)-1e-7
                assert fit['n_iter']==1, 'mandatory final M-step missing'
                assert np.all(fit['mu']>=w.q.min()-1e-12)
                assert np.all(fit['mu']<=w.q.max()+1e-12)
                assert np.all(fit['weights']>=1e-8-1e-12)
                assert abs(fit['weights'].sum()-1)<1e-12
                assert len(seed_ids)==(k-1 if audit['selected']=='growing_global_gain_EM' else k)
                for run in audit['initialization_runs'][1:]:
                    assert len(set(run['seed_ids']))==k
                row=dict(model=model,n=n,k=k,seed=seed,selected=audit['selected'],
                         growing_objective=audit['growing_objective'],
                         selected_objective=audit['selected_objective'],
                         final_objective=audit['final_objective'],
                         certified_errors=errors(fit['labels'],z,k),
                         restart_count=audit['restarts'])
                rows.append(row)
                print(json.dumps(row),flush=True)
with (ROOT/'certified_verification.csv').open('w',newline='',encoding='utf8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
report=dict(cases=len(rows),all_invariants_passed=True,
            winner_never_lower_than_original=True,
            mandatory_final_M_step_verified=True,
            fresh_errors_total=sum(r['certified_errors'] for r in rows),
            selected_residual_runs=sum(r['selected'].startswith('likelihood_') for r in rows),
            elapsed_seconds=time.monotonic()-started,
            scope='24 fresh SBM cases; implementation verification, not replacement for the archived 240-graph benchmark')
(ROOT/'certified_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report),flush=True)

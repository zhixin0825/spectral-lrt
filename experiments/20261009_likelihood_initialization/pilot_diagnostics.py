"""PILOT ONLY: explain fixed likelihood objectives; never select final methods.

Truth labels are used solely for post-fit error diagnostics and explicitly
marked oracle-start optimization trajectories.  Neither benchmark settings
nor final-test jobs are read or modified.  Source data: pilot_v2/jobs only.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from scipy.special import xlogy
from scipy.stats import spearmanr, kurtosis, skew
from experiment import features, metrics, DegreeMatchedSpectralMatrix, generate
from poisson_gaussian import _fit_one, _m_step, _e_step, _floor_covariance

ROOT=Path(__file__).resolve().parent
PILOT=ROOT/'pilot_v2'/'jobs'


def parameters_from_labels(T,D,labels,k):
    p=T.shape[1]
    init=(np.full(k,1/k),np.full(k,D.mean()),np.zeros((k,p)),
          np.repeat(np.eye(p)[None],k,axis=0))
    return _m_step(T,D,np.eye(k)[labels],init,.02)


def trajectory(T,D,labels,truth,k,steps=150):
    params=parameters_from_labels(T,D,labels,k)
    history=[]
    old=-np.inf
    for it in range(steps+1):
        obj,resp=_e_step(T,D,params)
        pred=resp.argmax(1)
        history.append(dict(iteration=it,objective=obj,**metrics(truth,pred,k)))
        if it and obj-old <= 1e-5*(1+abs(old)):
            break
        old=obj
        params=_m_step(T,D,resp,params,.02)
    return history


def partition_info(T,D,z,truth,k):
    table=np.zeros((k,k),dtype=int)
    np.add.at(table,(truth,z),1)
    details=[]
    for a in range(k):
        keep=z==a
        details.append(dict(n=int(keep.sum()),degree_mean=float(D[keep].mean()) if keep.any() else None,
            degree_sd=float(D[keep].std()) if keep.any() else None,
            T_mean=T[keep].mean(0).tolist() if keep.any() else None,
            T_cov=np.cov(T[keep],rowvar=False,bias=True).tolist() if keep.sum()>1 else None))
    return dict(confusion=table.tolist(),groups=details)


def diagnose_pg(stem):
    obs=np.load(PILOT/(stem+'.npz'))
    meta=json.loads((PILOT/(stem+'.json')).read_text())['metadata']
    D,U,vals,truth=(obs[q] for q in ('D','U','vals','truth'))
    W=np.minimum(1,2*D.mean()/np.maximum(D,1))
    T=features(D,W,U,vals)['neighbor']; k=U.shape[1]
    out=dict(stem=stem,restarts=[])
    seq=np.random.SeedSequence(meta['seed']+7919).spawn(6)
    best=None
    for run,child in enumerate(seq):
        seed=int(child.generate_state(1,dtype=np.uint32)[0])
        mode='weighted_kmeans++_seeding' if run%2==0 else 'random_data'
        fit=_fit_one(T,D,k,seed,150,.02,1e-5,mode)
        diag=dict(run=run,objective=fit['objective'],**metrics(truth,fit['labels'],k),
                  iterations=fit['n_iter'],converged=fit['converged'],
                  degree_rates=fit['degree_rates'].tolist(),weights=fit['weights'].tolist())
        out['restarts'].append(diag)
        if best is None or fit['objective']>best['objective']:
            best=fit
    out['matches_saved_prediction']=bool(np.array_equal(best['labels'],obs['poisson_gaussian']))
    out['best_partition']=partition_info(T,D,best['labels'],truth,k)
    out['true_partition']=partition_info(T,D,truth,truth,k)
    out['best_model_means']=best['means'].tolist()
    out['best_model_covariances']=best['covariances'].tolist()
    out['best_cov_eigenvalues']=np.linalg.eigvalsh(best['covariances']).tolist()
    out['oracle_start_DIAGNOSTIC_ONLY']=trajectory(T,D,truth,truth,k)
    out['classic_kmeans_start_DIAGNOSTIC_ONLY']=trajectory(T,D,obs['km_classic_U'],truth,k)
    out['fraction_degree_le2']=float(np.mean(D<=2))
    out['fraction_degree_le4']=float(np.mean(D<=4))
    # Model requires E[T|D,z] constant and Var[T|D,z] proportional to 1/D.
    # Quantify the actual degree trend within each true group.
    trend=[]
    for a in range(k):
        keep=(truth==a)&(D>0)
        bins=np.unique(np.quantile(D[keep],[0,.25,.5,.75,1]).astype(int))
        for lo,hi in zip(bins[:-1],bins[1:]):
            b=keep&(D>=lo)&(D<=hi)
            trend.append(dict(group=a,degree_lo=int(lo),degree_hi=int(hi),n=int(b.sum()),
                mean=T[b].mean(0).tolist(),cov_scaled_by_mean_degree=(np.cov(T[b],rowvar=False,bias=True)*D[b].mean()).tolist()))
    out['degree_conditional_trend']=trend
    out['degree_vs_feature_correlation_by_truth']=[np.corrcoef(np.c_[D[truth==a],T[truth==a]].T)[0,1:].tolist() for a in range(k)]
    return out


def profile(A,z,k):
    r=np.eye(k)[z]
    nk=r.sum(0)
    expo=np.outer(nk,nk)-r.T@r
    edges=r.T@(A@r)
    raw=np.divide(edges,expo,out=np.zeros_like(edges),where=expo>0)
    B=np.clip(raw,1e-6/len(z),1-1e-6)
    edge_ll=.5*np.sum(edges*np.log(B)+(expo-edges)*np.log1p(-B))
    prior=np.sum(xlogy(nk,nk/len(z)))
    return dict(edge_loglik=float(edge_ll),prior_loglik=float(prior),total=float(edge_ll+prior),
                Braw=raw.tolist(),block_counts=edges.tolist(),sizes=nk.tolist(),
                negative_block_count_fraction=float(np.mean(edges<0)))


def diagnose_surrogate(stem):
    obs=np.load(PILOT/(stem+'.npz'))
    data=json.loads((PILOT/(stem+'.json')).read_text());meta=data['metadata']
    D,U,vals,truth=(obs[q] for q in ('D','U','vals','truth'))
    W=np.minimum(1,2*D.mean()/np.maximum(D,1));k=U.shape[1];n=len(D)
    S=DegreeMatchedSpectralMatrix(D,W,U,vals)
    dense=S@np.eye(n)
    off=dense[np.triu_indices(n,1)]
    # Raw A is reconstructed ONLY to diagnose which block signal was removed.
    A,z,B=generate(meta['model'],n,meta['degree_target'],meta['seed'])
    assert np.array_equal(z,truth)
    null=np.zeros(n,dtype=int)
    out=dict(stem=stem,negative_entry_fraction=float(np.mean(off<0)),
             over_one_fraction=float(np.mean(off>1)),entry_min=float(off.min()),entry_max=float(off.max()),
             negative_mass_fraction=float(-off[off<0].sum()/off[off>0].sum()),
             max_degree_reconstruction_error=float(np.max(np.abs(dense.sum(1)-D))),
             true_B=B.tolist())
    for name,pred in [('truth',truth),('kmeans',obs['km_classic_U']),
                      ('surrogate_fit',obs['spectral_bernoulli_surrogate']),('null',null)]:
        out[name]=dict(compressed=profile(S,pred,k),raw=profile(A,pred,k),
                       error=metrics(truth,pred,k)['error'])
    out['selected_surrogate_record']=next(r for r in data['records'] if r['method']=='spectral_bernoulli_surrogate' and r['stage']=='initial')
    return out


def main():
    pg_stems=[f'k2_symmetric_n1200_d6_s{s}' for s in range(3)]
    pg_stems += [f'k3_heterogeneous_n1200_d24_s{s}' for s in range(3)]
    pg_stems += ['k2_symmetric_n1200_d24_s1','k2_symmetric_n1200_d96_s1']
    pg=[diagnose_pg(stem) for stem in pg_stems]
    surrogate_stems=[f'k2_symmetric_n1200_d{d}_s{s}' for d in [6,24,96] for s in range(3)]
    surrogate_stems += ['k3_heterogeneous_n1200_d24_s1','k4_hierarchical_n1200_d24_s1']
    sur=[diagnose_surrogate(stem) for stem in surrogate_stems]
    result=dict(scope='PILOT ONLY; no final-test data inspected; no hyperparameter or method changes',
                poisson_gaussian=pg,spectral_bernoulli_surrogate=sur)
    path=ROOT/'pilot_diagnostics.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Wrote '+str(path))
    for d in pg:
        best=max(d['restarts'],key=lambda r:r['objective']);besterr=min(d['restarts'],key=lambda r:r['error'])
        tr=d['oracle_start_DIAGNOSTIC_ONLY']
        print(json.dumps(dict(stem=d['stem'],best_objective_run=best,
            best_error_run_DIAGNOSTIC_ONLY=besterr,
            oracle_start_first=tr[0],oracle_start_last=tr[-1],
            degree_correlation=d['degree_vs_feature_correlation_by_truth'])))
    for d in sur:
        print(json.dumps(dict(stem=d['stem'],negative_fraction=d['negative_entry_fraction'],negative_mass_fraction=d['negative_mass_fraction'],
            compressed_true_minus_null=d['truth']['compressed']['total']-d['null']['compressed']['total'],
            raw_true_minus_null=d['truth']['raw']['total']-d['null']['raw']['total'],
            compressed_B_true_labels=d['truth']['compressed']['Braw'],raw_B_true_labels=d['truth']['raw']['Braw'],
            surrogate_sizes=d['surrogate_fit']['compressed']['sizes'],negative_block_fraction=d['surrogate_fit']['compressed']['negative_block_count_fraction'])))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Paired, unsupervised SBM likelihood-initialization benchmark.

Compressed methods receive ONLY (degree, eigenvectors, eigenvalues) of W A W.
Ground truth is used only by the evaluator. A separate Bernoulli VEM arm reads A.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import concurrent.futures
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits


MODELS = {
    'k2_symmetric': dict(B0=[[4, 1], [1, 4]], pi=[.5, .5]),
    'k3_equal_degree': dict(B0=[[8, 1, 3], [1, 9, 2], [3, 2, 7]], pi=[1/3]*3),
    'k3_heterogeneous': dict(B0=[[10, 2, 1], [2, 5, 2], [1, 2, 3]], pi=[1/3]*3),
    'k4_equal_degree': dict(B0=[[8, 4, 1, 3], [4, 8, 3, 1], [1, 3, 8, 4], [3, 1, 4, 8]], pi=[.25]*4),
    'k4_hierarchical': dict(B0=[[10, 5, 1, 1], [5, 7, 1, 1], [1, 1, 6, 3], [1, 1, 3, 4]], pi=[.25]*4),
    'k3_disassortative': dict(B0=[[1, 6, 2], [6, 1, 3], [2, 3, 1]], pi=[1/3]*3),
}


def spd_floor(cov, floor):
    vals, vecs = eigh((cov+cov.T)*.5, check_finite=False)
    vals = np.maximum(vals, floor)
    return (vecs*vals)@vecs.T


def standardize(x):
    x = np.asarray(x, float)
    mu = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-10] = 1.
    return (x-mu)/scale


def fit_gmm(x, k, seed, covariance='full', n_init=6, max_iter=150,
            floor=.01, tol=1e-5):
    """EM with constrained covariance eigenvalues >= floor.

    Starts use k-means++ center selection ONLY, never Lloyd iterations or truth.
    Eigenvalue clipping is the exact constrained covariance M step. Restarts
    are selected by Gaussian-mixture log likelihood of the supplied features.
    """
    n, p = x.shape
    global_cov = spd_floor(np.cov(x, rowvar=False, bias=True).reshape(p,p), floor)
    best = None
    for run in range(n_init):
        rng_seed = int(seed + 104729*run)
        mu, _ = kmeans_plusplus(x, n_clusters=k, random_state=rng_seed)
        covs = np.repeat(global_cov[None], k, 0)
        weights = np.ones(k)/k
        prev = -np.inf
        converged = False
        decreases = 0
        for it in range(max_iter+1):
            logp = np.empty((n,k))
            for a in range(k):
                vals, vecs = eigh(covs[a], check_finite=False)
                diff = (x-mu[a])@vecs
                quad = ((diff*diff)/vals).sum(1)
                logp[:,a] = np.log(weights[a])-.5*(p*np.log(2*np.pi)+np.log(vals).sum()+quad)
            norms = logsumexp(logp, axis=1)
            objective = float(norms.sum())
            if objective < prev-1e-6*max(1,abs(prev)):
                decreases += 1
            if it and abs(objective-prev) <= tol*n:
                converged = True
                break
            if it == max_iter:
                break
            resp = np.exp(logp-norms[:,None])
            nk = np.maximum(resp.sum(0), 1e-12)
            weights = nk/nk.sum()
            mu = resp.T@x/nk[:,None]
            if covariance == 'tied':
                cov = np.zeros((p,p))
                for a in range(k):
                    diff=x-mu[a]
                    cov += (diff.T*resp[:,a])@diff
                covs[:] = spd_floor(cov/n, floor)
            else:
                for a in range(k):
                    diff=x-mu[a]
                    covs[a] = spd_floor((diff.T*resp[:,a])@diff/nk[a], floor)
            prev=objective
        result = dict(labels=logp.argmax(1), objective=objective, n_iter=it,
                      converged=converged, decreases=decreases, selected_restart=run)
        if best is None or objective > best['objective']:
            best=result
    return best


def initial_resp(x,k,seed):
    centers,_=kmeans_plusplus(x,n_clusters=k,random_state=seed)
    d2=((x[:,None,:]-centers[None,:,:])**2).sum(2)
    labels=d2.argmin(1)
    r=np.full((len(x),k),.05/k)
    r[np.arange(len(x)),labels]+=.95
    return r


def fit_kmeans(x,k,seed,n_init=6,max_iter=150):
    """Exactly match GMM's per-restart center selection on the same inputs."""
    best=None
    for run in range(n_init):
        centers,_=kmeans_plusplus(x,n_clusters=k,random_state=int(seed+104729*run))
        est=KMeans(n_clusters=k,init=centers,n_init=1,max_iter=max_iter,
                   algorithm='lloyd',random_state=seed).fit(x)
        res=dict(labels=est.labels_,objective=-float(est.inertia_),
                 n_iter=int(est.n_iter_),converged=est.n_iter_<max_iter,
                 selected_restart=run)
        if best is None or res['objective']>best['objective']:
            best=res
    return best


class DegreeMatchedSpectralMatrix:
    """Signed low-rank reconstruction with zero diagonal and exact degrees.

    Its Bernoulli-shaped objective is a SURROGATE, not a graph likelihood:
    off-diagonal entries need not belong to [0,1]. No entrywise clipping.
    """
    def __init__(self,D,W,U,vals):
        self.F=U/W[:,None]
        self.vals=vals
        self.n=len(D)
        self.diag=(self.F*self.F)@vals
        rowsum=(self.F*vals)@self.F.sum(0)-self.diag
        self.r=D-rowsum
        self.c=self.r.sum()/((self.n-1)*(self.n-2))
        self.kdiag=2*self.r/(self.n-2)-self.c
    def __matmul__(self,R):
        sums=R.sum(0)
        base=(self.F*self.vals)@(self.F.T@R)-self.diag[:,None]*R
        corr=(self.r[:,None]*sums[None,:]+(self.r@R)[None,:])/(self.n-2)
        corr-=self.c*sums[None,:]
        return base+corr-self.kdiag[:,None]*R


def fit_vem(A, x, k, seed, n_init=6, max_iter=120, tol=1e-5):
    """Raw-A Bernoulli SBM mean-field variational EM.

    Spectral centers seed responsibilities without a fitted k-means partition.
    Batch E proposals are backtracked to ensure increasing fixed-B ELBO.
    This arm intentionally has MORE information than compressed methods.
    """
    n=len(x)
    def mstep(r):
        nk=r.sum(0)
        exposure=np.outer(nk,nk)-r.T@r
        edges=r.T@(A@r)
        B=np.clip(edges/np.maximum(exposure,1e-12),1e-6/n,1-1e-6)
        return B,nk/n
    def objective(r,B,pi):
        nk=r.sum(0)
        edges=r.T@(A@r)
        exposure=np.outer(nk,nk)-r.T@r
        return float(.5*(edges*np.log(B)+(exposure-edges)*np.log1p(-B)).sum()
                     +(r*np.log(np.maximum(pi,1e-300))).sum()
                     -(r*np.log(np.maximum(r,1e-300))).sum())
    best=None
    for run in range(n_init):
        r=initial_resp(x,k,int(seed+104729*run))
        B,pi=mstep(r)
        obj=objective(r,B,pi)
        converged=False
        backtracks=0
        for it in range(max_iter):
            nk=r.sum(0)
            scores=(A@r)@(np.log(B)-np.log1p(-B)).T
            scores+=(nk[None,:]-r)@np.log1p(-B).T+np.log(np.maximum(pi,1e-300))
            prop=np.exp(scores-logsumexp(scores,axis=1)[:,None])
            step=1.
            trial=obj
            for bt in range(14):
                rn=(1-step)*r+step*prop
                trial=objective(rn,B,pi)
                if trial>=obj-1e-8:
                    break
                step*=.5
                backtracks+=1
            if trial<obj-1e-8:
                break
            Bn,pin=mstep(rn)
            newobj=objective(rn,Bn,pin)
            improvement=newobj-obj
            r,B,pi,obj=rn,Bn,pin,newobj
            if improvement<=tol*n:
                converged=True
                break
        res=dict(labels=r.argmax(1),objective=obj,n_iter=it+1,converged=converged,
                 backtracks=backtracks,selected_restart=run)
        if best is None or res['objective']>best['objective']:
            best=res
    return best


def generate(model,n,degree,seed):
    spec=MODELS[model]
    B0=np.asarray(spec['B0'],float)
    pi=np.asarray(spec['pi'])
    k=len(pi)
    counts=np.floor(n*pi).astype(int)
    counts[:n-counts.sum()]+=1
    labels=np.repeat(np.arange(k),counts)
    rng=np.random.default_rng(np.random.SeedSequence([20261009,seed,list(MODELS).index(model),int(degree*100)]))
    rng.shuffle(labels)
    C=B0/(pi@B0@pi)
    B=degree*C/n
    uu,vv=np.triu_indices(n,1)
    keep=rng.random(len(uu))<B[labels[uu],labels[vv]]
    rows=np.concatenate([uu[keep],vv[keep]])
    cols=np.concatenate([vv[keep],uu[keep]])
    A=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n))
    return A,labels,B


def compress(A,k,seed):
    D=np.asarray(A.sum(1)).ravel()
    mean=D.mean()
    W=np.minimum(1,2*mean/np.maximum(D,1))
    M=A.multiply(W[:,None]).multiply(W[None,:]).tocsr()
    vals,U=eigsh(M,k=k,which='LM',tol=1e-7,maxiter=10000,
                 v0=np.random.default_rng(seed+1701).normal(size=len(D)))
    order=np.argsort(-np.abs(vals)); vals=vals[order];U=U[:,order]
    return D,W,U,vals


def features(D,W,U,vals):
    n=len(D)
    X=np.sqrt(n)*U/W[:,None]
    S=standardize(X)
    Z=standardize(np.c_[S,D])
    Y=(U*vals)/W[:,None]
    T=np.sqrt(n)*Y/np.maximum(D,1)[:,None]
    active=D>0
    center=T[active].mean(0)
    cov=np.cov(T[active],rowvar=False,bias=True)
    ev,vec=eigh(cov,check_finite=False)
    transform=vec/np.sqrt(np.maximum(ev, max(ev.max()*1e-4,1e-10)))
    T=(T-center)@transform
    T[~active]=0
    TD=standardize(np.c_[T,D])
    ASE=X*np.sqrt(np.abs(vals))[None,:]
    RRE=X*vals[None,:]
    return dict(spec=S,spec_degree=Z,neighbor=T,neighbor_degree=TD,
                classic=X,ase=ASE,rre=RRE)


def decoded_llr(labels,D,W,U,vals):
    """One shared compressed-information decoder, no A argument.

    Nuisance parameters and weighted centroids estimated from supplied labels.
    Signed inferred counts retained; degree correction enforces row sums.
    Explicit fallback on empty clusters / ill-conditioned centroid matrix.
    """
    n,k=U.shape
    Z=np.eye(k)[labels]
    counts=Z.sum(0)
    if counts.min()<max(k+2,.01*n):
        return labels.copy(),dict(decoder_fallback=1,decoder_reason='small_cluster')
    H=(Z.T@(W[:,None]*U))/counts[:,None]
    cond=float(np.linalg.cond(H))
    if not np.isfinite(cond) or cond>1e6:
        return labels.copy(),dict(decoder_fallback=1,decoder_reason='ill_conditioned',
                                  centroid_condition=cond if np.isfinite(cond) else None)
    N0=np.linalg.solve(H.T,((U*vals)/W[:,None]).T).T
    pi=counts/n
    N=N0+(D-N0.sum(1))[:,None]*pi
    Q=Z.T@U
    sums=Z.T@W
    numerator=(Q*vals)@Q.T
    diag_m=(U*U)@vals
    numerator[np.diag_indices(k)]-=Z.T@diag_m
    denom=np.outer(sums,sums)
    denom[np.diag_indices(k)]-=Z.T@(W*W)
    Braw=numerator/np.maximum(denom,1e-12)
    B=np.clip(Braw,1e-4*max(D.mean(),1)/n,1-1e-6)
    sizes=counts[None,:]-Z
    score=N@(np.log(B)-np.log1p(-B)).T+sizes@np.log1p(-B).T+np.log(pi)
    return score.argmax(1),dict(decoder_fallback=0,centroid_condition=cond,
                                block_clipped_fraction=float(np.mean(Braw!=B)),
                                negative_count_fraction=float(np.mean(N<0)))


def metrics(truth,pred,k):
    tab=np.zeros((k,k),dtype=int)
    np.add.at(tab,(truth,pred),1)
    rr,cc=linear_sum_assignment(-tab)
    err=1-tab[rr,cc].sum()/len(truth)
    sizes=np.bincount(pred,minlength=k)
    return dict(error=float(err),ari=float(adjusted_rand_score(truth,pred)),
                min_cluster=int(sizes.min()),max_cluster=int(sizes.max()))


def run_one(job):
    with threadpool_limits(limits=1):
        model,n,degree,seed,n_init,max_iter,outdir,include_raw=job
        path=Path(outdir)/'jobs'/f'{model}_n{n}_d{degree:g}_s{seed}.json'
        if path.exists():
            return dict(cached=True,job=path.stem)
        t0=time.perf_counter()
        A,truth,B=generate(model,n,degree,seed)
        k=B.shape[0]
        te=time.perf_counter()
        D,W,U,vals=compress(A,k,seed)
        eig_time=time.perf_counter()-te
        F=features(D,W,U,vals)
        meta=dict(model=model,n=n,k=k,degree_target=degree,seed=seed,
                  degree_realized=float(D.mean()),fraction_capped=float(np.mean(W<1)),
                  fraction_isolated=float(np.mean(D==0)),eigenvalues=vals.tolist(),
                  eig_seconds=eig_time,negative_eigenvalues=int(np.sum(vals<0)))
        methods=[('km_classic_U','km','classic'),('km_ASE','km','ase'),('km_RRE','km','rre'),
                 ('km_spec','km','spec'),('gmm_tied_spec','tied','spec'),
                 ('gmm_full_spec','full','spec'),('km_spec_degree','km','spec_degree'),
                 ('gmm_full_spec_degree','full','spec_degree'),
                 ('km_neighbor_degree','km','neighbor_degree'),
                 ('gmm_full_neighbor_degree','full','neighbor_degree'),
                 ('poisson_gaussian','pg','neighbor'),
                 ('spectral_bernoulli_surrogate','surrogate','neighbor_degree')]
        if include_raw:
            methods.append(('raw_bernoulli_vem','raw','neighbor_degree'))
        records=[]
        predictions={}
        for name,kind,feat in methods:
            ts=time.perf_counter()
            algseed=seed+7919
            try:
                if kind=='km':
                    fit=fit_kmeans(F[feat],k,algseed,n_init,max_iter)
                elif kind in ('full','tied'):
                    fit=fit_gmm(F[feat],k,algseed,kind,n_init,max_iter)
                elif kind=='pg':
                    from poisson_gaussian import fit_poisson_gaussian
                    fit=fit_poisson_gaussian(F[feat],D,k,algseed,n_init=n_init,max_iter=max_iter)
                elif kind=='surrogate':
                    S=DegreeMatchedSpectralMatrix(D,W,U,vals)
                    fit=fit_vem(S,F[feat],k,algseed,n_init,max_iter)
                else:
                    fit=fit_vem(A,F[feat],k,algseed,n_init,max_iter)
                labels=np.asarray(fit.pop('labels'),int)
                fit={key:(value.item() if isinstance(value,np.generic) else value) for key,value in fit.items()
                     if key not in ('means','covariances','weights','degree_means','responsibilities','history','all_runs','objective_history')}
                fit={key:value for key,value in fit.items() if np.isscalar(value) or value is None}
                fit={(('fit_'+key) if key in meta else key):value for key,value in fit.items()}
                runtime=time.perf_counter()-ts
                records.append(dict(**meta,method=name,stage='initial',seconds=runtime,
                                    failed=0,**metrics(truth,labels,k),**fit))
                predictions[name]=labels.astype(np.int16)
                if kind!='raw':
                    td=time.perf_counter()
                    refined,diag=decoded_llr(labels,D,W,U,vals)
                    records.append(dict(**meta,method=name,stage='decoded',
                                        seconds=runtime+time.perf_counter()-td,failed=0,
                                        **metrics(truth,refined,k),**diag))
                    predictions[name+'_decoded']=refined.astype(np.int16)
            except Exception as ex:
                import traceback
                records.append(dict(**meta,method=name,stage='initial',failed=1,
                                    error_message=str(ex),traceback=traceback.format_exc(),
                                    seconds=time.perf_counter()-ts))
        # Edge-only oracle benchmark: other labels are provided solely to form
        # edge counts; do not use exact-balance inference to recover the target.
        Ztrue=np.eye(k)[truth]
        Ntrue=A@Ztrue
        exposure=Ztrue.sum(0)[None,:]-Ztrue
        oracle=(Ntrue@(np.log(B)-np.log1p(-B)).T+exposure@np.log1p(-B).T).argmax(1)
        records.append(dict(**meta,method='edge_only_oracle',stage='reference',failed=0,seconds=0.,
                            **metrics(truth,oracle,k)))
        path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(metadata=meta,records=records,elapsed=time.perf_counter()-t0)
        tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2))
        tmp.replace(path)
        np.savez_compressed(path.with_suffix('.npz'),truth=truth.astype(np.int16),D=D,
                            U=U,vals=vals,**predictions)
        return dict(job=path.stem,seconds=payload['elapsed'],
                    failures=sum(r.get('failed',0) for r in records))


def aggregate(outdir):
    paths=sorted((Path(outdir)/'jobs').glob('*.json'))
    recs=[]
    for path in paths:
        recs.extend(json.loads(path.read_text())['records'])
    if recs:
        df=pd.DataFrame(recs)
        df.to_csv(Path(outdir)/'per_run_results.csv',index=False)
        ok=df[df.failed==0]
        summary=ok.groupby(['model','n','degree_target','method','stage'],dropna=False).agg(
            repetitions=('error','size'),error_mean=('error','mean'),error_sd=('error','std'),
            ari_mean=('ari','mean'),seconds_mean=('seconds','mean'),
            cap_mean=('fraction_capped','mean'),isolates_mean=('fraction_isolated','mean')).reset_index()
        summary.to_csv(Path(outdir)/'summary.csv',index=False)
        return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=1800)
    p.add_argument('--degrees',default='4,6,12,24')
    p.add_argument('--models',default=','.join(MODELS))
    p.add_argument('--seeds',default='100:120')
    p.add_argument('--n-init',type=int,default=6)
    p.add_argument('--max-iter',type=int,default=150)
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--no-raw',action='store_true')
    p.add_argument('--aggregate-only',action='store_true')
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.aggregate_only:
        print(aggregate(out).to_string(index=False));return
    lo,hi=map(int,args.seeds.split(':'))
    config=dict(vars(args));config['model_definitions']=MODELS
    config['notes']=['Known K, exactly balanced labels for declared equal proportions.',
        'No true parameters or labels used for fitting, restarting, or tuning.',
        'Covariance eigenvalue floor .01 for standardized GMM inputs.',
        'Spectral features sqrt(n) W^-1 U, per-coordinate observed standardization.',
        'Raw Bernoulli VEM is explicitly an additional-A comparator.',
        'KMeans and GMM on matching features use exactly the same per-restart center seeds.',
        'Spectral Bernoulli-shaped surrogate uses a signed degree-matched zero-diagonal low-rank matrix, not actual Bernoulli observations.',
        'Common decoded stage is a finite-sample plug-in of the proposed spectral-degree LLR decoder.']
    (out/'protocol.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    jobs=[(model,args.n,float(d),seed,args.n_init,args.max_iter,str(out),not args.no_raw)
          for model in args.models.split(',') for d in args.degrees.split(',') for seed in range(lo,hi)]
    print(json.dumps(dict(jobs=len(jobs),workers=args.workers,out=str(out))),flush=True)
    started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(run_one,j):j for j in jobs}
        for idx,f in enumerate(concurrent.futures.as_completed(futures),1):
            try: result=f.result()
            except Exception as e: result=dict(fatal=str(e),job=str(futures[f]))
            print(json.dumps(dict(done=idx,total=len(jobs),wall_seconds=round(time.perf_counter()-started,2),**result)),flush=True)
    summary=aggregate(out)
    print('Complete. Aggregated '+str(len(summary))+' condition-method-stage rows.',flush=True)


if __name__=='__main__':
    main()

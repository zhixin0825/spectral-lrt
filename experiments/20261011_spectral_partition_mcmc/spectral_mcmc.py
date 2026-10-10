"""Global independent MH proposals from random spectral directions.

Draw a fresh Gaussian K-by-K matrix W, turn centered retained eigenvector
rows times W into strictly positive categorical probabilities, and propose
all labels independently. For each fixed W, both q(old) and q(new) are
known. The exact independence-MH correction preserves the SAME collapsed
working target. Gaussian directions use no labels, k-means, or true P.
"""
import time
import math
import numpy as np
from numba import njit
from engine import _counts,_score,_sweep


@njit(cache=True)
def _run(Q,X,k,z0,sweeps,seed,scale,noise,every):
    np.random.seed(seed)
    n=len(z0)
    z=z0.copy();sizes,edges=_counts(Q,z,k)
    records=1+sweeps//every+int(sweeps%every!=0)
    labels=np.empty((records,n),np.int64)
    scores=np.empty(records)
    times=np.empty(records,np.int64)
    accepted_trace=np.empty(records,np.int64)
    labels[0],scores[0],times[0],accepted_trace[0]=z,_score(sizes,edges),0,0
    best,best_score=z.copy(),scores[0]
    accepted=0;index=1
    for sweep in range(1,sweeps+1):
        _sweep(Q,z,sizes,edges,1.,False)
        score=_score(sizes,edges)
        W=np.random.normal(0.,1.,(X.shape[1],k))
        fields=scale*(X@W)
        proposed=np.empty(n,np.int64)
        logq_old,logq_new=0.,0.
        for i in range(n):
            probabilities=np.exp(fields[i]-np.max(fields[i]))
            probabilities=(1.-noise)*probabilities/probabilities.sum()+noise/k
            u=np.random.random()
            chosen=k-1
            for a in range(k):
                u-=probabilities[a]
                if u<=0:
                    chosen=a
                    break
            proposed[i]=chosen
            logq_old+=math.log(probabilities[z[i]])
            logq_new+=math.log(probabilities[chosen])
        ns,ne=_counts(Q,proposed,k)
        new_score=_score(ns,ne)
        log_accept=new_score-score+logq_old-logq_new
        if math.log(np.random.random())<min(0.,log_accept):
            z,sizes,edges,score=proposed,ns,ne,new_score
            accepted+=1
        if score>best_score:
            best,best_score=z.copy(),score
        if sweep%every==0 or sweep==sweeps:
            labels[index],scores[index],times[index],accepted_trace[index]=z,score,sweep,accepted
            index+=1
    return z,best,labels,scores,times,accepted,accepted_trace


def spectral_chain(Q,k,z0,sweeps,seed,U,scale=8.,noise=.2,record_every=5):
    if not 0<noise<1:
        raise ValueError('Proposal noise must be strictly between zero and one')
    start=time.perf_counter()
    X=np.sqrt(len(Q))*(U-U.mean(axis=0))
    z,best,labels,scores,times,accepted,accepted_trace=_run(Q,np.ascontiguousarray(X),k,np.asarray(z0,dtype=np.int64),sweeps,seed,scale,noise,record_every)
    return dict(labels=z,best_labels=best,trace_labels=labels,trace_score=scores,trace_sweep=times,copy_accepted=np.array([accepted]),copy_tries=np.array([sweeps]),donor_trace_score=accepted_trace[:,None],elapsed=time.perf_counter()-start)


if __name__=='__main__':
    Q=np.array([[0.,.2,.7,.1],[.2,0.,.1,.8],[.7,.1,0.,.2],[.1,.8,.2,0.]])
    U=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])/np.sqrt(2.)
    r=spectral_chain(Q,2,np.array([0,1,0,1]),1,11,U,record_every=1)
    print('spectral proposal warmup completed',flush=True)

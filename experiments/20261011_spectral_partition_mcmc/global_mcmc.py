"""Global partition proposals with an exact MH correction.

Six independently initialized replicas use inverse temperatures >=1. Replica
zero targets the original collapsed working posterior. An auxiliary replica
supplies a noisy product proposal for the entire partition. Its density is
evaluated in both directions; donor replicas remain unchanged. Thus every
conditional move preserves the product target, even though proposals depend
on other replicas. No spectral clustering or ground truth is used to propose.
"""
import time
import math
import numpy as np
from numba import njit
from engine import _counts, _score, _sweep


@njit(cache=True)
def _run(Q,k,z0,sweeps,seed,betas,noise,interval,every):
    np.random.seed(seed)
    m,n=len(betas),len(z0)
    z=np.empty((m,n),np.int64)
    sizes=np.empty((m,k),np.int64)
    edges=np.empty((m,k,k))
    for r in range(m):
        z[r]=z0 if r==0 else np.random.randint(0,k,n)
        sizes[r],edges[r]=_counts(Q,z[r],k)
    records=1+sweeps//every+int(sweeps%every!=0)
    labels=np.empty((records,n),np.int64)
    scores=np.empty(records)
    times=np.empty(records,np.int64)
    labels[0],scores[0],times[0]=z[0],_score(sizes[0],edges[0]),0
    best,best_score=z[0].copy(),scores[0]
    accepted=np.zeros(m,np.int64)
    tried=np.zeros(m,np.int64)
    donor_scores=np.empty((records,m))
    for r in range(m):
        donor_scores[0,r]=_score(sizes[r],edges[r])
    same=math.log(1.-noise+noise/k)
    other=math.log(noise/k)
    index=1
    for sweep in range(1,sweeps+1):
        for r in range(m):
            _sweep(Q,z[r],sizes[r],edges[r],betas[r],False)
        score=_score(sizes[0],edges[0])
        if sweep%interval==0:
            donor=np.random.randint(1,m)
            proposed=z[donor].copy()
            logq_old,logq_new=0.,0.
            for i in range(n):
                if np.random.random()<noise:
                    proposed[i]=np.random.randint(k)
                logq_old+=same if z[0,i]==z[donor,i] else other
                logq_new+=same if proposed[i]==z[donor,i] else other
            ns,ne=_counts(Q,proposed,k)
            new_score=_score(ns,ne)
            tried[donor]+=1
            if math.log(np.random.random())<min(0.,betas[0]*(new_score-score)+logq_old-logq_new):
                z[0],sizes[0],edges[0]=proposed,ns,ne
                score=new_score
                accepted[donor]+=1
        if score>best_score:
            best,best_score=z[0].copy(),score
        if sweep%every==0 or sweep==sweeps:
            labels[index],scores[index],times[index]=z[0],score,sweep
            for r in range(m):
                donor_scores[index,r]=_score(sizes[r],edges[r])
            index+=1
    return z[0],best,labels,scores,times,accepted,tried,donor_scores


def global_chain(Q,k,z0,sweeps,seed,betas=(1.,2.,4.,8.,16.,32.),noise=.05,proposal_interval=5,record_every=5):
    if betas[0]!=1 or len(betas)<2 or not 0<noise<1:
        raise ValueError('Cold target beta must be1, with at least one donor and0<noise<1')
    t=time.perf_counter()
    z,best,labels,scores,times,accepted,tried,donor_scores=_run(Q,k,np.asarray(z0,dtype=np.int64),sweeps,seed,np.array(betas),noise,proposal_interval,record_every)
    return dict(labels=z,best_labels=best,trace_labels=labels,trace_score=scores,trace_sweep=times,copy_accepted=accepted,copy_tries=tried,copy_acceptance=accepted/np.maximum(tried,1),donor_trace_score=donor_scores,elapsed=time.perf_counter()-t)


def validate_global_proposal():
    import itertools
    from engine import log_target
    Q=np.array([[0.,.2,.7,.1],[.2,0.,.1,.8],[.7,.1,0.,.2],[.1,.8,.2,0.]])
    states=np.array(list(itertools.product(range(2),repeat=4)),dtype=np.int64)
    logs=np.array([log_target(Q,z,2) for z in states])
    p=np.exp(logs-logs.max());p/=p.sum()
    residual=0.
    for donor in states:
        lp=np.array([np.where(z==donor,np.log(.975),np.log(.025)).sum() for z in states])
        for a in range(len(states)):
            for b in range(len(states)):
                lab=logs[b]-logs[a]+lp[a]-lp[b]
                forward=p[a]*np.exp(lp[b]+min(0.,lab))
                backward=p[b]*np.exp(lp[a]+min(0.,-lab))
                residual=max(residual,abs(forward-backward))
    # Long small-state check of the actual interacting sampler.
    r=global_chain(Q,2,states[3],30000,710,betas=(1.,2.,4.),record_every=1)
    integer=np.array([8,4,2,1])
    observed=np.bincount(r['trace_labels'][1:]@integer,minlength=16)/30000
    tv=.5*np.abs(observed-p).sum()
    return dict(max_detailed_balance_residual=float(residual),cold_marginal_total_variation=float(tv),passed=residual<1e-12 and tv<.05)


if __name__=='__main__':
    print(validate_global_proposal(),flush=True)

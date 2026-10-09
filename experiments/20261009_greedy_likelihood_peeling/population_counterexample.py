"""Exact class-compressed population calculation; no n-by-n matrix needed."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

def example(n=6000):
    sizes=np.array([n//2,3*n//10,n-n//2-3*n//10])
    c=np.ones((3,3));np.fill_diagonal(c,10.)
    rho=np.log(n)/n;p=rho*c
    # Every observation of true class a has this identical population profile.
    # broadcasting subtracts candidate exposure on columns, not observation rows.
    lp=(p*sizes)@np.log(p).T-(p@sizes)[None,:]
    lb=(p*sizes)@np.log(p/(1-p)).T+(np.log1p(-p)@sizes)[None,:]
    records={}
    for name,score in [('poisson',lp),('bernoulli',lb)]:
        remaining=sizes.copy();seeds=[];events=[]
        for r in [3,2,1]:
            count=min(remaining.sum()//r,remaining.sum()-r) if r>1 else remaining.sum()-1
            choices=[]
            for candidate in np.flatnonzero(remaining):
                available=remaining.copy();available[candidate]-=1
                allocation=np.zeros(3,int);need=count
                for observation in np.lexsort((np.arange(3),-score[:,candidate])):
                    take=min(need,available[observation]);allocation[observation]=take;need-=take
                total=float(allocation@score[:,candidate])
                choices.append((candidate,total,allocation))
            candidate,total,allocation=max(choices,key=lambda x:(x[1],-x[0]))
            removed=allocation.copy();removed[candidate]+=1
            remaining-=removed;seeds.append(int(candidate))
            events.append(dict(remaining_groups=r,scored_other_count=int(count),
                selected_class_one_based=int(candidate+1),candidate_class_totals={str(x[0]+1):x[1] for x in choices},
                removed_class_counts=removed.tolist()))
        records[name]=dict(selected_classes_one_based=[x+1 for x in seeds],events=events)
    assert records['poisson']['selected_classes_one_based']==[3,1,1]
    assert records['bernoulli']['selected_classes_one_based']==[3,1,1]
    mu=p[[2,0,0]].copy();weights=np.ones(3)/3
    max_difference=0.
    for _ in range(100):
        score=(p*sizes)@np.log(mu).T-(mu@sizes)[None,:]+np.log(weights)[None,:]
        responsibility=np.exp(score-logsumexp(score,axis=1)[:,None])
        mass=(sizes[:,None]*responsibility).sum(axis=0)
        mu=((sizes[:,None]*responsibility).T@p)/mass[:,None]
        weights=mass/n
        max_difference=max(max_difference,float(np.max(np.abs(mu[1]-mu[2]))),float(abs(weights[1]-weights[2])))
    assert max_difference==0.
    return dict(n=n,sizes=sizes.tolist(),rho=float(rho),C=c.tolist(),
        selected_classes_one_based=records['poisson']['selected_classes_one_based'],
        removed_class_counts=[x['removed_class_counts'] for x in records['poisson']['events']],
        scores=records,duplicate_components_max_difference_after_100_EM=max_difference,
        CH_halfpoint_lower_bound=float(.25*(np.sqrt(10)-1)**2),
        interpretation='Ideal rank-3 population Q=ZBZ.T including diagonal; standard unperturbed soft EM; no original random adjacency is claimed.')

if __name__=='__main__':
    value=example();Path(__file__).with_name('population_counterexample.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value,indent=2))

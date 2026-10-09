"""Class-compressed population check of the unchanged raw-score rules."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from partial_peeling import other_count,RULES

def example(sizes):
    sizes=np.asarray(sizes,int)
    n=int(sizes.sum()); k=len(sizes)
    c=np.ones((k,k)); np.fill_diagonal(c,10.)
    p=c*np.log(n)/n
    scores=dict(poisson=(p*sizes)@np.log(p).T-(p@sizes)[None,:],
        bernoulli=(p*sizes)@np.log(p/(1-p)).T+(np.log1p(-p)@sizes)[None,:])
    results={}
    for likelihood,score in scores.items():
        for rule in RULES:
            remaining=sizes.copy();seeds=[];events=[]
            for stage in range(k):
                r=k-stage;count=other_count(int(remaining.sum()),k,r,rule)
                choices=[]
                for candidate in np.flatnonzero(remaining):
                    available=remaining.copy();available[candidate]-=1
                    allocation=np.zeros(k,int);need=count
                    for observation in np.lexsort((np.arange(k),-score[:,candidate])):
                        take=min(need,int(available[observation]))
                        allocation[observation]=take;need-=take
                    choices.append((int(candidate),float(allocation@score[:,candidate]),allocation))
                winner,total,allocation=max(choices,key=lambda x:(x[1],-x[0]))
                removed=allocation.copy();removed[winner]+=1
                remaining-=removed;seeds.append(winner)
                events.append(dict(remaining_groups=r,count=count,selected_class=winner+1,
                    removed_class_counts=removed.tolist(),candidate_totals={str(x[0]+1):x[1] for x in choices}))
            mu=p[seeds].copy(); weights=np.ones(k)/k
            objective_trace=[]
            duplicates=[(i,j) for i in range(k) for j in range(i+1,k) if seeds[i]==seeds[j]]
            max_duplicate_difference=0.
            for _ in range(100):
                component=((p*sizes)@np.log(mu).T-(mu@sizes)[None,:] if likelihood=='poisson' else
                           (p*sizes)@np.log(mu/(1-mu)).T+(np.log1p(-mu)@sizes)[None,:])
                value=component+np.log(weights)[None,:]
                responsibility=np.exp(value-logsumexp(value,axis=1)[:,None])
                objective_trace.append(float(sizes@logsumexp(value,axis=1)))
                mass=(sizes[:,None]*responsibility).sum(0)
                mu=((sizes[:,None]*responsibility).T@p)/mass[:,None]
                weights=mass/n
                for i,j in duplicates:
                    max_duplicate_difference=max(max_duplicate_difference,float(np.max(np.abs(mu[i]-mu[j]))),float(abs(weights[i]-weights[j])))
            results[likelihood+'_'+rule]=dict(seeds_one_based=[s+1 for s in seeds],
                covered_classes=len(set(seeds)),leftover_class_counts=remaining.tolist(),events=events,
                duplicate_component_max_difference=max_duplicate_difference,
                final_class_assignment_one_based=(value.argmax(1)+1).tolist(),
                EM_objective_trace=objective_trace)
    return dict(n=n,sizes=sizes.tolist(),C=c.tolist(),results=results,
        caveat='Ideal rank-K Q=Z B Z.T includes diagonal; not a random adjacency matrix; score uses original spectral profile parameters')

def main():
    payload={name:example(sizes) for name,sizes in
        [('old_unequal_example',[3000,1800,1200]),('balanced3',[2000,2000,2000]),
         ('more_unequal3',[4800,900,300]),('balanced4',[1500]*4)]}
    Path(__file__).with_name('population.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    for name,item in payload.items():
        print(name,{m:r['seeds_one_based'] for m,r in item['results'].items()})

if __name__=='__main__': main()

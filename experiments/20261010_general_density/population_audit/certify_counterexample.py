"""Exact rational interval certificate for the K=5 high-signal limiting path.

No external arithmetic package is needed. log bounds use the atanh series
on [1,2] with an explicit geometric tail. Every asserted inequality compares
rational endpoints, not floating-point approximations.
"""
from fractions import Fraction as F
from functools import lru_cache
import json
from pathlib import Path

M = [[71,914,119,129,132], [914,27252,2538,2125,2722],
     [119,2538,285,276,300], [129,2125,276,294,289],
     [132,2722,300,289,321]]
B = [tuple(F(v,10000) for v in row) for row in M]
PI = [F(v,200) for v in [4,114,13,44,25]]
TERMS = 32


def add(x,y): return x[0]+y[0],x[1]+y[1]
def sub(x,y): return x[0]-y[1],x[1]-y[0]
def scale(x,a): return (a*x[0],a*x[1]) if a>=0 else (a*x[1],a*x[0])
def summation(values):
    out = (F(0),F(0))
    for value in values: out = add(out,value)
    return out


def log_unit(x):
    assert 1<=x<=2
    z=(x-1)/(x+1)
    total=sum((z**(2*r+1)/F(2*r+1) for r in range(TERMS)),F(0))*2
    tail=2*z**(2*TERMS+1)/(F(2*TERMS+1)*(1-z*z))
    return total,total+tail


@lru_cache(maxsize=None)
def log_bounds(x):
    power=0
    while x<1: x*=2;power-=1
    while x>2: x/=2;power+=1
    return add(log_unit(x),scale(log_unit(F(2)),power))


@lru_cache(maxsize=None)
def divergence(a,b):
    if a==b:return F(0),F(0)
    terms=[]
    for p,x,y in zip(PI,a,b):
        terms.append(scale(add(scale(log_bounds(x/y),x),(y-x,y-x)),p))
    return summation(terms)


def mean(indices):
    total=sum((PI[i] for i in indices),F(0))
    return tuple(sum((PI[i]*B[i][j] for i in indices),F(0))/total for j in range(5))


def gain_intervals(centers):
    out=[]
    losses=[]
    for row in B:
        ds=[divergence(row,mu) for mu in centers]
        losses.append((min(d[0] for d in ds),min(d[1] for d in ds)))
    for candidate in B:
        values=[]
        for p,row,loss in zip(PI,B,losses):
            g=sub(loss,divergence(row,candidate))
            values.append(scale((max(g[0],0),max(g[1],0)),p))
        out.append(summation(values))
    return out


def winners(centers,expected):
    gaps=[]
    for row,which in zip(B,expected):
        ds=[divergence(row,mu) for mu in centers]
        for rival in range(len(centers)):
            if rival==which or centers[rival]==centers[which]:continue
            gap=sub(ds[rival],ds[which])
            assert gap[0]>0,(which,rival,gap)
            gaps.append(gap)
    return min(float(g[0]) for g in gaps)


def main():
    stage_specs=[
        (2,[mean(range(5))],3,[1,0,1,1,1]),
        (3,[B[1],mean([0,2,3,4])],0,[2,0,1,1,1]),
        (4,[B[1],mean([2,3,4]),B[0]],4,[2,0,1,1,3]),
        (5,[B[1],mean([2,3]),B[0],B[4]],3,[2,0,3,4,3]),
    ]
    certificates=[]
    for stage,old,chosen,expected in stage_specs:
        gains=gain_intervals(old)
        margin=min(gains[chosen][0]-gains[c][1] for c in range(5) if c!=chosen)
        assert margin>0
        gap=winners(old+[B[chosen]],expected)
        certificates.append(dict(stage=stage,chosen_block=chosen+1,
                                 gain_intervals=[[float(lo),float(hi)] for lo,hi in gains],
                                 gain_margin_lower=float(margin),
                                 first_estep_margin_lower=gap))
    # After the stage-4 M-step block 3 switches to the current pure block-5
    # component, leaving the old mixed center as block 4's sole current winner.
    stage4_end=[B[1],mean([2,3]),B[0],B[4]]
    after4_margin=winners(stage4_end,[2,0,3,1,3])
    # At stage 5 the old mixed component wins no row. Its soft M-step is
    # dominated by the unique row with smallest score deficit, namely block 4.
    initial5=stage4_end+[B[3]]
    deficits=[]
    for i,row in enumerate(B):
        ds=[divergence(row,mu) for mu in initial5]
        best=(min(d[0] for d in ds),min(d[1] for d in ds))
        deficits.append(sub(ds[1],best))
    dominant=3
    tail_margin=min(deficits[i][0]-deficits[dominant][1]
                    for i in range(5) if i!=dominant)
    assert tail_margin>0
    final=[B[1],B[3],B[0],mean([2,4]),B[3]]
    final_margin=winners(final,[2,0,3,4,3])
    result=dict(assertions='All comparisons passed using exact Fraction interval endpoints.',
                terms=TERMS, indexing='one-based in this certificate',
                stages=certificates,
                stage4_endpoint_assignment=[3,1,4,2,4],
                stage4_endpoint_margin_lower=after4_margin,
                stage5_empty_component=2,
                stage5_empty_component_score_deficit_intervals=[[float(a),float(b)] for a,b in deficits],
                dominating_block_in_its_soft_mstep=4,
                domination_margin_lower=float(tail_margin),
                final_assignment=[3,1,4,5,4],
                final_nonduplicate_margin_lower=final_margin,
                misclassified_mass='13/200',
                duplicate_components=[2,5], merged_blocks=[3,5],
                caveat='This certifies the high-signal population limiting path for one M-step per stage; it does not claim failure when every intermediate stage is iterated to convergence.')
    Path(__file__).with_name('counterexample_exact_certificate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

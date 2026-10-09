"""Independent numerical checks of the saved residual-seeding experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from residual_seed import Work, ROOT


def main():
    old_path=ROOT.parent/'spectral_partial_peel_20261009/results/per_graph.csv'
    if not old_path.exists():old_path=ROOT.parent/'20261009_partial_likelihood_peeling/results/per_graph.csv'
    old=pd.read_csv(old_path)
    old=old[old.method=='poisson_scaled_fixed_k']
    design=pd.read_csv(ROOT/'design_results/per_graph.csv')
    merged=design[design.method=='raw_peel'].merge(old[['graph','errors']],on='graph',suffixes=('_new','_old'))
    assert len(merged)==120 and (merged.errors_new==merged.errors_old).all()
    total_traces=0;total_points=0;minimum=0.;selected=0;accepts={};results={}
    for subset in ('design_results','validation_results'):
        f=pd.read_csv(ROOT/subset/'per_graph.csv')
        assert len(f)==1200 and f.graph.nunique()==120
        obj=f.pivot(index='graph',columns='method',values='objective')
        assert (obj.raw_peel_repaired>=obj.raw_peel-1e-7).all()
        accepted=0
        for path in sorted((ROOT/subset/'runs').glob('*.json')):
            p=json.loads(path.read_text(encoding='utf-8'))
            def visit(value):
                nonlocal total_traces,total_points,minimum
                if isinstance(value,dict):
                    if 'iteration' in value and 'objective_delta' in value:
                        minimum=min(minimum,value['objective_delta']);total_points+=1
                        assert value['objective_delta']>=-1e-7
                    for key,item in value.items():
                        if key in ('trace','fit_trace'):total_traces+=1
                        visit(item)
                elif isinstance(value,list):
                    for item in value:visit(item)
            visit(p['records'])
            chosen=p['records']['residual_greedy_pp_best10']['events']['selected_restart']
            restart=p['restart_rows']
            assert chosen==max(range(len(restart)),key=lambda i:restart[i]['objective'])
            selected+=1
            accepted+=int(p['records']['raw_peel_repaired']['events']['accepted'] is not None)
        accepts[subset]=accepted
        results[subset]=json.loads((ROOT/subset/'paired.json').read_text())
    # Explicit scalar sum checks on a small positive, non-degenerate profile.
    rng=np.random.default_rng(34791)
    u,_=np.linalg.qr(rng.normal(size=(11,3)))
    w=Work(u,np.array([5.,2.,-1.]))
    mu=w.q[[1,4]]
    direct=np.array([[sum(float(a*np.log(a/b)-a+b) for a,b in zip(row,center))
        for center in mu] for row in w.q])
    assert np.allclose(w.distance(mu),direct,atol=1e-11,rtol=1e-11)
    current=w.score(mu).max(1)
    oracle=np.array([sum(max(float(sum(a*np.log(b) for a,b in zip(row,c))-c.sum())-current[i],0.)
        for i,row in enumerate(w.q)) for c in w.q])
    _,event=w.global_candidate(mu)
    assert np.allclose(event['candidate_gains'],oracle,atol=1e-10,rtol=1e-10)
    report=dict(passed=True,baseline_graph_errors_match=120,graphs_checked=240,
        saved_fit_traces=total_traces,saved_trace_points=total_points,
        minimum_objective_increment=minimum,best10_selected_by_likelihood=selected,
        repair_never_decreases_final_likelihood=True,repair_accepted=accepts,
        scalar_deviance_max_abs_error=float(np.max(np.abs(w.distance(mu)-direct))),
        scalar_global_gain_max_abs_error=float(np.max(np.abs(event['candidate_gains']-oracle))),
        paired=results)
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='paired'},indent=2))


if __name__=='__main__':main()

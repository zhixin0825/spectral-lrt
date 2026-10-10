"""Fresh arbitrary-K and density checks for deterministic certified fitting.

This suite evaluates fitted labels only after the spectral-only method returns.
The historical 240-graph benchmark is not rerun or relabeled by these checks.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from residual_seed import Work,fit_spectral_profiles,_floored_weights


def _error(labels,truth,k):
    table=np.zeros((k,k),dtype=int)
    np.add.at(table,(labels,truth),1)
    rows,cols=linear_sum_assignment(-table)
    return int(len(truth)-table[rows,cols].sum())


def _check_fit(work,fitted,audit):
    assert np.all(np.isfinite(fitted['mu'])) and np.all(fitted['mu'] >= work.profile_floor*(1-1e-12))
    assert np.all(fitted['weights'] >= 1e-8-1e-14)
    assert abs(fitted['weights'].sum()-1) < 1e-12
    assert fitted['objective'] >= audit['selected_objective']-1e-7
    assert fitted['n_iter']==1,"Mandatory final E/M update must run exactly once"
    traces=[s['fit_trace'] for s in audit['growing_stages']]
    for round_event in audit['rounds']:
        assert len(round_event['proposals'])==(work.k+1 if work.k > 1 else 1)
        assert round_event['proposals'][0]['method']=='uniform_refit'
        for proposal in round_event['proposals']:
            assert np.allclose(proposal['initial_weights'],np.full(work.k,1/work.k),atol=0,rtol=0)
            traces.append(proposal['fit_trace'])
        if round_event['accepted']:
            assert round_event['best_improvement'] >= audit['threshold']
        else:
            assert max(p['improvement'] for p in round_event['proposals']) < audit['threshold']
    traces.append(fitted['trace'])
    worst=min(t['objective_delta'] for trace in traces for t in trace)
    assert worst >= -1e-7
    assert audit['stop_certificate'] and audit['stop_reason']=='no_threshold_improvement'
    return worst


def _spectrum(matrix,k):
    values,vectors=np.linalg.eigh(matrix)
    selected=np.argsort(np.abs(values))[-k:][::-1]
    return vectors[:,selected],values[selected]


def run_graph(n,k,density,seed):
    rng=np.random.default_rng(seed)
    truth=np.arange(n)%k
    rng.shuffle(truth)
    # Unequal expected degrees, positive full-rank shape, balanced classes.
    shape=np.full((k,k),.12)
    shape[np.diag_indices(k)]=np.linspace(.8,1.1,k)
    shape+=.008*(np.arange(k)[:,None]+np.arange(k)[None,:])
    target=4*math.log(n) if density=='logarithmic' else 2*math.sqrt(n)
    proportions=np.bincount(truth,minlength=k)/n
    probabilities=shape*(target/(n*float(proportions@shape@proportions)))
    assert probabilities.max() < 1
    graph_prob=probabilities[truth[:,None],truth[None,:]]
    np.fill_diagonal(graph_prob,0.)
    graph=np.triu(rng.random((n,n)) < graph_prob,1)
    graph=graph+graph.T
    u,lam=_spectrum(graph.astype(float),k)
    work=Work(u,lam)
    fitted,_,audit=work.iterative_certified_growing()
    worst=_check_fit(work,fitted,audit)
    # All truth-based evaluation begins after fitting and selection finish.
    errors=_error(fitted['labels'],truth,k)
    return dict(n=n,k=k,density=density,seed=seed,
        expected_degree_min=float(graph_prob.sum(1).min()),
        expected_degree_max=float(graph_prob.sum(1).max()),
        probability_max=float(probabilities.max()),profile_floor=work.profile_floor,
        errors=errors,error_rate=errors/n,objective=float(fitted['objective']),
        accepted_rounds=audit['accepted_rounds'],rounds_evaluated=len(audit['rounds']),
        stop_certificate=audit['stop_certificate'],worst_objective_delta=worst)


def forced_collapsed_endpoint():
    """Exercise multiple accepted rounds without changing the production path."""
    n,k=240,3
    truth=np.arange(n)%k
    probabilities=np.full((k,k),1e-4)
    np.fill_diagonal(probabilities,.75)
    mean=probabilities[truth[:,None],truth[None,:]]
    u,lam=_spectrum(mean,k)
    work=Work(u,lam)
    collapsed=work.fit(np.repeat(work.q.mean(0)[None,:],k,axis=0))
    # Inject a deliberately collapsed endpoint solely to check the repair loop.
    work.growing=lambda:(collapsed,np.array([],dtype=int),[])
    fitted,_,audit=work.iterative_certified_growing()
    worst=_check_fit(work,fitted,audit)
    assert audit['accepted_rounds'] >= 2
    return dict(kind='injected_collapsed_endpoint',n=n,k=k,
        accepted_rounds=audit['accepted_rounds'],
        accepted_improvements=[r['best_improvement'] for r in audit['rounds'] if r['accepted']],
        threshold=audit['threshold'],stop_certificate=audit['stop_certificate'],
        final_errors=_error(fitted['labels'],truth,k),worst_objective_delta=worst)


def edge_checks():
    counts=np.array([0.,1.,10.,1e-12])
    weights=_floored_weights(counts)
    assert np.isclose(weights.sum(),1.) and weights.min() >= 1e-8
    u=np.eye(4)[:,:2]
    work=Work(u,np.zeros(2))
    assert work.profile_floor > 0 and np.all(work.q > 0)
    fitted,_,audit=work.iterative_certified_growing()
    _check_fit(work,fitted,audit)
    historical=Work(u,np.zeros(2),floor_mode='historical')
    assert historical.profile_floor==1e-4*math.log(4)/4
    public=fit_spectral_profiles(u,np.zeros(2),method='global_gain_certified')
    assert public['events']['method']=='iterative_global_gain_certified'
    return dict(zero_spectrum_positive_floor=True,historical_floor_preserved=True,
                uniform_zero_mass_fallback=True,public_api_deterministic=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('general_k_fresh_checks.json'))
    args=parser.parse_args()
    rows=[]
    with threadpool_limits(limits=1):
        edges=edge_checks()
        injected=forced_collapsed_endpoint()
        for n in (256,512):
            for k in (3,4,6):
                for density in ('logarithmic','square_root'):
                    for seed in (202610101,202610102):
                        result=run_graph(n,k,density,seed)
                        rows.append(result)
                        print(json.dumps(result),flush=True)
    payload=dict(protocol=dict(fitting_inputs=['U','Lambda'],
        method='global_gain_certified',floor='1e-4*max(max(abs(Lambda)),1)/n',
        threshold='sum(q)/sqrt(log(n))',
        purpose='Fresh finite-sample implementation and accuracy checks; not an asymptotic theorem test',
        historical_benchmark_unchanged=True),
        edge_checks=edges,injected_endpoint_check=injected,graphs=rows,
        aggregate=dict(graphs=len(rows),exact_graphs=sum(r['errors']==0 for r in rows),
            total_errors=sum(r['errors'] for r in rows),total_nodes=sum(r['n'] for r in rows),
            max_error_rate=max(r['error_rate'] for r in rows),
            all_stop_certificates=all(r['stop_certificate'] for r in rows),
            worst_objective_delta=min(r['worst_objective_delta'] for r in rows)))
    args.out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload['aggregate']),flush=True)


if __name__=='__main__':main()

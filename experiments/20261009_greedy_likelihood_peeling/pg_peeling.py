"""Deterministic top-fraction raw-likelihood peeling, then all-node PG EM.

Each remaining node supplies its bounded singleton-MLE rate vector. Its own
density is excluded from the score; the selected other nodes maximize that
candidate's component log density. The largest raw total wins. No regret
normalization, group-average substitution, or truth-based selection is used.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures
import itertools
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

_OLD_CANDIDATES = [Path(__file__).resolve().parent.parent / 'spectral_em_20261009',
                   Path(__file__).resolve().parent.parent / '20261009_spectral_em']
OLD = next((directory for directory in _OLD_CANDIDATES
            if (directory/'likelihood_pp.py').is_file()), _OLD_CANDIDATES[0])
sys.path.insert(0, str(OLD))
from likelihood_pp import LikelihoodSeeder, fit_pg_rates
from benchmark import recovery


def peel_logpdf(logpdf, k):
    """Peel using rows=observations, columns=candidate singleton parameters.

    Ties choose the smallest global node ID. Each candidate selects
    floor(number remaining / number groups remaining) OTHER nodes, capped
    to reserve one node for each later group. The candidate is removed too.
    At the final step all other remaining nodes are scored and removed.
    """
    logpdf = np.asarray(logpdf, dtype=float)
    n = len(logpdf)
    if logpdf.shape != (n, n) or not 1 <= k <= n or not np.isfinite(logpdf).all():
        raise ValueError('invalid finite square logpdf matrix or k')
    remaining = np.arange(n)
    labels = np.full(n, -1, dtype=int)
    seeds, steps = [], []
    for group in range(k):
        remaining_k = k - group
        other_count = (min(len(remaining)//remaining_k, len(remaining)-remaining_k)
                       if remaining_k > 1 else len(remaining)-1)
        sub = logpdf[np.ix_(remaining, remaining)].copy()
        np.fill_diagonal(sub, -np.inf)
        if other_count == 0:
            scores = np.zeros(len(remaining))
        else:
            # Sort-free exact sum of the largest off-diagonal values.
            top = np.partition(sub, len(remaining) - other_count, axis=0)[-other_count:]
            scores = top.sum(axis=0)
        chosen_local = int(np.argmax(scores))
        chosen = int(remaining[chosen_local])
        other = remaining[remaining != chosen]
        ranked = np.lexsort((other, -logpdf[other, chosen]))
        selected_other = other[ranked[:other_count]]
        members = np.r_[chosen, selected_other].astype(int)
        direct_score = float(logpdf[selected_other, chosen].sum())
        if not np.isclose(direct_score, scores[chosen_local], atol=1e-9, rtol=1e-12):
            raise AssertionError('top-sum selection mismatch')
        labels[members] = group
        seeds.append(chosen)
        steps.append({'remaining_ids': remaining.copy(),
                      'candidate_scores': scores.copy(), 'selected_seed': chosen,
                      'members': members.copy(), 'selected_score': direct_score,
                      'group_size': other_count+1, 'scored_other_count': other_count,
                      'remaining_groups': remaining_k})
        remaining = remaining[~np.isin(remaining, members)]
    if len(remaining) or np.any(labels < 0) or len(set(seeds)) != k:
        raise AssertionError('invalid peeling partition')
    return np.asarray(seeds, dtype=int), labels, steps


def verify_top_rule():
    """Exhaustive independent subset oracle on small signed log densities."""
    rng = np.random.default_rng(832776)
    cases = 0
    for n in range(2, 10):
        for k in range(1, n+1):
            lp = -rng.exponential(4., size=(n, n))
            # Include exact ties to test deterministic ranking.
            if n % 3 == 0: lp[:2, :2] = -2.
            ids, labels, steps = peel_logpdf(lp, k)
            remaining = np.arange(n)
            for g, step in enumerate(steps):
                remaining_k = k-g
                other_count = (min(len(remaining)//remaining_k, len(remaining)-remaining_k)
                               if remaining_k > 1 else len(remaining)-1)
                brute = []
                for c in remaining:
                    others = [i for i in remaining if i != c]
                    scores = [sum(lp[i, c] for i in subset)
                              for subset in itertools.combinations(others, other_count)]
                    brute.append(max(scores))
                np.testing.assert_allclose(step['candidate_scores'], brute,
                                           atol=1e-12, rtol=1e-12)
                assert step['selected_seed'] == remaining[int(np.argmax(brute))]
                assert step['selected_seed'] in step['members']
                assert len(step['members']) == other_count+1
                remaining = remaining[~np.isin(remaining, step['members'])]
                cases += 1
    return {'oracle_steps_checked': cases, 'oracle': 'enumerate every other-node subset, exclude candidate self-density',
            'passed': True}


def pairwise_logpdf(seeder, chunk):
    n = len(seeder.model.y)
    lp = np.empty((n, n), dtype=float)
    for start in range(0, n, chunk):
        values, _ = seeder.model.evaluate(seeder.singletons[start:start+chunk])
        lp[:, start:start+chunk] = values
    return lp


def run_one(task):
    path, output, max_iter, chunk = task
    path, out = Path(path), Path(output)
    model, n, ch, seed = re.fullmatch(r'(.+)_n(\d+)_ch([\d.]+)_s(\d+)', path.stem).groups()
    common = {'graph': path.stem, 'model': model, 'n': int(n), 'ch': float(ch), 'seed': int(seed)}
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as data:
        u, lam, h = data['u'], data['lam'], data['h0']
        y = np.linalg.solve(h.T, (np.sqrt(len(u))*u*lam).T).T
        k = len(lam)
        with threadpool_limits(limits=1):
            seeder = LikelihoodSeeder(y)
            pair_start = time.perf_counter()
            logpdf = pairwise_logpdf(seeder, chunk)
            pair_elapsed = time.perf_counter()-pair_start
            peel_start = time.perf_counter()
            ids, peeled, steps = peel_logpdf(logpdf, k)
            peel_elapsed = time.perf_counter()-peel_start
            rates = seeder.singletons[ids].copy()
            init_lp, _ = seeder.model.evaluate(rates)
            initial_hard = init_lp.argmax(axis=1)
            fit = fit_pg_rates(y, rates, max_iter=max_iter)
        final = fit.pop('labels')
        fit.pop('r')
        trace = fit.pop('trace')
        final_rates, final_weights = fit.pop('rates'), fit.pop('weights')
        # Access truth only after initialization and all-node EM are finished.
        truth = data['truth_evaluation_only']
        row = common | fit | recovery(final, truth, k) | {
            'method': 'raw_total_likelihood_peeling_PG',
            'family': 'raw_total_likelihood_peeling_PG',
            'pairwise_elapsed': pair_elapsed, 'peeling_elapsed': peel_elapsed,
            'precompute_elapsed': seeder.precompute_elapsed,
            'seed_elapsed': seeder.precompute_elapsed+pair_elapsed+peel_elapsed,
            'total_elapsed': time.perf_counter()-started,
            'peeled_error_rate': recovery(peeled, truth, k)['error_rate'],
            'initial_hard_error_rate': recovery(initial_hard, truth, k)['error_rate'],
            'initial_objective': trace[0]['objective'],
            'rates': final_rates.tolist(), 'weights': final_weights.tolist()}
        arrays = {'u': u, 'lam': lam, 'h0': h, 'y': y,
                  'singleton_rates': seeder.singletons,
                  'seed_ids': ids, 'initial_rates': rates,
                  'peeled_labels': peeled, 'initial_hard_labels': initial_hard,
                  'final_labels': final, 'final_rates': final_rates,
                  'final_weights': final_weights, 'truth_evaluation_only': truth}
        short_steps = []
        for g, step in enumerate(steps):
            for key in ('remaining_ids', 'candidate_scores', 'members'):
                arrays[f'step_{g}_{key}'] = step[key]
            short_steps.append({key: value for key, value in step.items()
                                if key not in ('remaining_ids', 'candidate_scores', 'members')})
        np.savez_compressed(out/'inputs'/f'{path.stem}.npz', **arrays)
        (out/'runs'/f'{path.stem}.json').write_text(json.dumps({
            'graph': path.stem, 'results': [row], 'trace': trace,
            'peeling': short_steps, 'seed_ids': ids.tolist(),
            'saturation_report': seeder.saturation_report,
            'features': 'fixed inherited Y from original k-means decoder H0',
            'selection': 'largest raw total component logpdf on top OTHER nodes; exclude candidate self-density'}, indent=2))
    return row


def analyze(out, skip_baselines=False):
    ours = pd.read_csv(out/'per_run.csv')
    graph_set = set(ours.graph)
    if skip_baselines:
        allrows = ours.copy()
    else:
        prev_pp = pd.read_csv(OLD/'likelihood_pp_20000'/'per_run.csv')
        prev_random = pd.read_csv(OLD/'random_initialization_20000'/'per_run.csv')
        prev_pp = prev_pp[prev_pp.graph.isin(graph_set)].copy()
        prev_random = prev_random[prev_random.graph.isin(graph_set) &
            (prev_random.method.str.match(r'random_PG_\d+$') |
             prev_random.method.isin(['random_PG_best', 'km_start_PG', 'km_X']))].copy()
        prev_random['family'] = np.where(prev_random.method.str.match(r'random_PG_\d+$'),
                                        'random_labels', prev_random.method)
        allrows = pd.concat([ours, prev_pp, prev_random], ignore_index=True)
    allrows.to_csv(out/'comparison_per_run.csv', index=False)
    summary = allrows.groupby('family').agg(fits=('graph', 'size'), graphs=('graph', 'nunique'),
        mean_error=('error_rate', 'mean'), exact=('exact', 'sum'),
        converged=('converged', 'sum'), mean_iter=('n_iter', 'mean'),
        max_iter=('n_iter', 'max'), mean_em_elapsed=('elapsed', 'mean'),
        mean_seed_elapsed=('seed_elapsed', 'mean'))
    summary.to_csv(out/'overall_summary.csv')
    allrows.groupby(['model', 'n', 'ch', 'family']).agg(
        fits=('graph', 'size'), mean_error=('error_rate', 'mean'), exact=('exact', 'sum'),
        mean_iter=('n_iter', 'mean')).to_csv(out/'condition_summary.csv')
    pairs = []
    for baseline in ('km_X', 'km_start_PG', 'random_PG_best', 'euclidean_pp_best',
                     'greedy_euclidean_pp_best', 'likelihood_pp_best', 'greedy_likelihood_pp_best'):
        if skip_baselines: break
        b = allrows[allrows.method == baseline]
        merged = ours.merge(b[['graph', 'errors', 'objective']], on='graph', suffixes=('_peel', '_base'))
        diff = merged.errors_peel - merged.errors_base
        od = merged.objective_peel - merged.objective_base
        pairs.append({'baseline': baseline, 'paired_graphs': len(merged),
            'better_errors': int((diff<0).sum()), 'equal_errors': int((diff==0).sum()),
            'worse_errors': int((diff>0).sum()),
            'higher_objective_1e_3': int((od>1e-3).sum()),
            'lower_objective_1e_3': int((od< -1e-3).sum())})
    result = {'graphs': len(ours), 'mean_error': float(ours.error_rate.mean()),
        'exact': int(ours.exact.sum()), 'converged': int(ours.converged.sum()),
        'mean_iter': float(ours.n_iter.mean()), 'max_iter': int(ours.n_iter.max()),
        'mean_seed_elapsed': float(ours.seed_elapsed.mean()),
        'mean_pairwise_elapsed': float(ours.pairwise_elapsed.mean()),
        'mean_peeled_error': float(ours.peeled_error_rate.mean()),
        'mean_initial_hard_error': float(ours.initial_hard_error_rate.mean()),
        'no_material_EM_decrease': bool(ours.worst_objective_delta.min() >= -1e-7),
        'comparisons': pairs}
    (out/'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(OLD/'independent_20000'/'inputs'))
    p.add_argument('--out', default=str(Path(__file__).resolve().parent/'pg_results'))
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--skip-baselines', action='store_true',
                   help='summarize this method without reading archived baseline CSVs')
    args = p.parse_args()
    files = sorted(Path(args.input).glob('*.npz'))
    if args.limit: files = files[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ('inputs', 'runs'): (out/sub).mkdir(exist_ok=True)
    validation = verify_top_rule()
    (out/'validation.json').write_text(json.dumps(validation, indent=2))
    protocol = vars(args) | {'graphs': len(files), 'tau': 1., 'tol': 1e-6,
        'scored_other_count': 'min(floor(M/r), M-r) if r>1; M-1 if r=1',
        'group_size': 'scored_other_count + 1, candidate also removed',
        'include_candidate_self_in_score': False, 'score': 'raw total component logpdf on selected others',
        'candidate_rate': 'bounded singleton MLE; no group-average replacement',
        'tie_break': 'smallest global node ID', 'mixture_weights_init': 'uniform',
        'EM': 'all nodes freely reassigned; frozen inherited Y',
        'truth_access': 'after fitting only', 'created_before_fitting': True}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows, todo, failures = [], [], []
    for f in files:
        saved = out/'runs'/f'{f.stem}.json'
        if args.resume and saved.exists():
            rows.extend(json.loads(saved.read_text())['results'])
        else: todo.append(f)
    print(json.dumps({'pending': len(todo), 'validation': validation}), flush=True)
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, (str(f), str(out), args.max_iter, args.chunk)): f.stem for f in todo}
        pending = set(futures)
        last_report = 0.
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=30.,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try: rows.append(future.result())
                except Exception as exc: failures.append({'graph': futures[future], 'error': repr(exc)})
            now = time.perf_counter()-started
            if done or now-last_report >= 30.:
                print(json.dumps({'completed': len(rows), 'total': len(files),
                    'failed': len(failures), 'elapsed': round(now, 1)}), flush=True)
                last_report = now
                pd.DataFrame(rows).to_csv(out/'per_run.csv', index=False)
                (out/'failures.json').write_text(json.dumps(failures, indent=2))
    if failures: raise RuntimeError(failures)
    analyze(out, skip_baselines=args.skip_baselines)


if __name__ == '__main__': main()

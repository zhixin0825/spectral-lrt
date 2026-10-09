"""Fixed-feature full-Gaussian EM control; fitting never sees adjacency or truth.

X = sqrt(n) U Lambda is frozen. Coordinates are centered and divided by their
global standard deviations once. The constraint Sigma_a >= floor I is fixed,
and its exact Gaussian covariance M step clips sample-covariance eigenvalues.
The common observed mixture log likelihood is recorded after every M step.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits


def gaussian_mstep(x, r, covariance_floor):
    n, d = x.shape
    nk = r.sum(axis=0)
    if np.any(nk <= np.finfo(float).tiny):
        raise FloatingPointError('A component has zero posterior weight')
    weights = nk / n
    means = r.T @ x / nk[:, None]
    covariances = np.empty((r.shape[1], d, d))
    for a in range(r.shape[1]):
        centered = x - means[a]
        sample_covariance = (centered.T * r[:, a]) @ centered / nk[a]
        eigenvalues, eigenvectors = np.linalg.eigh(sample_covariance)
        covariances[a] = (eigenvectors * np.maximum(eigenvalues, covariance_floor)) @ eigenvectors.T
    return weights, means, covariances


def gaussian_estep(x, weights, means, covariances):
    n, d = x.shape
    scores = np.empty((n, len(weights)))
    normalizing_constant = d * np.log(2 * np.pi)
    for a in range(len(weights)):
        eigenvalues, eigenvectors = np.linalg.eigh(covariances[a])
        centered = (x - means[a]) @ eigenvectors
        scores[:, a] = np.log(weights[a]) - 0.5 * (
            normalizing_constant + np.log(eigenvalues).sum()
            + (centered ** 2 / eigenvalues).sum(axis=1))
    point_likelihoods = logsumexp(scores, axis=1)
    r = np.exp(scores - point_likelihoods[:, None])
    return float(point_likelihoods.sum()), r


def fit_direct_gaussian(u, lam, initial_labels, max_iter=150, tol=1e-6,
                        covariance_floor=1e-4):
    """Use spectral data + initial labels only. Fixed objective, exact EM steps."""
    started = time.perf_counter()
    n, k = u.shape
    x_original = np.sqrt(n) * u * lam
    center = x_original.mean(axis=0)
    scale = np.maximum(x_original.std(axis=0), np.finfo(float).eps)
    x = (x_original - center) / scale
    r0 = np.eye(k)[initial_labels]
    weights, means, covariances = gaussian_mstep(x, r0, covariance_floor)
    objective, r = gaussian_estep(x, weights, means, covariances)
    labels = r.argmax(axis=1)
    trace = [{'iteration': 0, 'objective': objective, 'gain': None,
              'parameter_change': None, 'label_changes': int((labels != initial_labels).sum())}]
    converged = False
    minimum_gain = float('inf')
    for iteration in range(1, max_iter + 1):
        weights_new, means_new, covariances_new = gaussian_mstep(x, r, covariance_floor)
        objective_new, r_new = gaussian_estep(x, weights_new, means_new, covariances_new)
        labels_new = r_new.argmax(axis=1)
        gain = objective_new - objective
        minimum_gain = min(minimum_gain, gain)
        parameter_change = max(float(np.max(np.abs(weights_new - weights))),
                               float(np.max(np.abs(means_new - means))),
                               float(np.max(np.abs(covariances_new - covariances))))
        label_changes = int((labels_new != labels).sum())
        trace.append({'iteration': iteration, 'objective': objective_new,
                      'gain': gain, 'parameter_change': parameter_change,
                      'label_changes': label_changes})
        if gain < -1e-8 * (1 + abs(objective)):
            raise FloatingPointError(f'EM objective decreased by {gain}')
        converged = (abs(gain) <= tol * (1 + abs(objective))
                     and parameter_change <= np.sqrt(tol) and label_changes == 0)
        weights, means, covariances = weights_new, means_new, covariances_new
        objective, r, labels = objective_new, r_new, labels_new
        if converged:
            break
    stats = dict(method='gmm_direct_X', objective=objective,
                 initial_objective=trace[0]['objective'],
                 objective_gain=objective-trace[0]['objective'],
                 iterations=iteration, converged=bool(converged),
                 minimum_gain=float(minimum_gain),
                 min_covariance_eigenvalue=float(np.linalg.eigvalsh(covariances).min()),
                 covariance_floor=covariance_floor, elapsed=time.perf_counter()-started,
                 weights=weights.tolist(), means=means.tolist(),
                 covariances=covariances.tolist(),
                 standardization_center=center.tolist(), standardization_scale=scale.tolist())
    return labels, stats, trace


def recovery(labels, truth, k):
    table = np.zeros((k, k), int)
    np.add.at(table, (truth, labels), 1)
    rows, cols = linear_sum_assignment(-table)
    errors = int(len(truth)-table[rows, cols].sum())
    return dict(errors=errors, error_rate=errors/len(truth), exact=int(errors == 0),
                ari=float(adjusted_rand_score(truth, labels)))


def run_one(task):
    path, max_iter, tol, covariance_floor = task
    path = Path(path)
    match = re.fullmatch(r'(.+)_n(\d+)_ch([\d.]+)_s(\d+)', path.stem)
    if match is None:
        raise ValueError(f'Unrecognized input filename: {path.name}')
    model, n, ch, seed = match.groups()
    with np.load(path, allow_pickle=False) as data:
        # Explicitly avoid passing evaluator-only data to fitting.
        u, lam, labels0 = data['u'], data['lam'], data['labels_km_X']
        with threadpool_limits(limits=1):
            labels, stats, trace = fit_direct_gaussian(
                u, lam, labels0, max_iter, tol, covariance_floor)
        truth = data['truth_evaluation_only']
    record = dict(graph=path.stem, model=model, n=int(n), ch=float(ch), seed=int(seed),
                  **recovery(labels, truth, u.shape[1]), **stats)
    labels_dir = path.parent.parent/'direct_gaussian_labels'
    labels_dir.mkdir(exist_ok=True)
    np.savez_compressed(labels_dir/f'{path.stem}.npz', labels_gmm_direct_X=labels)
    return dict(record=record, trace=trace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-iter', type=int, default=150)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--covariance-floor', type=float, default=1e-4)
    args = parser.parse_args()
    out = Path(args.out)
    inputs = sorted((out/'inputs').glob('*.npz'))
    protocol = dict(feature='X=sqrt(n) U Lambda, fixed throughout EM',
                    start='labels_km_X from the same saved compressed input',
                    standardization='fixed global centering and coordinate standard deviation',
                    model='full Gaussian mixture, a row-wise working likelihood',
                    constraints='each standardized covariance eigenvalue >= covariance_floor',
                    m_step='exact weights, means and eigenvalue-clipped covariances',
                    stopping='relative objective gain <= tol, max parameter change <= sqrt(tol), labels unchanged',
                    truth_access='evaluation after fitting; no adjacency read',
                    number_inputs=len(inputs), **vars(args))
    (out/'direct_gaussian_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    tasks = [(str(path), args.max_iter, args.tol, args.covariance_floor) for path in inputs]
    rows = []
    started = time.perf_counter()
    with (out/'direct_gaussian.jsonl').open('w', encoding='utf-8') as stream:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, task): task[0] for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                payload = future.result()
                rows.append(payload['record'])
                stream.write(json.dumps(payload)+'\n')
                stream.flush()
                if len(rows) % 20 == 0 or len(rows) == len(tasks):
                    print(json.dumps({'complete':len(rows), 'total':len(tasks),
                                      'elapsed':time.perf_counter()-started}), flush=True)
    frame = pd.DataFrame(rows).sort_values(['model','n','ch','seed'])
    frame.to_csv(out/'direct_gaussian.csv', index=False)
    base = pd.read_csv(out/'per_run.csv')
    base = base[base.method == 'km_X'][['graph','errors','error_rate','exact']]
    paired = frame.merge(base, on='graph', suffixes=('', '_km_X'))
    paired['error_rate_delta'] = paired.error_rate - paired.error_rate_km_X
    paired['comparison'] = np.where(paired.errors < paired.errors_km_X, 'improved',
                                   np.where(paired.errors > paired.errors_km_X, 'worsened', 'tied'))
    paired[['graph','model','n','ch','seed','errors','errors_km_X','error_rate',
            'error_rate_km_X','error_rate_delta','exact','exact_km_X','comparison',
            'iterations','converged','minimum_gain']].to_csv(out/'direct_gaussian_paired.csv', index=False)
    summary = paired.groupby(['model','n','ch']).agg(
        runs=('graph','size'), mean_error=('error_rate','mean'), mean_km_X_error=('error_rate_km_X','mean'),
        mean_delta=('error_rate_delta','mean'), exact=('exact','sum'), exact_km_X=('exact_km_X','sum'),
        converged=('converged','sum'), mean_iterations=('iterations','mean'),
        improved=('comparison', lambda s: int((s == 'improved').sum())),
        worsened=('comparison', lambda s: int((s == 'worsened').sum())),
        tied=('comparison', lambda s: int((s == 'tied').sum())))
    summary.to_csv(out/'direct_gaussian_summary.csv')
    print(summary.to_string(), flush=True)
    print(json.dumps({'runs':len(frame), 'converged':int(frame.converged.sum()),
                      'minimum_gain':float(frame.minimum_gain.min()),
                      'mean_error':float(frame.error_rate.mean()),
                      'mean_km_X_error':float(paired.error_rate_km_X.mean()),
                      'comparison':paired.comparison.value_counts().to_dict()}), flush=True)


if __name__ == '__main__':
    main()

"""Finite-graph audit of the retained-eigenpair likelihood decoder.

The design has 60 independent SBM graphs: K=2,...,6; two densities;
three fixed matrix families; two graph seeds; n=320. All six methods
receive the same retained K adjacency eigenpairs. Truth is used only
after every fit, replacement, and likelihood selection is complete.

Files preserve A, P, labels, eigenpairs, output labels/profiles/weights,
every candidate-scan loss, and every EM/replacement loss trace.
This benchmark is a numerical audit, not evidence for an asymptotic rate.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from spectral_likelihood import SpectralLikelihood

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'benchmark_results'
FAMILIES = ('assortative', 'disassortative_unequal', 'weak_general_unequal')


class AuditedLikelihood(SpectralLikelihood):
    """Record the same candidate scan as the engine, with no scoring changes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scans = []
        self.scan_totals = []
        self.scan_current_losses = []

    def global_candidate(self, retained_profiles, *, excluded=()):
        current = self.divergences(retained_profiles).min(axis=1)
        totals = np.minimum(current[:, None], self.candidate_divergences()).sum(axis=0)
        if len(excluded):
            totals[np.asarray(excluded, dtype=int)] = np.inf
        candidate = int(np.argmin(totals))
        self.scans.append(dict(scan=len(self.scans), retained_components=len(retained_profiles),
                               excluded_ids=list(map(int, excluded)), candidate=candidate,
                               chosen_hard_loss=float(totals[candidate])))
        self.scan_totals.append(totals.copy())
        self.scan_current_losses.append(current.copy())
        return candidate, float(totals[candidate])


def model(k, family):
    if family == 'assortative':
        B = np.ones((k, k)) + 5 * np.eye(k)
        pi = np.ones(k) / k
    elif family == 'disassortative_unequal':
        scale = np.linspace(.9, 1.15, k)
        B = (1.3 * np.ones((k, k)) - 1.1 * np.eye(k)) * np.outer(scale, scale)
        pi = np.linspace(1., 2.2, k)
        pi /= pi.sum()
    elif family == 'weak_general_unequal':
        rng = np.random.default_rng(2026101000 + k)
        R = rng.normal(0, .20, (k, k))
        B = np.exp((R + R.T) / 2.) + .08 * np.eye(k)
        pi = rng.permutation(np.linspace(.7, 2.4, k))
        pi /= pi.sum()
    else:
        raise ValueError(family)
    assert np.all(B > 0) and np.linalg.matrix_rank(B) == k
    return B, pi


def design():
    tasks = []
    for k in range(2, 7):
        for density in ('log_degree', 'dense'):
            for family_id, family in enumerate(FAMILIES):
                for repeat in range(2):
                    seed = 202610100 + 1000 * k + 100 * (density == 'dense') + 10 * family_id + repeat
                    tasks.append(dict(n=320, k=k, density=density, family=family, repeat=repeat,
                                      seed=seed,
                                      graph_id=f'k{k}_{density}_{family}_s{seed}'))
    return tasks


def generate(task):
    n, k = task['n'], task['k']
    B, pi = model(k, task['family'])
    sizes = np.floor(n * pi).astype(int)
    residual = n - sizes.sum()
    sizes[np.argsort(-(n * pi - sizes))[:residual]] += 1
    actual_pi = sizes / n
    target_degree = 3.0 * math.log(n) if task['density'] == 'log_degree' else .18 * (n-1)
    alpha = target_degree / (n * (actual_pi @ B @ actual_pi) - actual_pi @ np.diag(B))
    P = alpha * B
    if not np.all((P > 0) & (P < .95)):
        raise ValueError(f'Invalid probability scale: max P={P.max()}')
    rng = np.random.default_rng(task['seed'])
    truth = np.repeat(np.arange(k), sizes)
    rng.shuffle(truth)
    probs = P[truth[:, None], truth[None, :]]
    upper = np.triu(rng.random((n, n)) < probs, 1)
    A = (upper | upper.T).astype(np.uint8)
    eigenvalues, eigenvectors = np.linalg.eigh(A.astype(float))
    retained = np.argsort(-np.abs(eigenvalues))[:k]
    lam = eigenvalues[retained]
    u = eigenvectors[:, retained]
    expected_degrees = P @ sizes - np.diag(P)
    return A, B, P, truth, u, lam, dict(block_sizes=sizes.tolist(),
                                       block_proportions=actual_pi.tolist(),
                                       target_mean_degree=target_degree,
                                       expected_degree_by_block=expected_degrees.tolist(),
                                       empirical_mean_degree=float(A.sum()/n),
                                       P_eigenvalues=np.linalg.eigvalsh(P).tolist())


def metric(predicted, truth, k):
    table = np.zeros((k, k), dtype=int)
    np.add.at(table, (truth, predicted), 1)
    r, c = linear_sum_assignment(-table)
    return dict(misclassification=1-float(table[r,c].sum()/len(truth)),
                ari=float(adjusted_rand_score(truth, predicted)),
                n_output_classes=int(np.unique(predicted).size))


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def run_one(task):
    folder = RESULTS / task['graph_id']
    folder.mkdir(parents=True, exist_ok=True)
    result_file = folder / 'metrics.json'
    if result_file.exists():
        return json.loads(result_file.read_text())
    started = time.perf_counter()
    arrays, all_audits, outputs, scan_arrays = {}, {}, {}, {}
    audits = dict(em_updates=0, largest_em_loss_increase=0., replacement_rounds=0,
                  accepted_replacements=0, largest_replacement_loss_increase=0.,
                  largest_final_loss_increase=0., final_updates_have_two_trace_entries=True,
                  candidate_argmin_checks=0)
    with threadpool_limits(limits=1):
        A, B, P, truth, u, lam, graph_meta = generate(task)
        arrays.update(A=A, B=B, P=P, truth_evaluation_only=truth, u=u, lam=lam)
        q_reference = None
        for score_family, prefix in [('bernoulli', 'full'), ('sparse', 'sparse')]:
            work = AuditedLikelihood(u, lam, score_family=score_family)
            if q_reference is None:
                q_reference = work.q.copy()
            else:
                assert np.array_equal(q_reference, work.q)
            grown, growing_records = work.growing(max_iter=100)
            # The baseline receives the same mandatory final E/M update.
            grown_final = work.fit(grown.profiles, weights=grown.weights, max_iter=1, tol=0.)
            replaced, replacement_records = work.replace(grown, rounds=math.ceil(math.log(work.n)), trial_iter=1)
            final = work.fit(replaced.profiles, weights=replaced.weights, max_iter=1, tol=0.)
            for method, fit in [(prefix+'_growing', grown_final), (prefix+'_replacement', final)]:
                outputs[method] = dict(labels=fit.labels, final_soft_loss=fit.loss)
                arrays.update({method+'_labels':fit.labels, method+'_profiles':fit.profiles,
                               method+'_weights':fit.weights})
            trace_lists = [r['em_loss_trace'] for r in growing_records]
            trace_lists += [trial['em_loss_trace'] for r in replacement_records for trial in r['trials']]
            trace_lists += [grown_final.trace, final.trace]
            for trace in trace_lists:
                audits['em_updates'] += len(trace)-1
                if len(trace)>1:
                    audits['largest_em_loss_increase'] = max(audits['largest_em_loss_increase'],
                                                             float(np.diff(trace).max()))
            for r in replacement_records:
                audits['replacement_rounds'] += 1
                audits['accepted_replacements'] += int(r['accepted'] is not None)
                audits['largest_replacement_loss_increase'] = max(
                    audits['largest_replacement_loss_increase'],
                    r['after_soft_loss']-r['before_soft_loss'])
            audits['largest_final_loss_increase'] = max(audits['largest_final_loss_increase'],
                                                        grown_final.loss-grown.loss,
                                                        final.loss-replaced.loss)
            audits['final_updates_have_two_trace_entries'] &= len(final.trace)==2 and len(grown_final.trace)==2
            for rec, totals in zip(work.scans, work.scan_totals):
                assert int(totals.argmin()) == rec['candidate']
                audits['candidate_argmin_checks'] += 1
            scan_arrays[prefix+'_candidate_loss_by_scan'] = np.asarray(work.scan_totals)
            scan_arrays[prefix+'_current_loss_by_scan_and_node'] = np.asarray(work.scan_current_losses)
            all_audits[score_family] = dict(epsilon=work.epsilon, spectral_scale=work.spectral_scale,
                                            round_budget=math.ceil(math.log(work.n)),
                                            growing_loss=grown.loss, growing_final_loss=grown_final.loss,
                                            selected_loss=replaced.loss, final_loss=final.loss,
                                            growing=growing_records, replacement=replacement_records,
                                            growing_final_em_trace=grown_final.trace,
                                            final_em_trace=final.trace, candidate_scans=work.scans)
        for name, embedding in [('u_kmeans_10', math.sqrt(task['n'])*u),
                                ('u_lambda_kmeans_10', math.sqrt(task['n'])*u*lam)]:
            km = KMeans(n_clusters=task['k'], n_init=10, max_iter=300, algorithm='lloyd',
                        random_state=task['seed']+9107).fit(embedding)
            outputs[name] = dict(labels=km.labels_, inertia=float(km.inertia_))
            arrays[name+'_labels'] = km.labels_
            all_audits[name] = dict(inertia=float(km.inertia_), n_iter=int(km.n_iter_))
    # Access the truth only after all model selection has completed.
    metrics = {}
    for method, output in outputs.items():
        metrics[method] = metric(output['labels'], truth, task['k'])
        for key in ('final_soft_loss', 'inertia'):
            if key in output: metrics[method][key] = output[key]
    tolerance = 1e-7 * task['n']
    assert audits['largest_em_loss_increase'] <= tolerance
    assert audits['largest_replacement_loss_increase'] <= 0
    assert audits['largest_final_loss_increase'] <= tolerance
    assert audits['final_updates_have_two_trace_entries']
    arrays.update(scan_arrays)
    np.savez_compressed(folder / 'arrays.npz', **arrays)
    with gzip.open(folder / 'loss_audit.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(all_audits, handle, default=json_default)
    result = dict(**task, graph=graph_meta, methods=metrics, audit=audits,
                  seconds=time.perf_counter()-started,
                  files=['arrays.npz', 'loss_audit.json.gz'])
    result_file.write_text(json.dumps(result, indent=2))
    return result


def summarize(rows):
    if not rows: return
    methods = list(rows[0]['methods'])
    lines = []
    for row in rows:
        for method, values in row['methods'].items():
            lines.append({key:row[key] for key in ('graph_id','n','k','family','density','seed')}
                         | {'method':method} | values)
    fields = sorted({field for line in lines for field in line})
    with (RESULTS/'per_graph_methods.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(lines)
    summary = dict(graphs=len(rows), expected_graphs=60, n=320, methods={}, group_results={},
                   audit=dict(em_updates=sum(r['audit']['em_updates'] for r in rows),
                              candidate_argmin_checks=sum(r['audit']['candidate_argmin_checks'] for r in rows),
                              replacement_rounds=sum(r['audit']['replacement_rounds'] for r in rows),
                              accepted_replacements=sum(r['audit']['accepted_replacements'] for r in rows),
                              maximum_em_loss_increase=max(r['audit']['largest_em_loss_increase'] for r in rows),
                              maximum_replacement_loss_increase=max(r['audit']['largest_replacement_loss_increase'] for r in rows),
                              maximum_final_loss_increase=max(r['audit']['largest_final_loss_increase'] for r in rows),
                              all_mandatory_final_updates_executed=all(r['audit']['final_updates_have_two_trace_entries'] for r in rows)))
    for method in methods:
        error=np.array([r['methods'][method]['misclassification'] for r in rows])
        summary['methods'][method] = dict(mean_misclassification=float(error.mean()),
                                          median_misclassification=float(np.median(error)),
                                          perfect_graphs=int(np.sum(error==0)))
    for family in FAMILIES:
        for density in ('log_degree','dense'):
            chosen=[r for r in rows if r['family']==family and r['density']==density]
            if chosen:
                summary['group_results'][family+'_'+density]={m:float(np.mean([r['methods'][m]['misclassification'] for r in chosen])) for m in methods}
    for prefix in ('full','sparse'):
        delta=np.array([r['methods'][prefix+'_replacement']['misclassification']-r['methods'][prefix+'_growing']['misclassification'] for r in rows])
        summary[prefix+'_replacement_vs_growing'] = dict(better=int(np.sum(delta < -1e-12)),
                                                        equal=int(np.sum(abs(delta)<=1e-12)),
                                                        worse=int(np.sum(delta > 1e-12)),
                                                        mean_error_change=float(delta.mean()))
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    tasks=design()
    protocol=dict(created_before_fitting=True, graph_tasks=tasks, score_clipping='same q for both score controls',
                  methods=['full_growing','full_replacement','sparse_growing','sparse_replacement','u_kmeans_10','u_lambda_kmeans_10'],
                  growing_max_iter=100, replacement_rounds='ceil(log(n))', replacement_trial_iter=1,
                  baseline_extra_final_mstep=1, replacement_extra_final_mstep=1,
                  candidate_selection='all n candidate losses saved for every scan; only selected IDs excluded in growing',
                  limitations=['n=320 only','two graph seeds per K-family-density cell','weak general matrices can fall below useful spectral separation at this n',
                               'likelihood improvement need not improve the misclassification rate'],
                  engine_sha256=hashlib.sha256((ROOT/'spectral_likelihood.py').read_bytes()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (RESULTS/'protocol.json').write_text(json.dumps(protocol,indent=2))
    selected=[tasks[0],tasks[-1]] if args.pilot else tasks
    if args.workers==1:
        rows=[]
        for task in selected:
            row=run_one(task);rows.append(row)
            print(json.dumps(dict(done=len(rows),total=len(selected),graph=row['graph_id'],seconds=row['seconds'],
                                  errors={m:v['misclassification'] for m,v in row['methods'].items()})),flush=True)
    else:
        rows=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(run_one,selected):
                rows.append(row)
                print(json.dumps(dict(done=len(rows),total=len(selected),graph=row['graph_id'],seconds=row['seconds'],
                                      errors={m:v['misclassification'] for m,v in row['methods'].items()})),flush=True)
    summary=summarize(rows)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

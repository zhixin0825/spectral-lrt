"""Independent exact-target checks for uniform spectral-prototype MH.

The ordered list of distinct prototype IDs is independent of current C.
Both MH proposal probabilities condition on the same list. Each such
kernel, their uniform mixture, and composition with Gibbs preserve the
collapsed working target. Validation below checks mechanics, not recovery.
"""
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax

from independent_validation import reference_target
from independent_spectral_validation import eigenvectors
from prototype_mcmc import prototype_chain


ROOT = Path(__file__).resolve().parent


def conditional_checks(Q, k, U, scale=2., noise=.2):
    n = len(Q)
    states = np.array(list(itertools.product(range(k), repeat=n)), dtype=np.int64)
    logp = np.array([reference_target(Q, z, k) for z in states])
    p = softmax(logp)
    X = np.sqrt(n)*(U-U.mean(0))
    uncentered = np.sqrt(n)*U
    templates = list(itertools.permutations(range(n), k))
    maximum_balance = maximum_stationary = maximum_row = maximum_q = 0.
    maximum_center_difference = 0.
    proposal_min = 1.
    mixture = np.zeros((len(states), len(states)))
    for ids in templates:
        centers = X[list(ids)]
        distances = ((X[:, None, :]-centers[None, :, :])**2).sum(2)
        uncentered_distances = ((uncentered[:, None, :]-uncentered[list(ids)][None, :, :])**2).sum(2)
        maximum_center_difference = max(maximum_center_difference,
                                        float(abs(distances-uncentered_distances).max()))
        node_q = (1-noise)*softmax(-scale*distances, axis=1)+noise/k
        proposal_min = min(proposal_min, float(node_q.min()))
        logq = np.log(node_q[np.arange(n)[None, :], states]).sum(1)
        q = np.exp(logq)
        logalpha = logp[None, :]-logp[:, None]+logq[:, None]-logq[None, :]
        transition = q[None, :]*np.exp(np.minimum(0., logalpha))
        transition[np.diag_indices(len(states))] += 1-transition.sum(1)
        mixture += transition/len(templates)
        flow = p[:, None]*transition
        maximum_balance = max(maximum_balance, float(abs(flow-flow.T).max()))
        maximum_stationary = max(maximum_stationary, float(abs(p@transition-p).sum()/2))
        maximum_row = max(maximum_row, float(abs(transition.sum(1)-1).max()))
        maximum_q = max(maximum_q, abs(float(q.sum())-1))
    mixture_flow = p[:, None]*mixture
    mixture_error = float(abs(mixture_flow-mixture_flow.T).max())
    checks = [maximum_balance, maximum_stationary, maximum_row, maximum_q,
              maximum_center_difference, mixture_error]
    return dict(n=n, k=k, states=len(states), ordered_templates=len(templates),
                max_detailed_balance_residual=maximum_balance,
                max_stationary_total_variation_residual=maximum_stationary,
                max_transition_row_sum_error=maximum_row,
                max_proposal_normalization_error=maximum_q,
                max_centered_vs_uncentered_squared_distance_error=maximum_center_difference,
                min_node_proposal_probability=proposal_min,
                uniform_template_mixture_detailed_balance_residual=mixture_error,
                passed=max(checks)<1e-11)


def empirical_checks(Q, k, U, seed, samples=40000):
    states = np.array(list(itertools.product(range(k), repeat=len(Q))), dtype=np.int64)
    p = softmax([reference_target(Q, z, k) for z in states])
    initial = np.random.default_rng(seed).integers(k, size=len(Q), dtype=np.int64)
    run = prototype_chain(Q, k, initial, samples+2000, seed+951, U=U, record_every=1)
    weights = k**np.arange(len(Q)-1,-1,-1)
    recorded = run['trace_labels'][2001:]
    empirical = np.bincount(recorded@weights, minlength=len(states))/len(recorded)
    tv = float(abs(empirical-p).sum()/2)
    trace_error = max(abs(reference_target(Q, z, k)-score)
                      for z, score in zip(run['trace_labels'][::100], run['trace_score'][::100]))
    accepted = int(run['copy_accepted'].sum())
    tried = int(run['copy_tries'].sum())
    accepted_trace = run['global_accepted_trace']
    acceptance_record_consistent = (int(accepted_trace[-1])==accepted
                                    and bool(np.all(np.diff(accepted_trace)>=0)))
    return dict(n=len(Q), k=k, samples=len(recorded), seed=seed,
                cold_marginal_total_variation=tv,
                max_recorded_score_error=float(trace_error),
                global_accepted=accepted, global_tried=tried,
                accepted_trace_consistent=acceptance_record_consistent,
                passed=bool(tv<.06 and trace_error<1e-9 and tried==samples+2000
                            and acceptance_record_consistent))


def main():
    start = time.perf_counter()
    Q = np.array([[0., .0, .71, .22], [.0, 0., 1., .33],
                  [.71, 1., 0., .91], [.22, .33, .91, 0.]])
    conditional = [conditional_checks(Q, k, eigenvectors(Q, k)) for k in (2,3)]
    empirical = [empirical_checks(Q, k, eigenvectors(Q, k), seed)
                 for k, seed in ((2,83032),(3,83033))]
    report = dict(passed=all(x['passed'] for x in conditional+empirical),
                  elapsed_seconds=time.perf_counter()-start,
                  scope='All ordered small-network prototype templates, independent-MH target, and actual prototype/Gibbs sampler; no large-network mixing or recovery guarantee.',
                  target_reason='Uniform ordered prototype list is current-label independent. Both proposal densities use the same list. Corrected kernel, list mixture and Gibbs composition preserve the same p(C).',
                  conditional_checks=conditional, empirical_checks=empirical)
    (ROOT/'independent_prototype_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

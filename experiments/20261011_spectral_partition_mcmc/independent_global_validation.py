"""Independent small-state validation of fixed-donor global MH moves.

Reference densities are computed from explicit unordered vertex pairs. These
checks validate the target and implementation, not large-network mixing.
"""
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax

from independent_validation import reference_target
from global_mcmc import global_chain


ROOT = Path(__file__).resolve().parent


def donor_transition_checks(Q, k, noise):
    states = np.array(list(itertools.product(range(k), repeat=len(Q))), dtype=np.int64)
    logp = np.array([reference_target(Q, z, k) for z in states])
    p = softmax(logp)
    maximum_balance = maximum_stationary = maximum_row_error = maximum_q_error = 0.
    for donor in states:
        logq = np.where(states == donor, np.log(1-noise+noise/k), np.log(noise/k)).sum(1)
        q = np.exp(logq)
        # Rows are source states x; columns are proposed destination states y.
        logalpha = logp[None, :] - logp[:, None] + logq[:, None] - logq[None, :]
        transition = q[None, :] * np.exp(np.minimum(0., logalpha))
        transition[np.diag_indices(len(states))] += 1-transition.sum(1)
        flow = p[:, None] * transition
        maximum_balance = max(maximum_balance, float(abs(flow-flow.T).max()))
        maximum_stationary = max(maximum_stationary, float(abs(p@transition-p).sum()/2))
        maximum_row_error = max(maximum_row_error, float(abs(transition.sum(1)-1).max()))
        maximum_q_error = max(maximum_q_error, abs(float(q.sum())-1))
    return dict(n=len(Q), k=k, states=len(states), donors_checked=len(states), noise=noise,
                max_detailed_balance_residual=maximum_balance,
                max_stationary_total_variation_residual=maximum_stationary,
                max_transition_row_sum_error=maximum_row_error,
                max_proposal_probability_sum_error=maximum_q_error,
                passed=max(maximum_balance, maximum_stationary, maximum_row_error, maximum_q_error)<1e-11)


def empirical_checks(Q, k, betas, noise, interval, samples=40000):
    states = np.array(list(itertools.product(range(k), repeat=len(Q))), dtype=np.int64)
    p = softmax([reference_target(Q, z, k) for z in states])
    initial = np.random.default_rng(64811).integers(k, size=len(Q), dtype=np.int64)
    run = global_chain(Q, k, initial, samples+2000, 16433,
                       betas=betas, noise=noise, proposal_interval=interval, record_every=1)
    weights = k**np.arange(len(Q)-1,-1,-1)
    recorded = run['trace_labels'][2001:]
    empirical = np.bincount(recorded@weights, minlength=len(states))/len(recorded)
    tv = float(abs(empirical-p).sum()/2)
    trace_error = max(abs(reference_target(Q, z, k)-score)
                      for z, score in zip(run['trace_labels'][::100], run['trace_score'][::100]))
    return dict(n=len(Q), k=k, samples=len(recorded), betas=list(betas), noise=noise,
                proposal_interval=interval, cold_marginal_total_variation=tv,
                max_recorded_score_error=float(trace_error),
                donor_copy_accepted=run['copy_accepted'].tolist(),
                donor_copy_tries=run['copy_tries'].tolist(),
                passed=bool(tv<.06 and trace_error<1e-9))


def main():
    start = time.perf_counter()
    Q = np.array([[0., .0, .71, .22], [.0, 0., 1., .33],
                  [.71, 1., 0., .91], [.22, .33, .91, 0.]])
    transition = [donor_transition_checks(Q, k, noise)
                  for k, noise in ((2,.05),(3,.05),(3,.4))]
    empirical = [empirical_checks(Q, 2, (1.,2.,4.), .05, 5),
                 empirical_checks(Q, 3, (1.,2.,4.,8.,16.,32.), .05, 5),
                 empirical_checks(Q, 3, (1.,2.,4.), .4, 1)]
    report = dict(passed=all(x['passed'] for x in transition+empirical),
                  elapsed_seconds=time.perf_counter()-start,
                  scope='Exact conditional global MH density and actual interacting-chain cold marginal; no mixing guarantee.',
                  conditional_target_reason='For each fixed donor d, the corrected kernel preserves p(z_0); donor remains unchanged. Independent replica Gibbs updates and these conditional moves preserve the product target.',
                  transition_checks=transition, empirical_checks=empirical)
    (ROOT/'independent_global_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

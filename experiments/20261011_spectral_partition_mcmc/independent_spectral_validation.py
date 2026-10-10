"""Independent validation of random-direction spectral independence MH.

Each Gaussian W is independent of current labels. For a fixed W, the
proposal is an explicit product distribution, and its own MH kernel
preserves the collapsed target. Random W mixtures and Gibbs compositions
therefore preserve that same target. These tests do not establish mixing.
"""
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax

from independent_validation import reference_target
from spectral_mcmc import spectral_chain


ROOT = Path(__file__).resolve().parent


def eigenvectors(Q, k):
    values, vectors = np.linalg.eigh(Q)
    indices = np.argsort(-abs(values))[:k]
    return vectors[:, indices]


def conditional_checks(Q, k, U, scale=8., noise=.2):
    n = len(Q)
    states = np.array(list(itertools.product(range(k), repeat=n)), dtype=np.int64)
    logp = np.array([reference_target(Q, z, k) for z in states])
    p = softmax(logp)
    X = np.sqrt(n)*(U-U.mean(0))
    rng = np.random.default_rng(6973)
    matrices = [np.zeros((U.shape[1], k)), np.ones((U.shape[1], k))]
    matrices += [rng.normal(size=(U.shape[1], k)) for _ in range(8)]
    reports = []
    kernels = []
    for index, W in enumerate(matrices):
        node_q = (1-noise)*softmax(scale*(X@W), axis=1) + noise/k
        logq = np.log(node_q[np.arange(n)[None, :], states]).sum(1)
        q = np.exp(logq)
        logalpha = logp[None, :]-logp[:, None]+logq[:, None]-logq[None, :]
        transition = q[None, :]*np.exp(np.minimum(0., logalpha))
        transition[np.diag_indices(len(states))] += 1-transition.sum(1)
        kernels.append(transition)
        flow = p[:, None]*transition
        balance = float(abs(flow-flow.T).max())
        stationary = float(abs(p@transition-p).sum()/2)
        norm = abs(float(q.sum())-1)
        row_error = float(abs(transition.sum(1)-1).max())
        reports.append(dict(W_index=index, uniform_proposal=index<2,
                            min_node_proposal_probability=float(node_q.min()),
                            max_detailed_balance_residual=balance,
                            stationary_total_variation_residual=stationary,
                            proposal_normalization_error=norm,
                            max_transition_row_sum_error=row_error,
                            passed=max(balance, stationary, norm, row_error)<1e-11))
    mixture = np.mean(kernels, axis=0)
    flow = p[:, None]*mixture
    mixture_residual = float(abs(flow-flow.T).max())
    return dict(n=n, k=k, states=len(states), fixed_W_checks=reports,
                finite_W_mixture_detailed_balance_residual=mixture_residual,
                passed=all(r['passed'] for r in reports) and mixture_residual<1e-11)


def empirical_checks(Q, k, U, seed, samples=40000):
    states = np.array(list(itertools.product(range(k), repeat=len(Q))), dtype=np.int64)
    p = softmax([reference_target(Q, z, k) for z in states])
    initial = np.random.default_rng(seed).integers(k, size=len(Q), dtype=np.int64)
    run = spectral_chain(Q, k, initial, samples+2000, seed+517, U=U, record_every=1)
    weights = k**np.arange(len(Q)-1,-1,-1)
    recorded = run['trace_labels'][2001:]
    empirical = np.bincount(recorded@weights, minlength=len(states))/len(recorded)
    tv = float(abs(empirical-p).sum()/2)
    trace_error = max(abs(reference_target(Q, z, k)-score)
                      for z, score in zip(run['trace_labels'][::100], run['trace_score'][::100]))
    accepted = int(run['copy_accepted'].sum())
    tried = int(run['copy_tries'].sum())
    return dict(n=len(Q), k=k, samples=len(recorded), seed=seed,
                cold_marginal_total_variation=tv,
                max_recorded_score_error=float(trace_error),
                global_accepted=accepted, global_tried=tried,
                passed=bool(tv<.06 and trace_error<1e-9 and tried==samples+2000))


def main():
    start = time.perf_counter()
    Q = np.array([[0., .0, .71, .22], [.0, 0., 1., .33],
                  [.71, 1., 0., .91], [.22, .33, .91, 0.]])
    conditional = [conditional_checks(Q, k, eigenvectors(Q, k)) for k in (2,3)]
    empirical = [empirical_checks(Q, k, eigenvectors(Q, k), seed)
                 for k, seed in ((2,9183),(3,9191))]
    report = dict(passed=all(x['passed'] for x in conditional+empirical),
                  elapsed_seconds=time.perf_counter()-start,
                  scope='Fixed-W independence-MH target and actual random-W/Gibbs sampler; no large-network mixing or recovery guarantee.',
                  target_reason='W has a current-label-independent distribution. Both proposal densities use the same W; each corrected move preserves p(C). The W mixture and Gibbs composition preserve p(C).',
                  conditional_checks=conditional, empirical_checks=empirical)
    (ROOT/'independent_spectral_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

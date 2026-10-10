"""Independent checks of the fractional undirected SBM sampler.

This file intentionally computes reference targets from explicit unordered
vertex pairs; it does not use engine._score or engine._candidate_gains for
reference answers. Small-state enumeration checks mechanics, not recovery.
"""
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import betaln, softmax

import engine


ROOT = Path(__file__).resolve().parent


def reference_counts(Q, labels, k):
    sizes = np.array([sum(int(a) == c for a in labels) for c in range(k)])
    counts = np.zeros((k, k))
    opportunities = np.zeros((k, k))
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = sorted((int(labels[i]), int(labels[j])))
            counts[a, b] += Q[i, j]
            opportunities[a, b] += 1
    return sizes, counts, opportunities


def reference_target(Q, labels, k):
    _, counts, opportunities = reference_counts(Q, labels, k)
    return float(sum(betaln(counts[a, b] + 1, opportunities[a, b] - counts[a, b] + 1)
                     for a in range(k) for b in range(a, k)))


def exact_state_checks(Q, k):
    n = len(Q)
    states = np.array(list(itertools.product(range(k), repeat=n)), dtype=np.int64)
    reference = np.array([reference_target(Q, labels, k) for labels in states])
    target_error = 0.
    count_error = 0.
    conditional_error = 0.
    probability_error = 0.
    permutation_error = 0.
    checked_conditionals = 0
    for index, labels in enumerate(states):
        sizes, counts, _ = reference_counts(Q, labels, k)
        engine_sizes, engine_counts = engine._counts(Q, labels, k)
        count_error = max(count_error, float(np.max(abs(counts - engine_counts))),
                          float(np.max(abs(sizes - engine_sizes))))
        target_error = max(target_error, abs(engine.log_target(Q, labels, k) - reference[index]))
        for i in range(n):
            full = []
            for a in range(k):
                changed = labels.copy()
                changed[i] = a
                full.append(reference_target(Q, changed, k))
            full = np.array(full)
            fast = engine.conditional_scores(Q, labels, i, k)
            conditional_error = max(conditional_error,
                                    float(np.max(abs((fast - fast[0]) - (full - full[0])))))
            probability_error = max(probability_error, float(np.max(abs(softmax(fast) - softmax(full)))))
            checked_conditionals += 1
        # Nontrivial fixed label permutation leaves the unknown-P target unchanged.
        relabeled = (labels + 1) % k
        permutation_error = max(permutation_error, abs(reference_target(Q, relabeled, k) - reference[index]))
    return states, reference, dict(
        n=n, k=k, states=len(states), conditionals=checked_conditionals,
        max_count_error=count_error, max_log_target_error=target_error,
        max_relative_conditional_error=conditional_error,
        max_conditional_probability_error=probability_error,
        max_target_label_permutation_error=permutation_error,
        passed=max(target_error, count_error, conditional_error, permutation_error) < 1e-10,
    )


def transition_checks(Q, k, states, reference, beta):
    n = len(Q)
    index = {tuple(labels): i for i, labels in enumerate(states)}
    transition = np.zeros((len(states), len(states)))
    for row, labels in enumerate(states):
        for i in range(n):
            probabilities = softmax(beta * engine.conditional_scores(Q, labels, i, k))
            for a in range(k):
                changed = labels.copy()
                changed[i] = a
                transition[row, index[tuple(changed)]] += probabilities[a] / n
    weights = softmax(beta * reference)
    flow = weights[:, None] * transition
    balance = float(np.max(abs(flow - flow.T)))
    stationary = float(np.sum(abs(weights @ transition - weights)) / 2)
    row_error = float(np.max(abs(transition.sum(1) - 1)))
    return dict(beta=beta, max_detailed_balance_residual=balance,
                stationary_total_variation_residual=stationary,
                max_transition_row_sum_error=row_error,
                passed=max(balance, stationary, row_error) < 1e-11)


def empirical_checks(Q, k, samples=30000, tempering=False):
    n = len(Q)
    states = np.array(list(itertools.product(range(k), repeat=n)), dtype=np.int64)
    target = softmax([reference_target(Q, labels, k) for labels in states])
    labels = np.random.default_rng(827).integers(k, size=n, dtype=np.int64)
    if tempering:
        output = engine.tempered_chain(Q, k, labels, samples + 1000, 1453,
                                       betas=(1., .5, .1), record_every=1)
    else:
        output = engine.collapsed_chain(Q, k, labels, samples + 1000, 1453, record_every=1)
    ids = output['trace_labels'][1001:] @ (k ** np.arange(n - 1, -1, -1))
    empirical = np.bincount(ids, minlength=len(states)) / len(ids)
    tv = float(abs(empirical - target).sum() / 2)
    trace_error = float(max(abs(score - reference_target(Q, labels, k))
                            for labels, score in zip(output['trace_labels'][::100], output['trace_score'][::100])))
    result = dict(method='parallel_tempering' if tempering else 'collapsed_gibbs',
                  samples=len(ids), target_total_variation=tv,
                  sampled_trace_score_max_error=trace_error,
                  passed=tv < .04 and trace_error < 1e-9)
    if tempering:
        result['swap_acceptance'] = output['swap_acceptance'].tolist()
    return result


def swap_checks():
    # Algebraic detailed balance for the exact swap expression used by engine.
    residual = 0.
    for b1, b2 in ((1., .7), (.7, .2), (.2, .01)):
        for s1, s2 in ((-3., -8.), (-8., -3.), (-7., -7.), (-100., -120.)):
            x = b1 * s1 + b2 * s2
            y = b1 * s2 + b2 * s1
            forward = (b1 - b2) * (s2 - s1)
            reverse = -forward
            # Log equilibrium flows avoid underflow and unnecessary normalization.
            residual = max(residual, abs(x + min(0., forward) - y - min(0., reverse)))
    return dict(max_swap_log_flow_balance_residual=residual, passed=residual < 1e-11)


def algorithm_checks():
    rng = np.random.default_rng(641)
    reports = []
    for case in range(8):
        n, k = 18, (2 if case % 2 else 3)
        raw = rng.uniform(0., .8, size=(n, n))
        Q = (raw + raw.T) / 2
        if case > 3:
            Q = (Q > .4).astype(float)
        np.fill_diagonal(Q, 0.)
        z0 = rng.integers(k, size=n, dtype=np.int64)
        if case == 2:
            z0[:] = 0  # Explicit empty-group case.
        c = engine.collapsed_chain(Q, k, z0, 57, 834 + case, record_every=7)
        pt = engine.tempered_chain(Q, k, z0, 57, 335 + case, record_every=7)
        icm = engine.collapsed_chain(Q, k, z0, 57, 762 + case, greedy=True, record_every=1)
        em = engine.variational_em(Q, k, z0, 100, 662 + case)
        target_error = max(abs(reference_target(Q, z, k) - score)
                           for run in (c, pt, icm, em)
                           for z, score in zip(run['trace_labels'], run['trace_score']))
        best_selection = min(reference_target(Q, run['best_labels'], k) - max(run['trace_score'])
                             for run in (c, pt, icm, em))
        icm_delta = float(np.diff(icm['trace_score']).min())
        em_delta = float(np.diff(em['elbo_trace']).min())
        matrix = em['responsibilities']
        mass_error = float(np.max(abs(matrix.sum(1) - 1)))
        # Recover the ELBO independently from unordered vertex pairs and entropy.
        elbo = 0.
        for i in range(n):
            for j in range(i + 1, n):
                for a in range(k):
                    for b in range(k):
                        p = em['P'][a, b]
                        elbo += matrix[i, a] * matrix[j, b] * (Q[i, j] * np.log(p) + (1-Q[i, j]) * np.log1p(-p))
        positive = matrix > 0
        elbo -= float((matrix[positive] * np.log(matrix[positive])).sum())
        elbo_error = abs(elbo - em['elbo_trace'][-1])
        truth = np.arange(n, dtype=int) % k
        labels = c['labels']
        metric_before = engine.metrics(labels, truth, k)
        metric_after = engine.metrics((labels + 1) % k, truth, k)
        metric_invariant = metric_before == metric_after
        passed = (target_error < 1e-9 and best_selection > -1e-9 and icm_delta > -1e-9
                  and em_delta > -1e-8 and elbo_error < 1e-8 and mass_error < 1e-10
                  and metric_invariant and c['trace_sweep'][-1] == 57
                  and pt['trace_sweep'][-1] == 57)
        reports.append(dict(case=case, k=k, binary=case > 3, empty_initial_groups=case == 2,
                            max_trace_score_error=float(target_error),
                            best_score_minus_recorded_max=float(best_selection),
                            minimum_icm_score_change=icm_delta, minimum_em_elbo_change=em_delta,
                            final_elbo_reference_error=float(elbo_error),
                            max_responsibility_row_sum_error=mass_error,
                            metric_label_permutation_invariant=metric_invariant, passed=bool(passed)))
    return reports


def main():
    start = time.perf_counter()
    Q = np.array([[0., .0, .14, .71, .22], [.0, 0., 1., .33, .05],
                  [.14, 1., 0., .59, .91], [.71, .33, .59, 0., .4],
                  [.22, .05, .91, .4, 0.]])
    states, scores, exhaustive = exact_state_checks(Q, 3)
    transitions = [transition_checks(Q, 3, states, scores, beta) for beta in (1., .37)]
    empirical_Q = Q[:4, :4].copy()
    empirical = [empirical_checks(empirical_Q, 2, tempering=pt) for pt in (False, True)]
    algorithms = algorithm_checks()
    swap = swap_checks()
    passed = (exhaustive['passed'] and swap['passed']
              and all(x['passed'] for x in transitions + empirical + algorithms))
    result = dict(passed=passed, scope='Sampler and optimizer mechanics; not an SBM recovery guarantee.',
                  elapsed_seconds=time.perf_counter()-start, exhaustive=exhaustive,
                  random_scan_transition=transitions, parallel_tempering_swap=swap,
                  empirical_small_state_stationarity=empirical, algorithm_checks=algorithms)
    path = ROOT / 'independent_validation.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

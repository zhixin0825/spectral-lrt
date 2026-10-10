"""Reproducible K=5 finite-stage growing counterexample and deterministic repair.

All fitted scores remain the working profile score.  The repair tries every
component removal, inserts the all-node positive-gain maximizing row, resets
weights uniformly, performs the prescribed EM steps, and accepts only an
improving endpoint under the same objective.  It uses no restart.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from population_growing import divergence, em, growing

B = np.array([
    [.0071, .0914, .0119, .0129, .0132],
    [.0914, 2.7252, .2538, .2125, .2722],
    [.0119, .2538, .0285, .0276, .0300],
    [.0129, .2125, .0276, .0294, .0289],
    [.0132, .2722, .0300, .0289, .0321],
])
PI = np.array([.02, .57, .065, .22, .125])


def objective(B, pi, mu, weights, degree):
    return float(pi @ logsumexp(-degree * divergence(B, mu, pi)
                                + np.log(weights)[None, :], axis=1))


def replacement(B, pi, fit, degree, rounds=3, steps=1):
    mu = np.array(fit['final_mu'])
    weights = np.array(fit['final_weights'])
    current = objective(B, pi, mu, weights, degree)
    records = []
    pairwise = divergence(B, B, pi)
    for round_id in range(rounds):
        trials = []
        best = None
        for removed in range(len(B)):
            other = np.delete(mu, removed, axis=0)
            losses = divergence(B, other, pi).min(1)
            gains = pi @ np.maximum(losses[:, None] - pairwise, 0.)
            candidate = int(gains.argmax())
            initial = mu.copy()
            initial[removed] = B[candidate]
            trial = em(B, pi, initial, degree, steps=steps, floor=1e-8,
                       weights=np.full(len(B), 1 / len(B)), trace=True)
            trials.append(dict(removed=removed, candidate=candidate, gains=gains.tolist(),
                               objective=trial['objective'], labels=trial['labels'].tolist(),
                               weights=trial['weights'].tolist(), mu=trial['mu'].tolist(),
                               history=trial['history']))
            if trial['objective'] > current + 1e-10 and (best is None or
                                                        trial['objective'] > best['fit']['objective']):
                best = dict(removed=removed, candidate=candidate, fit=trial)
        old = current
        if best is not None:
            mu = best['fit']['mu']
            weights = best['fit']['weights']
            current = best['fit']['objective']
        records.append(dict(round=round_id + 1, previous_objective=old,
                            objective=current,
                            accepted=None if best is None else dict(removed=best['removed'],
                                                                  candidate=best['candidate']),
                            trials=trials))
    return dict(records=records, final_mu=mu.tolist(), final_weights=weights.tolist(),
                final_labels=(-degree * divergence(B, mu, pi)
                              + np.log(weights)[None, :]).argmax(1).tolist(),
                objective=current)


def main():
    runs = []
    for degree in [1e4, 1e5, 1e6, 1e8, 1e10, 1e12]:
        base = growing(B, PI, degree=degree, steps=1, trace=True)
        control = growing(B, PI, degree=degree, steps=2, trace=False)
        extra = em(B, PI, np.array(base['final_mu']), degree, steps=100,
                   weights=np.array(base['final_weights']), trace=False)
        repaired = replacement(B, PI, base, degree, rounds=3, steps=1)
        row = dict(degree=degree, growing=base, two_steps_each_stage=control,
                   extra_100_final_steps_labels=extra['labels'].tolist(), repair=repaired,
                   true_parameters_objective=objective(B, PI, B, PI, degree))
        runs.append(row)
        print(json.dumps(dict(degree=degree, error=base['error'],
                              selected=base['selected'], base_objective=base['objective_minus_self_entropy'],
                              repaired_objective=repaired['objective'],
                              repair_accepts=[r['accepted'] for r in repaired['records']],
                              repaired_labels=repaired['final_labels'])), flush=True)
    def determinant_integer(matrix):
        # Fraction-free Bareiss elimination; all leading pivots are nonzero here.
        a = [list(map(int, row)) for row in matrix]
        previous = 1
        for k in range(len(a)-1):
            pivot = a[k][k]
            assert pivot
            for i in range(k+1, len(a)):
                for j in range(k+1, len(a)):
                    numerator = pivot*a[i][j] - a[i][k]*a[k][j]
                    assert numerator % previous == 0
                    a[i][j] = numerator // previous
                a[i][k] = 0
            previous = pivot
        return a[-1][-1]
    integer_matrix = np.rint(10000 * B).astype(int)
    minors = [determinant_integer(integer_matrix[:i, :i].tolist()) for i in range(1, 6)]
    result = dict(description='Population profile audit; this is not a sampled graph experiment.',
                  B=B.tolist(), pi=PI.tolist(), P='P = degree * B / n, choose n > degree * max(B)',
                  component_and_block_indices='zero-based in JSON',
                  eigenvalues=np.linalg.eigvalsh(B).tolist(),
                  integer_matrix_leading_principal_determinants=minors,
                  row_divergence=divergence(B, B, PI).tolist(), runs=runs)
    Path(__file__).with_name('replacement_counterexample_results.json').write_text(
        json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

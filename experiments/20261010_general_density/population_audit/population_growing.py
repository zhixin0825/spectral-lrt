"""Audit original growing global-gain initialization on SBM population profiles.

Rows B_a have type masses pi_a and coordinate masses pi_b.  For P=d B/n,
the score differences are -d D_pi(B_a || mu), exactly the compressed
all-node working score.  Each block has many nodes; excluding selected
node IDs does not exclude another candidate row of that same type.
No restart or substitution of Euclidean seeding is used.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment


def divergence(x, mu, coord):
    ratio_minus_one = (mu[None, :, :] - x[:, None, :]) / x[:, None, :]
    f = ratio_minus_one - np.log1p(ratio_minus_one)
    small = abs(ratio_minus_one) < 1e-4
    v = ratio_minus_one[small]
    f[small] = v*v*(.5 + v*(-1/3 + v*(.25 + v*(-.2 + v/6))))
    return np.maximum((x[:, None, :] * f) @ coord, 0.)


def weights_floor(mass, floor):
    if floor == 0:
        return mass / mass.sum()
    free = np.ones(len(mass), dtype=bool)
    out = np.full(len(mass), floor)
    for _ in range(len(mass) + 1):
        left = 1. - (~free).sum() * floor
        trial = mass[free] * left / mass[free].sum()
        below = trial < floor
        if not below.any():
            out[free] = trial
            return out
        free[np.flatnonzero(free)[below]] = False
    raise RuntimeError('floored weights')


def em(x, pi, mu, degree, steps, hard=False, floor=1e-8, trace=False,
       weights=None, coord=None):
    coord = pi if coord is None else coord
    weights = np.full(len(mu), 1 / len(mu)) if weights is None else weights.copy()
    mu = mu.copy()
    history = []
    for it in range(steps + 1):
        ds = divergence(x, mu, coord)
        with np.errstate(divide='ignore'):
            sc = -degree * ds + np.log(weights)[None, :]
        normal = logsumexp(sc, axis=1)
        labels = ds.argmin(1) if hard else sc.argmax(1)
        objective = float(pi @ normal)
        if hard:
            tau = np.eye(len(mu))[labels]
        else:
            tau = np.exp(sc - normal[:, None])
        mass = pi @ tau
        log_responsibility = sc - normal[:, None]
        log_mass = logsumexp(np.log(pi)[:, None] + log_responsibility, axis=0)
        if trace:
            history.append(dict(iteration=it, mu=mu.tolist(), weights=weights.tolist(),
                                labels=labels.tolist(), mass=mass.tolist(),
                                objective_minus_self_entropy=objective,
                                best_divergences=ds.min(1).tolist(),
                                responsibility=tau.tolist(), log_mass=log_mass.tolist()))
        if it == steps:
            break
        new = mu.copy()
        if hard:
            active = mass > 0
            new[active] = (tau[:, active] * pi[:, None]).T @ x / mass[active, None]
        else:
            # The profile M-step must not discard exponentially small components.
            # This log normalization preserves the exact-arithmetic limiting mean
            # even when every raw responsibility underflows in ordinary doubles.
            local_normalized_mass = np.exp(np.log(pi)[:, None] + log_responsibility
                                           - log_mass[None, :])
            new = local_normalized_mass.T @ x
        new_weights = weights_floor(mass, floor)
        if np.max(abs(new - mu)) < 1e-14 and np.max(abs(new_weights - weights)) < 1e-14:
            mu, weights = new, new_weights
            if not trace:
                break
            continue
        mu, weights = new, new_weights
    return dict(mu=mu, weights=weights, labels=labels, objective=objective,
                history=history, responsibility=tau, mass=mass)


def growing(B, pi, degree=1e6, steps=10, hard=False, floor=1e-8, trace=False):
    mu = (pi @ B)[None, :]
    stages = []
    selected = []
    pair_ds = divergence(B, B, pi)
    for size in range(2, len(B) + 1):
        loss = divergence(B, mu, pi).min(1)
        gains = pi @ np.maximum(loss[:, None] - pair_ds, 0.)
        candidate = int(gains.argmax())
        selected.append(candidate)
        initial = np.vstack([mu, B[candidate]])
        fit = em(B, pi, initial, degree, steps, hard=hard, floor=floor, trace=trace)
        mu = fit['mu']
        if trace:
            stages.append(dict(size=size, candidate=candidate, gains=gains.tolist(),
                               initial_mu=initial.tolist(), history=fit['history']))
    # For pure population rows an injective labels map is exactly perfect recovery.
    labels = fit['labels']
    confusion = np.zeros((len(B), len(B)))
    confusion[np.arange(len(B)), labels] = pi
    ri, ci = linear_sum_assignment(-confusion)
    error = float(1 - confusion[ri, ci].sum())
    return dict(error=max(error, 0.), selected=selected, stages=stages,
                final_mu=mu.tolist(), final_weights=fit['weights'].tolist(),
                final_labels=labels.tolist(),
                objective_minus_self_entropy=fit['objective'])


def sample_model(rng, K, family):
    # All probabilities remain positive and the matrices are almost surely full rank.
    pi = rng.dirichlet(np.full(K, rng.choice([.3, 1., 5.])))
    pi = .02 + .98 * pi
    pi /= pi.sum()
    if family == 'lognormal':
        upper = rng.normal(0, 1.5, (K, K))
        B = np.exp((upper + upper.T) / 2)
    elif family == 'euclidean':
        X = rng.normal(0, 1, (K, rng.integers(1, K + 1)))
        dist = np.sum((X[:, None] - X[None]) ** 2, axis=-1)
        B = np.exp(-dist / rng.uniform(.3, 10)) + rng.uniform(.1, 1)
        B += np.eye(K) * rng.uniform(.0001, .05)
    elif family == 'lowrank':
        X = rng.lognormal(0, 1.5, (K, rng.integers(1, K + 1)))
        B = X @ X.T + np.eye(K) * rng.uniform(.0001, .01)
    elif family == 'nearconstant':
        upper = rng.normal(0, 1, (K, K))
        B = 20 + upper + upper.T
        B = np.maximum(B, .01)
    elif family == 'sqrtgram':
        dim = int(rng.choice([1, 2, 2, 2, 3, 4]))
        X = rng.normal(size=(K, dim)) * rng.lognormal(0, 1.5, (K, 1))
        X -= pi @ X
        G = X @ X.T
        weightedG = np.sqrt(pi[:, None] * pi[None, :]) * G
        val, vec = np.linalg.eigh(weightedG)
        S = (vec * np.sqrt(np.maximum(val, 0))) @ vec.T
        S /= np.sqrt(pi[:, None] * pi[None, :])
        S /= np.max(abs(S))
        B = np.ones((K, K)) + .05 * S + np.eye(K) * 1e-6
    else:
        raise ValueError(family)
    B = B / (pi @ B @ pi)
    return B, pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--degree', type=float, default=1e8)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ks', type=int, nargs='+', default=[3, 4, 5, 6])
    parser.add_argument('--families', nargs='+', default=['lognormal', 'euclidean', 'lowrank', 'nearconstant'])
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    found = []
    counts = {}
    for K in args.ks:
        for family in args.families:
            bad = 0
            for trial in range(args.count):
                B, pi = sample_model(rng, K, family)
                result = growing(B, pi, degree=args.degree, steps=args.steps, hard=args.hard)
                if result['error'] > 1e-8:
                    bad += 1
                    rec = dict(K=K, family=family, trial=trial, B=B.tolist(), pi=pi.tolist(),
                               degree=args.degree, steps=args.steps, hard=args.hard,
                               result=growing(B, pi, degree=args.degree, steps=args.steps,
                                              hard=args.hard, trace=True))
                    if bad <= 5:
                        found.append(rec)
                        print(json.dumps(dict(found_bad=True, K=K, family=family, trial=trial,
                                             error=result['error'], selected=result['selected'])), flush=True)
            counts[f'{K}_{family}'] = dict(total=args.count, failed=bad)
            args.output.write_text(json.dumps(dict(config=vars(args) | {'output':str(args.output)},
                                                  counts=counts, examples=found), indent=2))
            print(json.dumps(dict(K=K, family=family, total=args.count, failed=bad)), flush=True)


if __name__ == '__main__':
    main()

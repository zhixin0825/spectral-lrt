"""Poisson--Gaussian likelihood-regret ++ on one fixed feature matrix.

The regret is already a deviance/squared-distance analogue: do not square it.
Singleton saturation is a constrained global MLE, not clipping the observation.
No truth, adjacency, degree, or k-means is used by this initializer.
The caller is responsible for the provenance of the fixed coordinates Y.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.special import gammaln, logsumexp, wrightomega
from poisson_em import PoissonGaussianConvolution, RATE_FLOOR, _floored_weights


def aligned_observation(model, rates):
    """Scalar log densities and posterior count moments for aligned (n,d) pairs."""
    rates = np.asarray(rates, dtype=float)
    if rates.shape != model.y.shape:
        raise ValueError('aligned rates must match Y')
    if np.any(rates < RATE_FLOOR) or np.any(rates > model.rate_cap):
        raise ValueError('rates outside parameter bounds')
    lr = np.log(rates)
    root = model.tau2 * wrightomega((model.y + .5) / model.tau2
                                  + lr - np.log(model.tau2)) - 1
    modes = np.maximum(0, np.ceil(root)).astype(np.int64)
    ns = modes[..., None] + model.offsets
    safe = np.maximum(ns, 0)
    lg = (model.gamma_table[safe] if model.gamma_table is not None
          and safe.max() <= model.gamma_cap else gammaln(safe + 1))
    terms = (safe * lr[..., None] - rates[..., None] - lg
             - (model.y[..., None] - safe)**2 / (2 * model.tau2)
             + model.normal_logconst)
    terms = np.where(ns >= 0, terms, -np.inf)
    lp = logsumexp(terms, axis=-1)
    p = np.exp(terms - lp[..., None])
    mean = np.sum(p * safe, axis=-1)
    var = np.sum(p * (safe - mean[..., None])**2, axis=-1)
    return lp, mean, var


def singleton_mle(model, steps=42):
    """Maximize each scalar log f(y;lambda), using its monotone score.

    score=m/lambda-1; curvature=(Var(N|y)-m)/lambda**2 <=0.
    Boundary solutions satisfy the one-sided KKT conditions.
    """
    lo = np.full(model.y.shape, RATE_FLOOR)
    hi = np.full(model.y.shape, model.rate_cap)
    _, ml, _ = aligned_observation(model, lo)
    _, mh, _ = aligned_observation(model, hi)
    floor = ml <= lo
    cap = mh >= hi
    for _ in range(steps):
        mid = (lo + hi) / 2
        _, m, _ = aligned_observation(model, mid)
        positive = m > mid
        lo = np.where(positive, mid, lo)
        hi = np.where(positive, hi, mid)
    rates = np.where(floor, RATE_FLOOR,
                     np.where(cap, model.rate_cap, (lo + hi) / 2))
    lp, mean, var = aligned_observation(model, rates)
    score = mean / rates - 1
    interior = ~(floor | cap)
    report = {'floor_coordinates': int(floor.sum()), 'cap_coordinates': int(cap.sum()),
              'interior_coordinates': int(interior.sum()), 'bisection_steps': steps,
              'interior_score_max_abs': float(np.abs(score[interior]).max(initial=0)),
              'floor_score_max': float(score[floor].max()) if floor.any() else None,
              'cap_score_min': float(score[cap].min()) if cap.any() else None,
              'variance_minus_mean_max': float((var-mean).max())}
    return rates, lp.sum(axis=1), report


class LikelihoodSeeder:
    def __init__(self, y, tau=1.):
        started = time.perf_counter()
        self.model = PoissonGaussianConvolution(y, tau)
        self.singletons, self.saturated, self.saturation_report = singleton_mle(self.model)
        self.precompute_elapsed = time.perf_counter() - started

    def regret(self, indices):
        lp, _ = self.model.evaluate(self.singletons[np.asarray(indices, dtype=int)])
        raw = self.saturated[:, None] - lp
        if raw.min() < -1e-7:
            raise FloatingPointError('singleton MLE does not dominate seed density')
        return np.maximum(raw, 0)

    def seed(self, k, rng, method='likelihood_pp', local_trials=1):
        """Select observation-index prototypes. Greedy trials cost extra density calls."""
        started = time.perf_counter()
        n = len(self.model.y)
        if not 1 <= k <= n or local_trials < 1:
            raise ValueError('invalid seed count/trials')
        ids = [int(rng.integers(n))]
        if method not in ('likelihood_pp', 'euclidean_pp', 'uniform'):
            raise ValueError(method)
        if method == 'euclidean_pp':
            closest = np.sum((self.model.y - self.model.y[ids[0]])**2, axis=1)
        else:
            closest = self.regret(ids)[:, 0]
        potentials = [float(closest.sum())]
        candidate_evaluations = 1
        while len(ids) < k:
            weights = closest.copy()
            weights[ids] = 0
            remaining = np.setdiff1d(np.arange(n), ids)
            if method == 'uniform' or weights.sum() <= 1e-12:
                candidates = rng.choice(remaining, size=local_trials, replace=True)
            else:
                candidates = rng.choice(n, size=local_trials, p=weights/weights.sum())
            if method == 'euclidean_pp':
                distance = np.sum((self.model.y[:, None, :]
                                   - self.model.y[candidates][None, :, :])**2, axis=2)
            else:
                distance = self.regret(candidates)
            improved = np.minimum(closest[:, None], distance)
            chosen = int(np.argmin(improved.sum(axis=0)))
            ids.append(int(candidates[chosen]))
            closest = improved[:, chosen]
            potentials.append(float(closest.sum()))
            candidate_evaluations += len(candidates)
        if len(set(ids)) != k or np.any(np.diff(potentials) > 1e-7):
            raise FloatingPointError('invalid ++ trajectory')
        rates = self.singletons[ids].copy()
        final_pg_potential = float(self.regret(ids).min(axis=1).sum())
        return rates, {'indices': ids, 'potential_trace': potentials,
                       'final_pg_regret': final_pg_potential,
                       'seed_elapsed': time.perf_counter()-started,
                       'local_trials': local_trials,
                       'candidate_evaluations': candidate_evaluations,
                       'sampling': method}


def fit_pg_rates(y, initial_rates, tau=1., max_iter=300, tol=1e-6,
                 initial_weights=None):
    """Same soft latent-count EM/stopping rule as fit_poisson_em, seeded by rates."""
    started = time.perf_counter()
    model = PoissonGaussianConvolution(y, tau)
    rates = np.asarray(initial_rates, dtype=float).copy()
    k = len(rates)
    weights = (_floored_weights(np.ones(k)) if initial_weights is None
               else _floored_weights(np.asarray(initial_weights, dtype=float)))
    lp, counts = model.evaluate(rates)
    scores = lp + np.log(weights)[None, :]
    norm = logsumexp(scores, axis=1)
    r = np.exp(scores - norm[:, None])
    labels = r.argmax(axis=1)
    objective = float(norm.sum())
    trace = [{'iteration': 0, 'objective': objective, 'n_changed': 0,
              'param_delta': 0., 'objective_delta': 0.,
              'class_min': int(np.bincount(labels, minlength=k).min()),
              'expected_class_min': float(r.sum(axis=0).min())}]
    converged = False
    worst = 0.
    for iteration in range(1, max_iter+1):
        mass = r.sum(axis=0)
        sums = np.einsum('na,nab->ab', r, counts, optimize=True)
        updated = rates.copy()
        active = mass > 0
        updated[active] = sums[active]/mass[active, None]
        updated = np.clip(updated, RATE_FLOOR, model.rate_cap)
        weights_new = _floored_weights(mass)
        pdelta = float(max(np.max(np.abs(updated-rates)/(1+np.abs(rates))),
                           np.max(np.abs(weights_new-weights))))
        lp, new_counts = model.evaluate(updated)
        scores = lp + np.log(weights_new)[None, :]
        norm = logsumexp(scores, axis=1)
        new_r = np.exp(scores-norm[:, None])
        new_labels = scores.argmax(axis=1)
        new_objective = float(norm.sum())
        delta = new_objective-objective
        worst = min(worst, delta)
        if delta < -1e-9*max(1., abs(objective)):
            raise FloatingPointError(f'EM likelihood decreased: {delta}')
        changed = int(np.count_nonzero(new_labels != labels))
        trace.append({'iteration': iteration, 'objective': new_objective,
                      'n_changed': changed, 'param_delta': pdelta,
                      'objective_delta': delta,
                      'class_min': int(np.bincount(new_labels, minlength=k).min()),
                      'expected_class_min': float(new_r.sum(axis=0).min())})
        old_objective = objective
        rates, weights, r, labels, counts, objective = (
            updated, weights_new, new_r, new_labels, new_counts, new_objective)
        if abs(delta) <= tol*(1+abs(old_objective)) and changed == 0 and pdelta <= max(np.sqrt(tol), 1e-5):
            converged = True
            break
    return {'labels': labels, 'rates': rates, 'weights': weights, 'r': r,
            'trace': trace, 'converged': converged,
            'reason': 'objective_and_parameter_tolerance' if converged else 'max_iter',
            'elapsed': time.perf_counter()-started, 'n_iter': len(trace)-1,
            'objective': objective, 'worst_objective_delta': worst,
            'objective_kind': 'mixture_log_likelihood',
            'numerical_support_report': model.support_report()}

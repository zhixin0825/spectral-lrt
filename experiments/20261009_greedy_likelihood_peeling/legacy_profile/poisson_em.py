"""Fixed-feature Poisson--Gaussian mixture EM.

The observation model is Y_ib = N_ib + Normal(0, tau**2), with independent
N_ib | class=a ~ Poisson(rates[a,b]).  It is a working likelihood for real
spectral features, including negative values; it is not a graph likelihood.
No adjacency matrix, degree vector, or changing feature calibration is used.

The infinite convolution is evaluated around its exact discrete posterior
mode.  Its omitted relative mass is bounded using discrete log concavity;
the Poisson mass is never renormalized on the numerical count grid.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp, wrightomega

RATE_FLOOR = 1e-6
WEIGHT_FLOOR = 1e-8


def _floored_weights(counts: np.ndarray, floor: float = WEIGHT_FLOOR) -> np.ndarray:
    """Exact argmax sum(counts * log(weights)) on the floored simplex."""
    counts = np.asarray(counts, dtype=float)
    k = len(counts)
    if not 0 <= floor < 1 / k or np.any(counts < 0):
        raise ValueError("Invalid simplex floor or class counts")
    if counts.sum() <= 0:
        return np.full(k, 1 / k)
    free = np.ones(k, dtype=bool)
    result = np.full(k, floor)
    for _ in range(k + 1):
        mass = 1 - floor * np.count_nonzero(~free)
        if counts[free].sum() <= 0:
            result[free] = mass / np.count_nonzero(free)
            break
        trial = counts[free] * (mass / counts[free].sum())
        below = trial < floor
        if not np.any(below):
            result[free] = trial
            break
        indices = np.flatnonzero(free)
        free[indices[below]] = False
    return result


class PoissonGaussianConvolution:
    """Reusable convolution evaluator for one fixed feature matrix."""

    def __init__(self, y: np.ndarray, tau: float = 1.0, *, window_sigma: float = 12.0,
                 rate_cap: float | None = None):
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim != 2 or not self.y.size or not np.all(np.isfinite(self.y)):
            raise ValueError("y must be a nonempty, finite, two-dimensional matrix")
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be positive and finite")
        if window_sigma < 8:
            raise ValueError("At least eight Gaussian standard deviations are required")
        self.tau = float(tau)
        self.tau2 = self.tau * self.tau
        self.rate_cap = float(rate_cap if rate_cap is not None
                              else max(1.0, float(self.y.max())) + 100.0)
        if not np.isfinite(self.rate_cap) or self.rate_cap <= RATE_FLOOR:
            raise ValueError("rate_cap must be finite and greater than RATE_FLOOR")
        self.radius = int(np.ceil(window_sigma * self.tau + 3))
        self.offsets = np.arange(-self.radius, self.radius + 1, dtype=np.int64)
        self.normal_logconst = -0.5 * np.log(2 * np.pi * self.tau2)
        self.gamma_cap = int(np.ceil(max(self.y.max(), self.rate_cap, 0))) + self.radius + 4
        self.gamma_table = (gammaln(np.arange(self.gamma_cap + 1, dtype=float) + 1)
                            if self.gamma_cap <= 1_000_000 else None)
        # If m is a mode, w(m+d)/w(m) <= exp[-d(d-1)/(2*tau**2)].
        # The same inequality applies on the left.  Sum the two tail bounds.
        tail_exponent = -self.radius * (self.radius + 1) / (2 * self.tau2)
        self.relative_tail_bound = float(
            2 * np.exp(tail_exponent) /
            (-np.expm1(-(self.radius + 1) / self.tau2)))
        self.last_mode_min = 0
        self.last_mode_max = 0

    def evaluate(self, rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return component log densities (n,K) and posterior counts (n,K,d)."""
        rates = np.asarray(rates, dtype=float)
        if rates.ndim != 2 or rates.shape[1] != self.y.shape[1]:
            raise ValueError("rates must have shape (number of classes, feature dimension)")
        if (not np.all(np.isfinite(rates)) or np.any(rates < RATE_FLOOR)
                or np.any(rates > self.rate_cap * (1 + 1e-12))):
            raise ValueError("rates are outside the declared parameter bounds")
        log_rates = np.log(rates)
        # w(k+1)/w(k) = rates/(k+1) * exp((y-k-1/2)/tau**2).
        # The real root k0 of this ratio=1 is found without exponent overflow:
        # k0+1 = tau**2 * WrightOmega((y+1/2)/tau**2 + log(rate/tau**2)).
        argument = ((self.y[:, None, :] + 0.5) / self.tau2
                    + log_rates[None, :, :] - np.log(self.tau2))
        root = self.tau2 * wrightomega(argument) - 1
        modes = np.maximum(0, np.ceil(root)).astype(np.int64)
        self.last_mode_min = int(modes.min())
        self.last_mode_max = int(modes.max())
        counts = modes[..., None] + self.offsets
        valid = counts >= 0
        counts_safe = np.maximum(counts, 0)
        if self.gamma_table is not None and counts_safe.max() <= self.gamma_cap:
            log_factorial = self.gamma_table[counts_safe]
        else:
            log_factorial = gammaln(counts_safe + 1)
        log_terms = (counts_safe * log_rates[None, :, :, None]
                     - rates[None, :, :, None] - log_factorial
                     - (self.y[:, None, :, None] - counts_safe) ** 2 / (2 * self.tau2)
                     + self.normal_logconst)
        log_terms = np.where(valid, log_terms, -np.inf)
        marginal_logpdf = logsumexp(log_terms, axis=-1)
        posterior = np.exp(log_terms - marginal_logpdf[..., None])
        posterior_counts = np.sum(posterior * counts_safe, axis=-1)
        return marginal_logpdf.sum(axis=-1), posterior_counts

    def support_report(self) -> dict[str, Any]:
        return {
            "method": "adaptive exact discrete posterior mode plus symmetric count window",
            "count_window_radius": self.radius,
            "relative_density_error_bound": self.relative_tail_bound,
            "log_density_error_bound": float(np.log1p(self.relative_tail_bound)),
            "mode_min_last_evaluation": self.last_mode_min,
            "mode_max_last_evaluation": self.last_mode_max,
            "poisson_grid_renormalized": False,
            "rate_floor": RATE_FLOOR,
            "rate_cap": self.rate_cap,
            "weight_floor": WEIGHT_FLOOR,
        }


def poisson_observation(y: np.ndarray, rates: np.ndarray, tau: float = 1.0,
                        *, window_sigma: float = 12.0,
                        rate_cap: float | None = None) -> dict[str, Any]:
    """Public single-evaluation interface, convenient for full-grid checks."""
    rates = np.asarray(rates, dtype=float)
    if rate_cap is None:
        rate_cap = max(max(1.0, float(np.max(y))) + 100.0,
                       float(np.max(rates)) + 1.0)
    model = PoissonGaussianConvolution(y, tau, window_sigma=window_sigma, rate_cap=rate_cap)
    logpdf, means = model.evaluate(rates)
    return {"component_logpdf": logpdf, "posterior_counts": means,
            "numerical_support_report": model.support_report()}


def fit_poisson_em(y: np.ndarray, labels_init: np.ndarray, tau: float = 1.0,
                   max_iter: int = 200, tol: float = 1e-6,
                   hard: bool = False) -> dict[str, Any]:
    """Fit a fixed-feature smoothed-Poisson mixture by EM or classification EM.

    Component labels are integer IDs 0,...,K-1; every initial component must
    occur. The M steps exactly respect the declared rate and simplex bounds.
    Hard EM maximizes classified likelihood including class weights; soft EM
    maximizes observed mixture likelihood. Every trace includes iteration 0.
    A material objective decrease raises FloatingPointError.
    """
    started = time.perf_counter()
    y = np.asarray(y, dtype=float)
    initial = np.asarray(labels_init)
    if initial.ndim != 1 or len(initial) != len(y):
        raise ValueError("labels_init must have one entry per observation")
    if not np.issubdtype(initial.dtype, np.integer):
        if np.any(initial != np.round(initial)):
            raise ValueError("labels_init must contain integer class IDs")
        initial = initial.astype(int)
    labels = initial.astype(int, copy=True)
    if labels.min() < 0:
        raise ValueError("Initial labels must be nonnegative")
    k = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=k).astype(float)
    if np.any(sizes == 0):
        raise ValueError("Every initial component must be present")
    if max_iter < 0 or tol <= 0:
        raise ValueError("max_iter must be nonnegative and tol positive")
    model = PoissonGaussianConvolution(y, tau)
    rates = np.vstack([y[labels == a].mean(axis=0) for a in range(k)])
    rates = np.clip(rates, RATE_FLOOR, model.rate_cap)
    weights = _floored_weights(sizes)
    component_logpdf, posterior_counts = model.evaluate(rates)
    scores = component_logpdf + np.log(weights)[None, :]
    if hard:
        r = np.eye(k)[labels]
        objective = float(scores[np.arange(len(y)), labels].sum())
    else:
        lognorm = logsumexp(scores, axis=1)
        r = np.exp(scores - lognorm[:, None])
        objective = float(lognorm.sum())
        labels = r.argmax(axis=1)
    trace = [{"iteration": 0, "objective": objective, "n_changed": 0,
              "param_delta": 0.0, "class_min": int(np.bincount(labels, minlength=k).min()),
              "expected_class_min": float(r.sum(axis=0).min()), "objective_delta": 0.0}]
    converged = False
    reason = "max_iter"
    worst_objective_delta = 0.0
    for iteration in range(1, max_iter + 1):
        mass = r.sum(axis=0)
        weighted_count_sums = np.einsum("na,nab->ab", r, posterior_counts, optimize=True)
        new_rates = rates.copy()
        nonempty = mass > 0
        new_rates[nonempty] = weighted_count_sums[nonempty] / mass[nonempty, None]
        new_rates = np.clip(new_rates, RATE_FLOOR, model.rate_cap)
        new_weights = _floored_weights(mass)
        param_delta = float(max(np.max(np.abs(new_rates - rates) / (1 + np.abs(rates))),
                                np.max(np.abs(new_weights - weights))))
        new_logpdf, new_posterior_counts = model.evaluate(new_rates)
        new_scores = new_logpdf + np.log(new_weights)[None, :]
        new_labels = new_scores.argmax(axis=1)
        if hard:
            new_r = np.eye(k)[new_labels]
            new_objective = float(new_scores[np.arange(len(y)), new_labels].sum())
        else:
            lognorm = logsumexp(new_scores, axis=1)
            new_r = np.exp(new_scores - lognorm[:, None])
            new_objective = float(lognorm.sum())
        delta = new_objective - objective
        worst_objective_delta = min(worst_objective_delta, delta)
        numerical_slack = 1e-9 * max(1.0, abs(objective))
        if delta < -numerical_slack:
            raise FloatingPointError(
                f"Likelihood decreased at iteration {iteration}: {delta:.12g}; "
                f"objective={objective:.12g}")
        changed = int(np.count_nonzero(new_labels != labels))
        trace.append({"iteration": iteration, "objective": new_objective,
                      "n_changed": changed, "param_delta": param_delta,
                      "class_min": int(np.bincount(new_labels, minlength=k).min()),
                      "expected_class_min": float(new_r.sum(axis=0).min()),
                      "objective_delta": float(delta)})
        rates, weights, r, labels = new_rates, new_weights, new_r, new_labels
        component_logpdf, posterior_counts = new_logpdf, new_posterior_counts
        old_objective = objective
        objective = new_objective
        if (abs(delta) <= tol * (1 + abs(old_objective)) and changed == 0
                and param_delta <= max(np.sqrt(tol), 1e-5)):
            converged = True
            reason = "objective_and_parameter_tolerance"
            break
    return {
        "labels": labels, "rates": rates, "weights": weights, "r": r,
        "trace": trace, "converged": converged, "reason": reason,
        "elapsed": time.perf_counter() - started,
        "n_iter": len(trace) - 1, "objective": objective,
        "objective_kind": "classification_log_likelihood" if hard else "mixture_log_likelihood",
        "worst_objective_delta": worst_objective_delta,
        "numerical_support_report": model.support_report(),
    }


"""Poisson degree / conditional Gaussian mixture initialization.

The only observations are D_i and T_i, where T_i is a transformed average
of the spectral features of node i's neighbours.  The working model is

    Z_i ~ Categorical(pi)
    D_i | Z_i=a ~ Poisson(delta_a)
    T_i | D_i>0, Z_i=a ~ N(mu_a, Sigma_a / D_i).

At D_i=0, T_i is deterministic/undefined and contributes NO Gaussian term.
This is an explicit composite likelihood approximation to an SBM, not the
Bernoulli graph likelihood.  The fitting routine never accesses A or labels.

Each EM covariance update solves its constrained M-step exactly under
Sigma_a >= reg_covar * I by clipping its eigenvalues from below.  Mixture
weights and degree rates have no artificial positive floors during EM;
zero-weight / zero-rate components are valid boundary cases.  Initial
centres come from data-point sampling or k-means++ sampling, without Lloyd
iterations or an initial k-means partition.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy


def _floor_covariance(cov: np.ndarray, floor: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh((cov + cov.T) * 0.5)
    return (vecs * np.maximum(vals, floor)) @ vecs.T


def _seed_centres(T, D, k, rng, mode):
    positive = np.flatnonzero(D > 0)
    if mode == "random_data":
        return rng.choice(positive, size=k, replace=len(positive) < k)
    # Precision-weighted k-means++ seeding only.  There is no Lloyd step.
    # D-weighting makes isolated, noisy low-degree averages less dominant.
    point_weights = D[positive].astype(float)
    point_weights /= point_weights.sum()
    first = int(rng.choice(len(positive), p=point_weights))
    selected = [first]
    closest = np.sum((T[positive] - T[positive[first]]) ** 2, axis=1)
    for _ in range(1, k):
        probabilities = point_weights * closest
        total = probabilities.sum()
        if not np.isfinite(total) or total <= np.finfo(float).tiny:
            # Duplicate feature rows are legitimate in sparse graphs.
            remaining = np.setdiff1d(np.arange(len(positive)), selected)
            nxt = int(rng.choice(remaining)) if len(remaining) else int(
                rng.choice(len(positive), p=point_weights)
            )
        else:
            nxt = int(rng.choice(len(positive), p=probabilities / total))
        selected.append(nxt)
        closest = np.minimum(
            closest,
            np.sum((T[positive] - T[positive[nxt]]) ** 2, axis=1),
        )
    return positive[np.asarray(selected)]


def _log_scores(T, D, weights, rates, means, covs):
    n, p = T.shape
    k = len(weights)
    log_weights = np.full(k, -np.inf)
    np.log(weights, out=log_weights, where=weights > 0)
    scores = (
        log_weights[None, :]
        + xlogy(D[:, None], rates[None, :])
        - rates[None, :]
        - gammaln(D[:, None] + 1)
    )
    positive = D > 0
    if np.any(positive):
        Tp, Dp = T[positive], D[positive]
        common = 0.5 * p * (np.log(Dp) - np.log(2 * np.pi))
        for a in range(k):
            if weights[a] == 0 or rates[a] == 0:
                continue
            vals, vecs = np.linalg.eigh(covs[a])
            # covs are constrained positive definite by construction.
            if np.min(vals) <= 0:
                raise FloatingPointError("Nonpositive covariance eigenvalue")
            rotated = (Tp - means[a]) @ vecs
            mahal = np.sum(rotated * rotated / vals, axis=1)
            scores[positive, a] += (
                common - 0.5 * np.log(vals).sum() - 0.5 * Dp * mahal
            )
    return scores


def _e_step(T, D, params):
    scores = _log_scores(T, D, *params)
    normalizers = logsumexp(scores, axis=1)
    if not np.all(np.isfinite(normalizers)):
        raise FloatingPointError("Nonfinite mixture log-likelihood")
    return float(normalizers.sum()), np.exp(scores - normalizers[:, None])


def _m_step(T, D, responsibilities, old_params, reg_covar):
    n, p = T.shape
    weights_old, rates_old, means_old, covs_old = old_params
    k = len(weights_old)
    mass = responsibilities.sum(axis=0)
    degree_mass = responsibilities.T @ D
    weights = mass / n
    rates = rates_old.copy()
    np.divide(degree_mass, mass, out=rates, where=mass > 0)
    means = means_old.copy()
    numerator = responsibilities.T @ (D[:, None] * T)
    np.divide(
        numerator,
        degree_mass[:, None],
        out=means,
        where=degree_mass[:, None] > 0,
    )
    positive = D > 0
    positive_mass = responsibilities[positive].sum(axis=0)
    covs = covs_old.copy()
    for a in range(k):
        if positive_mass[a] > 0 and degree_mass[a] > 0:
            diff = T[positive] - means[a]
            precision_weight = responsibilities[positive, a] * D[positive]
            raw = (diff.T * precision_weight) @ diff / positive_mass[a]
            covs[a] = _floor_covariance(raw, reg_covar)
    return weights, rates, means, covs


def _fit_one(T, D, k, seed, max_iter, reg_covar, tol, mode):
    n, p = T.shape
    rng = np.random.default_rng(seed)
    selected = _seed_centres(T, D, k, rng, mode)
    positive = D > 0
    pooled_mean = np.average(T[positive], axis=0, weights=D[positive])
    diff = T[positive] - pooled_mean
    pooled_cov = _floor_covariance(
        (diff.T * D[positive]) @ diff / positive.sum(), reg_covar
    )
    params = (
        np.full(k, 1.0 / k),
        # Positive, mildly jittered rates only initialize the optimizer.
        np.maximum(D[selected], 0.1) * np.exp(rng.normal(0, 0.15, size=k)),
        T[selected].copy(),
        np.repeat(pooled_cov[None, :, :], k, axis=0),
    )
    objective, responsibilities = _e_step(T, D, params)
    history = [objective]
    converged = False
    status = "max_iter"
    numerical_rejections = 0
    n_iter = 0
    for iteration in range(1, max_iter + 1):
        new_params = _m_step(T, D, responsibilities, params, reg_covar)
        new_objective, new_responsibilities = _e_step(T, D, new_params)
        improvement = new_objective - objective
        roundoff = 1e-10 * (1.0 + abs(objective))
        if improvement < 0:
            # Never retain a decreasing iterate.  A substantial decline
            # indicates a numerical failure, not successful convergence.
            numerical_rejections += 1
            converged = improvement >= -roundoff
            status = "numerical_plateau" if converged else "numerical_decrease"
            break
        params, responsibilities = new_params, new_responsibilities
        old_objective = objective
        objective = new_objective
        history.append(objective)
        n_iter = iteration
        if improvement <= tol * (1.0 + abs(old_objective)):
            converged = True
            status = "converged"
            break
    weights, rates, means, covs = params
    labels = np.argmax(responsibilities, axis=1)
    return {
        "labels": labels,
        "objective": objective,
        "objective_per_node": objective / n,
        "n_iter": n_iter,
        "converged": converged,
        "status": status,
        "weights": weights,
        "degree_rates": rates,
        "means": means,
        "covariances": covs,
        "responsibilities": responsibilities,
        "objective_history": history,
        "initialization": mode,
        "initial_seed_indices": selected,
        "seed": int(seed),
        "numerical_rejections": numerical_rejections,
        "effective_components": int(np.count_nonzero(weights > 1e-8)),
        "empty_hard_clusters": k - len(np.unique(labels)),
    }


def fit_poisson_gaussian(
    T,
    D,
    k,
    seed,
    n_init=8,
    max_iter=150,
    reg_covar=0.02,
    tol=1e-5,
):
    """Fit a degree-Poisson / conditional-Gaussian mixture by multistart EM.

    Parameters
    ----------
    T : (n,p) array
        Neighbourhood averages, already globally transformed/whitened by
        the caller.  The values of T at zero-degree nodes are ignored.
    D : (n,) array
        Nonnegative integer degrees.
    k : int
        Known number of mixture components.
    seed : int
        Master seed. Independent reproducible initialization seeds are
        generated with numpy.random.SeedSequence.spawn.
    n_init, max_iter : int
        Number of starts and maximum EM iterations per start.
    reg_covar : float
        Strictly positive lower bound on each covariance eigenvalue.
        The constrained covariance M-step is exact for this objective.
    tol : float
        Relative observed-likelihood improvement tolerance.

    Returns
    -------
    dict
        Best run selected solely by its model likelihood, including labels,
        objective, n_iter, converged, parameters, and per-start diagnostics.
        No true labels, adjacency entries, or external initial partition
        are consumed by this function.
    """
    T = np.asarray(T, dtype=float)
    D = np.asarray(D, dtype=float)
    if T.ndim != 2 or D.ndim != 1 or len(D) != len(T):
        raise ValueError("T must be n x p and D must be length n")
    n, p = T.shape
    if n == 0 or p == 0 or not (1 <= k <= n) or int(k) != k:
        raise ValueError("Require n,p >= 1 and integer 1 <= k <= n")
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Degrees must be finite and nonnegative")
    if not np.allclose(D, np.rint(D), rtol=0, atol=1e-10):
        raise ValueError("Degrees must be integer-valued")
    if not np.all(np.isfinite(T[D > 0])):
        raise ValueError("Positive-degree feature rows must be finite")
    if n_init < 1 or int(n_init) != n_init or max_iter < 1 or int(max_iter) != max_iter:
        raise ValueError("n_init and max_iter must be positive integers")
    if reg_covar <= 0 or not np.isfinite(reg_covar) or tol < 0 or not np.isfinite(tol):
        raise ValueError("Require finite reg_covar > 0 and tol >= 0")
    T = T.copy()
    T[D == 0] = 0.0
    k, n_init, max_iter = int(k), int(n_init), int(max_iter)
    if not np.any(D > 0):
        return {
            "labels": np.zeros(n, dtype=int),
            "objective": 0.0,
            "objective_per_node": 0.0,
            "n_iter": 0,
            "converged": True,
            "status": "all_isolates",
            "weights": np.full(k, 1.0 / k),
            "degree_rates": np.zeros(k),
            "means": np.zeros((k, p)),
            "covariances": np.repeat((reg_covar * np.eye(p))[None], k, axis=0),
            "responsibilities": np.full((n, k), 1.0 / k),
            "objective_history": [0.0],
            "initialization": "none",
            "seed": int(seed),
            "n_init": n_init,
            "runs": [],
            "numerical_rejections": 0,
            "effective_components": k,
            "empty_hard_clusters": k - 1,
            "covariance_eigenvalue_floor": float(reg_covar),
        }
    child_sequences = np.random.SeedSequence(int(seed)).spawn(n_init)
    best = None
    runs = []
    for run, child in enumerate(child_sequences):
        child_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        mode = "weighted_kmeans++_seeding" if run % 2 == 0 else "random_data"
        result = _fit_one(
            T, D, k, child_seed, max_iter, reg_covar, tol, mode
        )
        runs.append({key: result[key] for key in (
            "objective", "n_iter", "converged", "status", "initialization",
            "seed", "effective_components", "empty_hard_clusters",
            "numerical_rejections",
        )})
        if best is None or result["objective"] > best["objective"]:
            best = result
            best["best_run"] = run
    best["runs"] = runs
    best["n_init"] = n_init
    best["master_seed"] = int(seed)
    best["covariance_eigenvalue_floor"] = float(reg_covar)
    return best

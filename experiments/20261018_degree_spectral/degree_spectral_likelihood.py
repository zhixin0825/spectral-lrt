"""Degree-augmented spectral Bernoulli working-score decoder.

The only retained observations are degrees D and signed eigenpairs (U, Lambda)
of W A W, where w_i=min(1, 2 mean(D)/D_i), with w_i=1 for D_i=0.
The exact retained action is A X=Y for X=[1,WU], Y=[D,W^{-1}U Lambda].
Its minimum-Frobenius-norm extension is R=Y pinv(X)=A P_X. R is generally
signed and asymmetric. It is never clipped or symmetrized: R 1=D remains exact.
Only fitted Bernoulli parameters are clipped to a compact interval.

This is a reference implementation of a working criterion, not an independent
Bernoulli likelihood for R, or a claim of a sparse sharp-rate theorem. It uses
quadratic storage and cubic candidate preprocessing; no performance claims are
made. Labels are zero based. Deterministic ties select the first candidate,
and an incumbent fit survives objective ties in replacement rounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def degree_weights(degrees):
    """Return the prescribed diagonal of W, including the all-isolated case."""
    d = np.asarray(degrees, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("degrees must be a finite nonnegative vector, n>=2")
    w = np.ones_like(d)
    positive = d > 0
    w[positive] = np.minimum(1.0, 2.0 * d.mean() / d[positive])
    return w


def reconstruct_degree_spectrum(degrees, eigenvectors, eigenvalues, rcond=1e-12):
    """Recover A P_X using only the retained degrees and weighted eigenpairs.

    The Moore-Penrose inverse also covers dependent feature columns, for example
    a regular graph whose retained eigenspace already contains the constant.
    Consistency of Y with X is checked numerically; no adjacency input is read.
    """
    d = np.asarray(degrees, dtype=float)
    w = degree_weights(d)
    u = np.asarray(eigenvectors, dtype=float)
    lam = np.asarray(eigenvalues, dtype=float)
    if u.ndim != 2 or u.shape[0] != len(d) or lam.shape != (u.shape[1],):
        raise ValueError("expected U.shape=(n,k) and eigenvalues.shape=(k,)")
    if not 1 <= u.shape[1] <= len(d) or not np.isfinite(u).all() or not np.isfinite(lam).all():
        raise ValueError("invalid finite spectral inputs")
    if not np.allclose(u.T @ u, np.eye(u.shape[1]), rtol=1e-8, atol=1e-9):
        raise ValueError("the retained eigenvectors must be orthonormal")
    x = np.column_stack((np.ones(len(d)), w[:, None] * u))
    y = np.column_stack((d, (u * lam[None, :]) / w[:, None]))
    r = y @ np.linalg.pinv(x, rcond=rcond)
    residual = r @ x - y
    rel = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(y)))
    if rel > 1e-7:
        raise ValueError("inconsistent retained actions: Y must annihilate ker(X)")
    return r, {
        "feature_rank": int(np.linalg.matrix_rank(x, tol=rcond * np.linalg.norm(x, 2))),
        "feature_columns": int(x.shape[1]),
        "action_relative_residual": rel,
        "degree_max_absolute_residual": float(np.max(np.abs(r.sum(axis=1) - d))),
        "min_reconstructed_entry": float(r.min()),
        "max_reconstructed_entry": float(r.max()),
        "max_absolute_asymmetry": float(np.max(np.abs(r - r.T))),
    }


def row_scores(reconstruction, profiles):
    """Complete Bernoulli row scores, valid for signed fractional observations."""
    r = np.asarray(reconstruction, dtype=float)
    mu = np.asarray(profiles, dtype=float)
    if r.ndim != 2 or mu.ndim != 2 or r.shape[1] != mu.shape[1]:
        raise ValueError("R and profiles must have the same number of columns")
    if not np.isfinite(r).all() or not np.isfinite(mu).all() or (mu <= 0).any() or (mu >= 1).any():
        raise ValueError("R must be finite and fitted parameters lie in (0,1)")
    # No Bernoulli KL expression: KL is undefined when entries of R are signed.
    return r @ (np.log(mu) - np.log1p(-mu)).T + np.log1p(-mu).sum(axis=1)[None, :]


def _evaluate(r, profiles, weights):
    with np.errstate(divide="ignore"):
        logits = row_scores(r, profiles) + np.log(weights)[None, :]
    normalizers = logsumexp(logits, axis=1)
    responsibilities = np.exp(logits - normalizers[:, None])
    return float(normalizers.sum()), responsibilities, logits.argmax(axis=1)


def fit_profile_mixture(reconstruction, initial_profiles, epsilon, *,
                        initial_weights=None, max_iter=25, tol=1e-10):
    """Perform at least one exact E/M update of the bounded working objective.

    The signed weighted-mean coordinate maximizer is clipped to [eps,1-eps].
    If its unbounded value lies below zero or above one, the objective is
    monotone on the parameter interval and the corresponding endpoint is exact.
    A component with zero mass retains its previous profile and has zero weight.
    """
    r = np.asarray(reconstruction, dtype=float)
    mu = np.array(initial_profiles, dtype=float, copy=True)
    if not 0 < epsilon < 0.5 or int(max_iter) != max_iter or max_iter < 1 or tol < 0:
        raise ValueError("require 0<epsilon<1/2, max_iter>=1 and tol>=0")
    if mu.ndim != 2 or r.ndim != 2 or mu.shape[1] != r.shape[1] or len(mu) < 1:
        raise ValueError("invalid reconstruction/profile shapes")
    if not np.isfinite(mu).all() or not np.isfinite(r).all():
        raise ValueError("profiles and R must be finite")
    mu = np.clip(mu, epsilon, 1.0 - epsilon)
    k = len(mu)
    weights = np.ones(k) / k if initial_weights is None else np.array(initial_weights, dtype=float, copy=True)
    if weights.shape != (k,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be a finite nonnegative vector with positive mass")
    weights /= weights.sum()
    objective, resp, labels = _evaluate(r, mu, weights)
    trace = [objective]
    for _ in range(int(max_iter)):
        mass = resp.sum(axis=0)
        active = mass > 0
        new_mu = mu.copy()
        new_mu[active] = np.clip((resp[:, active].T @ r) / mass[active, None], epsilon, 1.0 - epsilon)
        new_weights = mass / len(r)
        new_objective, new_resp, new_labels = _evaluate(r, new_mu, new_weights)
        delta = new_objective - objective
        if delta < -1e-9 * max(1.0, abs(objective)):
            raise FloatingPointError(f"exact E/M decreased the working objective by {delta}")
        trace.append(new_objective)
        scale = 1.0 + abs(objective)
        mu, weights, objective, resp, labels = new_mu, new_weights, new_objective, new_resp, new_labels
        if abs(delta) <= tol * scale:
            break
    return {"profiles": mu, "weights": weights, "responsibilities": resp,
            "labels": labels, "objective": objective, "objective_trace": trace}


def degree_spectral_likelihood(degrees, eigenvectors, eigenvalues, *, c0=0.02,
                               max_em_iter=25, tol=1e-10, replacement_rounds=None):
    """Fit K communities from D and K signed eigenpairs of W A W alone."""
    if not 0 < c0 < 1.0 / 8.0:
        raise ValueError("c0 must lie in (0,1/8)")
    r, reconstruction_metadata = reconstruct_degree_spectrum(degrees, eigenvectors, eigenvalues)
    n, k = len(r), len(eigenvalues)
    eps = c0 * max(1.0, float(np.mean(degrees))) / n
    if eps >= 0.5:
        raise ValueError("degree-derived fitted-parameter interval is empty")
    rounds = int(np.ceil(np.log(n))) if replacement_rounds is None else replacement_rounds
    if int(rounds) != rounds or rounds < 0:
        raise ValueError("replacement_rounds must be a nonnegative integer")
    candidates = np.clip(r, eps, 1.0 - eps)
    candidate_scores = row_scores(r, candidates)
    initial = np.clip(r.mean(axis=0, keepdims=True), eps, 1.0 - eps)
    fit = fit_profile_mixture(r, initial, eps, max_iter=max_em_iter, tol=tol)
    growing, replacement, chosen = [], [], []
    for components in range(2, k + 1):
        hard_baseline = row_scores(r, fit["profiles"]).max(axis=1)
        gains = np.maximum(candidate_scores - hard_baseline[:, None], 0).sum(axis=0)
        gains[chosen] = -np.inf
        node = int(np.argmax(gains))
        chosen.append(node)
        initial = np.vstack((fit["profiles"], candidates[node]))
        fit = fit_profile_mixture(r, initial, eps, max_iter=max_em_iter, tol=tol)
        growing.append({"components": components, "node": node, "gain": float(gains[node]),
                        "objective_trace": fit["objective_trace"]})
    # K=1 has no meaningful deletion baseline and is already exactly fitted.
    for round_id in range(int(rounds) if k > 1 else 0):
        incumbent = fit
        selected = incumbent
        trials = []
        for deleted in range(k):
            hard_baseline = row_scores(r, np.delete(incumbent["profiles"], deleted, axis=0)).max(axis=1)
            gains = np.maximum(candidate_scores - hard_baseline[:, None], 0).sum(axis=0)
            node = int(np.argmax(gains))
            initial = incumbent["profiles"].copy()
            initial[deleted] = candidates[node]
            trial = fit_profile_mixture(r, initial, eps, max_iter=max_em_iter, tol=tol)
            trials.append({"deleted": deleted, "node": node, "gain": float(gains[node]),
                           "objective_trace": trial["objective_trace"]})
            if trial["objective"] > selected["objective"]:
                selected = trial
        fit = selected
        replacement.append({"round": round_id + 1, "before": incumbent["objective"],
                            "after": fit["objective"], "trials": trials})
        if selected is incumbent:
            break
    # Mandatory selected-endpoint E step and exact M step precede classification.
    final = fit_profile_mixture(r, fit["profiles"], eps, initial_weights=fit["weights"], max_iter=1, tol=0)
    final.update({"epsilon": eps, "growing": growing, "replacement": replacement,
                  "reconstruction_metadata": reconstruction_metadata,
                  "inputs": "degrees and signed eigenpairs of W A W only",
                  "criterion": "complete Bernoulli working score of signed A P_X"})
    return final


class DegreeSpectralLikelihood:
    """Public decoder facade retaining only degrees and signed eigenpairs."""

    def __init__(self, degrees, eigenvectors, eigenvalues, *, c0=0.02):
        self.degrees = np.array(degrees, dtype=float, copy=True)
        self.eigenvectors = np.array(eigenvectors, dtype=float, copy=True)
        self.eigenvalues = np.array(eigenvalues, dtype=float, copy=True)
        self.c0 = c0

    def reconstruct(self):
        """Return the unmodified signed, generally asymmetric matrix A P_X."""
        return reconstruct_degree_spectrum(self.degrees, self.eigenvectors, self.eigenvalues)[0]

    def decode(self, *, max_em_iter=25, tol=1e-10, replacement_rounds=None):
        return degree_spectral_likelihood(self.degrees, self.eigenvectors, self.eigenvalues,
                                          c0=self.c0, max_em_iter=max_em_iter,
                                          tol=tol, replacement_rounds=replacement_rounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ containing D, U, Lambda only")
    parser.add_argument("output", type=Path, help="output NPZ for labels, profiles, weights")
    parser.add_argument("--em-steps", type=int, default=25)
    parser.add_argument("--c0", type=float, default=0.02)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        result = degree_spectral_likelihood(data["D"], data["U"], data["Lambda"],
                                            max_em_iter=args.em_steps, c0=args.c0)
    arrays = {name: result.pop(name) for name in ("labels", "profiles", "weights", "responsibilities")}
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

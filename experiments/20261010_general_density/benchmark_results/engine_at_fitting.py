"""Spectral likelihood decoding with deterministic parameter replacement.

Only the K retained adjacency eigenpairs are inputs. The default score keeps
both Bernoulli terms. After global-mean growing, at most ceil(log(n)) rounds
test one replacement for each component, selected by the all-node likelihood
gain. Each trial starts with uniform weights and uses exact EM updates.
The final E/M update is mandatory. No restart sampling or labels are used.

The ``sparse`` score is a controlled comparison on the same profiles; it is
not the default general-density decoder. Historical experiments are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class Fit:
    profiles: np.ndarray
    weights: np.ndarray
    labels: np.ndarray
    loss: float
    trace: list[float]


class SpectralLikelihood:
    def __init__(self, eigenvectors, eigenvalues, *, floor=1e-4,
                 score_family="bernoulli"):
        self.u = np.asarray(eigenvectors, dtype=float)
        self.lam = np.asarray(eigenvalues, dtype=float)
        if self.u.ndim != 2 or self.lam.shape != (self.u.shape[1],):
            raise ValueError("Supply an n by K eigenvector array and K eigenvalues.")
        self.n, self.k = self.u.shape
        if self.n < 2 or not 1 <= self.k <= self.n:
            raise ValueError("Require n >= 2 and 1 <= K <= n.")
        if not np.isfinite(self.u).all() or not np.isfinite(self.lam).all():
            raise ValueError("Eigenpairs must be finite.")
        if not 0 < floor < 0.25:
            raise ValueError("The floor coefficient must lie in (0, 1/4).")
        if score_family not in {"bernoulli", "sparse"}:
            raise ValueError("Unknown score family.")
        self.score_family = score_family
        self.spectral_scale = max(1.0, float(np.max(np.abs(self.lam))))
        self.epsilon = floor * self.spectral_scale / self.n
        if self.epsilon >= 0.5:
            raise ValueError("The supplied eigenpairs do not give a valid clipping interval.")
        self.q = np.clip((self.u * self.lam) @ self.u.T,
                         self.epsilon, 1.0 - self.epsilon)
        if score_family == "bernoulli":
            self.entropy = (self.q * np.log(self.q)
                            + (1.0 - self.q) * np.log1p(-self.q)).sum(axis=1)
        else:
            self.entropy = (self.q * np.log(self.q) - self.q).sum(axis=1)
        self._candidate_divergences = None

    def scores(self, profiles):
        profiles = np.asarray(profiles)
        if self.score_family == "bernoulli":
            return (self.q @ (np.log(profiles) - np.log1p(-profiles)).T
                    + np.log1p(-profiles).sum(axis=1)[None, :])
        return self.q @ np.log(profiles).T - profiles.sum(axis=1)[None, :]

    def divergences(self, profiles):
        # The exact divergences are nonnegative; remove roundoff at zero.
        return np.maximum(self.entropy[:, None] - self.scores(profiles), 0.0)

    def candidate_divergences(self):
        if self._candidate_divergences is None:
            self._candidate_divergences = self.divergences(self.q)
        return self._candidate_divergences

    def fit(self, initial_profiles, *, weights=None, max_iter=100, tol=1e-9):
        if max_iter < 1:
            raise ValueError("A fit must include at least one M-step.")
        profiles = np.array(initial_profiles, dtype=float, copy=True)
        count = len(profiles)
        if count == 0 or profiles.shape != (count, self.n):
            raise ValueError("Each component profile must have n entries.")
        if not ((profiles >= self.epsilon).all()
                and (profiles <= 1 - self.epsilon).all()):
            raise ValueError("Initial profiles must lie in the retained profile box.")
        weights = (np.full(count, 1.0 / count) if weights is None
                   else np.array(weights, dtype=float, copy=True))
        if weights.shape != (count,) or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Weights must belong to the simplex.")
        trace = []
        previous_labels = None
        for iteration in range(max_iter + 1):
            log_weights = np.full(count, -np.inf)
            np.log(weights, out=log_weights, where=weights > 0)
            log_joint = -self.divergences(profiles) + log_weights[None, :]
            log_normal = logsumexp(log_joint, axis=1)
            responsibility = np.exp(log_joint - log_normal[:, None])
            labels = np.argmax(log_joint, axis=1)
            loss = -float(log_normal.sum())
            if trace and loss > trace[-1] + 1e-7 * self.n:
                raise FloatingPointError("An exact EM update increased soft deviance.")
            trace.append(loss)
            settled = (iteration > 0 and np.array_equal(labels, previous_labels)
                       and abs(trace[-2] - loss) <= tol * self.n)
            if iteration == max_iter or settled:
                return Fit(profiles, weights, labels, loss, trace)
            previous_labels = labels.copy()
            mass = responsibility.sum(axis=0)
            active = mass > 0
            profiles[active] = responsibility[:, active].T @ self.q / mass[active, None]
            # A zero-mass component retains its previous, feasible profile.
            # Clipping below only corrects floating-point convex-combination error.
            np.clip(profiles, self.epsilon, 1 - self.epsilon, out=profiles)
            weights = mass / mass.sum()
        raise AssertionError("Unreachable fitting state.")

    def global_candidate(self, retained_profiles, *, excluded=()):
        current = self.divergences(retained_profiles).min(axis=1)
        totals = np.minimum(current[:, None], self.candidate_divergences()).sum(axis=0)
        if len(excluded):
            totals[np.asarray(excluded, dtype=int)] = np.inf
        candidate = int(np.argmin(totals))
        return candidate, float(totals[candidate])

    def growing(self, *, max_iter=100):
        profiles = self.q.mean(axis=0, keepdims=True)
        if self.k == 1:
            return self.fit(profiles, max_iter=max_iter), []
        selected = []
        records = []
        for count in range(2, self.k + 1):
            candidate, hard_loss = self.global_candidate(profiles, excluded=selected)
            selected.append(candidate)
            fitted = self.fit(np.vstack([profiles, self.q[candidate]]), max_iter=max_iter)
            profiles = fitted.profiles
            records.append({"components": count, "candidate": candidate,
                            "initial_hard_loss": hard_loss, "final_soft_loss": fitted.loss,
                            "em_loss_trace": fitted.trace})
        return fitted, records

    def replace(self, fitted, *, rounds=None, trial_iter=1):
        rounds = math.ceil(math.log(self.n)) if rounds is None else int(rounds)
        if rounds < 0 or trial_iter < 1:
            raise ValueError("Require nonnegative rounds and at least one trial M-step.")
        if self.k == 1:
            return fitted, []
        records = []
        for step in range(rounds):
            before = fitted
            best = before
            accepted = None
            trials = []
            for removed in range(self.k):
                retained = np.delete(before.profiles, removed, axis=0)
                candidate, hard_loss = self.global_candidate(retained)
                initial = before.profiles.copy()
                initial[removed] = self.q[candidate]
                # Uniform trial weights give F_trial <= G_trial + n log K
                # before fitting, irrespective of the incumbent weights.
                trial = self.fit(initial, max_iter=trial_iter, tol=0.0)
                trials.append({"removed": removed, "candidate": candidate,
                               "initial_hard_loss": hard_loss,
                               "final_soft_loss": trial.loss, "em_loss_trace": trial.trace})
                if trial.loss < best.loss:
                    best = trial
                    accepted = {"removed": removed, "candidate": candidate}
            fitted = best
            records.append({"round": step + 1, "before_soft_loss": before.loss,
                            "after_soft_loss": fitted.loss, "accepted": accepted,
                            "trials": trials})
            if accepted is None:
                # Repeating these deterministic trials on the unchanged fit
                # gives the same result, so this is equivalent to the full budget.
                break
        return fitted, records

    def decode(self, *, growing_iter=100, trial_iter=1, rounds=None):
        grown, growing_records = self.growing(max_iter=growing_iter)
        replaced, replacement_records = self.replace(grown, rounds=rounds,
                                                      trial_iter=trial_iter)
        final = self.fit(replaced.profiles, weights=replaced.weights,
                         max_iter=1, tol=0.0)
        return final, {"score_family": self.score_family,
                       "epsilon": self.epsilon,
                       "spectral_scale": self.spectral_scale,
                       "round_budget": math.ceil(math.log(self.n)) if rounds is None else rounds,
                       "growing_loss": grown.loss,
                       "selected_loss": replaced.loss,
                       "final_loss": final.loss,
                       "growing_labels": grown.labels,
                       "growing": growing_records,
                       "replacement": replacement_records,
                       "final_em_loss_trace": final.trace}

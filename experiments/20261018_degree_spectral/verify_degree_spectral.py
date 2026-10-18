"""Reproducible algebraic/numerical checks; not a community-recovery benchmark."""
from __future__ import annotations

import hashlib
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from degree_spectral_likelihood import (
    DegreeSpectralLikelihood, degree_weights, fit_profile_mixture,
    reconstruct_degree_spectrum, row_scores,
)


SEED = 20261018


def retained_fixture(a, k):
    """Adjacency is used only here to prepare the retained-input test fixtures."""
    d = a.sum(axis=1)
    w = degree_weights(d)
    lam, u = np.linalg.eigh(w[:, None] * a * w[None, :])
    order = np.argsort(-np.abs(lam), kind="stable")[:k]
    return d, u[:, order], lam[order]


def require_small(value, name, tolerance=2e-10):
    assert value < tolerance, f"{name}: {value} >= {tolerance}"
    return float(value)


def verify_reconstruction(a, k, name):
    d, u, lam = retained_fixture(a, k)
    r, meta = reconstruct_degree_spectrum(d, u, lam)
    decoder = DegreeSpectralLikelihood(d, u, lam)
    np.testing.assert_allclose(decoder.reconstruct(), r, atol=1e-12, rtol=1e-12)
    w = degree_weights(d)
    x = np.column_stack((np.ones(len(d)), w[:, None] * u))
    y = np.column_stack((d, u * lam[None, :] / w[:, None]))
    projector = x @ np.linalg.pinv(x, rcond=1e-12)
    z = np.eye(3)[np.arange(len(d)) % 3]
    errors = {
        "retained_action": require_small(np.max(np.abs(a @ x - y)), name + " A X=Y"),
        "reconstructed_action": require_small(np.max(np.abs(r @ x - y)), name + " R X=Y"),
        "projection_identity": require_small(np.max(np.abs(r - a @ projector)), name + " R=A P_X"),
        "degree_identity": require_small(np.max(np.abs(r.sum(axis=1) - d)), name + " R1=D"),
        "projector_idempotence": require_small(np.max(np.abs(projector @ projector - projector)), name + " projector"),
        "blockcount_error_identity": require_small(
            np.max(np.abs((r @ z - a @ z) - a @ (projector - np.eye(len(d))) @ z)),
            name + " block counts"),
        "blockcount_total_degree": require_small(np.max(np.abs((r @ z).sum(axis=1) - d)), name + " count totals"),
    }
    result = decoder.decode(max_em_iter=8)
    assert np.isfinite(result["objective"])
    assert result["labels"].shape == (len(d),)
    assert result["profiles"].shape == (k, len(d))
    eps = result["epsilon"]
    assert np.all(result["profiles"] >= eps) and np.all(result["profiles"] <= 1 - eps)
    np.testing.assert_allclose(result["responsibilities"].sum(axis=1), 1, atol=1e-12)
    em_traces = [result["objective_trace"]]
    em_traces += [item["objective_trace"] for item in result["growing"]]
    for round_record in result["replacement"]:
        assert round_record["after"] >= round_record["before"]
        assert len(round_record["trials"]) == k
        em_traces += [trial["objective_trace"] for trial in round_record["trials"]]
    worst_em_delta = min(float(np.min(np.diff(trace))) for trace in em_traces)
    assert all(len(trace) >= 2 for trace in em_traces)
    assert worst_em_delta > -1e-9
    repeat = decoder.decode(max_em_iter=8)
    np.testing.assert_array_equal(result["labels"], repeat["labels"])
    np.testing.assert_array_equal(result["profiles"], repeat["profiles"])
    return {"name": name, "n": len(d), "k": k, "metadata": meta, "errors": errors,
            "negative_eigenvalues_retained": int((lam < -1e-10).sum()),
            "worst_em_delta": worst_em_delta,
            "replacement_rounds_completed": len(result["replacement"]),
            "same_input_deterministic": True}, r, d, u, lam


def run_checks():
    rng = np.random.default_rng(SEED)
    n = 24
    probs = np.linspace(.06, .55, n)
    upper = np.triu(rng.random((n, n)) < (probs[:, None] + probs[None, :]) / 2, 1)
    heterogeneous = (upper + upper.T).astype(float)
    cycle = np.zeros((10, 10))
    for i in range(len(cycle)):
        cycle[i, (i + 1) % len(cycle)] = cycle[(i + 1) % len(cycle), i] = 1
    empty = np.zeros((8, 8))
    fixtures = [(heterogeneous, 3, "heterogeneous_signed_asymmetric"),
                (cycle, 2, "regular_cycle_rank_deficient_features"),
                (empty, 3, "all_isolated"),
                (heterogeneous[:7, :7], 7, "full_spectrum")]
    outputs = [verify_reconstruction(*fixture) for fixture in fixtures]
    reports = [item[0] for item in outputs]
    assert reports[0]["metadata"]["min_reconstructed_entry"] < -1e-4
    assert reports[0]["metadata"]["max_absolute_asymmetry"] > 1e-4
    assert reports[1]["metadata"]["feature_rank"] < reports[1]["metadata"]["feature_columns"]
    assert reports[1]["negative_eigenvalues_retained"] >= 1
    np.testing.assert_allclose(outputs[-1][1], fixtures[-1][0], atol=1e-10)
    # The cycle's two extreme eigenvectors span its bipartition indicators.
    cycle_groups = np.eye(2)[np.arange(len(cycle)) % 2]
    oracle_count_error = require_small(np.max(np.abs(outputs[1][1] @ cycle_groups - cycle @ cycle_groups)), "exact in-span block counts")
    # Unconstrained signed observations force both fitted endpoints. A dense
    # coordinate grid independently confirms the clipped-mean global optimizer.
    signed = np.array([[-4., .1, 2.], [-2., .3, 4.], [-3., .5, 1.], [-1., .7, 5.]])
    epsilon = .02
    fit = fit_profile_mixture(signed, np.full((1, 3), .5), epsilon, max_iter=1)
    expected = np.clip(signed.mean(axis=0), epsilon, 1 - epsilon)
    np.testing.assert_allclose(fit["profiles"][0], expected, atol=1e-14)
    grid = np.linspace(epsilon, 1 - epsilon, 10001)
    for j in range(3):
        t = signed[:, j].sum()
        grid_values = t * np.log(grid) + (len(signed) - t) * np.log1p(-grid)
        exact = t * np.log(expected[j]) + (len(signed) - t) * np.log1p(-expected[j])
        assert exact + 1e-10 >= grid_values.max()
    direct_scores = np.stack([np.sum(signed * np.log(mu) + (1 - signed) * np.log1p(-mu), axis=1)
                              for mu in np.array([[.2, .4, .6], [.7, .3, .1]])], axis=1)
    np.testing.assert_allclose(row_scores(signed, [[.2, .4, .6], [.7, .3, .1]]), direct_scores, atol=1e-12)
    multiple = fit_profile_mixture(signed, [[.1, .5, .8], [.9, .2, .1]], epsilon, max_iter=12, tol=0)
    assert min(np.diff(multiple["objective_trace"])) > -1e-9
    # Decoder's public entry points have no adjacency argument or graph fixture.
    public_api = {"constructor": str(inspect.signature(DegreeSpectralLikelihood)),
                  "reconstruct": str(inspect.signature(DegreeSpectralLikelihood.reconstruct)),
                  "decode": str(inspect.signature(DegreeSpectralLikelihood.decode))}
    assert "adjacency" not in public_api["constructor"]
    return {"status": "PASS", "scope": "algebra, retained inputs, deterministic fitting and exact E/M only; no recovery-rate benchmark",
            "seed": SEED, "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "fixtures": reports, "exact_in_span_blockcount_error": oracle_count_error,
            "signed_coordinate_optimizer": expected.tolist(), "signed_em_objective_trace": multiple["objective_trace"],
            "public_api": public_api,
            "source_sha256": hashlib.sha256(Path(__file__).with_name("degree_spectral_likelihood.py").read_bytes()).hexdigest(),
            "verification_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    report = run_checks()
    destination = Path(__file__).with_name("CHECKS.json")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "fixtures": len(report["fixtures"]),
                      "checks_file": str(destination)}, indent=2))

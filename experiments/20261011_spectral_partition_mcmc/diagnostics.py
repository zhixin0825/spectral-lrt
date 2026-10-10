"""Explicit oracle diagnostics separating basin stability from random starts.

Truth-initialized runs and P fitted using true blocks are diagnostics only.
They are excluded from the random-initialization benchmark recovery counts.
Every numerical decoder still uses Q reconstructed from saved U and lam.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from numba import njit
from threadpoolctl import threadpool_limits

import engine
from benchmark import exact_information, write_csv


def fitted_P(Q: np.ndarray, z: np.ndarray, k: int) -> np.ndarray:
    """Beta(1,1) posterior mean with unordered block-pair counts."""
    counts = np.bincount(z, minlength=k)
    indicator = np.eye(k)[z]
    total = indicator.T @ Q @ indicator
    P = np.empty((k, k))
    for a in range(k):
        for b in range(a, k):
            m = counts[a] * counts[b] if a != b else counts[a] * (counts[a] - 1) / 2
            s = float(total[a, b]) if a != b else float(total[a, a]) / 2
            P[a, b] = P[b, a] = (s + 1.0) / (m + 2.0)
    return P


def separation(P: np.ndarray, z: np.ndarray, k: int) -> dict:
    counts = np.bincount(z, minlength=k)
    distance = min(float(np.linalg.norm(P[a] - P[b])) for a in range(k) for b in range(a + 1, k))
    info, _ = exact_information(P, counts)
    return {
        "minimum_pair_row_l2": distance,
        "minimum_pair_row_l2_divided_by_pmax": distance / float(P.max()),
        "minimum_pair_chernoff_divided_by_logn": info / math.log(len(z)),
        "p_min": float(P.min()),
        "p_max": float(P.max()),
    }


@njit(cache=True)
def _frozen_score(Q, z, logP, log1P):
    value = 0.0
    for i in range(len(z)):
        for j in range(i):
            value += Q[i, j] * logP[z[i], z[j]] + (1 - Q[i, j]) * log1P[z[i], z[j]]
    return value


@njit(cache=True)
def _frozen_chain(Q, P, z0, sweeps, seed, every):
    np.random.seed(seed)
    n, k = len(z0), len(P)
    z = z0.copy()
    sizes = np.zeros(k, np.int64)
    for i in range(n):
        sizes[z[i]] += 1
    logP, log1P = np.log(P), np.log(1 - P)
    value = _frozen_score(Q, z, logP, log1P)
    best, best_value = z.copy(), value
    records = 1 + sweeps // every + int(sweeps % every != 0)
    labels = np.empty((records, n), np.int64)
    values = np.empty(records)
    times = np.empty(records, np.int64)
    labels[0], values[0], times[0] = z, value, 0
    index = 1
    for sweep in range(1, sweeps + 1):
        for i in np.random.permutation(n):
            old = z[i]
            sizes[old] -= 1
            sums = np.zeros(k)
            for j in range(n):
                if j != i:
                    sums[z[j]] += Q[i, j]
            scores = np.zeros(k)
            for a in range(k):
                for c in range(k):
                    scores[a] += sums[c] * logP[a, c] + (sizes[c] - sums[c]) * log1P[a, c]
            probabilities = np.exp(scores - np.max(scores))
            draw = np.random.random() * probabilities.sum()
            chosen = k - 1
            for a in range(k):
                draw -= probabilities[a]
                if draw <= 0:
                    chosen = a
                    break
            z[i] = chosen
            sizes[chosen] += 1
            value += scores[chosen] - scores[old]
        # Recompute to prevent accumulation error in posterior-best selection.
        value = _frozen_score(Q, z, logP, log1P)
        if value > best_value:
            best, best_value = z.copy(), value
        if sweep % every == 0 or sweep == sweeps:
            labels[index], values[index], times[index] = z, value, sweep
            index += 1
    return z, best, labels, values, times


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "results_pilot")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results_diagnostics")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--models", nargs="+", default=["balanced2", "unequal3", "hierarchy4", "disassort3"])
    parser.add_argument("--signal", type=float, nargs="+", default=[0.65, 1.35])
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=200)
    parser.add_argument("--record-every", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    threadpool_limits(limits=1)
    engine.warmup()
    _frozen_chain(np.array([[0.0, 0.2], [0.2, 0.0]]), np.array([[0.3, 0.1], [0.1, 0.3]]), np.array([0, 1], np.int64), 1, 1, 1)
    rows, summaries = [], []
    for model in args.models:
        for signal in args.signal:
            signal_slug = f"{signal:.4g}".replace(".", "p")
            slug = f"{model}_n{args.n}_signal{signal_slug}_seed{args.graph_seed}"
            source = args.input / slug
            saved = np.load(source / "graph.npz")
            U, lam = saved["U"], saved["lam"]
            truth, k = np.asarray(saved["truth"], dtype=np.int64), int(saved["k"])
            Q = engine.reconstruction(U, lam)
            z0 = np.asarray(np.load(source / "initial_labels_start0.npy"), dtype=np.int64)
            oracle_P, random_P = fitted_P(Q, truth, k), fitted_P(Q, z0, k)
            with (source / "diagnostics.json").open(encoding="utf-8") as handle:
                diagnostics = json.load(handle)
            oracle_separation = separation(oracle_P, truth, k)
            random_separation = separation(random_P, z0, k)
            baseline = {}
            with (source / "final.csv").open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if int(row["start"]) in (0, -1):
                        baseline[f"random_benchmark_{row['method']}_final_error_rate"] = float(row["error_rate"])
            common = {
                "graph_id": slug, "model": model, "n": len(truth), "k": k,
                "signal_target": signal, "graph_seed": args.graph_seed,
                "sweeps": args.sweeps,
                "clipping_relative_sum_change": diagnostics["relative_clip_sum_change"],
                "negative_spectral_entries_fraction": diagnostics["negative_spectral_entries_fraction"],
                "truth_log_collapsed_target": engine.log_target(Q, truth, k),
                "random_start_log_collapsed_target": engine.log_target(Q, z0, k),
                **{f"true_block_P_{key}": value for key, value in oracle_separation.items()},
                **{f"random_block_P_{key}": value for key, value in random_separation.items()},
                **baseline,
            }
            out = args.out / slug
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "fitted_parameters.npz", truth_block_P=oracle_P, random_block_P=random_P, random_initial_labels=z0, truth=truth)
            for mode in ("truth_initialized_collapsed_gibbs", "random_initialized_oracle_P_gibbs"):
                start = time.perf_counter()
                seed = 110328 + list(args.models).index(model) * 100 + int(round(signal * 10))
                if mode == "truth_initialized_collapsed_gibbs":
                    initial = truth.copy()
                    result = engine.collapsed_chain(Q, k, initial, args.sweeps, seed, record_every=args.record_every)
                    target_name = "beta_collapsed_fractional_Bernoulli"
                    truth_selection_target = common["truth_log_collapsed_target"]
                else:
                    initial = z0.copy()
                    final, best, traces, scores, times = _frozen_chain(Q, oracle_P, initial, args.sweeps, seed, args.record_every)
                    result = {"labels": final, "best_labels": best, "trace_labels": traces, "trace_score": scores, "trace_sweep": times}
                    target_name = "frozen_oracle_P_fractional_Bernoulli"
                    truth_selection_target = float(_frozen_score(Q, truth, np.log(oracle_P), np.log1p(-oracle_P)))
                elapsed = time.perf_counter() - start
                trace_metrics = [engine.metrics(z, truth, k) for z in result["trace_labels"]]
                errors = np.asarray([entry["error_rate"] for entry in trace_metrics])
                trace_sweeps = np.asarray(result["trace_sweep"])
                burn = trace_sweeps >= args.sweeps // 2
                collapsed_trace = np.asarray([engine.log_target(Q, z, k) for z in result["trace_labels"]])
                np.savez_compressed(out / f"{mode}.npz", initial_labels=initial, final_labels=result["labels"], best_target_labels=result["best_labels"], trace_labels=result["trace_labels"], trace_selection_target=result["trace_score"], trace_collapsed_target=collapsed_trace, trace_sweep=trace_sweeps, trace_error_rate=errors)
                run_summary = {
                    **common, "diagnostic": mode, "selection_target": target_name,
                    "elapsed": elapsed,
                    "initial_error_rate": engine.metrics(initial, truth, k)["error_rate"],
                    "near_truth_5pct_occupancy_after_burnin": float(np.mean(errors[burn] <= 0.05)),
                    "near_truth_1pct_occupancy_after_burnin": float(np.mean(errors[burn] <= 0.01)),
                    "exact_occupancy_after_burnin": float(np.mean(errors[burn] == 0)),
                    "mean_error_rate_after_burnin": float(np.mean(errors[burn])),
                    "max_error_rate_after_burnin": float(np.max(errors[burn])),
                }
                summaries.append(run_summary)
                for selection, z in (("final", result["labels"]), ("best_target", result["best_labels"])):
                    score = engine.log_target(Q, z, k)
                    selection_score = score if mode == "truth_initialized_collapsed_gibbs" else float(_frozen_score(Q, z, np.log(oracle_P), np.log1p(-oracle_P)))
                    sizes = np.bincount(z, minlength=k)
                    rows.append({
                        **run_summary, "selection": selection, **engine.metrics(z, truth, k),
                        "log_collapsed_target": score,
                        "collapsed_score_minus_truth": score - common["truth_log_collapsed_target"],
                        "log_selection_target": selection_score,
                        "selection_score_minus_truth": selection_score - truth_selection_target,
                        "min_group_size": int(sizes.min()), "empty_groups": int((sizes == 0).sum()),
                    })
                print(f"{slug} {mode}: final error={rows[-2]['error_rate']:.4f}, best error={rows[-1]['error_rate']:.4f}, truth-basin occupancy={run_summary['near_truth_5pct_occupancy_after_burnin']:.3f}", flush=True)
                write_csv(args.out / "diagnostics.csv", rows)
                write_csv(args.out / "basin_occupancy.csv", summaries)
    with (args.out / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump({"configuration": vars(args), "caveat": "Truth-initialized chains and true-block-fitted P are explicit oracle diagnostics, excluded from random-start recovery counts.", "results": rows}, handle, default=jsonable, indent=2)
    print(f"Complete: {len(summaries)} diagnostic chains; {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

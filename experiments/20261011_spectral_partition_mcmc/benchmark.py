"""Random-start partition updates using only retained signed spectral pairs.

The engines consume Q = clip(U diag(lam) U.T, 0, 1), with zero diagonal.
The beta-collapsed fractional Bernoulli target is a working posterior, not
the exact posterior for the original binary graph. Ground truth is used
only for evaluation and the explicitly marked truth-score diagnostic.

Example:
    python benchmark.py --n 256 512 --seeds 0,1 --starts 2 --sweeps 400
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln

import engine

_WORKER_WARMED = False


MODELS = {
    "balanced2": {
        "proportions": [0.5, 0.5],
        "base_P": [[5, 1], [1, 5]],
        "description": "Equal-size two-block assortative SBM",
    },
    "unequal3": {
        "proportions": [0.5, 0.3, 0.2],
        "base_P": [[6, 1, 1.5], [1, 4, 0.8], [1.5, 0.8, 5]],
        "description": "Unequal community sizes and heterogeneous degrees",
    },
    "hierarchy4": {
        "proportions": [0.25, 0.25, 0.25, 0.25],
        "base_P": [[7, 3, 1, 1], [3, 7, 1, 1], [1, 1, 7, 3], [1, 1, 3, 7]],
        "description": "Four blocks nested in two more strongly connected pairs",
    },
    "disassort3": {
        "proportions": [1 / 3, 1 / 3, 1 / 3],
        "base_P": [[1, 4, 3], [4, 1, 4], [3, 4, 1]],
        "description": "Disassortative three-block SBM requiring signed eigenvalues",
    },
}


def balanced_counts(n: int, proportions: np.ndarray) -> np.ndarray:
    unrounded = n * proportions
    counts = np.floor(unrounded).astype(int)
    order = np.argsort(-(unrounded - counts), kind="stable")
    counts[order[: n - int(counts.sum())]] += 1
    return counts


def exact_information(P: np.ndarray, counts: np.ndarray) -> tuple[float, np.ndarray]:
    """Minimum ordered-pair Chernoff information for a true block-a node.

    For each a != b, its row has counts[c] - 1{c=a} independent Bernoulli
    observations with parameters P[a,c] and P[b,c]. Optimize the Chernoff
    parameter rather than replacing it with the t=1/2 affinity.
    """
    k = len(counts)
    info = np.full((k, k), np.inf)
    for a in range(k):
        trials = counts.copy()
        trials[a] -= 1
        for b in range(k):
            if a == b:
                continue
            pa, pb = P[a], P[b]
            log_pa, log_pb = np.log(pa), np.log(pb)
            log_qa, log_qb = np.log1p(-pa), np.log1p(-pb)

            def log_affinity(t: float) -> float:
                one = t * log_pa + (1 - t) * log_pb
                zero = t * log_qa + (1 - t) * log_qb
                return float(trials @ np.logaddexp(one, zero))

            fit = minimize_scalar(log_affinity, bounds=(0.0, 1.0), method="bounded")
            info[a, b] = max(0.0, -float(fit.fun))
    return float(info.min()), info


def scaled_model(model: str, n: int, signal: float, pmax_cap: float) -> dict:
    spec = MODELS[model]
    proportions = np.asarray(spec["proportions"], dtype=float)
    counts = balanced_counts(n, proportions)
    base = np.asarray(spec["base_P"], dtype=float)
    target = signal * math.log(n)
    upper = pmax_cap / float(base.max())
    achieved_upper, _ = exact_information(upper * base, counts)
    capped = achieved_upper < target
    if capped:
        scale = upper
    else:
        lower = 1e-12
        for _ in range(55):
            middle = (lower + upper) / 2
            achieved, _ = exact_information(middle * base, counts)
            if achieved < target:
                lower = middle
            else:
                upper = middle
        scale = (lower + upper) / 2
    P = scale * base
    achieved, information_matrix = exact_information(P, counts)
    expected_degree = P @ counts - np.diag(P)
    return {
        "model": model,
        "description": spec["description"],
        "n": n,
        "k": len(counts),
        "counts": counts,
        "proportions": counts / n,
        "P": P,
        "scale": scale,
        "signal_target": signal,
        "information": achieved,
        "signal_achieved": achieved / math.log(n),
        "information_matrix": information_matrix,
        "pmax_capped": capped,
        "expected_degree": expected_degree,
    }


def collapsed_score(Q: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Exact beta(1,1)-collapsed fractional upper-triangle objective."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=k)
    indicator = np.eye(k)[labels]
    double_sum = indicator.T @ Q @ indicator
    score = 0.0
    for a in range(k):
        for b in range(a, k):
            if a == b:
                trials = counts[a] * (counts[a] - 1) // 2
                successes = float(double_sum[a, a] / 2)
            else:
                trials = counts[a] * counts[b]
                successes = float(double_sum[a, b])
            # The clipped symmetric Q guarantees valid fractional counts.
            # Tolerate only floating-point summation error at the boundaries.
            successes = min(float(trials), max(0.0, successes))
            score += float(betaln(successes + 1, trials - successes + 1))
    return score


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _label_rows(result: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(result["labels"], dtype=np.int32)
    trace = np.asarray(result.get("trace_labels", []), dtype=np.int32)
    if trace.size == 0:
        trace = labels.reshape(1, n)
    elif trace.ndim == 1:
        trace = trace.reshape(1, n)
    return labels, trace


def run_graph(job: dict) -> dict:
    # Each worker limits independent BLAS pools before expensive calculations.
    # threadpoolctl also covers already-imported NumPy/SciPy libraries.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
    global _WORKER_WARMED
    if not _WORKER_WARMED:
        engine.warmup()
        _WORKER_WARMED = True
    started = time.perf_counter()
    model, n, graph_seed, signal = job["model"], job["n"], job["graph_seed"], job["signal"]
    spec = scaled_model(model, n, signal, job["pmax_cap"])
    k, counts, P = spec["k"], spec["counts"], spec["P"]
    signal_slug = f"{signal:.4g}".replace(".", "p")
    slug = f"{model}_n{n}_signal{signal_slug}_seed{graph_seed}"
    out = Path(job["out"]) / slug
    out.mkdir(parents=True, exist_ok=True)
    model_index = list(MODELS).index(model)
    signal_id = int(round(signal * 1_000_000))
    graph_rng = np.random.default_rng(np.random.SeedSequence([110326, model_index, n, signal_id, graph_seed]))
    truth = np.repeat(np.arange(k, dtype=np.int32), counts)
    graph_rng.shuffle(truth)
    upper_i, upper_j = np.triu_indices(n, 1)
    upper = graph_rng.random(len(upper_i)) < P[truth[upper_i], truth[upper_j]]
    A = np.zeros((n, n), dtype=float)
    A[upper_i, upper_j] = upper
    A[upper_j, upper_i] = upper
    spectral_seed = int(graph_rng.integers(0, 2**31 - 1))
    U, lam = engine.spectrum(A, k, spectral_seed)
    U, lam = np.asarray(U), np.asarray(lam)
    raw = (U * lam) @ U.T
    Q = engine.reconstruction(U, lam)
    raw_off_diagonal = raw[upper_i, upper_j]
    clipped_off_diagonal = Q[upper_i, upper_j]
    raw_sum = float(raw_off_diagonal.sum())
    clipped_sum = float(clipped_off_diagonal.sum())
    diagnostics = {
        "graph_id": slug,
        "model": model,
        "n": n,
        "k": k,
        "graph_seed": graph_seed,
        "signal_target": signal,
        "signal_achieved": spec["signal_achieved"],
        "pmax_capped": spec["pmax_capped"],
        "p_min": float(P.min()),
        "p_max": float(P.max()),
        "expected_degree_min": float(spec["expected_degree"].min()),
        "expected_degree_max": float(spec["expected_degree"].max()),
        "observed_degree_min": float(A.sum(axis=1).min()),
        "observed_degree_mean": float(A.sum(axis=1).mean()),
        "negative_spectral_entries_fraction": float((raw_off_diagonal < 0).mean()),
        "spectral_entries_above_one_fraction": float((raw_off_diagonal > 1).mean()),
        "original_upper_edge_sum": float(upper.sum()),
        "raw_spectral_upper_sum": raw_sum,
        "clipped_spectral_upper_sum": clipped_sum,
        "relative_clip_sum_change": (clipped_sum - raw_sum) / max(abs(raw_sum), 1e-12),
        "relative_clip_frobenius_change": float(np.linalg.norm(clipped_off_diagonal - raw_off_diagonal) / max(np.linalg.norm(raw_off_diagonal), 1e-12)),
        "spectral_eigenvalues": ";".join(f"{x:.12g}" for x in lam),
        "truth_log_posterior_diagnostic": collapsed_score(Q, truth, k),
    }
    np.savez_compressed(
        out / "graph.npz", n=n, k=k, truth=truth, P=P, counts=counts,
        U=U, lam=lam, upper_triangle_bits=np.packbits(upper),
        upper_triangle_bit_count=len(upper), graph_seed=graph_seed,
        spectral_seed=spectral_seed, information_matrix=spec["information_matrix"],
        information=spec["information"], signal_target=signal,
    )
    with (out / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump({**spec, **diagnostics}, handle, default=jsonable, indent=2)

    final_rows, best_rows, restart_rows = [], [], []

    def record_result(method: str, start: int, z0: np.ndarray, result: dict, publish: bool = True) -> None:
        final, trace = _label_rows(result, n)
        # Selection uses only the working-posterior value, never truth labels.
        candidates = [final, *trace]
        if "best_labels" in result:
            candidates.append(np.asarray(result["best_labels"], dtype=np.int32))
        candidate_scores = [collapsed_score(Q, labels, k) for labels in candidates]
        best = candidates[int(np.argmax(candidate_scores))].copy()
        trace_score = np.asarray([collapsed_score(Q, labels, k) for labels in trace])
        trace_metrics = [engine.metrics(labels, truth, k) for labels in trace]
        trace_sweep = np.asarray(result.get("trace_sweep", []))
        if trace_sweep.size != len(trace):
            trace_sweep = np.arange(len(trace)) * job["record_every"]
        payload = {
            "initial_labels": z0,
            "final_labels": final,
            "best_posterior_labels": best,
            "trace_labels": trace,
            "trace_log_posterior": trace_score,
            "trace_sweep": trace_sweep,
            "trace_errors": np.asarray([m["errors"] for m in trace_metrics]),
            "trace_ari": np.asarray([m["ari"] for m in trace_metrics]),
        }
        for key in ("trace_score", "trace_elbo", "elbo_trace", "elbo", "P", "final_P", "responsibilities", "changed_fraction", "swap_acceptance", "swap_accepted", "swap_tries", "restart_initial_labels", "restart_final_labels", "restart_best_labels", "restart_final_scores", "restart_best_scores", "selected_final_restart", "selected_best_restart"):
            if key in result:
                payload["engine_" + key] = np.asarray(result[key])
        np.savez_compressed(out / f"{method}_start{start}.npz", **payload)
        common = {
            "graph_id": slug, "model": model, "n": n, "k": k,
            "signal_target": signal, "signal_achieved": spec["signal_achieved"],
            "graph_seed": graph_seed, "start": start, "method": method,
            "elapsed": float(result.get("elapsed", float("nan"))),
            "initial_error_rate": engine.metrics(z0, truth, k)["error_rate"],
            "initial_log_posterior": collapsed_score(Q, z0, k),
            "truth_log_posterior_diagnostic": diagnostics["truth_log_posterior_diagnostic"],
            "p_max": float(P.max()),
            "expected_degree_min": diagnostics["expected_degree_min"],
            "negative_spectral_entries_fraction": diagnostics["negative_spectral_entries_fraction"],
        }
        if "swap_acceptance" in result:
            common["swap_acceptance_mean"] = float(np.mean(result["swap_acceptance"]))
        final_destination = final_rows if publish else restart_rows
        best_destination = best_rows if publish else restart_rows
        for selection, labels, rows in (("final", final, final_destination), ("best_posterior", best, best_destination)):
            sizes = np.bincount(labels, minlength=k)
            metrics = engine.metrics(labels, truth, k)
            score = collapsed_score(Q, labels, k)
            rows.append({
                **common, "selection": selection, **metrics,
                "log_posterior": score,
                "score_minus_truth": score - diagnostics["truth_log_posterior_diagnostic"],
                "min_group_size": int(sizes.min()), "max_group_size": int(sizes.max()),
                "empty_groups": int((sizes == 0).sum()),
                "group_sizes": ";".join(str(int(x)) for x in sizes),
            })

    # The spectral baseline uses its own standard k-means++ initialization.
    if "spectral_kmeans" in job["methods"]:
        baseline_begin = time.perf_counter()
        labels = engine.spectral_kmeans(U, lam, k, spectral_seed)
        result = {"labels": labels, "best_labels": labels, "trace_labels": np.asarray([labels]), "trace_sweep": [0], "elapsed": time.perf_counter() - baseline_begin}
        baseline_z0 = graph_rng.integers(0, k, size=n, dtype=np.int32)
        record_result("spectral_kmeans", -1, baseline_z0, result)
    for start in range(job["starts"]):
        chain_rng = np.random.default_rng(np.random.SeedSequence([110327, model_index, n, signal_id, graph_seed, start]))
        z0 = chain_rng.integers(0, k, size=n, dtype=np.int32)
        seeds = {name: int(chain_rng.integers(0, 2**31 - 1)) for name in ("gibbs", "icm", "em", "pt", "gibbs6")}
        np.save(out / f"initial_labels_start{start}.npy", z0)
        for method in job["methods"]:
            if method == "spectral_kmeans":
                continue
            if method in ("gibbs", "icm"):
                result = engine.collapsed_chain(Q, k, z0.copy(), job["sweeps"], seeds[method], beta=1.0, greedy=(method == "icm"), record_every=job["record_every"])
            elif method == "em":
                result = engine.variational_em(Q, k, z0.copy(), job["em_iters"], seeds[method])
            elif method == "pt":
                result = engine.tempered_chain(Q, k, z0.copy(), job["sweeps"], seeds[method], betas=tuple(job["betas"]), record_every=job["record_every"])
            elif method == "gibbs6":
                restart_rng = np.random.default_rng(seeds[method])
                restart_results, restart_initial = [], []
                for restart in range(job["independent_restarts"]):
                    initial = z0.copy() if restart == 0 else restart_rng.integers(0, k, size=n, dtype=np.int32)
                    seed = int(restart_rng.integers(0, 2**31 - 1))
                    one = engine.collapsed_chain(Q, k, initial, job["sweeps"], seed, beta=1.0, greedy=False, record_every=job["record_every"])
                    restart_results.append(one)
                    restart_initial.append(initial)
                    record_result(f"gibbs6_restart{restart}", start, initial, one, publish=False)
                endpoint_scores = np.asarray([collapsed_score(Q, r["labels"], k) for r in restart_results])
                best_scores = np.asarray([collapsed_score(Q, r["best_labels"], k) for r in restart_results])
                selected_final = int(np.argmax(endpoint_scores))
                selected_best = int(np.argmax(best_scores))
                result = dict(restart_results[selected_final])
                result["best_labels"] = restart_results[selected_best]["best_labels"].copy()
                result["elapsed"] = sum(float(r["elapsed"]) for r in restart_results)
                result["restart_initial_labels"] = np.asarray(restart_initial)
                result["restart_final_labels"] = np.asarray([r["labels"] for r in restart_results])
                result["restart_best_labels"] = np.asarray([r["best_labels"] for r in restart_results])
                result["restart_final_scores"] = endpoint_scores
                result["restart_best_scores"] = best_scores
                result["selected_final_restart"] = selected_final
                result["selected_best_restart"] = selected_best
            else:
                raise ValueError(method)
            record_result(method, start, z0, result)
            write_csv(out / "final.csv", final_rows)
            write_csv(out / "best_posterior.csv", best_rows)
            write_csv(out / "independent_restarts.csv", restart_rows)
    diagnostics["elapsed_total"] = time.perf_counter() - started
    write_csv(out / "final.csv", final_rows)
    write_csv(out / "best_posterior.csv", best_rows)
    write_csv(out / "independent_restarts.csv", restart_rows)
    return {"final": final_rows, "best": best_rows, "restarts": restart_rows, "diagnostics": diagnostics}


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["model"], row["n"], row["signal_target"], row["method"], row["selection"])
        groups.setdefault(key, []).append(row)
    out = []
    for key, items in sorted(groups.items()):
        item = dict(zip(("model", "n", "signal_target", "method", "selection"), key))
        item["runs"] = len(items)
        for field in ("error_rate", "ari", "elapsed", "empty_groups", "score_minus_truth"):
            values = np.asarray([r[field] for r in items], dtype=float)
            item[field + "_mean"] = float(np.mean(values))
            item[field + "_std"] = float(np.std(values))
        item["exact_fraction"] = float(np.mean([r["exact"] for r in items]))
        item["error_rate_median"] = float(np.median([r["error_rate"] for r in items]))
        item["error_rate_max"] = float(np.max([r["error_rate"] for r in items]))
        out.append(item)
    return out


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        fields = [int(x) for x in value.split(":")]
        return list(range(*fields))
    return [int(x) for x in value.split(",") if x]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1"), help="Comma-separated graph seeds, or a start:stop range")
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=400)
    parser.add_argument("--em-iters", type=int, default=200)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--signal", nargs="+", type=float, default=[0.65, 1.35], help="Exact oracle Chernoff information divided by log(n)")
    parser.add_argument("--pmax-cap", type=float, default=0.8)
    parser.add_argument("--methods", nargs="+", choices=["spectral_kmeans", "gibbs", "gibbs6", "icm", "em", "pt"], default=["spectral_kmeans", "gibbs", "gibbs6", "icm", "em", "pt"])
    parser.add_argument("--independent-restarts", type=int, default=6, help="Independent plain Gibbs chains for the equal-budget gibbs6 arm")
    parser.add_argument("--betas", type=lambda value: [float(x) for x in value.split(",")], default=[1.0, 0.7, 0.45, 0.25, 0.12, 0.04])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results_pilot")
    args = parser.parse_args()
    if args.starts < 1 or args.sweeps < 1 or args.record_every < 1:
        parser.error("starts, sweeps, and record-every must be positive")
    if not 0 < args.pmax_cap < 1:
        parser.error("pmax-cap must be strictly between zero and one")
    if min(args.n) < 20 or min(args.signal) <= 0:
        parser.error("n must be at least 20 and signal must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    with (args.out / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, default=jsonable, indent=2)
    jobs = [
        {**config, "model": model, "n": n, "signal": signal, "graph_seed": seed, "out": str(args.out.resolve())}
        for model, n, signal, seed in itertools.product(args.models, args.n, args.signal, args.seeds)
    ]
    final_rows, best_rows, restart_rows, diagnostics, failures = [], [], [], [], []
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    print(f"Running {len(jobs)} graphs, {args.starts} random starts per method; output: {args.out}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        submitted = {pool.submit(run_graph, job): job for job in jobs}
        for finished, future in enumerate(as_completed(submitted), start=1):
            job = submitted[future]
            try:
                result = future.result()
                final_rows.extend(result["final"])
                best_rows.extend(result["best"])
                restart_rows.extend(result["restarts"])
                diagnostics.append(result["diagnostics"])
                print(f"[{finished}/{len(jobs)}] {result['diagnostics']['graph_id']}: {result['diagnostics']['elapsed_total']:.1f}s", flush=True)
            except Exception:
                failures.append({"model": job["model"], "n": job["n"], "signal": job["signal"], "graph_seed": job["graph_seed"], "traceback": traceback.format_exc()})
                print(f"[{finished}/{len(jobs)}] FAILED {job['model']} n={job['n']} signal={job['signal']} seed={job['graph_seed']}\n{failures[-1]['traceback']}", flush=True)
            write_csv(args.out / "final.csv", final_rows)
            write_csv(args.out / "best_posterior.csv", best_rows)
            write_csv(args.out / "independent_restarts.csv", restart_rows)
            write_csv(args.out / "graph_diagnostics.csv", diagnostics)
            write_csv(args.out / "summary.csv", summarize(final_rows + best_rows))
            write_csv(args.out / "failures.csv", failures)
    print(f"Complete: {len(diagnostics)} graphs succeeded, {len(failures)} failed.", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

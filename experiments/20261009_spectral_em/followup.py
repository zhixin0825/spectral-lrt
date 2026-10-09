"""Change only EM initialization on the ORIGINAL fixed U-calibrated features.

Reads saved compressed spectral observations Y and km_X labels, never A.
Y and H_U stay unchanged, so tau-matched terminal likelihoods are directly
comparable to the main soft_tau1 and soft_tau2 fits.  This differs from the
main soft_X_tau1/soft_X_tau2 arms, which use a different fixed H_X decoder.
Ground truth is accessed only after both EM fits have completed.
"""

from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from poisson_em import fit_poisson_em

METHODS = (("soft_Ucal_Xinit_tau1", 1.), ("soft_Ucal_Xinit_tau2", 2.))


def recovery(labels: np.ndarray, truth: np.ndarray, k: int) -> dict:
    table = np.zeros((k, k), dtype=int)
    np.add.at(table, (truth, labels), 1)
    rows, cols = linear_sum_assignment(-table)
    errors = len(truth) - int(table[rows, cols].sum())
    return {"errors": errors, "error_rate": errors / len(truth),
            "exact": int(errors == 0), "ari": float(adjusted_rand_score(truth, labels))}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON type: {type(value)}")


def run_one(task: tuple[str, int, float, str]) -> dict:
    root_string, max_iter, tol, key = task
    root = Path(root_string)
    start = time.perf_counter()
    source = json.loads((root / "runs" / f"{key}.json").read_text(encoding="utf-8"))
    match = re.fullmatch(r"(.+)_n(\d+)_ch([0-9.]+)_s(\d+)", key)
    if match is None:
        raise ValueError(f"Unexpected graph key: {key}")
    name, n, ch, seed = match.groups()
    common = {"graph": key, "model": name, "n": int(n), "ch": float(ch), "seed": int(seed),
              "initial_method": "km_X", "fixed_calibration": "H_U from km_U",}
    common.update({k: v for k, v in source["diagnostics"].items() if k != "h0"})
    fitted = []
    traces = {}
    labels_by_method = {}
    with np.load(root / "inputs" / f"{key}.npz", allow_pickle=False) as saved:
        y = saved["y"]
        initial_labels = saved["labels_km_X"]
        # The inference calls receive only the frozen observed feature matrix
        # and spectral initialization. No truth or adjacency argument exists.
        with threadpool_limits(limits=1):
            for method, tau in METHODS:
                result = fit_poisson_em(y, initial_labels, tau=tau,
                                        max_iter=max_iter, tol=tol, hard=False)
                final_labels = result.pop("labels")
                result.pop("r", None)
                traces[method] = result.pop("trace")
                result["iterations"] = result["n_iter"]
                labels_by_method[method] = final_labels
                fitted.append((method, result, final_labels))
        # Access ground truth only AFTER all inference and parameter fitting.
        truth = saved["truth_evaluation_only"]
        k = len(np.unique(initial_labels))
        initial_metrics = recovery(initial_labels, truth, k)
        records = []
        for method, result, final_labels in fitted:
            record = {**common, "method": method, **recovery(final_labels, truth, k), **result,
                      "initial_error_rate": initial_metrics["error_rate"],
                      "initial_exact": initial_metrics["exact"]}
            records.append(record)
    payload = {"graph": key, "inference_input": "saved y and labels_km_X only",
               "feature_calibration_changed": False, "results": records,
               "traces": traces, "labels": labels_by_method,
               "elapsed": time.perf_counter() - start}
    (root / "followup_runs" / f"{key}.json").write_text(
        json.dumps(payload, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="Existing benchmark directory containing inputs/ and runs/")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--expected-graphs", type=int)
    args = parser.parse_args()
    root = args.out.resolve()
    inputs = sorted((root / "inputs").glob("*.npz"))
    if not inputs:
        raise FileNotFoundError(f"No saved spectral inputs under {root / 'inputs'}")
    if args.expected_graphs is not None and len(inputs) != args.expected_graphs:
        raise ValueError(f"Expected {args.expected_graphs} graphs, found {len(inputs)}")
    missing = [p.stem for p in inputs if not (root / "runs" / f"{p.stem}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Main records not yet saved for graphs: {missing}")
    (root / "followup_runs").mkdir(exist_ok=True)
    protocol = {"methods": [name for name, _ in METHODS], "taus": [tau for _, tau in METHODS],
                "max_iter": args.max_iter, "tol": args.tol,
                "initialization": "labels_km_X from each saved spectral input",
                "fixed_features": "original y = sqrt(n)*U*Lambda*H_U^{-1}",
                "calibration_changes": False,
                "likelihood_comparison": "tau 1 vs main soft_tau1; tau 2 vs main soft_tau2",
                "truth_access": "evaluation after all fitting only",
                "graphs": len(inputs), "graph_keys": [p.stem for p in inputs],
                "kernel_sha256": hashlib.sha256(Path(__file__).with_name("poisson_em.py").read_bytes()).hexdigest()}
    (root / "followup_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    tasks = [(str(root), args.max_iter, args.tol, p.stem) for p in inputs]
    records, failures, all_labels = [], [], {}
    started = time.perf_counter()
    with (root / "followup_traces.jsonl").open("w", encoding="utf-8") as trace_file:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, task): task[-1] for task in tasks}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                key = futures[future]
                try:
                    payload = future.result()
                    records.extend(payload["results"])
                    trace_file.write(json.dumps({"graph": key, "traces": payload["traces"]},
                                                default=_json_default) + "\n")
                    trace_file.flush()
                    for method, labels in payload["labels"].items():
                        all_labels[f"{key}__{method}"] = labels
                    errors = {r["method"]: round(100 * r["error_rate"], 3) for r in payload["results"]}
                    print(json.dumps({"completed": index, "total": len(tasks), "graph": key,
                                      "error_percent": errors, "elapsed": payload["elapsed"]}), flush=True)
                except Exception as error:
                    failures.append({"graph": key, "error": repr(error)})
                    print(json.dumps({"completed": index, "total": len(tasks), "graph": key,
                                      "failure": repr(error)}), flush=True)
    frame = pd.DataFrame(records).sort_values(["model", "n", "ch", "seed", "method"])
    frame.to_csv(root / "followup.csv", index=False)
    np.savez_compressed(root / "followup_labels.npz", **all_labels)
    summary = {"input_graphs": len(inputs), "completed_graphs": len(inputs) - len(failures),
               "fit_records": len(records), "failures": failures,
               "elapsed_seconds": time.perf_counter() - started}
    (root / "followup_completion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if failures:
        raise RuntimeError(f"Follow-up failed for {len(failures)} graphs")


if __name__ == "__main__":
    main()

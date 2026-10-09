"""Separate the initial likelihood E step from later EM parameter updates.

No tuning: fixed H_U decoder, KM_X initial labels, tau=1, max_iter=0/1.
The fitting calls finish before this script reads truth for evaluation.
"""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from poisson_em import fit_poisson_em


FINAL_METHOD = "soft_Ucal_Xinit_tau1"
STAGES = ["km_X", "initial_E", "one_EM_iteration", "converged_EM"]
CONDITION = ["model", "n", "ch"]


def error_count(labels, truth, k):
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (truth, labels), 1)
    rows, columns = linear_sum_assignment(-counts)
    return int(len(truth) - counts[rows, columns].sum())


def one_graph(task):
    path, metadata, base_errors, final_errors = task
    with np.load(path) as data:
        y = np.asarray(data["y"], float)
        labels_init = np.asarray(data["labels_km_X"], int)
        result0 = fit_poisson_em(y, labels_init, tau=1.0, max_iter=0)
        result1 = fit_poisson_em(y, labels_init, tau=1.0, max_iter=1)
        truth = np.asarray(data["truth_evaluation_only"], int)
        k = y.shape[1]
        errors0 = error_count(result0["labels"], truth, k)
        errors1 = error_count(result1["labels"], truth, k)
    rows = []
    for stage, errors, iterations, objective in [
        ("km_X", base_errors, np.nan, np.nan),
        ("initial_E", errors0, 0, result0["objective"]),
        ("one_EM_iteration", errors1, 1, result1["objective"]),
        ("converged_EM", final_errors, np.nan, np.nan),
    ]:
        rows.append(dict(**metadata, stage=stage, errors=int(errors),
                         error_rate=float(errors / metadata["n"]),
                         exact=int(errors == 0), iterations=iterations,
                         objective=objective))
    return rows


def compare(pivot, before, after):
    delta = pivot[before] - pivot[after]
    return dict(before=before, after=after, n_graphs=len(pivot),
                after_better=int((delta > 0).sum()),
                same_recovery_error=int((delta == 0).sum()),
                after_worse=int((delta < 0).sum()))


def summarize(rows):
    pivot = rows.pivot(index="graph", columns="stage", values="errors")
    means = {stage: float(100 * rows.loc[rows["stage"] == stage, "error_rate"].mean()) for stage in STAGES}
    exact = {stage: int(rows.loc[rows["stage"] == stage, "exact"].sum()) for stage in STAGES}
    net_gain = means["km_X"] - means["converged_EM"]
    return dict(n_graphs=len(pivot), error_mean_pct=means, exact_total=exact,
                first_E_improvement_pct=means["km_X"] - means["initial_E"],
                final_improvement_pct=net_gain,
                additional_iteration_improvement_pct=means["initial_E"] - means["converged_EM"],
                first_E_fraction_of_final_net_gain=(means["km_X"] - means["initial_E"]) / net_gain if net_gain != 0 else None,
                comparisons=[compare(pivot, "km_X", "initial_E"),
                             compare(pivot, "initial_E", "one_EM_iteration"),
                             compare(pivot, "initial_E", "converged_EM"),
                             compare(pivot, "one_EM_iteration", "converged_EM")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-results", action="store_true", help="summarize the already computed one_step.csv without repeating fits")
    args = parser.parse_args()
    out = Path(args.out)
    baseline = pd.read_csv(out / "per_run.csv")
    baseline = baseline[baseline["method"] == "km_X"].drop_duplicates("graph")
    followup = pd.read_csv(out / "followup.csv")
    followup = followup[followup["method"] == FINAL_METHOD][["graph", "errors"]].rename(columns={"errors": "final_errors"})
    matched = baseline.merge(followup, on="graph", how="inner").sort_values("graph")
    tasks = []
    for row in matched.to_dict("records"):
        metadata = {field: row[field] for field in ["graph", *CONDITION, "seed"]}
        tasks.append((str(out / "inputs" / (row["graph"] + ".npz")), metadata, int(row["errors"]), int(row["final_errors"])))
    started = time.perf_counter()
    if args.reuse_results:
        rows = pd.read_csv(out / "one_step.csv")
        chunks = None
    elif args.workers == 1:
        chunks = list(map(one_graph, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            chunks = list(pool.map(one_graph, tasks, chunksize=4))
    if chunks is not None:
        rows = pd.DataFrame([record for chunk in chunks for record in chunk])
        rows.to_csv(out / "one_step.csv", index=False)
    conditions = []
    for condition, group in rows.groupby(CONDITION, sort=True):
        metadata = dict(zip(CONDITION, condition))
        metadata["n"] = int(metadata["n"])
        metadata["ch"] = float(metadata["ch"])
        conditions.append({**metadata, **summarize(group)})
    result = dict(protocol="fixed H_U/Y; KM_X start; tau=1; no parameter tuning; truth read only after both fitting calls",
                  elapsed_seconds=time.perf_counter() - started,
                  cached_fits=args.reuse_results,
                  final_method=FINAL_METHOD, overall=summarize(rows), conditions=conditions,
                  interpretation="First E uses parameters estimated from KM_X labels; later stages update parameters. Same recovery error need not mean identical partitions.")
    (out / "one_step_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"], **result["overall"]}))


if __name__ == "__main__":
    main()

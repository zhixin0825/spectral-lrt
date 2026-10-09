"""Audit a completed or partial paired spectral-only benchmark.

Usage: python analyze.py --out final
Reads the current CSV and each JSON once; it never waits for running jobs.
Positive paired improvement means fewer errors than the named baseline.
Objective comparisons are restricted to one fixed calibration/tau/model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20261009
BOOTSTRAP_REPLICATES = 10000
DECREASE_TOL = 1e-8
MATERIAL_RELATIVE_TOL = 1e-9
CONDITION = ["model", "n", "ch"]


def finite_number(value, default=np.nan):
    try:
        answer = float(value)
        return answer if np.isfinite(answer) else default
    except (ValueError, TypeError):
        return default


def boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and np.isfinite(value):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "1"):
            return True
        if value.lower() in ("false", "0"):
            return False
    return None


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def best_trace_key(record, traces):
    if record["method"] != "soft_tau1_best4":
        return record["method"] if record["method"] in traces else None
    starts = record.get("restart_objectives", [])
    if isinstance(starts, str):
        try:
            starts = json.loads(starts)
        except json.JSONDecodeError:
            starts = []
    if starts:
        winner = max(starts, key=lambda item: finite_number(item.get("objective"), -np.inf))
        start = winner.get("start", "")
        key = "soft_tau1" if start == "km_U" else start.replace("plus_plus_", "restart_")
        if key in traces:
            return key
    candidates = [key for key in traces if key == "soft_tau1" or key.startswith("restart_")]
    if not candidates:
        return None
    target = finite_number(record.get("objective"))
    return min(candidates, key=lambda key: abs(finite_number(traces[key][-1].get("objective")) - target))


def audit_trace(record, traces, cap):
    method = record["method"]
    key = best_trace_key(record, traces)
    trace = traces.get(key, []) if key else []
    iteration = finite_number(record.get("n_iter"))
    if not np.isfinite(iteration):
        iteration = finite_number(record.get("iterations"))
    if trace:
        iteration = finite_number(trace[-1].get("iteration"), len(trace) - 1)
    converged = boolean(record.get("converged"))
    reason = record.get("reason")
    objectives = np.asarray([finite_number(item.get("objective")) for item in trace])
    steps = np.diff(objectives)
    relative_steps = steps / (1 + np.abs(objectives[:-1]))
    material = relative_steps < -MATERIAL_RELATIVE_TOL
    first, last = (trace[0], trace[-1]) if trace else ({}, {})
    class_mins = [finite_number(item.get("class_min")) for item in trace]
    expected_mins = [finite_number(item.get("expected_class_min")) for item in trace]
    changed = [finite_number(item.get("n_changed")) for item in trace]
    cap_for_method = 300 if method.startswith("km_") else cap
    final_objective = finite_number(last.get("objective", record.get("objective")))
    initial_objective = finite_number(first.get("objective"))
    return dict(
        **{field: record.get(field) for field in ["graph", *CONDITION, "seed"]},
        method=method, trace_key=key, trace_available=bool(trace),
        is_restart=method.startswith("restart_"),
        is_alias=method == "soft_tau1_best4", iterations=iteration,
        converged=converged, stop_reason=reason,
        max_iter_cap=cap_for_method,
        max_iter_hit=bool(reason == "max_iter" or (np.isfinite(iteration) and iteration >= cap_for_method)),
        n_trace_points=len(trace), initial_objective=initial_objective,
        final_objective=final_objective,
        objective_gain=final_objective - initial_objective,
        objective_gain_per_node=(final_objective - initial_objective) / float(record["n"]),
        min_objective_step=float(np.min(steps)) if len(steps) else np.nan,
        max_objective_step=float(np.max(steps)) if len(steps) else np.nan,
        objective_decreases_lt_neg1e8=int(np.sum(steps < -DECREASE_TOL)),
        material_objective_decreases=int(material.sum()),
        min_relative_objective_step=float(np.min(relative_steps)) if len(steps) else np.nan,
        raw_decreases_within_relative_slack=int(((steps < -DECREASE_TOL) & ~material).sum()),
        material_relative_tolerance=MATERIAL_RELATIVE_TOL,
        final_objective_step=float(steps[-1]) if len(steps) else np.nan,
        final_param_delta=finite_number(last.get("param_delta")),
        min_class_over_trace=min(class_mins) if class_mins else np.nan,
        final_class_min=finite_number(last.get("class_min")),
        min_expected_class_over_trace=min(expected_mins) if expected_mins else np.nan,
        final_expected_class_min=finite_number(last.get("expected_class_min")),
        final_label_change=finite_number(last.get("n_changed")),
        max_label_change=max(changed) if changed else np.nan,
        labels_changed_total=sum(changed) if changed else np.nan,
    )


def bootstrap_mean(values, rng):
    values = np.asarray(values, float)
    if len(values) == 0:
        return np.full(BOOTSTRAP_REPLICATES, np.nan)
    if np.ptp(values) == 0:
        return np.full(BOOTSTRAP_REPLICATES, values[0])
    return values[rng.integers(len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)


def bootstrap_interval(values, rng):
    return np.quantile(bootstrap_mean(values, rng), [.025, .975])


def comparison_baselines(method):
    if method == "gmm_direct_X":
        return ["km_U", "km_X"]
    if method.startswith("soft_Ucal_Xinit_"):
        matched = "soft_tau2" if method.endswith("tau2") else "soft_tau1"
        return ["km_U", "km_X", "km_Y", "km_Y_from_U", matched] + (["soft_tau1_best4"] if method.endswith("tau1") else [])
    if method in ("soft_X_tau1", "soft_X_tau2", "gmm_YX_from_X"):
        return ["km_U", "km_X", "km_YX_from_X"]
    if method in ("soft_tau1", "soft_tau2", "hard_tau1", "soft_tau1_best4", "gmm_Y"):
        return ["km_U", "km_X", "km_Y", "km_Y_from_U"]
    if method == "km_YX_from_X":
        return ["km_U", "km_X"]
    if method in ("km_Y", "km_Y_from_U", "km_X"):
        return ["km_U"]
    return []


def paired_row(group, method, baseline, rng, scope, condition):
    improvements = group["improvement_pct"].to_numpy(float)
    if scope == "per_condition":
        low, high = bootstrap_interval(improvements, rng)
        mean_delta = improvements.mean()
        method_mean = 100 * group["error_rate"].mean()
        baseline_mean = 100 * group["baseline_error_rate"].mean()
        exact_delta = 100 * (group["exact"] - group["baseline_exact"]).mean()
        n_conditions = 1
    else:
        strata = list(group.groupby(CONDITION, sort=True, dropna=False))
        replicates = np.zeros(BOOTSTRAP_REPLICATES)
        for _, stratum in strata:
            replicates += bootstrap_mean(stratum["improvement_pct"], rng) / len(strata)
        low, high = np.quantile(replicates, [.025, .975])
        mean_delta = np.mean([item["improvement_pct"].mean() for _, item in strata])
        method_mean = np.mean([100 * item["error_rate"].mean() for _, item in strata])
        baseline_mean = np.mean([100 * item["baseline_error_rate"].mean() for _, item in strata])
        exact_delta = np.mean([100 * (item["exact"] - item["baseline_exact"]).mean() for _, item in strata])
        n_conditions = len(strata)
    error_deltas = group["baseline_errors"].to_numpy() - group["errors"].to_numpy()
    return dict(scope=scope, **dict(zip(CONDITION, condition)), method=method,
                baseline=baseline, n_pairs=len(group), n_conditions=n_conditions,
                method_error_mean_pct=method_mean, baseline_error_mean_pct=baseline_mean,
                mean_improvement_pct=mean_delta, ci95_low_pct=low, ci95_high_pct=high,
                wins=int((error_deltas > 0).sum()), ties=int((error_deltas == 0).sum()),
                losses=int((error_deltas < 0).sum()),
                method_exact_total=int(group["exact"].sum()),
                baseline_exact_total=int(group["baseline_exact"].sum()),
                macro_exact_improvement_pct=exact_delta,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
                bootstrap_seed=BOOTSTRAP_SEED)


def correlation(first, second):
    mask = np.isfinite(first) & np.isfinite(second)
    first, second = np.asarray(first)[mask], np.asarray(second)[mask]
    if len(first) < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def normalize_trace(trace):
    return [dict(item,
                 param_delta=item.get("param_delta", item.get("parameter_change")),
                 n_changed=item.get("n_changed", item.get("label_changes")),
                 objective_delta=item.get("objective_delta", item.get("gain")))
            for item in trace]


def load_extra_traces(out, payloads, read_errors):
    for filename in ["followup_traces.jsonl", "direct_gaussian.jsonl"]:
        path = out / filename
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if "record" in item:
                    record = item["record"]
                    graph = record["graph"]
                    target = payloads.setdefault(graph, {"graph": graph, "results": [], "traces": {}})
                    target.setdefault("results", []).append(record)
                    target.setdefault("traces", {})[record["method"]] = normalize_trace(item.get("trace", []))
                else:
                    graph = item["graph"]
                    target = payloads.setdefault(graph, {"graph": graph, "results": [], "traces": {}})
                    for method, trace in item.get("traces", {}).items():
                        target.setdefault("traces", {})[method] = normalize_trace(trace)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                read_errors.append({"file": filename, "line": line_number, "error": repr(exc)})


def same_objective_comparison(records, method, baseline):
    own = records[records["method"] == method]
    base = records[records["method"] == baseline][["graph", "errors", "error_rate", "objective"]].rename(columns={key: "base_" + key for key in ["errors", "error_rate", "objective"]})
    comp = own.merge(base, on="graph")
    if comp.empty:
        return dict(method=method, baseline=baseline, n_pairs=0)
    objective_delta = comp["objective"] - comp["base_objective"]
    error_delta = comp["errors"] - comp["base_errors"]
    higher = objective_delta > DECREASE_TOL
    material_higher = objective_delta > MATERIAL_RELATIVE_TOL * (1 + comp["base_objective"].abs())
    corr = []
    for condition, sub in comp.groupby(CONDITION, sort=True, dropna=False):
        corr.append(dict(**dict(zip(CONDITION, condition)), n_pairs=len(sub),
                         correlation=correlation((sub["objective"] - sub["base_objective"]).to_numpy(), (sub["base_errors"] - sub["errors"]).to_numpy())))
    return dict(method=method, baseline=baseline, n_pairs=len(comp),
                same_fixed_features_and_tau=True,
                higher_objective_graphs=int(higher.sum()),
                materially_higher_objective_graphs=int(material_higher.sum()),
                lower_objective_graphs=int((objective_delta < -DECREASE_TOL).sum()),
                materially_lower_objective_graphs=int((objective_delta < -MATERIAL_RELATIVE_TOL * (1 + comp["base_objective"].abs())).sum()),
                higher_objective_better_errors=int((higher & (error_delta < 0)).sum()),
                higher_objective_equal_errors=int((higher & (error_delta == 0)).sum()),
                higher_objective_worse_errors=int((higher & (error_delta > 0)).sum()),
                mean_error_improvement_pct=float(100 * (comp["base_error_rate"] - comp["error_rate"]).mean()),
                within_same_model_objective_error_correlation=correlation(objective_delta.to_numpy(), (-error_delta).to_numpy()),
                descriptive_within_condition_correlations=corr,
                higher_objective_worse_examples=comp[higher & (error_delta > 0)][["graph", *CONDITION, "errors", "base_errors", "objective", "base_objective"]].to_dict("records"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="benchmark output directory")
    args = parser.parse_args()
    out = Path(args.out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8")) if (out / "protocol.json").exists() else {}
    tables = []
    read_errors = []
    inputs_read = []
    for filename in ["per_run.csv", "followup.csv", "direct_gaussian.csv"]:
        path = out / filename
        if not path.exists():
            continue
        try:
            tables.append(pd.read_csv(path))
            inputs_read.append(filename)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
            read_errors.append({"file": filename, "error": repr(exc)})
    if not tables:
        raise SystemExit("No per-run result CSV is available yet.")
    records = pd.concat(tables, ignore_index=True, sort=False)
    if records.empty:
        raise SystemExit("No per-run results are available yet.")
    records = records.drop_duplicates(["graph", "method"], keep="last")
    for column in ("errors", "error_rate", "exact", "ari", "n", "ch", "seed", "objective", "iterations", "n_iter"):
        if column in records:
            records[column] = pd.to_numeric(records[column], errors="coerce")
    payloads = {}
    for path in sorted((out / "runs").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payloads[payload["graph"]] = payload
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            read_errors.append({"file": path.name, "error": repr(exc)})
    load_extra_traces(out, payloads, read_errors)
    audited = []
    cap = int(protocol.get("max_iter", 150))
    for record in records.to_dict("records"):
        payload = payloads.get(record["graph"], {})
        native = next((item for item in payload.get("results", []) if item.get("method") == record["method"]), {})
        record = record | native
        audited.append(audit_trace(record, payload.get("traces", {}), cap))
    for graph, payload in sorted(payloads.items()):
        if graph not in set(records["graph"]):
            continue
        if not payload.get("results"):
            continue
        prototype = payload["results"][0]
        for key, trace in payload.get("traces", {}).items():
            if key.startswith("restart_"):
                native = {**prototype, "method": key, "n_iter": len(trace) - 1,
                          "converged": None, "reason": None}
                audited.append(audit_trace(native, {key: trace}, cap))
    convergence = pd.DataFrame(audited)
    convergence.to_csv(out / "convergence.csv", index=False)
    main_audit = convergence[~convergence["is_restart"]]
    merged = records.merge(main_audit.drop(columns=["model", "n", "ch", "seed"]), on=["graph", "method"], how="left", suffixes=("", "_audit"))
    summary = []
    for condition, group in merged.groupby([*CONDITION, "method"], sort=True, dropna=False):
        iterations = pd.to_numeric(group["iterations_audit"], errors="coerce").dropna()
        conv = [boolean(value) for value in group["converged_audit"]] if "converged_audit" in group else [boolean(value) for value in group["converged"]]
        valid_conv = [value for value in conv if value is not None]
        summary.append(dict(**dict(zip([*CONDITION, "method"], condition)), n_graphs=len(group),
                            error_mean_pct=100 * group["error_rate"].mean(),
                            error_median_pct=100 * group["error_rate"].median(),
                            error_mean_count=group["errors"].mean(), exact_total=int(group["exact"].sum()),
                            exact_pct=100 * group["exact"].mean(), ari_mean=group["ari"].mean(),
                            convergence_observed=len(valid_conv), converged_total=sum(valid_conv),
                            convergence_pct=100 * np.mean(valid_conv) if valid_conv else np.nan,
                            iterations_mean=iterations.mean(), iterations_p10=iterations.quantile(.1),
                            iterations_median=iterations.median(), iterations_p90=iterations.quantile(.9),
                            iterations_max=iterations.max(), max_iter_hits=int(group["max_iter_hit"].sum()),
                            final_label_change_mean=group["final_label_change"].mean(),
                            final_label_change_max=group["final_label_change"].max(),
                            trace_count=int(group["trace_available"].sum()),
                            monotonicity_decreases=int(group["objective_decreases_lt_neg1e8"].sum()),
                            material_monotonicity_decreases=int(group["material_objective_decreases"].sum()),
                            raw_decreases_within_relative_slack=int(group["raw_decreases_within_relative_slack"].sum()),
                            min_objective_step=group["min_objective_step"].min(),
                            min_relative_objective_step=group["min_relative_objective_step"].min(),
                            final_param_delta_max=group["final_param_delta"].max(),
                            min_class_over_trace=group["min_class_over_trace"].min()))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired, pair_frames = [], {}
    for method in sorted(records["method"].unique()):
        for baseline in comparison_baselines(method):
            own = records[records["method"] == method]
            other = records[records["method"] == baseline][["graph", "errors", "error_rate", "exact"]].rename(columns={key: "baseline_" + key for key in ["errors", "error_rate", "exact"]})
            group = own.merge(other, on="graph", how="inner").sort_values("graph")
            if group.empty:
                continue
            group["improvement_pct"] = 100 * (group["baseline_error_rate"] - group["error_rate"])
            pair_frames[(method, baseline)] = group
            for condition, sub in group.groupby(CONDITION, sort=True, dropna=False):
                paired.append(paired_row(sub, method, baseline, rng, "per_condition", condition))
            paired.append(paired_row(group, method, baseline, rng, "macro_fixed_conditions", ("__all__", 0, np.nan)))
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)
    correlations = []
    for method in sorted(records["method"].unique()):
        baseline = "km_X" if "_X_" in method or "_Xinit_" in method else "km_U"
        frame = pair_frames.get((method, baseline))
        if frame is None or not method.startswith(("soft_", "hard_")):
            continue
        frame = frame.merge(main_audit[["graph", "method", "objective_gain_per_node"]], on=["graph", "method"])
        for condition, sub in frame.groupby(CONDITION, sort=True, dropna=False):
            correlations.append(dict(**dict(zip(CONDITION, condition)), method=method, baseline=baseline,
                                     n_pairs=len(sub), statistic="descriptive Pearson within condition/method",
                                     objective_gain_vs_error_improvement=correlation(sub["objective_gain_per_node"].to_numpy(), sub["improvement_pct"].to_numpy())))
    best4 = same_objective_comparison(records, "soft_tau1_best4", "soft_tau1")
    same_model_pairs = [("soft_Ucal_Xinit_tau1", "soft_tau1"),
                        ("soft_Ucal_Xinit_tau1", "soft_tau1_best4"),
                        ("soft_Ucal_Xinit_tau2", "soft_tau2")]
    same_model_comparisons = [same_objective_comparison(records, method, baseline) for method, baseline in same_model_pairs
                              if {method, baseline}.issubset(set(records["method"]))]
    failure_path = out / "failures.json"
    failures = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else []
    raw_traces = convergence[~convergence["is_alias"] & convergence["trace_available"]]
    expected_graphs = len(protocol.get("models", [])) * len(protocol.get("ns", [])) * len(protocol.get("chs", [])) * int(protocol.get("seeds", 0))
    analysis = dict(n_graphs=int(records["graph"].nunique()), expected_graphs=expected_graphs,
                    complete_graph_count=bool(expected_graphs and records["graph"].nunique() == expected_graphs),
                    n_records=len(records), methods=sorted(records["method"].unique()),
                    graph_count_by_method=records.groupby("method")["graph"].nunique().to_dict(),
                    csv_inputs_read=inputs_read,
                    n_json_payloads=len(payloads), json_read_errors=read_errors,
                    csv_graphs_without_json=sorted(set(records["graph"]) - set(payloads)),
                    json_graphs_not_yet_in_csv=sorted(set(payloads) - set(records["graph"])),
                    failure_count=len(failures), failures=failures,
                    bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES,
                                   per_condition="paired resampling of graphs within the named condition",
                                   macro="equal weight per tested condition; graph resampling within each fixed condition",
                                   interpretation="descriptive uncertainty for this finite benchmark design, not a universal population claim"),
                    trace_audit=dict(unique_actual_traces=len(raw_traces),
                                     traces_with_decreases=int((raw_traces["objective_decreases_lt_neg1e8"] > 0).sum()),
                                     total_decreases=int(raw_traces["objective_decreases_lt_neg1e8"].sum()),
                                     min_objective_step=raw_traces["min_objective_step"].min(),
                                     traces_with_material_decreases=int((raw_traces["material_objective_decreases"] > 0).sum()),
                                     total_material_decreases=int(raw_traces["material_objective_decreases"].sum()),
                                     min_relative_objective_step=raw_traces["min_relative_objective_step"].min(),
                                     material_relative_tolerance=MATERIAL_RELATIVE_TOL,
                                     raw_decreases_within_relative_slack=int(raw_traces["raw_decreases_within_relative_slack"].sum()),
                                     traces_with_empty_hard_class=int((raw_traces["min_class_over_trace"] == 0).sum())),
                    objective_gain_correlations=correlations, best4_vs_single_tau1=best4,
                    same_fixed_model_initialization_comparisons=same_model_comparisons,
                    objective_comparison_rule="never compare raw likelihood values across calibration, tau, hard/soft objective, or Gaussian/Poisson methods")
    (out / "analysis.json").write_text(json.dumps(clean_json(analysis), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(clean_json({"graphs": analysis["n_graphs"], "expected": expected_graphs,
                                "methods": len(analysis["methods"]), "failures": len(failures),
                                "trace_audit": analysis["trace_audit"],
                                "same_model_comparisons": [{key: value for key, value in item.items()
                                                            if "correlation" not in key and "examples" not in key}
                                                           for item in [best4, *same_model_comparisons]]}), ensure_ascii=False))
    macro = paired[paired["scope"] == "macro_fixed_conditions"]
    display = macro[(macro["baseline"].isin(["km_U", "km_X"])) & macro["method"].str.startswith(("soft_", "hard_", "gmm_"))]
    print(display[["method", "baseline", "n_pairs", "mean_improvement_pct", "ci95_low_pct", "ci95_high_pct", "wins", "ties", "losses"]].to_string(index=False))


if __name__ == "__main__":
    main()

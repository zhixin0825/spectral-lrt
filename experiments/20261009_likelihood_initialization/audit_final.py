#!/usr/bin/env python3
"""Independent read-only statistical audit of the frozen SBM benchmark.

The audit reads completed JSON jobs, never alters or refits an algorithm, and
does not use pilot jobs. Confidence intervals are descriptive paired t
intervals, not familywise-adjusted hypothesis tests. Positive error deltas
mean that the first method is worse. Runtime excludes the shared spectral
compression, which is recorded separately by the experiment runner.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


PAIRS = [
    ("gmm_full_spec", "km_spec", "same_features_and_center_seeds"),
    ("gmm_tied_spec", "km_spec", "same_features_and_center_seeds"),
    ("gmm_full_spec_degree", "km_spec_degree", "same_features_and_center_seeds"),
    ("gmm_full_neighbor_degree", "km_neighbor_degree", "same_features_and_center_seeds"),
    ("poisson_gaussian", "km_neighbor_degree", "same_information_different_model_and_seeding"),
    ("gmm_full_spec", "km_classic_U", "different_geometry_same_spectral_information"),
    ("poisson_gaussian", "km_classic_U", "different_geometry_degree_augmented"),
    ("spectral_bernoulli_surrogate", "km_classic_U", "different_objective_degree_augmented"),
    ("raw_bernoulli_vem", "km_classic_U", "additional_original_edge_information"),
]


def paired_summary(frame):
    delta = frame.error_a.to_numpy() - frame.error_b.to_numpy()
    n = len(delta)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    half = float(t.ppf(.975, n-1) * se) if n > 1 else np.nan
    tolerance = 1e-12
    return dict(
        repetitions=n,
        error_a=float(frame.error_a.mean()),
        error_b=float(frame.error_b.mean()),
        delta=mean, delta_se=se,
        ci95_lower=mean-half, ci95_upper=mean+half,
        wins=int(np.count_nonzero(delta < -tolerance)),
        ties=int(np.count_nonzero(np.abs(delta) <= tolerance)),
        losses=int(np.count_nonzero(delta > tolerance)),
        seconds_a=float(frame.seconds_a.mean()),
        seconds_b=float(frame.seconds_b.mean()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="final")
    parser.add_argument("--output", default="final_audit")
    args = parser.parse_args()
    results = Path(args.results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((results / "protocol.json").read_text())
    lo, hi = map(int, protocol["seeds"].split(":"))
    expected_seeds = hi-lo
    expected_jobs = len(protocol["models"].split(",")) * len(protocol["degrees"].split(",")) * expected_seeds
    jobs = sorted((results / "jobs").glob("*.json"))
    records, metadata = [], []
    for path in jobs:
        item = json.loads(path.read_text())
        records.extend(item["records"])
        metadata.append(item["metadata"])
    if not records:
        print(json.dumps(dict(completed_jobs=0, expected_jobs=expected_jobs)))
        return
    frame = pd.DataFrame(records)
    jobframe = pd.DataFrame(metadata)
    keys = ["model", "n", "degree_target", "seed", "method", "stage"]
    duplicate_rows = int(frame.duplicated(keys).sum())
    failed = frame[frame.failed != 0]
    good = frame[frame.failed == 0].copy()
    summary = good.groupby(["model", "n", "degree_target", "method", "stage"]).agg(
        repetitions=("error", "size"), error_mean=("error", "mean"),
        error_sd=("error", "std"), seconds_mean=("seconds", "mean"),
        min_cluster_min=("min_cluster", "min"), ari_mean=("ari", "mean"),
    ).reset_index()
    summary.to_csv(output / "audit_condition_means.csv", index=False)
    failed.to_csv(output / "audit_failed_records.csv", index=False)
    pairs = []
    merge_keys = ["model", "n", "degree_target", "seed", "stage"]
    for method_a, method_b, interpretation in PAIRS:
        a = good[good.method == method_a][merge_keys + ["error", "seconds"]]
        b = good[good.method == method_b][merge_keys + ["error", "seconds"]]
        joined = a.merge(b, on=merge_keys, suffixes=("_a", "_b"), validate="one_to_one")
        for group, subset in joined.groupby(["model", "n", "degree_target", "stage"]):
            model, n, degree, stage = group
            pairs.append(dict(
                model=model, n=int(n), degree_target=degree, stage=stage,
                method_a=method_a, method_b=method_b, interpretation=interpretation,
                complete=len(subset) == expected_seeds,
                **paired_summary(subset),
            ))
    pairframe = pd.DataFrame(pairs)
    pairframe.to_csv(output / "audit_paired_comparisons.csv", index=False)
    initial = good[good.stage == "initial"].copy()
    initial["not_converged"] = ~initial.converged.astype(bool)
    initial["empty_cluster"] = initial.min_cluster == 0
    initial["small_cluster"] = initial.min_cluster < .01*initial.n
    initial["decreases"] = pd.to_numeric(initial.get("decreases", 0), errors="coerce").fillna(0)
    initial["numerical_rejections"] = pd.to_numeric(initial.get("numerical_rejections", 0), errors="coerce").fillna(0)
    diagnostics = initial.groupby(["model", "degree_target", "method"]).agg(
        repetitions=("error", "size"), not_converged=("not_converged", "sum"),
        empty_cluster=("empty_cluster", "sum"), small_cluster=("small_cluster", "sum"),
        monotonicity_decreases=("decreases", "sum"),
        numerical_rejections=("numerical_rejections", "sum"),
        n_iter_mean=("n_iter", "mean"),
    ).reset_index()
    diagnostics.to_csv(output / "audit_initial_diagnostics.csv", index=False)
    decoded = good[good.stage == "decoded"].copy()
    if len(decoded):
        decode_summary = decoded.groupby(["model", "degree_target", "method"]).agg(
            repetitions=("error", "size"), fallbacks=("decoder_fallback", "sum"),
            negative_count_fraction=("negative_count_fraction", "mean"),
            block_clipped_fraction=("block_clipped_fraction", "mean"),
            centroid_condition_mean=("centroid_condition", "mean"),
            centroid_condition_max=("centroid_condition", "max"),
        ).reset_index()
        decode_summary.to_csv(output / "audit_decoder_diagnostics.csv", index=False)
        effect = good[good.stage == "decoded"].merge(
            good[good.stage == "initial"], on=["model", "n", "degree_target", "seed", "method"],
            suffixes=("_decoded", "_initial"), validate="one_to_one",
        )
        effect["decoder_delta"] = effect.error_decoded - effect.error_initial
        effect["decoder_improved"] = effect.decoder_delta < -1e-12
        effect["decoder_worsened"] = effect.decoder_delta > 1e-12
        effect_summary = effect.groupby(["model", "degree_target", "method"]).agg(
            repetitions=("decoder_delta", "size"), delta_mean=("decoder_delta", "mean"),
            improved=("decoder_improved", "sum"), worsened=("decoder_worsened", "sum"),
        ).reset_index()
        effect_summary.to_csv(output / "audit_decoder_effect.csv", index=False)
    aggregate_diag = diagnostics.groupby("method")[[
        "repetitions", "not_converged", "empty_cluster", "small_cluster", "monotonicity_decreases", "numerical_rejections"
    ]].sum().reset_index()
    complete_pairs = pairframe[pairframe.complete]
    report = dict(
        completed_jobs=len(jobs), expected_jobs=expected_jobs,
        all_complete=len(jobs) == expected_jobs,
        duplicate_method_stage_records=duplicate_rows,
        failed_records=len(failed),
        completed_conditions=int((jobframe.groupby(["model", "degree_target"]).size() == expected_seeds).sum()),
        metadata_unique_shapes=sorted({(int(x["n"]), int(x["k"])) for x in metadata}),
        feature_and_seeding_comparisons=PAIRS,
        initial_diagnostics=aggregate_diag.to_dict("records"),
        fallback_total=int(decoded.decoder_fallback.sum()) if len(decoded) else 0,
        notes=[
            "Paired CI is descriptive, unadjusted for multiple comparisons.",
            "Positive delta means first method has higher error.",
            "Do not average currently completed jobs as representative of all models: scheduler is ordered by model and degree.",
            "Comparisons using poor standardized/degree-duplicated L2 geometry do not establish superiority to classic U/ASE/RRE clustering.",
            "d=4,6 are numerically below log(1800); d=12,24,64,96 are above. Fixed n grid does not verify asymptotic exponents.",
            "Monotonicity and convergence diagnostics cover only each method's selected restart; all restart trajectories are not saved.",
            "The raw Bernoulli VEM arm reads A; compressed-only arms must not be selected using its objective.",
        ],
    )
    (output / "audit_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in (
        "completed_jobs", "expected_jobs", "all_complete", "duplicate_method_stage_records", "failed_records", "completed_conditions", "fallback_total"
    )}, indent=2))
    print(aggregate_diag.to_string(index=False))
    print("\nCompleted matched-feature comparisons, initial stage:")
    cols = ["model", "degree_target", "method_a", "method_b", "error_a", "error_b", "delta", "ci95_lower", "ci95_upper", "wins", "ties", "losses"]
    print(complete_pairs[(complete_pairs.stage == "initial") & complete_pairs.interpretation.str.startswith("same_")][cols].round(5).to_string(index=False))


if __name__ == "__main__":
    main()

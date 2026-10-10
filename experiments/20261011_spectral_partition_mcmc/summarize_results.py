"""Independent endpoint and trajectory analysis of saved SBM experiments.

Example: python summarize_results.py --inputs results_pilot results_main --out analysis
Later input directories supersede duplicate graph/start/method endpoints.
The primary summary uses final.csv; best-posterior selection stays separate.
All recovery measurements use saved labels, only after inference has ended.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


METHODS = ['em', 'icm', 'gibbs', 'gibbs6', 'pt', 'spectral_kmeans']
NAMES = {'em': 'EM', 'icm': 'ICM', 'gibbs': 'Gibbs', 'gibbs6': 'Gibbs, 6 starts',
         'pt': 'Parallel tempering', 'spectral_kmeans': 'Spectral k-means'}
COLORS = dict(zip(METHODS, ['#9c755f', '#bab0ac', '#4e79a7', '#59a14f', '#e15759', '#b07aa1']))
MODEL_NAMES = {'balanced2': 'Balanced K=2', 'unequal3': 'Unequal K=3',
               'hierarchy4': 'Hierarchical K=4', 'disassort3': 'Disassortative K=3',
               'balanced6': 'Balanced K=6'}


def mismatch(labels, truth, k):
    table = np.zeros((k, k), dtype=np.int64)
    np.add.at(table, (np.asarray(truth, int), np.asarray(labels, int)), 1)
    row, col = linear_sum_assignment(-table)
    return int(len(truth) - table[row, col].sum())


def safe_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_rows(inputs, filename):
    chunks = []
    for priority, directory in enumerate(inputs):
        frame = safe_csv(directory / filename)
        if frame.empty:
            continue
        frame['source_dir'] = str(directory.resolve())
        frame['source_priority'] = priority
        chunks.append(frame)
    if not chunks:
        return pd.DataFrame(), 0
    raw = pd.concat(chunks, ignore_index=True)
    key = ['graph_id', 'start', 'method', 'selection']
    result = raw.drop_duplicates(key, keep='last').copy()
    return result, int(len(raw) - len(result))


def trajectory_row(row, near):
    directory = Path(row['source_dir']) / row['graph_id']
    archive = directory / f"{row['method']}_start{int(row['start'])}.npz"
    graph = directory / 'graph.npz'
    result = {key: row[key] for key in ('graph_id', 'model', 'n', 'k', 'signal_target',
                                       'signal_achieved', 'graph_seed', 'start', 'method', 'source_dir')}
    result['archive_found'] = archive.exists() and graph.exists()
    if not result['archive_found']:
        return result
    with np.load(graph) as data:
        truth = data['truth']
    with np.load(archive) as data:
        labels = data['trace_labels']
        times = data['trace_sweep'].astype(float)
        errors = np.array([mismatch(z, truth, int(row['k'])) for z in labels])
        rates = errors / int(row['n'])
        final_error = mismatch(data['final_labels'], truth, int(row['k']))
        initial_error = mismatch(data['initial_labels'], truth, int(row['k']))
        stored_error_max_difference = float(np.max(abs(errors - data['trace_errors']))) if 'trace_errors' in data else None
        paired_initial_match = bool(np.array_equal(data['initial_labels'], labels[0]))
        # A selected restart can legitimately have a different trace's initial labels.
        if row['method'] == 'gibbs6' and 'engine_selected_final_restart' in data:
            result['selected_final_restart'] = int(data['engine_selected_final_restart'])
        result.update(final_errors_recomputed=final_error,
                      initial_errors_recomputed=initial_error,
                      stored_trace_error_max_difference=stored_error_max_difference,
                      trace_initial_equals_saved_initial=paired_initial_match,
                      recorded_frames=len(times), final_recorded_update=float(times[-1]),
                      final_error_matches_csv=bool(final_error == int(row['errors'])))
        if row['method'] == 'spectral_kmeans':
            result.update(first_recorded_near_hit=np.nan, last20_near_fraction=np.nan,
                          last20_error_mean=np.nan, last20_error_median=np.nan,
                          last20_error_q90=np.nan, last20_exact_fraction=np.nan)
        else:
            hits = times[rates <= near + 1e-15]
            last = times >= .8 * times[-1]
            result.update(first_recorded_near_hit=float(hits[0]) if len(hits) else np.nan,
                          last20_near_fraction=float((rates[last] <= near + 1e-15).mean()),
                          last20_error_mean=float(rates[last].mean()),
                          last20_error_median=float(np.median(rates[last])),
                          last20_error_q90=float(np.quantile(rates[last], .9)),
                          last20_exact_fraction=float((errors[last] == 0).mean()))
        if 'engine_elbo_trace' in data:
            delta = np.diff(data['engine_elbo_trace'])
            result['minimum_saved_em_elbo_change'] = float(delta.min()) if len(delta) else np.nan
    return result


def summarize(frame, trajectory, near):
    if frame.empty:
        return frame
    merged = frame.copy()
    if not trajectory.empty:
        keys = ['graph_id', 'start', 'method', 'source_dir']
        fields = [x for x in ('first_recorded_near_hit', 'last20_near_fraction',
                             'last20_error_mean', 'last20_error_q90', 'last20_exact_fraction')
                  if x in trajectory.columns]
        merged = merged.merge(trajectory[keys + fields], on=keys, how='left')
    results = []
    keys = ['model', 'n', 'signal_target', 'method', 'selection']
    for group, items in merged.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, group))
        errors = items['error_rate'].to_numpy(float)
        record.update(runs=len(items), graphs=items['graph_id'].nunique(),
                      exact_runs=int((items['errors'] == 0).sum()),
                      near_1pct_runs=int((errors <= near + 1e-15).sum()),
                      exact_fraction=float((items['errors'] == 0).mean()),
                      near_1pct_fraction=float((errors <= near + 1e-15).mean()),
                      error_rate_mean=float(errors.mean()), error_rate_median=float(np.median(errors)),
                      error_rate_q10=float(np.quantile(errors, .1)),
                      error_rate_q90=float(np.quantile(errors, .9)),
                      error_rate_min=float(errors.min()), error_rate_max=float(errors.max()),
                      signal_achieved_min=float(items['signal_achieved'].min()),
                      signal_achieved_max=float(items['signal_achieved'].max()),
                      elapsed_mean=float(items['elapsed'].mean()),
                      empty_group_runs=int((items['empty_groups'] > 0).sum()))
        for name in ('first_recorded_near_hit', 'last20_near_fraction', 'last20_error_mean',
                     'last20_error_q90', 'last20_exact_fraction'):
            if name in items:
                valid = items[name].dropna().to_numpy(float)
                record[name + '_mean'] = float(valid.mean()) if len(valid) else np.nan
                record[name + '_median'] = float(np.median(valid)) if len(valid) else np.nan
                record[name + '_runs_observed'] = len(valid)
        results.append(record)
    return pd.DataFrame(results)


def table_text(summary):
    if summary.empty:
        return 'No completed results.\n'
    lines = ['| Model | n | I / log n | Method | Graphs / runs | Exact | <=1% error | Mean error | Median error |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in summary.to_dict('records'):
        signal = f"{row['signal_achieved_min']:.2f}"
        if abs(row['signal_achieved_max'] - row['signal_achieved_min']) > .005:
            signal += f"–{row['signal_achieved_max']:.2f}"
        lines.append(f"| {MODEL_NAMES.get(row['model'], row['model'])} | {int(row['n'])} | {signal} | "
                     f"{NAMES.get(row['method'], row['method'])} | {int(row['graphs'])} / {int(row['runs'])} | "
                     f"{int(row['exact_runs'])}/{int(row['runs'])} | {int(row['near_1pct_runs'])}/{int(row['runs'])} | "
                     f"{100*row['error_rate_mean']:.2f}% | {100*row['error_rate_median']:.2f}% |")
    return '\n'.join(lines) + '\n'


def strong_rows(final):
    # Strong means achieved oracle information >= log n, not the requested value.
    return final[final['signal_achieved'] >= 1.0 - 1e-8].copy()


def plot_traces(final, out):
    strong = strong_rows(final)
    selected = []
    if strong.empty:
        return selected
    models = [x for x in MODEL_NAMES if x in set(strong['model'])]
    fig, axes = plt.subplots(len(models), 1, figsize=(9.2, max(3.15*len(models), 3.5)), squeeze=False)
    for model, ax in zip(models, axes[:, 0]):
        available = strong[strong['model'] == model].copy()
        # Fixed selection by metadata only; no recovery statistic enters this rule.
        available = available.sort_values(['n', 'signal_target', 'graph_seed', 'start', 'source_priority'])
        first = available.iloc[0]
        graph_id = first['graph_id']
        random_rows = available[(available['graph_id'] == graph_id) & (available['start'] >= 0)]
        start = int(random_rows['start'].min())
        rows = available[(available['graph_id'] == graph_id) & (available['start'] == start)]
        selection = {key: first[key] for key in ('model', 'n', 'signal_target', 'signal_achieved', 'graph_seed', 'graph_id')}
        selection['start'] = start
        selection['rule'] = 'First metadata-sorted achieved-strong graph and start, never selected by outcome.'
        init = []
        for method in ('em', 'icm', 'gibbs', 'pt'):
            match = rows[rows['method'] == method]
            if match.empty:
                continue
            row = match.iloc[-1]
            directory = Path(row['source_dir']) / graph_id
            archive = directory / f'{method}_start{start}.npz'
            if not archive.exists():
                continue
            with np.load(directory/'graph.npz') as data:
                truth = data['truth']
            with np.load(archive) as data:
                times = data['trace_sweep']
                errors = np.array([mismatch(z, truth, int(row['k'])) for z in data['trace_labels']]) / int(row['n'])
                init.append(data['initial_labels'].copy())
            ax.plot(times, errors*100, label=NAMES[method], color=COLORS[method], linewidth=1.5,
                    linestyle='--' if method in ('em', 'icm') else '-')
        selection['methods_share_initial_labels'] = bool(all(np.array_equal(init[0], x) for x in init[1:])) if init else None
        selected.append(selection)
        ax.axhline(1., color='#555555', linestyle=':', linewidth=1)
        ax.set_ylim(bottom=-1)
        ax.set_ylabel('Misclassified nodes (%)')
        ax.set_title(f"{MODEL_NAMES[model]}, n={int(first['n'])}, graph seed={int(first['graph_seed'])}, "
                     f"I/log n={first['signal_achieved']:.2f}, start={start}", fontsize=10)
        ax.grid(axis='y', color='#e8e8e8')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper right', ncol=2, fontsize=9)
        ax.set_xlabel('Gibbs / ICM sweeps; EM coordinate-update rounds')
    fig.suptitle('Fixed graph choices: random-start partition trajectories', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out/'preregistered_strong_traces.png', dpi=190)
    plt.close(fig)
    return selected


def plot_success(final, out, near):
    strong = strong_rows(final)
    if strong.empty:
        return []
    models = [x for x in MODEL_NAMES if x in set(strong['model'])]
    methods = [m for m in METHODS if m in set(strong['method'])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    records = []
    width = .8/len(methods)
    positions = np.arange(len(models))
    for method_index, method in enumerate(methods):
        exact, near_values = [], []
        for model in models:
            rows = strong[(strong['model'] == model) & (strong['method'] == method)]
            if rows.empty:
                exact.append(np.nan); near_values.append(np.nan)
                continue
            e = float((rows['errors'] == 0).mean())
            h = float((rows['error_rate'] <= near + 1e-15).mean())
            exact.append(e); near_values.append(h)
            records.append(dict(model=model, method=method, runs=len(rows),
                                graphs=rows['graph_id'].nunique(), exact_fraction=e,
                                near_1pct_fraction=h, n_min=int(rows['n'].min()), n_max=int(rows['n'].max())))
        x = positions - .4 + width*(method_index+.5)
        axes[0].bar(x, exact, width=width, color=COLORS[method], label=NAMES[method])
        axes[1].bar(x, near_values, width=width, color=COLORS[method], label=NAMES[method])
    for ax, title in zip(axes, ('Exact recovery at final state', 'At most 1% error at final state')):
        ax.set_title(title, fontsize=11)
        ax.set_xticks(positions, [MODEL_NAMES[x].replace(' ', '\n', 1) for x in models])
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', color='#e8e8e8')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Fraction of graph/start runs')
    fig.suptitle('Achieved I/log n >= 1; final states only', fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .95))
    fig.savefig(out/'strong_endpoint_success.png', dpi=190)
    plt.close(fig)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--near', type=float, default=.01)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    final, final_duplicate = load_rows(args.inputs, 'final.csv')
    best, best_duplicate = load_rows(args.inputs, 'best_posterior.csv')
    if final.empty:
        raise SystemExit('No final.csv rows found; wait for at least one completed graph.')
    trajectory = pd.DataFrame([trajectory_row(row, args.near) for row in final.to_dict('records')])
    trajectory.to_csv(args.out/'trajectory_diagnostics.csv', index=False)
    final_summary = summarize(final, trajectory, args.near)
    best_summary = summarize(best, pd.DataFrame(), args.near)
    final_summary.to_csv(args.out/'final_summary.csv', index=False)
    best_summary.to_csv(args.out/'best_posterior_summary.csv', index=False)
    final.to_csv(args.out/'final_endpoints.csv', index=False)
    best.to_csv(args.out/'best_posterior_endpoints.csv', index=False)
    (args.out/'final_table.md').write_text(table_text(final_summary), encoding='utf-8')
    (args.out/'best_posterior_table.md').write_text(table_text(best_summary), encoding='utf-8')
    traces = plot_traces(final, args.out)
    chart_data = plot_success(final, args.out, args.near)
    if chart_data:
        pd.DataFrame(chart_data).to_csv(args.out/'strong_endpoint_success.csv', index=False)
    missing = trajectory[~trajectory['archive_found']]
    metadata = dict(inputs=[str(p.resolve()) for p in args.inputs],
                    final_rows=len(final), best_posterior_rows=len(best),
                    distinct_graphs=final['graph_id'].nunique(),
                    superseded_duplicate_final_rows=final_duplicate,
                    superseded_duplicate_best_rows=best_duplicate,
                    near_error_threshold=args.near,
                    missing_archives=missing[['graph_id', 'start', 'method']].to_dict('records'),
                    fixed_trace_choices=traces,
                    primary='Final states in final.csv. Gibbs6 endpoint selected by endpoint working posterior only.',
                    hitting='First recorded <=1% error hit; recordings can miss excursions between frames.',
                    occupation='Fraction of saved frames in the last 20% of the selected trajectory.',
                    strong='Achieved oracle Chernoff information I/log n >= 1.',
                    sample_unit='Runs share graphs; graph/start runs are not independent graph replicates.',
                    best='Best-posterior endpoints reported separately and never substituted for final samplers.',
                    limitation='Recovery measurements do not establish posterior mixing or a recovery theorem.')
    (args.out/'analysis_metadata.json').write_text(json.dumps(metadata, indent=2, default=lambda x: x.item()), encoding='utf-8')
    print(f"Analyzed {len(final)} final endpoints on {metadata['distinct_graphs']} graphs; "
          f"superseded {final_duplicate} duplicate rows; {len(missing)} missing trajectory archives.")
    print(table_text(final_summary))


if __name__ == '__main__':
    main()

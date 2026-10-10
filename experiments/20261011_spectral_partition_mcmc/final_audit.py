"""Read-only independent audit of archived spectral-MCMC endpoints.

No inference algorithm is executed. Upper-triangle target counts and errors
are recomputed independently from saved spectra and saved labels. The audit
also checks expected run coverage, paired random starts and source hashes.
"""
import ast
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln

from independent_validation import reference_counts
from summarize_results import mismatch, safe_csv


ROOT = Path(__file__).resolve().parent
MODEL_ORDER = ['balanced2', 'unequal3', 'hierarchy4', 'disassort3', 'balanced6']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return str(row['graph_id']), str(row['method']), int(row['start'])


def expected_keys(directory, config):
    if 'methods' in config:
        graphs = []
        for model,n,signal,seed in itertools.product(config['models'], config['n'], config['signal'], config['seeds']):
            slug = f"{model}_n{n}_signal{signal:.4g}_seed{seed}".replace('.', 'p')
            graphs.append(slug)
        keys = set()
        for graph in graphs:
            for method in config['methods']:
                starts = [-1] if method=='spectral_kmeans' else range(config['starts'])
                keys.update((graph,method,int(start)) for start in starts)
        return keys, {}
    # Supplemental graph-selection is read from its already frozen inputs.
    sources = {}
    for source_name in config['inputs']:
        source = Path(source_name)
        if not source.is_absolute():
            source = Path.cwd()/source
        frame = safe_csv(source/'final.csv')
        for gid, rows in frame.groupby('graph_id'):
            first = rows.iloc[0]
            if (first['model'] in config['models'] and int(first['graph_seed']) in config['seeds']
                    and abs(float(first['signal_target'])-config['signal'])<1e-8):
                sources[gid] = source/gid
    method = ('prototype_global' if 'prototype' in directory.name
              else 'spectral_global' if 'spectral' in directory.name else 'global')
    return {(gid,method,start) for gid in sources for start in range(config['starts'])}, sources


def independent_graph(path):
    with np.load(path) as saved:
        U, lam, truth = saved['U'], saved['lam'], saved['truth']
        raw = (U*lam)@U.T
        Q = np.clip((raw+raw.T)/2,0,1)
        np.fill_diagonal(Q,0)
        n,k = int(saved['n']),int(saved['k'])
    i,j = np.triu_indices(n,1)
    return dict(n=n,k=k,truth=truth,i=i,j=j,q=Q[i,j])


def independent_score(graph, z):
    a,b = z[graph['i']],z[graph['j']]
    codes = np.minimum(a,b)*graph['k']+np.maximum(a,b)
    total = np.bincount(codes,minlength=graph['k']**2)
    successes = np.bincount(codes,weights=graph['q'],minlength=graph['k']**2)
    # Empty unordered block pairs contribute log B(1,1)=0.
    return float(betaln(successes+1,total-successes+1).sum())


def aggregate(frame, keys):
    records=[]
    for group, rows in frame.groupby(keys,sort=True):
        group = group if isinstance(group,tuple) else (group,)
        item=dict(zip(keys,group))
        item.update(runs=len(rows),graphs=rows['graph_id'].nunique(),
                    exact=int((rows['errors']==0).sum()),
                    near_1pct=int((rows['error_rate']<=.01+1e-15).sum()),
                    mean_error=float(rows['error_rate'].mean()),median_error=float(rows['error_rate'].median()),
                    min_error=float(rows['error_rate'].min()),max_error=float(rows['error_rate'].max()))
        records.append(item)
    return records


def audit_source_boundary():
    functions = {
        'engine.py':['collapsed_chain','tempered_chain','variational_em','_run_chain','_run_tempered','_v_em','_sweep'],
        'global_mcmc.py':['global_chain','_run'],
        'spectral_mcmc.py':['spectral_chain','_run'],
        'prototype_mcmc.py':['prototype_chain','_run'],
    }
    violations=[]
    inspected=[]
    for filename, names in functions.items():
        source=(ROOT/filename).read_text(encoding='utf-8')
        tree=ast.parse(source)
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in names:
                identifiers={x.id for x in ast.walk(fn) if isinstance(x,ast.Name)}
                bad=identifiers.intersection({'A','truth','true_P','oracle_P'})
                if bad: violations.append(dict(file=filename,function=fn.name,identifiers=sorted(bad)))
                inspected.append(f'{filename}:{fn.name}')
    return dict(passed=not violations, inspected_functions=inspected, violations=violations,
                review='Driver fitting calls were manually inspected: decoders receive Q,k,z0 and optional retained U only; truth/P/binary A fields are used for simulation, spectral decomposition, baselines and post-fit diagnostics. Posterior selection uses target scores, never truth.',
                excluded_oracle_diagnostics='results_diagnostics explicitly uses truth-started chains and truth-block-fitted P; excluded from every random-start recovery table.')


def main():
    directories=sorted(p for p in ROOT.glob('results_*') if (p/'config.json').exists())
    audits=[]; all_final=[]; hashes={}; source_hash_issues=[]; unhistorical=[]
    for directory in directories:
        config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        final=safe_csv(directory/'final.csv'); best=safe_csv(directory/'best_posterior.csv')
        expected,sources=expected_keys(directory,config)
        actual={key(x) for x in final.to_dict('records')}
        best_keys={key(x) for x in best.to_dict('records')}
        issues=[]; score_error=0.; records_checked=0; hashes_matched=0; selected_checked=0; numerical_ties=[]
        graph_cache={}
        for row in final.to_dict('records')+best.to_dict('records'):
            gid,method,start=key(row)
            archive=directory/gid/f'{method}_start{start}.npz'
            graph_path=directory/gid/'graph.npz'
            if not archive.exists() or not graph_path.exists():
                issues.append(dict(graph=gid,method=method,start=start,error='missing saved labels or graph'))
                continue
            if gid not in graph_cache:
                graph_cache[gid]=independent_graph(graph_path)
                if gid in sources:
                    source_graph=sources[gid]/'graph.npz'
                    if digest(graph_path)!=digest(source_graph):
                        issues.append(dict(graph=gid,error='supplement graph differs from frozen source'))
                    else: hashes_matched+=1
            graph=graph_cache[gid]
            with np.load(archive) as saved:
                name='final_labels' if row['selection']=='final' else 'best_posterior_labels'
                labels=saved[name].astype(np.int64)
                errors=mismatch(labels,graph['truth'],graph['k'])
                score=independent_score(graph,labels)
                difference=abs(score-float(row['log_posterior']))
                score_error=max(score_error,difference)
                if errors!=int(row['errors']) or abs(errors/graph['n']-float(row['error_rate']))>1e-12:
                    issues.append(dict(graph=gid,method=method,start=start,error='CSV mismatch with archived labels'))
                if difference>1e-7:
                    issues.append(dict(graph=gid,method=method,start=start,error='CSV target mismatch',difference=difference))
                if row['selection']=='final' and start>=0:
                    initial=saved['initial_labels']
                    if 'methods' in config:
                        signal_id=int(round(float(row['signal_target'])*1_000_000))
                        rng=np.random.default_rng(np.random.SeedSequence([110327,MODEL_ORDER.index(row['model']),graph['n'],signal_id,int(row['graph_seed']),start]))
                        expected_initial=rng.integers(0,graph['k'],size=graph['n'],dtype=np.int32)
                    else:
                        expected_initial=np.load(sources[gid]/f'initial_labels_start{start}.npy')
                    if not np.array_equal(initial,expected_initial):
                        issues.append(dict(graph=gid,method=method,start=start,error='paired iid random initial labels mismatch'))
                    if method=='gibbs6':
                        final_labels=saved['engine_restart_final_labels']
                        calculated=np.array([independent_score(graph,z) for z in final_labels])
                        chosen=int(saved['engine_selected_final_restart'])
                        deficit=float(calculated.max()-calculated[chosen])
                        if (deficit>1e-7 or not np.array_equal(labels,final_labels[chosen])):
                            issues.append(dict(graph=gid,method=method,start=start,error='Gibbs6 endpoint selection mismatch'))
                        elif chosen!=int(np.argmax(calculated)):
                            numerical_ties.append(dict(graph=gid,start=start,score_deficit=deficit,
                                                       reason='Label-permutation equivalent endpoints tie within independent summation precision.'))
                        selected_checked+=1
                records_checked+=1
        failures=safe_csv(directory/'failures.csv')
        if 'source_hashes' in config:
            for filename,value in config['source_hashes'].items():
                actual_hash=digest(ROOT/filename)
                if value!=actual_hash:
                    source_hash_issues.append(dict(directory=directory.name,file=filename,recorded=value,current=actual_hash))
        else:
            unhistorical.append(directory.name)
        for path in sorted(directory.rglob('*')):
            if path.is_file(): hashes[str(path.relative_to(ROOT))]=digest(path)
        item=dict(directory=directory.name,expected_final_rows=len(expected),actual_final_rows=len(final),
                  expected_graphs=len({x[0] for x in expected}),actual_graphs=final['graph_id'].nunique() if not final.empty else 0,
                  missing_final_runs=[list(x) for x in sorted(expected-actual)],
                  extra_final_runs=[list(x) for x in sorted(actual-expected)],
                  final_duplicate_rows=len(final)-len(actual),
                  missing_best_runs=[list(x) for x in sorted(expected-best_keys)],
                  best_duplicate_rows=len(best)-len(best_keys),
                  recorded_failed_graphs=len(failures),endpoint_rows_independently_checked=records_checked,
                  max_absolute_target_score_error=score_error,frozen_source_graph_hash_matches=hashes_matched,
                  Gibbs6_endpoint_selections_checked=selected_checked,numerically_tied_Gibbs6_selections=numerical_ties,issues=issues,
                  passed=bool(actual==expected and best_keys==expected and len(final)==len(expected)
                              and len(best)==len(expected) and len(failures)==0 and not issues))
        audits.append(item)
        if not final.empty:
            final=final.copy();final['directory']=directory.name;all_final.append(final)
    all_frame=pd.concat(all_final,ignore_index=True)
    core=all_frame[all_frame['directory'].isin(['results_pilot','results_replication'])]
    core_strong=core[core['signal_target']>1]
    source_boundary=audit_source_boundary()
    manifest=ROOT/'final_result_sha256.json'
    manifest.write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    report=dict(passed=all(x['passed'] for x in audits) and not source_hash_issues and source_boundary['passed'],
                counts=dict(result_directories=len(audits),archived_files_hashed=len(hashes),
                            total_final_rows=len(all_frame),core_n512_graphs=core['graph_id'].nunique(),
                            core_n512_strong_graphs=core_strong['graph_id'].nunique()),
                main_n512_strong_final_table=aggregate(core_strong,['model','method']),
                main_n512_strong_final_aggregate=aggregate(core_strong,['method']),
                main_n512_weak_final_table=aggregate(core[core['signal_target']<1],['model','method']),
                extension_final_endpoints=aggregate(all_frame[all_frame['directory'].isin(['results_n1024','results_k6','results_long'])],['directory','model','n','method']),
                supplement_final_endpoints=aggregate(all_frame[all_frame['directory'].str.contains('global|spectral|prototype')],['directory','model','n','method']),
                source_boundary=source_boundary,source_hash_mismatches=source_hash_issues,
                original_core_configs_without_historical_source_hashes=unhistorical,
                directory_audits=audits,result_hash_manifest=manifest.name,
                result_hash_manifest_sha256=digest(manifest),
                interpretation='Final endpoints only. Six-start Gibbs uses endpoint target selection. Supplements are post-hoc spectral-informed global-MH analyses with distinct budgets; they do not replace original failed outcomes. No oracle diagnostics enter recovery tables. Historical source identity cannot be established for original configs lacking source hashes; current source was independently reviewed.',
                limitation='Static source audit and endpoint reproduction do not establish posterior mixing, statistical optimality or absence of every possible external run.')
    (ROOT/'final_audit.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item()),encoding='utf-8')
    brief={key:report[key] for key in ['passed','counts','main_n512_strong_final_table','main_n512_strong_final_aggregate','extension_final_endpoints','supplement_final_endpoints','source_hash_mismatches']}
    print(json.dumps(brief,indent=2,default=lambda x:x.item()))
    if not report['passed']:
        print('Incomplete or inconsistent directories:',[x['directory'] for x in audits if not x['passed']])
        raise SystemExit(1)


if __name__=='__main__':
    main()

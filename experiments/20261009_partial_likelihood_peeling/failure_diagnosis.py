"""Post-hoc diagnosis of equal3_n2000_ch0.8_s20004.

Truth labels enter centroid, distance, confusion and counterfactual analyses.
These are diagnostic interventions, not label-free fitted algorithm results.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'spectral_greedy_peel_20261009'
if not OLD.exists(): OLD=ROOT.parent/'20261009_greedy_likelihood_peeling'
sys.path.insert(0,str(OLD))
from profile_peeling import fit_soft,metric,as_json

GRAPH='equal3_n2000_ch0.8_s20004'
OUT=ROOT/'failure_diagnosis'

def confusion(truth,pred,k=3):
    table=np.zeros((k,k),int)
    np.add.at(table,(truth,pred),1)
    return table.tolist()

def objective(q,mu,weights):
    return float(logsumexp(q@np.log(mu).T-mu.sum(1)[None,:]+np.log(weights)[None,:],axis=1).sum())

def fit_record(q,initial,truth,max_iter=300,tol=1e-6):
    result=fit_soft(q,initial,max_iter=max_iter,tol=tol)
    labels=result.pop('labels');mu=result.pop('mu');weights=result.pop('weights');trace=result.pop('trace')
    # Remove the observation-only Gamma constant for raw-score reporting.
    from scipy.special import gammaln
    gamma=float(gammaln(q+1).sum())
    result['objective']+=gamma
    for t in trace:t['objective']+=gamma
    return dict(**result,**metric(labels,truth,3),confusion=confusion(truth,labels),trace=trace),mu,weights,labels

def main():
    OUT.mkdir(exist_ok=True)
    with np.load(ROOT/'results/arrays'/f'{GRAPH}.npz') as source:
        data={key:source[key] for key in source.files}
    payload=json.loads((ROOT/'results/runs'/f'{GRAPH}.json').read_text(encoding='utf-8'))
    u,lam,z=data['u'],data['lam'],data['truth_evaluation_only']
    n,k=u.shape; x=np.sqrt(n)*u*lam
    q=np.maximum((u*lam)@u.T,1e-4*np.log(n)/n)
    centers_x=np.array([x[z==a].mean(0) for a in range(k)])
    centers_q=np.array([q[z==a].mean(0) for a in range(k)])
    C0=np.array([[10,1,2],[1,11,1],[2,1,10]],float)
    def ch(a,b,c):
        f=lambda t: np.mean((1-t)*c[a]+t*c[b]-c[a]**(1-t)*c[b]**t)
        return float(-minimize_scalar(lambda t:-f(t),bounds=(0.,1.),method='bounded').fun)
    C=C0*(.8/min(ch(a,b,C0) for a in range(k) for b in range(a)))
    covariance=np.array([np.cov(x[z==a].T,bias=True) for a in range(k)])
    pairs=[]
    for a in range(k):
        for b in range(a+1,k):
            delta=centers_x[a]-centers_x[b]
            pooled=(covariance[a]+covariance[b])/2
            sigma=float(np.sqrt(delta@pooled@delta)/np.linalg.norm(delta))
            directed_ab=float(np.sum(centers_q[a]*np.log(centers_q[a]/centers_q[b])-centers_q[a]+centers_q[b]))
            directed_ba=float(np.sum(centers_q[b]*np.log(centers_q[b]/centers_q[a])-centers_q[b]+centers_q[a]))
            pairs.append(dict(pair=f'{a+1}-{b+1}',CH=ch(a,b,C),X_centroid_distance=float(np.linalg.norm(delta)),
                pooled_SD_along_separation=sigma,distance_over_directional_SD=float(np.linalg.norm(delta)/sigma),
                mahalanobis_distance=float(np.sqrt(delta@np.linalg.solve(pooled,delta))),
                Poisson_KL_a_to_b=directed_ab,Poisson_KL_b_to_a=directed_ba))
    ids=data['poisson_scaled_fixed_k_seed_ids']
    old_ids=data['poisson_original_seed_ids']
    allids=np.r_[ids,old_ids[-1]]
    blocks=lambda profiles: np.array([[row[z==a].mean()*n/np.log(n) for a in range(k)] for row in profiles])
    oracle_scores=q@np.log(centers_q).T-centers_q.sum(1)[None,:]
    oracle_labels=oracle_scores.argmax(1)
    seed_records=[]
    for j,i in enumerate(allids):
        best=z[i]
        own_dist=np.linalg.norm(x[z==best]-centers_x[best],axis=1)
        centered=np.linalg.norm(x[i]-centers_x[best])
        seed_records.append(dict(role=['new_first','shared_second','new_third','old_third'][j],
            node_zero_based=int(i),true_class_one_based=int(best+1),X=x[i].tolist(),
            percentile_distance_to_own_X_center=float(np.mean(own_dist<=centered)*100),
            profile_total=float(q[i].sum()),block_profile_C_units=blocks(q[[i]])[0].tolist(),
            directed_Poisson_KL_to_true_centers=[float(np.sum(q[i]*np.log(q[i]/m)-q[i]+m)) for m in centers_q]))
    row_scores=q@np.log(q).T-q.sum(1)[None,:]
    selection=[]
    for method in ('poisson_original','poisson_scaled_fixed_k','poisson_scaled_remaining'):
        steps=payload['methods'][method]['events']
        for step in steps:
            rem=np.asarray(step['remaining_candidates']);totals=np.asarray(step['candidate_totals'])
            count=step['scored_other_count']
            best_by_class=[]
            for a in range(k):
                eligible=np.flatnonzero(z[rem]==a)
                if not len(eligible):continue
                winner=eligible[np.argmax(totals[eligible])]
                best_by_class.append(dict(class_one_based=a+1,node_zero_based=int(rem[winner]),
                    max_total=float(totals[winner]),remaining_nodes=int(len(eligible))))
            # Test ideal true centroids on the SAME remaining top subsets.
            ideal=oracle_scores[rem]
            ideal_totals=np.sort(ideal,axis=0)[-count:].sum(0)
            saturation=(q[rem]*np.log(q[rem])-q[rem]).sum(1)
            selected=np.asarray(step['members'])[1:]
            raw=float(row_scores[selected,step['candidate']].sum())
            sat=float((q[selected]*np.log(q[selected])-q[selected]).sum())
            selection.append(dict(method=method,stage=step['stage'],candidate=step['candidate'],
                scored_other_count=count,winner_class_one_based=int(z[step['candidate']]+1),
                selected_core_composition=np.bincount(z[step['members']],minlength=k).tolist(),
                best_candidate_by_true_class=best_by_class,
                ideal_centroid_top_totals=ideal_totals.tolist(),
                winning_raw_total=raw,winning_saturation_total=sat,winning_total_deviance=sat-raw))
    with threadpool_limits(limits=1):
        # Strict continuation from the identical original seed profiles.
        strict,strict_mu,strict_weights,strict_labels=fit_record(q,q[ids],z,max_iter=3000,tol=1e-10)
        original,original_mu,original_weights,original_labels=fit_record(q,q[old_ids],z,max_iter=3000,tol=1e-10)
        # Counterfactual replacing one duplicate with a representative of the
        # missing true class. Truth access is explicit and diagnostic only.
        missing=next(a for a in range(k) if a not in z[ids])
        eligible=np.flatnonzero(z==missing)
        target_node=int(eligible[np.argmin(np.sum((x[eligible]-centers_x[missing])**2,axis=1))])
        intervention=q[ids].copy();intervention[2]=q[target_node]
        replaced,replaced_mu,replaced_weights,replaced_labels=fit_record(q,intervention,z,max_iter=3000,tol=1e-10)
        # Sharpen the mixed middle prototype only, without adding a missing seed.
        sharpened=q[ids].copy();sharpened[1]=centers_q[z[ids[1]]]
        sharpened_result,_,_,_=fit_record(q,sharpened,z,max_iter=3000,tol=1e-10)
        oracle,oracle_mu,oracle_weights,oracle_fit_labels=fit_record(q,centers_q,z,max_iter=3000,tol=1e-10)
    fitted_block_profiles=blocks(data['poisson_scaled_fixed_k_final_parameters'])
    blend=q[(z==0)|(z==2)].mean(0)
    bridge=q[ids[1]]
    bridge_to_blend=float(np.sum(bridge*np.log(bridge/blend)-bridge+blend))
    # A controlled interpolation from bad fit to correctly allocated centroids.
    # Component 0 -> missing class 1, component 1 -> class 3, component 2 -> class 2.
    bad_mu=data['poisson_scaled_fixed_k_final_parameters'];bad_weights=data['poisson_scaled_fixed_k_final_weights']
    good_mu=centers_q[[0,2,1]];good_weights=np.array([(z==a).mean() for a in [0,2,1]])
    path=[]
    for t in np.r_[0.,1e-5,1e-4,1e-3,np.linspace(.01,1,100)]:
        value=objective(q,(1-t)*bad_mu+t*good_mu,(1-t)*bad_weights+t*good_weights)
        path.append(dict(t=float(t),objective=value))
    diagnostic=dict(graph=GRAPH,C=C.tolist(),pair_geometry=pairs,
        X_centroids=centers_x.tolist(),X_within_RMS=[float(np.sqrt(np.sum((x[z==a]-centers_x[a])**2,axis=1).mean())) for a in range(k)],
        true_centroid_block_profiles_C_units=blocks(centers_q).tolist(),seeds=seed_records,
        mixed_1_3_block_profile_C_units=blocks(blend[None,:])[0].tolist(),
        shared_second_Poisson_KL_to_mixed_1_3=bridge_to_blend,
        oracle_centroid_initial_error=metric(oracle_labels,z,k),oracle_centroid_confusion=confusion(z,oracle_labels),
        bad_initial_confusion=confusion(z,data['poisson_scaled_fixed_k_initial_E_labels']),
        bad_final_confusion=confusion(z,data['poisson_scaled_fixed_k_final_labels']),
        bad_final_block_profiles_C_units=fitted_block_profiles.tolist(),
        bad_final_weights=bad_weights.tolist(),selection=selection,
        strict_same_seed= strict,strict_old_seed=original,
        replace_duplicate_with_missing_representative=dict(node_zero_based=target_node,result=replaced),
        sharpen_shared_second_only=sharpened_result,oracle_centroid_EM=oracle,
        interpolation=path,
        caveat='All true-community distances, centers, CH, confusion and interventions are post-hoc diagnostics; no truth-informed production method is claimed.')
    (OUT/'diagnostic.json').write_text(json.dumps(diagnostic,indent=2,default=as_json),encoding='utf-8')
    pd.DataFrame(pairs).to_csv(OUT/'pair_geometry.csv',index=False)
    pd.DataFrame(path).to_csv(OUT/'interpolation.csv',index=False)
    np.savez_compressed(OUT/'plot_data.npz',x=x,truth=z,centers_x=centers_x,seeds=ids,
        old_seeds=old_ids,bad_initial=data['poisson_scaled_fixed_k_initial_E_labels'],
        bad_final=data['poisson_scaled_fixed_k_final_labels'],strict_final=strict_labels,
        strict_mu=strict_mu,strict_weights=strict_weights)
    print(json.dumps(dict(pairs=pairs,seeds=seed_records,strict={name:{key:val[key] for key in ['errors','n_iter','converged','objective']}
        for name,val in [('same_seed',strict),('old_seed',original),('replace_duplicate',replaced),('sharpen_middle',sharpened_result),('oracle',oracle)]},
        bad_final_confusion=diagnostic['bad_final_confusion'],bad_final_block_profiles_C_units=fitted_block_profiles.tolist()),indent=2),flush=True)

if __name__=='__main__':main()

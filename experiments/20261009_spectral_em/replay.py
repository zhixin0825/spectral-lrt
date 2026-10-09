"""Replay the preferred fixed-U-calibration/X-initialization method from saved spectra."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from poisson_em import fit_poisson_em
p=argparse.ArgumentParser()
p.add_argument('input_npz')
p.add_argument('--tau',type=float,default=1.)
p.add_argument('--max-iter',type=int,default=150)
a=p.parse_args()
with np.load(a.input_npz) as data:
    u,lam,h=data['u'],data['lam'],data['h0']
    x=np.sqrt(len(u))*u*lam
    y=np.linalg.solve(h.T,x.T).T
    initial=data['labels_km_X']
    # These inputs alone determine the fit; truth is only read after fitting.
    fit=fit_poisson_em(y,initial,tau=a.tau,max_iter=a.max_iter,tol=1e-6)
    truth=data['truth_evaluation_only']
    k=len(lam)
    table=np.zeros((k,k),int)
    np.add.at(table,(truth,fit['labels']),1)
    rr,cc=linear_sum_assignment(-table)
    errors=int(len(u)-table[rr,cc].sum())
print(json.dumps({'graph':Path(a.input_npz).stem,'tau':a.tau,
                  'errors':errors,'error_rate':errors/len(u),
                  'converged':fit['converged'],'iterations':fit['n_iter'],
                  'objective':fit['objective'],'reason':fit['reason']}))

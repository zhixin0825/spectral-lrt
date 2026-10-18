"""Independent finite-sample check of the sharp-rate draft (not a rate proof)."""
import os
for _var in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[_var] = '1'
import json
from pathlib import Path
import numpy as np
from scipy.linalg import subspace_angles
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent

def eigenpairs(A, k):
    val, vec = eigsh(A, k=k, which='LM', tol=2e-11,
                     v0=np.random.default_rng(194).normal(size=len(A)))
    return val, vec

def one(n, scenario, seed):
    rng = np.random.default_rng(seed)
    z = np.arange(n) % 2
    if scenario == 'dense_assortative':
        P = np.array([[.30, .10], [.10, .22]])
    elif scenario == 'dense_disassortative':
        P = np.array([[.10, .30], [.30, .15]])
    elif scenario == 'sparse_assortative':
        P = (np.log(n)/n) * np.array([[8., 2.], [2., 5.]])
    else:
        raise ValueError(scenario)
    p = P.max()
    prob = P[z[:, None], z[None, :]]
    A = (rng.random((n, n)) < prob).astype(float)
    A = np.triu(A, 1)
    A += A.T
    pa, pb = P[0, z[1:]], P[1, z[1:]]
    weights = logit(pa) - logit(pb)
    base = np.log1p(-pa) - np.log1p(-pb)
    def cgf(alpha):
        return float(np.logaddexp(np.log1p(-pa)-alpha*base,
                                 np.log(pa)-alpha*(base+weights)).sum())
    alpha = float(minimize_scalar(cgf, bounds=(0., 1.), method='bounded').x)
    tiltprob = expit(logit(pa)-alpha*weights)
    # Vertex 0 is sampled from the Chernoff tilted row law; B stays original SBM.
    x = (rng.random(n-1) < tiltprob).astype(float)
    A[0, 1:] = A[1:, 0] = x
    B = A[1:, 1:]
    ld, V = eigenpairs(B, 2)
    la, U = eigenpairs(A, 2)
    F = (U*la) @ U.T
    w = logit(P[0, z]) - logit(P[1, z])
    wV = V.T @ w[1:]
    xV = V.T @ x
    xQ = x-V @ xV
    t = U[0]
    Y = V.T @ U[1:]
    Q = U[1:] - V @ Y
    NQ = B @ Q - V @ (ld[:, None]*(V.T @ Q))
    resolvent_equation_residual = np.linalg.norm(Q*la - NQ - xQ[:, None]*t)
    h = (xQ @ Q) / t
    exact = (xV @ (Y @ Y.T) @ wV
             + np.sum(h*t*(Y.T @ wV))
             + np.sum(la*t*t*w[0])
             + np.sum(la*t*(Q.T @ w[1:])))
    direct = F[0] @ w
    linear = xV @ wV
    angle = np.sin(subspace_angles(U, np.vstack((np.zeros((1, 2)), V))))
    s = max(1., float(abs(la).max()))
    eps = .001*s/n
    q = np.clip(F, eps, 1-eps)
    mu = np.vstack([q[z == a].mean(axis=0) for a in range(2)])
    fitted_score = (q[0] @ (logit(mu[0])-logit(mu[1]))
                    + (np.log1p(-mu[0])-np.log1p(-mu[1])).sum())
    population_qscore = (q[0] @ w
                    + (np.log1p(-P[0,z])-np.log1p(-P[1,z])).sum())
    raw_score = float(x @ weights + base.sum())
    return dict(n=n, scenario=scenario, seed=seed, np=float(n*p),
                alpha=alpha, oracle_information=-cgf(alpha),
                tilted_degree=float(x.sum()),
                raw_score=raw_score, spectral_fitted_score=float(fitted_score),
                fitted_minus_raw=float(fitted_score-raw_score),
                fitted_minus_qoracle=float(fitted_score-population_qscore),
                arrowhead_residual=float(direct-linear),
                arrowhead_identity_error=float(abs(direct-exact)),
                resolvent_equation_residual=float(resolvent_equation_residual),
                sqrt_n_times_angle=float(np.sqrt(n)*max(angle)),
                n_times_gram_error=float(n*np.linalg.norm(Y@Y.T-np.eye(2), 2)),
                max_h=float(abs(h).max()))

def main():
    rows=[]
    with threadpool_limits(limits=1):
        for scenario in ['dense_assortative','dense_disassortative','sparse_assortative']:
            for n in [200,400,800,1600]:
                vals=[one(n, scenario, 20261018+rep) for rep in range(12)]
                rows.extend(vals)
                summary={k: float(np.max(np.abs([v[k] for v in vals])))
                         for k in ['arrowhead_residual','fitted_minus_raw',
                                   'fitted_minus_qoracle','arrowhead_identity_error',
                                   'resolvent_equation_residual',
                                   'sqrt_n_times_angle','n_times_gram_error','max_h']}
                print(json.dumps(dict(scenario=scenario,n=n, summary=summary)),flush=True)
    target=OUT/'tilted_arrowhead_check.json'
    target.write_text(json.dumps(dict(description=__doc__, graphs=len(rows),
                                      repetitions=12, seed_start=20261018,
                                      records=rows),indent=2))
    print('saved',str(target),flush=True)

if __name__ == '__main__':
    main()

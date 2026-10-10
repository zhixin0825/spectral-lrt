"""Spectral-only partition inference, fixed K and unknown symmetric P.

The sampling target is a fractional Bernoulli *working* posterior, not the
posterior of the original binary graph. No decoder accesses A or true labels.
Beta(1,1) priors integrate the unordered block probabilities. Labels have an
iid uniform prior; empty groups are allowed. No binomial combinatorial factor
is added when comparing different partitions.
"""
import math
import time
import numpy as np
from numba import njit
from scipy.special import betaln
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def spectrum(A, k, seed):
    rng = np.random.default_rng(seed)
    values, vectors = eigsh(A.astype(float), k=k, which="LM", v0=rng.normal(size=len(A)), tol=1e-9)
    order = np.argsort(-np.abs(values))
    return np.ascontiguousarray(vectors[:, order]), values[order]


def reconstruction(U, lam):
    raw = (U * lam) @ U.T
    Q = np.clip((raw + raw.T) / 2, 0.0, 1.0)
    np.fill_diagonal(Q, 0.0)
    return np.ascontiguousarray(Q)


def spectral_kmeans(U, lam, k, seed):
    return KMeans(n_clusters=k, n_init=10, random_state=int(seed), max_iter=300).fit_predict(U * lam)


def metrics(labels, truth, k):
    table = np.zeros((k, k), dtype=np.int64)
    np.add.at(table, (truth, labels), 1)
    row, col = linear_sum_assignment(-table)
    errors = len(labels) - int(table[row, col].sum())
    return dict(errors=errors, error_rate=errors / len(labels), ari=float(adjusted_rand_score(truth, labels)), exact=errors == 0)


@njit(cache=True)
def _counts(Q, z, k):
    n = len(z)
    sizes = np.zeros(k, np.int64)
    edges = np.zeros((k, k))
    for i in range(n):
        sizes[z[i]] += 1
        for j in range(i):
            a, b = z[i], z[j]
            if a > b:
                a, b = b, a
            edges[a, b] += Q[i, j]
    return sizes, edges


@njit(cache=True)
def _term(s, m):
    # Endpoint epsilon only repairs accumulated numerical subtraction residue.
    s = min(float(m), max(0.0, s))
    return math.lgamma(s + 1.0) + math.lgamma(m - s + 1.0) - math.lgamma(m + 2.0)


@njit(cache=True)
def _score(sizes, edges):
    k = len(sizes)
    score = 0.0
    for a in range(k):
        for b in range(a, k):
            m = sizes[a] * sizes[b] if a != b else sizes[a] * (sizes[a] - 1) / 2
            score += _term(edges[a, b], m)
    return score


def log_target(Q, z, k):
    sizes, edges = _counts(Q, np.asarray(z, dtype=np.int64), k)
    return float(_score(sizes, edges))


@njit(cache=True)
def _remove(Q, z, i, sizes, edges):
    k = len(sizes)
    sums = np.zeros(k)
    for j in range(len(z)):
        if j != i:
            sums[z[j]] += Q[i, j]
    old = z[i]
    sizes[old] -= 1
    for c in range(k):
        a, b = min(old, c), max(old, c)
        edges[a, b] -= sums[c]
    return sums


@njit(cache=True)
def _candidate_gains(sizes, edges, sums):
    k = len(sizes)
    gains = np.zeros(k)
    for a in range(k):
        for c in range(k):
            r, t = min(a, c), max(a, c)
            m = sizes[a] * sizes[c] if a != c else sizes[a] * (sizes[a] - 1) / 2
            gains[a] += _term(edges[r, t] + sums[c], m + sizes[c]) - _term(edges[r, t], m)
    return gains


def conditional_scores(Q, z, i, k):
    """Scores relative to the same removed-node state, for independent QA."""
    z = np.asarray(z, dtype=np.int64)
    sizes, edges = _counts(Q, z, k)
    sums = _remove(Q, z, i, sizes, edges)
    return _candidate_gains(sizes, edges, sums)


@njit(cache=True)
def _sweep(Q, z, sizes, edges, beta, greedy):
    changes = 0
    for i in np.random.permutation(len(z)):
        old = z[i]
        sums = _remove(Q, z, i, sizes, edges)
        gains = _candidate_gains(sizes, edges, sums)
        if greedy:
            chosen = int(np.argmax(gains))
            # Preserve the incumbent at numerical ties.
            if gains[chosen] <= gains[old] + 1e-10:
                chosen = old
        else:
            top = np.max(gains)
            probabilities = np.exp(beta * (gains - top))
            u = np.random.random() * probabilities.sum()
            chosen = len(sizes) - 1
            for a in range(len(sizes)):
                u -= probabilities[a]
                if u <= 0:
                    chosen = a
                    break
        sizes[chosen] += 1
        for c in range(len(sizes)):
            a, b = min(chosen, c), max(chosen, c)
            edges[a, b] += sums[c]
        z[i] = chosen
        changes += int(chosen != old)
    return changes


@njit(cache=True)
def _run_chain(Q, k, z0, sweeps, seed, beta, greedy, every):
    np.random.seed(seed)
    z = z0.copy()
    sizes, edges = _counts(Q, z, k)
    records = 1 + sweeps // every + int(sweeps % every != 0)
    labels = np.empty((records, len(z)), np.int64)
    scores = np.empty(records)
    times = np.empty(records, np.int64)
    changes = np.zeros(records)
    labels[0], scores[0], times[0] = z, _score(sizes, edges), 0
    best_z, best_score = z.copy(), scores[0]
    r = 1
    total_changes = 0
    since = 0
    for sweep in range(1, sweeps + 1):
        total_changes += _sweep(Q, z, sizes, edges, beta, greedy)
        since += 1
        score = _score(sizes, edges)
        if score > best_score:
            best_z, best_score = z.copy(), score
        if sweep % every == 0 or sweep == sweeps:
            labels[r], scores[r], times[r] = z, score, sweep
            changes[r] = total_changes / (since * len(z))
            r += 1
            total_changes, since = 0, 0
    return z, best_z, labels, scores, times, changes


def collapsed_chain(Q, k, z0, sweeps, seed, beta=1.0, greedy=False, record_every=5):
    start = time.perf_counter()
    z, best, labels, scores, times, changes = _run_chain(Q, k, np.asarray(z0, dtype=np.int64), sweeps, seed, beta, greedy, record_every)
    return dict(labels=z, best_labels=best, trace_labels=labels, trace_score=scores, trace_sweep=times, changed_fraction=changes, elapsed=time.perf_counter()-start)


@njit(cache=True)
def _run_tempered(Q, k, z0, sweeps, seed, betas, every):
    np.random.seed(seed)
    rcount, n = len(betas), len(z0)
    replicas = np.empty((rcount, n), np.int64)
    all_sizes = np.empty((rcount, k), np.int64)
    all_edges = np.empty((rcount, k, k))
    for r in range(rcount):
        replicas[r] = z0 if r == 0 else np.random.randint(0, k, n)
        all_sizes[r], all_edges[r] = _counts(Q, replicas[r], k)
    records = 1 + sweeps // every + int(sweeps % every != 0)
    labels = np.empty((records, n), np.int64)
    scores = np.empty(records)
    times = np.empty(records, np.int64)
    changes = np.zeros(records)
    labels[0], scores[0], times[0] = replicas[0], _score(all_sizes[0], all_edges[0]), 0
    best_z, best_score = replicas[0].copy(), scores[0]
    tries = np.zeros(rcount-1, np.int64)
    accepted = np.zeros(rcount-1, np.int64)
    index, sum_changes, since = 1, 0, 0
    for sweep in range(1, sweeps+1):
        for r in range(rcount):
            c = _sweep(Q, replicas[r], all_sizes[r], all_edges[r], betas[r], False)
            if r == 0:
                sum_changes += c
        replica_scores = np.empty(rcount)
        for r in range(rcount):
            replica_scores[r] = _score(all_sizes[r], all_edges[r])
        for r in range(sweep % 2, rcount-1, 2):
            tries[r] += 1
            log_accept = (betas[r]-betas[r+1])*(replica_scores[r+1]-replica_scores[r])
            if math.log(np.random.random()) < min(0.0, log_accept):
                temp_z, temp_s, temp_e = replicas[r].copy(), all_sizes[r].copy(), all_edges[r].copy()
                replicas[r], all_sizes[r], all_edges[r] = replicas[r+1], all_sizes[r+1], all_edges[r+1]
                replicas[r+1], all_sizes[r+1], all_edges[r+1] = temp_z, temp_s, temp_e
                temp_score = replica_scores[r]
                replica_scores[r], replica_scores[r+1] = replica_scores[r+1], temp_score
                accepted[r] += 1
        score = replica_scores[0]
        if score > best_score:
            best_z, best_score = replicas[0].copy(), score
        since += 1
        if sweep % every == 0 or sweep == sweeps:
            labels[index], scores[index], times[index] = replicas[0], score, sweep
            changes[index] = sum_changes / (since*n)
            index += 1
            sum_changes, since = 0, 0
    return replicas[0], best_z, labels, scores, times, changes, accepted, tries


def tempered_chain(Q, k, z0, sweeps, seed, betas=(1., .7, .45, .25, .12, .04), record_every=5):
    start=time.perf_counter()
    z,best,labels,scores,times,changes,accepted,tries=_run_tempered(Q,k,np.asarray(z0,dtype=np.int64),sweeps,seed,np.array(betas),record_every)
    return dict(labels=z,best_labels=best,trace_labels=labels,trace_score=scores,trace_sweep=times,changed_fraction=changes,elapsed=time.perf_counter()-start,swap_acceptance=accepted/np.maximum(tries,1),swap_accepted=accepted,swap_tries=tries)


@njit(cache=True)
def _v_em(Q, k, z0, iterations, seed):
    np.random.seed(seed)
    n = len(z0)
    R = np.zeros((n,k))
    for i in range(n):
        R[i,z0[i]]=1.0
    history = np.empty(iterations+1)
    labels_history = np.empty((iterations+1,n),np.int64)
    P = np.empty((k,k))
    # Uniform iid label prior, same as collapsed Gibbs; no weight refit.
    for iteration in range(iterations+1):
        mass=R.sum(axis=0)
        S=Q@R
        edge=R.T@S
        pairs=np.outer(mass,mass)-R.T@R
        for a in range(k):
            for b in range(k):
                P[a,b]=min(1.-1e-6,max(1e-6,edge[a,b]/max(pairs[a,b],1e-12)))
        value=0.0
        for a in range(k):
            for b in range(k):
                value+=0.5*(edge[a,b]*math.log(P[a,b])+(pairs[a,b]-edge[a,b])*math.log(1.-P[a,b]))
        for i in range(n):
            for a in range(k):
                if R[i,a]>0:
                    value-=R[i,a]*math.log(R[i,a])
            labels_history[iteration,i]=np.argmax(R[i])
        history[iteration]=value
        if iteration==iterations or (iteration>1 and abs(history[iteration]-history[iteration-1])<1e-8*n):
            return labels_history[iteration],R,P,history[:iteration+1],labels_history[:iteration+1]
        logP=np.log(P)
        log1P=np.log(1.-P)
        for i in np.random.permutation(n):
            weights=np.zeros(k)
            for a in range(k):
                for c in range(k):
                    weights[a]+=S[i,c]*logP[a,c]+(mass[c]-R[i,c]-S[i,c])*log1P[a,c]
            weights=np.exp(weights-np.max(weights))
            weights/=weights.sum()
            delta=weights-R[i]
            for j in range(n):
                for c in range(k):
                    S[j,c]+=Q[j,i]*delta[c]
            mass+=delta
            R[i]=weights
    return labels_history[-1],R,P,history,labels_history


def variational_em(Q,k,z0,max_iter,seed):
    start=time.perf_counter()
    z,R,P,history,labels=_v_em(Q,k,np.asarray(z0,dtype=np.int64),max_iter,seed)
    scores=np.array([log_target(Q,row,k) for row in labels])
    # Selects by collapsed score only for common comparison, not ground truth.
    best=labels[int(np.argmax(scores))].copy()
    return dict(labels=z,best_labels=best,trace_labels=labels,trace_score=scores,trace_sweep=np.arange(len(history)),elbo_trace=history,P=P,responsibilities=R,elapsed=time.perf_counter()-start)


def warmup():
    Q=np.array([[0.,.2,.7,.1],[.2,0.,.1,.8],[.7,.1,0.,.2],[.1,.8,.2,0.]])
    z=np.array([0,1,0,1],np.int64)
    collapsed_chain(Q,2,z,1,1,record_every=1)
    tempered_chain(Q,2,z,1,1,record_every=1)
    variational_em(Q,2,z,1,1)


if __name__=="__main__":
    warmup()
    print("engine warmup completed",flush=True)

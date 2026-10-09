"""Top-fraction raw-score seeds; intentionally leave unselected nodes.

Rows are observations, columns are candidate node parameters. Candidate self
scores are excluded. K original seeds are selected, then all n nodes enter EM.
"""
from __future__ import annotations
import itertools
import numpy as np

RULES = ('original', 'scaled_remaining', 'scaled_fixed_k')

def other_count(m, k, r, rule):
    if rule == 'original':
        return min(m // r, m-r) if r > 1 else m-1
    if rule not in RULES:
        raise ValueError(rule)
    denominator = r if rule == 'scaled_remaining' else k
    # Exact floor(M/(1.5 denominator)); reserve a candidate for each later seed.
    return min(2*m // (3*denominator), m-r)

def peel(score, k, rule='scaled_remaining'):
    score = np.asarray(score, float)
    n = len(score)
    if score.shape != (n,n) or not np.isfinite(score).all() or not 1 <= k <= n:
        raise ValueError('finite square score and 1 <= K <= n required')
    remaining = np.arange(n)
    labels = np.full(n, -1, int)
    ids, events = [], []
    for stage in range(k):
        r = k-stage
        count = other_count(len(remaining), k, r, rule)
        sub = score[np.ix_(remaining, remaining)].copy()
        np.fill_diagonal(sub, -np.inf)
        if count:
            totals = np.partition(sub, len(remaining)-count, axis=0)[-count:].sum(0)
        else:
            totals = np.zeros(len(remaining))
        chosen = int(remaining[np.argmax(totals)])
        others = remaining[remaining != chosen]
        order = np.lexsort((others, -score[others,chosen]))
        members = np.r_[chosen, others[order[:count]]]
        np.testing.assert_allclose(score[members[1:],chosen].sum(), totals[np.argmax(totals)], rtol=1e-12, atol=1e-9)
        ids.append(chosen)
        labels[members] = stage
        events.append(dict(stage=stage, remaining_components=r,
            remaining_count=len(remaining), scored_other_count=count,
            candidate=chosen, members=members.copy(),
            remaining_candidates=remaining.copy(), candidate_totals=totals.copy(),
            winning_total=float(totals[np.argmax(totals)])))
        remaining = remaining[~np.isin(remaining, members)]
    assert len(set(ids)) == k
    assert np.array_equal(np.flatnonzero(labels < 0), remaining)
    if rule == 'original':
        assert len(remaining) == 0
    return np.asarray(ids), labels, events, remaining

def validate():
    rng = np.random.default_rng(20261009091)
    checks = 0
    for n in range(2, 10):
        for k in range(1,n+1):
            for rule in RULES:
                score = rng.normal(size=(n,n))
                if n % 3 == 0:
                    score[:2,:2] = 0
                ids, labels, events, leftover = peel(score,k,rule)
                remaining = np.arange(n)
                for stage,event in enumerate(events):
                    # Independently derive the count, then enumerate all subsets.
                    r = k-stage
                    denom = 1.5*(r if rule == 'scaled_remaining' else k)
                    expected = (min(int(np.floor(len(remaining)/denom)),len(remaining)-r)
                        if rule != 'original' else
                        min(len(remaining)//r,len(remaining)-r) if r>1 else len(remaining)-1)
                    assert event['scored_other_count'] == expected
                    brute = [max(sum(score[i,c] for i in subset)
                        for subset in itertools.combinations([i for i in remaining if i!=c],expected))
                        for c in remaining]
                    np.testing.assert_allclose(event['candidate_totals'], brute, atol=1e-12)
                    assert event['candidate'] == remaining[np.argmax(brute)]
                    remaining = remaining[~np.isin(remaining,event['members'])]
                    checks += 1
    examples = {}
    for rule in RULES:
        _,_,events,leftover = peel(np.zeros((1000,1000)),3,rule)
        examples[rule] = dict(group_sizes=[len(e['members']) for e in events],leftover=len(leftover))
    return dict(brute_force_steps=checks,passed=True,examples_n1000_k3=examples)

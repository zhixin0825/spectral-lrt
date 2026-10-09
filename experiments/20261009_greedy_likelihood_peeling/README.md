# Greedy likelihood initialization

The implementation follows the user's other-node top-fraction rule, retaining selected node parameters unchanged before full-data iterations.

Read [GREEDY_LIKELIHOOD_INITIALIZATION_20261009.md](GREEDY_LIKELIHOOD_INITIALIZATION_20261009.md) for results, decoder provenance, and the population counterexample.

- `pg_peeling.py`: actual Poisson–Gaussian component density on the inherited fixed Y; the upstream decoder still uses k-means.
- `profile_peeling.py`: no decoder, raw fractional Poisson score and a clearly separated deviance control.
- `profile_score_controls.py`: no decoder, expected Poisson and Bernoulli row scores, verifying that the failure does not depend on the fractional Gamma term.
- `population_counterexample.py`: exact class-compressed calculation at log(n)/n density and duplicate-component invariance.

The complete saved-input archive preserves eigenpairs, decoder matrices, selected centers, provisional groups, per-candidate totals, fitted parameters and iteration traces. The GitHub directory preserves code, compact per-graph results and checks; its code expects exact input arrays from that archive.

With NumPy, SciPy, pandas, scikit-learn and threadpoolctl installed:

```sh
python pg_peeling.py --input /path/to/independent_20000/inputs --out rerun_pg --skip-baselines --workers 4
python profile_peeling.py --input-dir /path/to/full120/inputs --out rerun_profile --workers 4
python profile_score_controls.py --input-dir /path/to/full120/inputs --out rerun_controls --workers 4
python population_counterexample.py
```

Only eigenpairs and fixed transformations enter fitting. Original adjacency and true labels do not enter center selection or iteration; saved truth labels are consulted afterwards for evaluation. These are working likelihoods/scores, not exact graph likelihoods or exact leave-one-out likelihoods. Numerical convergence does not imply a global or true optimum.

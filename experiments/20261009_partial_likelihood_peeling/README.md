# Smaller-fraction raw-likelihood peeling

Read [REPORT_20261009.md](REPORT_20261009.md) for the 120-graph paired comparison.

Only U and Lambda enter seeding and EM. The original, fixed-K 1/(1.5K), and decreasing-r 1/(1.5r) rules share the same expected Poisson or Bernoulli working score and unchanged all-node soft EM. Smaller-fraction rules select K parameter candidates and leave unselected observations for EM.

The code reuses the unchanged EM from the adjacent `20261009_greedy_likelihood_peeling` directory. With NumPy, SciPy, pandas, scikit-learn and threadpoolctl:

```sh
python run_profile.py --input-dir /path/to/numerical-records/arrays --out rerun --workers 4
python population.py
```

The saved numerical-records arrays contain exactly the required `u`, `lam`, `truth_evaluation_only`, and `baseline_km_X` keys; fitting never reads the last two. Compact tables are archived here; complete numerical inputs, selected groups, candidate totals, parameters and traces are in the separately saved numerical archive.

The comparison reuses the prior 120 spectra rather than an independent test set. These spectral working scores are not exact graph or exact leave-one-out likelihoods.

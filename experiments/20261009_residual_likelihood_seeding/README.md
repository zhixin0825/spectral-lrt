# Likelihood residual selection and parameter replacement (2026-10-09)

This experiment follows the failed fixed-K `1/(1.5K)` initializer. It tests
likelihood versions of k-means++ and deterministic selection by global gain,
then independently validates the frozen methods on another 120 graphs.

Read [REPORT_20261009.md](REPORT_20261009.md) for the explanation, all methods,
negative results, and independent validation. `verification.json` contains
numerical checks; the two results directories contain complete compact tables.
Full numeric arrays and candidate/EM traces are retained separately.

## Production entry point

```python
from residual_seed import fit_spectral_profiles

# u: n by K eigenvectors; lam: K eigenvalues, corresponding to u.
result = fit_spectral_profiles(u, lam, method='repair')
labels, profile_means, weights = result['labels'], result['mu'], result['weights']

# Alternative: grow centers using full-data likelihood gain and EM.
result = fit_spectral_profiles(u, lam, method='global_gain')
```

The repair route retains the user's smaller peeling initializer, fits on all
observations, and tries replacing one component at a time. It accepts only a
final mixture-likelihood improvement larger than `1e-5`. All proposals are
evaluated against the original endpoint; this is one pass, not repeated search.

Fitting uses only `U,Lambda`. The spectral profile score is a Poisson working
likelihood, not an exact graph/LOO likelihood. The classical k-means++ theorem
does not establish recovery guarantees for this adapted score.

Dependencies are numpy, scipy, pandas, scikit-learn and threadpoolctl. The
implementation imports the archived sibling experiments
`20261009_partial_likelihood_peeling` and `20261009_greedy_likelihood_peeling`.
Generation imports `20261009_spectral_em`.

## Reproduce independent validation

From this experiment directory in a checkout of the whole repository:

```bash
python generate_independent.py
python residual_seed.py --input-dir independent_inputs --out validation_results --workers 4
```

The graph seeds are `30000..30004`; they were declared before fitting.
The design spectra use the previous experiment's seeds `20000..20004`. Restore
the separately retained numerical archive to obtain the exact spectra:

```bash
python restore_records.py /path/to/spectral_residual_seeding_numeric_records_20261009.tar.xz
python residual_seed.py --input-dir design_inputs --out design_results --workers 4
python verify_results.py
```

Each input NPZ must contain `u`,
`lam`, `truth_evaluation_only`, `baseline_km_X`. The last two are accessed only
after fitting and selecting all methods. They are unnecessary for the production
entry point. Saved arrays include the spectra, every method's labels and means,
weights, seed IDs and all greedy restart labels; saved JSON includes global
candidate gains, EM traces, and every replacement proposal.

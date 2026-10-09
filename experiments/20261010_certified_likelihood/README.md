# Complete likelihood decoder implementation checks

The production method is Work.certified_growing() / method='global_gain_certified' in ../20261009_residual_likelihood_seeding/residual_seed.py.

The check loads that exact production class and its exact floored-simplex optimizer, then generates 24 new SBM graphs: six model shapes, n=400/800, two seeds each. Fitting and selection never access true labels. Truth is used afterward to record the misclassification count.

All 24 cases passed: ceil(log(n)) independent residual-start count; highest final working likelihood selection including original growing; mandatory final E/M; unique seed IDs; positive profile box; normalized floored weights; and final objective never lower than the original growing endpoint. The fresh numerical errors are retained in the CSV, including weak finite-sample conditions.

These are implementation checks, not a 240-graph benchmark rerun and not an empirical proof of a finite-sample accuracy advantage. The asymptotic rate follows from the manuscript's full probability proof.

Run from the repository with:
```bash
python experiments/20261010_certified_likelihood/verify_certified.py
```
Dependencies are numpy, scipy, and threadpoolctl. The inherited production imports are unchanged; the check isolates the exact class definitions to avoid unrelated archived entrypoint imports.

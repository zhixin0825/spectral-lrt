# General-density likelihood decoder: finite-graph audit

The implementation in `spectral_likelihood.py` uses the full Bernoulli profile score, the original global-mean growing fit, and deterministic component replacement. Every replacement trial chooses its new row by the same all-node likelihood gain, starts with uniform mixture weights, and takes one M-step. The round budget is `ceil(log(n))`; a final E/M update is mandatory. This audit uses the original adjacency eigenpairs, including negative signal eigenvalues, without degree regularization.

The 60-graph benchmark verifies the numerical behavior of this implementation and records its finite-sample limitations. It does not establish an asymptotic exponent or prove that replacement improves classification on these graphs. A separate exact population counterexample and its successful replacement repair are documented in [population_audit/README.md](population_audit/README.md).

The manuscript uses the actual symmetric Bernoulli matrix `P` and the scalar density `p = max_ab P_ab`. Its uniform assumptions are `n_a >= pi0*n`, `P_ab >= kappa*p`, `sigma_min(P) >= kappa*p`, `p <= 1-eta`, and `p = Ω(log n/n)`, with fixed positive constants. These imply expected degrees of order `np`, which may differ across blocks. Neither `P/p` nor community proportions must converge. The fixed matrices below specify this finite benchmark; they do not impose an additional convergence assumption on the theorem.

## Reproduce

From the repository root:

```bash
python experiments/20261010_general_density/benchmark.py --workers 2
```

The default writes a new `benchmark_rerun` directory; `--pilot` instead uses `benchmark_pilot`. An explicit `--output-dir PATH` chooses another destination. Results are resumed only when the complete saved protocol matches the current code and design, so the archived `benchmark_results` evidence is not overwritten by a rerun. The design, fitting-time implementation hashes, dependencies, all original adjacency matrices, and the exact retained eigenpairs are saved with the original results.

## Decode supplied eigenpairs

From the repository root, with `U` an n-by-K orthonormal array and `eigenvalues` the length-K array of its signed adjacency eigenvalues:

```python
import sys
sys.path.insert(0, "experiments/20261010_general_density")
from spectral_likelihood import SpectralLikelihood

fit, audit = SpectralLikelihood(U, eigenvalues).decode()
labels = fit.labels
```

The eigenpairs should be those with the K largest absolute eigenvalues. The fitting class has no adjacency or truth-label input. Its `rounds=None` default selects the proved round budget; smaller explicit budgets are experimental overrides.

## Fixed design

There are five choices of K, two density settings, three matrix families, and two independent graph seeds per cell:

| Item | Values |
|---|---|
| Number of nodes | 320 |
| K | 2, 3, 4, 5, 6 |
| Target average expected degree | `3 log(n)` or `0.18 (n-1)` |
| Graphs per K/family/density cell | 2 |
| Total graphs | 60 |
| Growing EM budget | At most 100 M-steps per stage |
| Replacement budget | At most `ceil(log(n)) = 6` rounds |
| Replacement fitting | One M-step per component-removal candidate |
| Final fitting | One mandatory M-step, also given to the growing baseline |

The assortative family has balanced blocks and a block matrix proportional to `11' + 5 I`. The disassortative family has unequal block sizes and a matrix proportional to `(1.3 11' - 1.1 I)` with unequal positive row/column scaling; its informative eigenvalues include negative values. The weak general family has a fixed positive symmetric, full-rank matrix close to a common connection probability, with unequal block sizes. Its weak directions make this a deliberately difficult finite-sample control.

For every graph, the proportionality constant is chosen using the actual integer block sizes so that the target average expected degree is exact. Expected degrees need not be equal across blocks. The complete probabilities and each block's expected degree are saved, so the density description does not hide a homogeneous-degree assumption.

Each method uses the same K adjacency eigenpairs with the largest absolute eigenvalues. Full-score and sparse-score methods receive exactly the same doubly clipped spectral reconstruction `q`; only their working likelihood score changes. The two K-means controls use ten Lloyd starts on `sqrt(n) U` and `sqrt(n) U Lambda`, respectively. Ground truth determines the simulated graph and is used to evaluate completed fits; it is not passed to fitting, likelihood comparison, replacement selection, or stopping.

## Mean misclassification fractions

Each family/density row averages the ten graphs spanning K=2,...,6 and the two seeds. These are descriptive averages over this design, not estimates for one homogeneous model.

| Family | Density | Full Bernoulli | Sparse score | K-means on U | K-means on U Lambda |
|---|---|---:|---:|---:|---:|
| Assortative | `3 log(n)` | 0.008125 | 0.008125 | 0.008125 | 0.008125 |
| Assortative | Dense | 0 | 0 | 0 | 0 |
| Disassortative, unequal sizes | `3 log(n)` | 0.378750 | 0.377500 | 0.407813 | 0.391875 |
| Disassortative, unequal sizes | Dense | 0.074063 | 0.073438 | 0.095313 | 0.084063 |
| Weak general, unequal sizes | `3 log(n)` | 0.663125 | 0.664375 | 0.662500 | 0.658438 |
| Weak general, unequal sizes | Dense | 0.657500 | 0.659063 | 0.664063 | 0.660000 |
| **All 60 graphs** | | **0.296927** | **0.297083** | **0.306302** | **0.300417** |

For the full Bernoulli score, replacement and growing have identical misclassification on all 60 graphs. The same is true for the sparse score. Therefore the table applies to either version within each score family. There were seven accepted replacement trials, but their apparent soft-loss improvements were only `2.27e-13` to `2.05e-12`, at floating-point scale. They are not empirical evidence of a substantive replacement benefit.

Against K-means on U, the full decoder has lower error on 26 graphs, equal error on 27, and higher error on 7. Against K-means on U Lambda, the counts are 21, 28, and 11. Against the sparse score, the full score wins on 8 graphs, ties on 43, and loses on 9. Thus this small benchmark does not demonstrate a systematic finite-sample advantage of the full score over the sparse control. Its purpose is to verify the declared full-score algorithm and disclose its behavior at both densities.

The weak general family performs poorly under every decoder. Full rank and positive entries alone do not imply strong separation at n=320. `model_signal_diagnostics.json` records the eigenvalues of the rank-K population signal matrix `N^(1/2) P N^(1/2)` and a degree-based scale for inspecting these examples. The diagnostic ratio is descriptive, not a recovery theorem. The poor cases remain in every summary and all original records.

## Numerical verification

- 15,444 EM updates were recorded. The largest positive soft-loss change was `2.7284841053187847e-12`.
- Across 127 evaluated replacement rounds, selected soft loss never increased.
- All 874 candidate scans saved every candidate's hard loss, including the excluded-node mask through infinite losses, and the stored selected ID agrees with the minimum.
- Every mandatory final update was executed, with the expected two objective entries. The largest positive final-update change was `2.0463630789890885e-12`.
- All 60 adjacency matrices are symmetric with zero diagonal and are distinct. Every P is a valid symmetric probability matrix.
- The largest saved eigenpair residual `||A U - U Lambda||_F` was `1.076751787811479e-13`; the largest orthogonality error was `4.52372910726312e-15`.
- Direct calls to the unmodified engine on two difficult archived graphs, for both score families, reproduce the benchmark labels exactly.

The positive EM changes above are rounding-level observations. Exact arithmetic supplies the monotonicity argument; the finite calculation only checks that the code respects it to numerical precision.

## Saved evidence

`benchmark_results/protocol.json` declares all 60 graph seeds and models, method settings, the score-control relationship, and SHA-256 hashes of the engine and benchmark. `environment.json` records Python, NumPy, SciPy, and scikit-learn versions.

`summary.json` contains aggregate results, while `per_graph_methods.csv` contains all 360 graph/method rows. `independent_validation.json` contains eigenpair checks, direct engine replays, and all paired win/tie/loss counts.

Every graph has its own directory with:

- `metrics.json`: graph seed, block sizes, expected and observed degree summaries, method errors, and numerical audit totals.
- `arrays.npz`: the original adjacency matrix A, model matrices B/P, evaluation labels, retained U/Lambda, every method's labels, fitted likelihood profiles/weights, and all candidate-scan losses.
- `loss_audit.json.gz`: growing-stage losses, every replacement candidate and its EM losses, accepted replacements, final-update losses, and the scan metadata that indexes the arrays.

No graphs or methods are omitted from the aggregate because of poor results. The complete graph evidence is about 7.8 MB before adding documentation and manifests.

## Fitting-time versions and later boundary maintenance

The protocol hashes identify the preserved `benchmark_results/engine_at_fitting.py` and `benchmark_results/benchmark_at_fitting.py`. After fitting, the engine gained a one-line feasibility correction for the initial rounded mean: an empty graph at n=11 can otherwise produce a mean infinitesimally below the lower floor. The correction is the same clipping already used after weighted M-steps. The empty graph now decodes successfully.

`post_fit_maintenance.json` verifies that the changed mean expression is bitwise identical before and after clipping on all 60 saved graphs. Thus the archived fitting paths and results are unaffected. The reproduction CLI also gained a selectable output directory and a protocol-match guard; no model, candidate selection or fitting calculation changed. Both the fitting-time hashes and current maintained-code hashes are recorded, rather than silently changing the frozen protocol.

The theoretical default floor requires the admissibility condition stated in the manuscript. This finite-graph audit describes the chosen numeric floor and the actual model matrices; it does not assert that positivity and full rank alone establish that condition, or that n=320 is already in an asymptotic regime.

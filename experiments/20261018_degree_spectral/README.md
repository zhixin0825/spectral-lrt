# Degree plus spectral likelihood: general branch

`degree_spectral_likelihood.py` implements a retained-degree, regularized-spectral
version of the complete Bernoulli working-score algorithm. It accompanies the
later general-density section; it does not replace the logarithmic-density
sharp theorem or establish a sharp theorem below that density.

The public API is:

```python
from degree_spectral_likelihood import DegreeSpectralLikelihood

# D: (n,), U: (n,K), Lambda: (K,). The signed eigenpairs are from W A W.
decoder = DegreeSpectralLikelihood(D, U, Lambda, c0=0.02)
R = decoder.reconstruct()
fit = decoder.decode(max_em_iter=25)
labels = fit["labels"]
```

The prescribed regularizer is `w_i = min(1, 2*mean(D)/D_i)`, with `w_i=1` for
isolated vertices. Computing `D` and the eigenpairs is a preprocessing operation.
The decoder receives only these retained arrays and has no adjacency argument.

With `X=[1,WU]` and `Y=[D,W^-1 U Lambda]`, it forms `R=Y pinv(X)=A P_X`.
This exact action identity also handles dependent feature columns. `R` is
generally signed and asymmetric; it is neither clipped nor symmetrized.
Consequently its row sums retain `D`, and its action on every retained feature
is exact. This does not imply exact recovery of arbitrary block edge counts.

Only fitted profiles are constrained to `[epsilon,1-epsilon]`, where
`epsilon=c0*max(1,mean(D))/n`. Scores contain both edge and nonedge terms.
The exact M step clips the responsibility-weighted mean of the signed rows.
Initialization grows profiles by all-node positive score gain, excluding node
IDs already appended during growing, and performs joint E/M fits. Replacement
rounds test each deletion position's best all-node-gain candidate, compare fitted
soft objectives against the incumbent, and retain the incumbent on ties. A
mandatory final E/M update precedes classification. Candidate and component
ties use the smallest array index. Responsibilities and output labels use
zero-based indices.

The default replacement budget is `ceil(log(n))`; a round with no improving
trial ends replacement. Each fit performs at least one exact M step and at
most `max_em_iter` updates. This reference implementation caches all node
candidate scores, using quadratic storage and cubic preprocessing.

For saved retained inputs, the command-line interface accepts an NPZ containing
exactly the arrays needed by the decoder:

```sh
python degree_spectral_likelihood.py retained_inputs.npz fit.npz --em-steps 25
```

It writes labels, fitted profiles, weights and responsibilities to `fit.npz`,
and fitting metadata to `fit.json`.

Run the verification with Python, NumPy and SciPy:

```sh
python verify_degree_spectral.py
```

`CHECKS.json` records the reproducible seed, dependency versions, source hashes,
and results for four small fixtures. These checks cover exact retained actions,
degree preservation, projector and block-count identities, an explicitly
signed/asymmetric reconstruction, dependent feature columns, negative retained
eigenvalues, isolated vertices, full-spectrum reconstruction, clipped-mean
coordinate optimization for signed observations, deterministic fitting, and
E/M objective monotonicity. They are not a community-recovery performance
comparison or evidence of a statistical rate below logarithmic degree.

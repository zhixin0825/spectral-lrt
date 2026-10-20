# Constant oracle-risk bounds for normalized spectral likelihood

Research proof note, 20 October 2026. This source builds the complete 39-page
note for the spectral-lrt project. It gives literal expected-risk bounds
\(\mathbb E\ell(\widehat z,z)\le C R_*(n)\), with \(C\) independent of \(n\)
on the explicitly stated parameter classes.

## Results and exact scope

| Result | Oracle reference | Conditions |
| --- | --- | --- |
| Theorem 1.1: ordinary SBM, signed Euclidean spectral reconstruction with finite growing/replacement likelihood fitting | Raw-edge Bernoulli LRT with true block parameters and neighbor labels | Fixed arbitrary \(K\), comparable group sizes, \(p/2\le P_{ab}\le p\le1/4\), \(\sigma_{\min}(P)\ge\kappa p\), \(np\ge20\log n\) |
| Theorem 1.2: DC-SBM, degree-metric reconstruction with estimated-parameter Bernoulli profile LRT | Raw-edge profile LRT with true neighbor parameters and the target degree profiled separately for each candidate | Positive bounded degree parameters, exact within-block mean normalization, positive full-rank compact block profiles, \(C_0\log n\le np\), \(np^2\le C_1\) |
| Corollary 1.3: stronger DC-SBM comparison | Raw-edge LRT additionally supplied with the target's true degree parameter | The preceding sparse conditions and equal actual finite-\(n\) block exposures |
| Proposition 6.1: broader ordinary SBM class | The same true-parameter ordinary oracle | General positive compact profiles bounded away from one, and a deterministic information-degree margin |
| Theorem 7.1: general DC-SBM obstruction | Known-target-degree local oracle | Two explicit exactly normalized Bernoulli parameter sequences make the risk ratio diverge for every full-adjacency algorithm |

The reference risks are local row likelihood risks. They do not use a global
exact-group-size constraint to infer a missing label. The profile oracle is a
generalized likelihood-ratio reference, not a fixed-known-target-degree Bayes
test.

## What is proved about the algorithm

The encoder retains only observed degrees and the informative normalized
adjacency eigenpairs. Both decoders can be computed from those retained
quantities. Leave-one-out systems and true labels are used only in the proof.

The ordinary theorem covers a finite replacement schedule, exact E/M
updates, and the mandatory endpoint order. It assumes no global maximizer
of the mixture criterion and no correct initialization.

The DC theorem uses the corrected reconstruction
\(D^{1/2}U\Lambda U^\top D^{1/2}\), its explicit hollow correction, a
deterministic radius-count initializer, estimated nuisance parameters, and
exact Bernoulli target profiling. One refinement pass suffices. The stated
finite repeated hard-refit procedure is also covered.

The new DC theorem must not be attached to an earlier experimental curve
computed from the Euclidean reconstruction \(AQQ^\top\). The retained input
is the same, but the DC decoder changes.

The manuscript proves the full-to-recomputed-LOO score bounds, the tilted
bias and quadratic-moment bounds, the \(d^{-1/2}\) tail prefactor, and the
unconditional exceptional-event comparison. It does not assume any of
these as unverified algorithmic success conditions.

## Limits of the result

The constants can depend on fixed signal conditioning, group proportions,
degree-parameter bounds, the floor coefficient, and \(C_1\). The result
does not assert a small numerical constant, convergence of the risk ratio,
or uniformity when the normalized signal singular value tends to zero.
The range \(np=o(\log n)\) and the general DC range \(np^2\to\infty\)
are not covered. General DC-SBM cannot satisfy the known-target oracle
claim without an additional restriction; the counterexample establishes
this even for a classifier observing the full graph.

The theorem uses exact arithmetic. Fixed floating-point rank cutoffs or
optimization tolerances require separate quantitative error control.

## Build

Use a standard TeX Live or MiKTeX installation with the packages named in
**main.tex**. From this directory run:

~~~sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
~~~

The output is **main.pdf**. The supplied **main.bbl** also permits a build
without running BibTeX. **references.bib** is the editable bibliography.

## Source map

- **statements.tex**: models, oracle definitions, algorithms, and main statements.
- **sharp_tail.tex**: sharp Bernoulli transfer and fixed-\(K\) oracle comparison.
- **normalized_geometry.tex**: actual normalized spectral geometry and LOO.
- **sbm_proof.tex**: finite ordinary likelihood fitting and sharp risk.
- **dcsbm_proof.tex**: corrected DC reconstruction, initialization, fitting, and profile risk.
- **probability_closure.tex**: every exceptional event paid at the oracle scale.
- **impossibility.tex**: full-adjacency known-target-oracle counterexample.
- **scope_audit.tex**: implementation changes, exact scope, and audit record.

The proof has undergone internal independent checks; it has not undergone
external peer review. No simulation estimate is used as a proof of a
bounded asymptotic risk ratio.

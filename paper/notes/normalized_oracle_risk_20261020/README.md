# Constant oracle-risk bounds for normalized spectral likelihood

Research proof note, 20 October 2026, revision 2. This source builds the 48-page
note for the spectral-lrt project. It gives literal expected-risk bounds
\(\mathbb E\ell(\widehat z,z)\le C R_*(n)\), with \(C\) independent of \(n\)
on the explicitly stated parameter classes.

## Results and exact scope

| Result | Oracle reference | Conditions |
| --- | --- | --- |
| Theorem 1.1: ordinary SBM, signed Euclidean spectral reconstruction with finite growing/replacement likelihood fitting | Raw-edge Bernoulli LRT with true block parameters and neighbor labels | Fixed arbitrary \(K\), comparable group sizes, \(p/2\le P_{ab}\le p\le1/4\), \(\sigma_{\min}(P)\ge\kappa p\), \(np\ge5\log n\) |
| Theorem 1.2: DC-SBM, degree-metric reconstruction with estimated-parameter Bernoulli profile LRT | Raw-edge profile LRT with true neighbor parameters and the target degree profiled separately for each candidate | Positive bounded degree parameters, exact within-block mean normalization, positive full-rank compact block profiles, \(C_0\log n\le np\), \(np^2\le C_1\) |
| Corollary 1.3: stronger DC-SBM comparison | Raw-edge LRT additionally supplied with the target's true degree parameter | The preceding sparse conditions and equal actual finite-\(n\) block exposures |
| Proposition 6.1: broader ordinary SBM class | The same true-parameter ordinary oracle | General positive compact profiles bounded away from one, and a deterministic information-degree margin |
| Proposition 5.11: classical balanced DC-SBM oracle alignment | Known-target-degree, profile, and true-group voting local references | Fixed arbitrary K, balanced planted partition, bounded positive theta with group means one, q/p in a fixed compact subinterval of (0,1), np tending to infinity, and bounded np^2; no logarithmic lower bound for this oracle comparison |
| Theorem 7.1: general DC-SBM obstruction | Known-target-degree local oracle | Two explicit exactly normalized Bernoulli parameter sequences outside the classical assortative class; this is not an obstruction to the balanced planted-partition result |
| Proposition 9.1 and Lemmas 9.2–9.3: sparse proof ingredients | Degree-floor geometry and sharp transfer with exceptional coefficients | Geometry is proved for np tending to infinity; the transfer lemmas require stated coefficient-tail hypotheses and do not by themselves prove the actual sparse algorithm theorem |

The reference risks are local row likelihood risks. They do not use a global
exact-group-size constraint to infer a missing label. The profile oracle is a
generalized likelihood-ratio reference, not a fixed-known-target-degree Bayes
test.

## Revision 2: oracle alignment and the sparse extension

Section 5.10 supplies a full sharp Bernoulli comparison between the classical
balanced DC-SBM local oracle of Gao–Ma–Zhang–Zhou, the capped profile oracle,
and true-group neighbor voting. Their common exact-risk scale retains the
square-root prefactor. This comparison works when np grows arbitrarily
slowly. The actual spectral algorithm bound still requires its separate
logarithmic-range theorem.

Section 6.2 improves the ordinary-SBM lower-degree condition from 20 log n
to 5 log n by averaging the oracle lower bound over every vertex. It changes
no algorithm step and is not presented as removal of the logarithmic scale.

Section 9 proves normalized degree-floor concentration, weighted degree
energy, block-space residuals, and DC tilted mean/variance cancellation
under np tending to infinity. It gives an exactly degree-preserving spectral
reconstruction and two new sharp-transfer lemmas, including an expected
probability of the tested row hitting large coefficients. The actual floor
coefficient-count tail, full-to-LOO score comparison, and fitted-parameter
control are explicitly left as outstanding mathematical estimates.

The audit notes are supporting independent working records. The TeX
statements and their scope qualifications are authoritative.

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
The actual algorithmic range \(np=o(\log n)\) and the general DC range
\(np^2\to\infty\) are not covered by the main theorems. The new oracle
comparison and degree-floor geometry do cover sublogarithmic degree on
their separately stated classes. General DC-SBM cannot satisfy the known-target oracle
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
- **dc_oracle_bridge.tex**: sharp classical balanced DC-SBM oracle comparison, with no logarithmic-degree premise.
- **probability_closure.tex**: every exceptional event paid at the oracle scale.
- **impossibility.tex**: full-adjacency known-target-oracle counterexample.
- **scope_audit.tex**: implementation changes, exact scope, and audit record.
- **sparse_extension_status.tex**: proved degree-floor geometry, exact degree correction, flexible sharp transfers, and the outstanding actual sparse theorem.
- **audits/**: independent DC-reference, probability-constant, regularized-encoder, and sparse-transfer working audits.

The proof has undergone internal independent checks; it has not undergone
external peer review. No simulation estimate is used as a proof of a
bounded asymptotic risk ratio.

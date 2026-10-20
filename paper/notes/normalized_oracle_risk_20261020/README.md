# Constant oracle-risk bounds for normalized spectral likelihood

Research proof note, 20 October 2026, **revision 3**.

**The primary regularized algorithm theorems have no logarithmic expected-degree lower bound.** On the fixed structural classes below they prove the literal inequality
\[
\mathbb E\ell(\widehat z,z)\le C R_*(n),
\]
where \(C\) is independent of \(n\), under \(d=np\to\infty\) and \(np^2\le C_1<\infty\).

- [Reviewed PDF](normalized_spectral_oracle_risk_20261020.pdf)
- [Reproducible TeX source archive](normalized_spectral_oracle_risk_20261020_source.zip)
- [Main source](main.tex)
- [Build and review record](BUILD.txt)

## Primary results and exact scope

| Model and actual decoder | Oracle reference | Conditions |
| --- | --- | --- |
| Ordinary SBM: empirical degree floor, exact-degree metric reconstruction, radius-count initialization, block Bernoulli fit, and unprofiled Bernoulli LRT | Raw-edge Bernoulli row LRT supplied with the true block matrix and neighbor labels | Fixed arbitrary \(K\), comparable group sizes, general positive compact full-rank block profiles, \(np\to\infty\), bounded \(np^2\) |
| DC-SBM: the same floor and reconstruction, radius-count initialization, clamped normalized degree estimates, exact Bernoulli block fitting, and target profile LRT | Raw-edge profile LRT supplied with true neighbor parameters, with the target degree profiled separately under each candidate | The preceding degree range; fixed bounded positive theta, exact within-block theta means one, positive compact full-rank block profiles |
| DC-SBM with equal finite-\(n\) block exposures | Raw-edge LRT additionally supplied with the target's true degree parameter | The preceding DC class and equal exposures; includes the classical balanced planted partition |
| General DC-SBM without an additional restriction | Known-target-degree local LRT | A uniform constant ratio is impossible; the retained two-point counterexample lies outside the equal-exposure class |

The main statement is **Theorem 1.1**. Section 10 specifies the regularized algorithm and gives the complete no-logarithmic-lower-bound proof. Both new decoders permit the specified finite hard refits, including finite adaptive stopping after at least one update.

The reference risks average the actual directional error at each fixed true vertex. They are local row likelihood risks; they do not use a global exact-group-size constraint to infer a missing label. The DC profile oracle is a generalized likelihood-ratio reference, distinct from the oracle that knows the target theta.

## What revision 3 repairs

The 9 October degree-plus-spectrum draft already contained a sparse same-threshold truncated-row-energy argument and local leave-one-out analysis. It covered \(np\to\infty\), including \(np=o(\log n)\), at the leading-exponent level. Archiving that argument did not establish that it was false. The later constant-ratio proof failed to bring its sparse machinery forward.

Revision 3 recovers that chain and supplies the stronger estimates required for a constant risk ratio:

1. **Positive empirical degree floor.** Set \(G_i=\max(D_i,a_0\bar D)\). Isolated vertices no longer trigger a whole-graph fallback. The previous fallback is shown to fail on a concrete \(np=\log\log n\) SBM.
2. **Actual floor-normalized row tails.** A same-threshold recursion gives exponentially small expected truncated spectral-row energy. It accounts for the square-root cap and the recomputed empirical threshold. No global row-incoherence event is assumed.
3. **Local scalar LOO at \(O(1)\) cost.** Only coordinates hit by the tested row need bounded deleted eigenvector rows. The proof includes degree correction and simple hollowing.
4. **Fitting from aggregate weighted errors.** Wrong-label mass, block totals, clamped nuisance estimation, exact signed Bernoulli fits, and the invariant basin are controlled without a global entrywise reconstruction bound.
5. **Unconditional probability closure at every diverging scale.** Polynomially accurate regularized geometry suffices below a fixed multiple of \(\log n\). Above that internal proof split, the centered raw operator-norm bound supplies an arbitrarily large fixed exponential rate, and the positive floor turns it into aggregate geometry. No minimum-raw-degree event or information-degree margin is needed for the new regularized theorem.

The sharp \(d^{-1/2}\) prefactor is retained. The result does not assert that the risk ratio tends to one.

## Actual algorithm and retained information

The encoder retains the raw degrees and the \(K\) eigenpairs of \(G^{-1/2}AG^{-1/2}\) with largest absolute eigenvalues. The decoder computes
\[
R_0=G^{1/2}U\Lambda U^\top G^{1/2},\qquad
R_1=R_0+(D-R_0\mathbf1)\mathbf1^\top/n,
\]
then applies the simple row hollowing defined in Section 10.4. The resulting \(R^h\) has zero diagonal and \(R^h\mathbf1=D\) exactly. No original-edge lookup is needed after encoding.

Radius-count initialization uses fixed radii \(1/8\) and \(1/4\). Ordinary SBM fixes theta at one and retains the full Bernoulli nonedge intercept. DC-SBM uses bounded clamped degree estimates normalized within fitted groups and profiles the target theta for each candidate. The mathematical decoder uses exact arithmetic and exact likelihood optimizers.

## Separate earlier algorithm results

Sections 3–6 retain the earlier unregularized proofs. These are different algorithm statements:

- The ordinary Euclidean free-profile growing/replacement/soft-EM schedule retains its earlier logarithmic-range theorem, including probabilities beyond bounded \(np^2\).
- The earlier unregularized DC metric decoder retains its separately proved logarithmic-range result.
- Those restrictions are **not assumptions of the primary regularized theorem**.
- The unchanged free-profile schedule is not silently identified with the new radius-count/block-LRT algorithm. Its previous global generated-profile box cannot be inferred from the aggregate floor geometry.

Earlier experimental curves evaluate their recorded algorithms. They should not be relabeled as experiments for this regularized variant.

## Limits

Constants may depend on fixed signal conditioning, group proportions, theta bounds, floor and clamping constants, and \(C_1\), but not on \(n\). The theorem does not cover a normalized signal singular value tending to zero or general DC-SBM with \(np^2\to\infty\). General DC-SBM still cannot uniformly match the oracle supplied with the unknown target theta without an additional restriction.

A fixed floating-point rank cutoff or optimization tolerance requires separate quantitative error control. The proofs and checks are internal research review, not external peer review. No simulation estimate is used as proof.

## Build

From this directory, with a standard TeX Live or MiKTeX installation:

~~~sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
~~~

This produces **main.pdf**. The supplied **main.bbl** also permits a build without BibTeX. The reviewed deliverable is copied to **normalized_spectral_oracle_risk_20261020.pdf**. Source and artifact checksums are recorded separately.

## Source map

- **statements.tex**: model, precise oracle definitions, primary regularized theorem, and separately labeled earlier results.
- **sharp_tail.tex**: sharp Bernoulli transfer and fixed-\(K\) oracle comparison.
- **normalized_geometry.tex**, **sbm_proof.tex**, **dcsbm_proof.tex**, **probability_closure.tex**: retained unregularized proofs and reusable spectral/oracle tools.
- **dc_oracle_bridge.tex**: sharp classical balanced DC-SBM oracle alignment without a logarithmic degree premise.
- **impossibility.tex**: general known-target-theta obstruction.
- **scope_audit.tex**: exact implementation and theorem boundaries.
- **sparse_extension_status.tex**: degree-floor geometry and sharp transfer tools; the former open interfaces are now cross-referenced to their proofs.
- **sparse_restoration.tex**: recovered history, explicit regularized algorithm, final theorems, and risk assembly.
- **restored_row_tail.tex**: same-threshold truncated row energy and local exceptional mass.
- **restored_local_loo.tex**: scalar full-to-LOO comparison with the empirical floor.
- **restored_dc_fit.tex**: weighted fitting, signed Bernoulli profiles, finite hard-refit invariance, and ordinary unprofiled Bernoulli LRT.
- **restored_high_range.tex**: exponentially accurate aggregate geometry and the final all-scale closure.
- **audits/**: historical evidence and independent mathematical review records.


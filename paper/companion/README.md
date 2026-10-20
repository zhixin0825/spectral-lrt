# Companion notes: degree correction and additional likelihood results

This 57-page working companion collects material separated from the ordinary-SBM article, [Spectral likelihood classification in stochastic block models](../sbm/sbm_spectral_likelihood.pdf). The first article is self-contained. This companion recalls shared tools by name and local statements; compiling it does not require the first article's auxiliary files.

## Files

- [Companion PDF](spectral_likelihood_companion.pdf)
- [Complete companion source package](spectral_likelihood_companion_source.zip)
- [Ordinary-SBM article](../sbm/sbm_spectral_likelihood.pdf)

## Current result

The current regularized DC-SBM theorem in Sections 1–5 assumes fixed arbitrary K, comparable groups, positive compact full-rank block profiles, bounded positive degree parameters with mean one in each block, **d = np → ∞ and np² ≤ C₁**. There is **no logarithmic lower bound**. The risk benchmark is the full-edge profile oracle. The known-target-degree oracle comparison additionally requires finite-n equal block exposures.

Sections 6–10 retain distinct older unregularized algorithms and their full proofs. Logarithmic hypotheses written there belong only to those historical decoders; they are not restrictions on the current theorem. The older ordinary free-profile result also covers probabilities outside bounded np².

## Coverage of the previous combined source

| Previous component | Current location |
| --- | --- |
| `statements.tex`: model and oracle definitions | `models.tex`, Section 1; historical encoder statements in `historical_statements.tex`, Section 6 |
| `sparse_restoration.tex`: current DC algorithm and theorem | `regularized_dc.tex`, Section 2; all-scale risk assembly in `dc_risk.tex`, Section 5 |
| `restored_dc_fit.tex`: DC nuisance fitting, profile reduction, invariant basin | `dc_fitting.tex`, Section 4; the ordinary-only specialization is in the first article |
| `sharp_tail.tex`: full sharp-transfer proof and ordinary oracle comparison | Full proof in the first article; local statement, formulas and comparison recalled in `shared_tools.tex`, Sections 3.1–3.3 |
| `sparse_extension_status.tex`: floor geometry and DC specialization | First article gives the shared full proof; Section 3.4 verifies the bounded-theta population space and all DC-specific consequences |
| `restored_row_tail.tex`, `restored_local_loo.tex`, `restored_high_range.tex` | Shared full proofs in the first article; needed statements and applicability checks in `shared_tools.tex`, Sections 3.5–3.7 |
| `sparse_extension_status.tex`: DC tilted cancellation | `shared_tools.tex`, Section 3.8, with full DC-specific derivation |
| `sparse_extension_status.tex`: optional exponential-moment and exceptional-hit transfers | `shared_tools.tex`, Section 3.9, with both proofs |
| `normalized_geometry.tex` | Full historical unregularized proof retained as `normalized_geometry.tex`, Section 7 |
| `sbm_proof.tex`: ordinary free-profile growing/replacement/soft-EM algorithm | Full proof retained as `sbm_proof.tex`, Section 8 |
| `dcsbm_proof.tex`, `dc_oracle_bridge.tex` | Full historical DC proof and classical balanced oracle comparison retained, Section 9 |
| `probability_closure.tex` | Full historical probability closure retained, Section 10 |
| `impossibility.tex` | Full known-target-degree oracle obstruction retained, Section 11 |
| `sparse_restoration.tex`: whole-graph isolated-vertex fallback counterexample | `additional_notes.tex`, Section 12.1, with full proof |
| `sparse_extension_status.tex`: existing degree-plus-spectrum action identity | `additional_notes.tex`, Section 12.2 |
| `scope_audit.tex` | Current scope summarized in Sections 12.3–12.4; original narrative preserved in `archive/scope_audit.tex` |
| Closing sparse-interface and restoration history narratives | `archive/sparse_interfaces.tex` and `archive/restoration_history.tex`; concise current account in Section 12.4 |
| `audits/` | All indexed review files and their README retained in `audits/` |
| Previous repository README: 18 October paper, implementation and experiment record | `archive/repository_history_20261018.md` |
| Complete 68-page revision 3 | Preserved separately at `paper/notes/normalized_oracle_risk_20261020/` in the repository |

This is an organized working collection. It is not presented as a second finished article, and no experiments were added for this editorial split. Internal audits are historical working records, not external peer review. The current TeX statements and proofs control where older snapshots describe an interface as open.

## Build

Run `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. The supplied `main.bbl` permits rebuilding with LaTeX alone when no citation changes are made. See `BUILD.txt` for exact commands.

The output deliverable is `spectral_likelihood_companion.pdf`. The source package includes the TeX/BibTeX inputs, bibliography, audit records and archived narratives; auxiliary compilation files are not required.

## Split validation

The final 57-page build has no undefined references, duplicate labels or overfull boxes, and embeds all fonts. Every page was rendered and inspected. The current DC information is denoted by its directional profile exponent J; historical ordinary information I is kept separate. The first paper compiles independently of this companion.

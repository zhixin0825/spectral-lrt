# Community summation index revision — October 29, 2026

Base commit: eb48287d75faa6010f2dbdaa1c3273d9e27415b3.

Replace the third community index ell-prime by r in article.tex (27 occurrences), experiments.tex (1 occurrence), and sharp_tail.tex (7 occurrences). This is a bound-index renaming, not a change to any distribution, assumption, algorithm, theorem or experiment.

The audit also found pre-existing local uses of r in the appendices: the polynomial tail parameter in geometry/row-tail/risk proofs; the eigenvector-column index in the LOO proof; tilted probabilities r_j and their vector; and the diagonal vector in block fitting. These local definitions are retained. The new community index is explicitly bound by the community count definition and the displayed sums; the edited formulas contain none of the other r objects. This revision does not claim r was previously unused or reserve it globally.

The updated manuscript was compiled locally with three PDFLaTeX passes using the included bibliography. It remains 35 pages, with no undefined references/citations or overfull/underfull warnings. Render comparison with the approved v21 PDF found changes only on pages 3, 4, 8, 10 and 14. The displayed T and I definitions remain single-line formulas. All other manuscript files and experimental code/data are unchanged. This check is not a new proof audit or experiment run.

This commit updates TeX sources only. The historical repository PDF and ZIP are not replaced. The revised compiled PDF and complete source package are supplied in the conversation as v22.

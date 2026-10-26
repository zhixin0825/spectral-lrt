# Notation and preservation audit — 26 October 2026

Base source: `0fe40da1fee92f7bec3ea3f7f6fdd53faae8c442`, `zhixin0825/spectral-lrt`, branch `proof/normalized-oracle-risk-20261020`.

## 1. Primary reference actually inspected

Zhou and Li's author preprint, arXiv:1812.11269v1, printed page 9, equations (20)–(23), explicitly uses beta for community balance, p* for the largest block probability, omega for the maximum/minimum probability ratio, pi in S_K for label permutations, and D_alpha for Chernoff quantities. The PDF page was inspected visually as well as through extracted text. Its title is *Non-Asymptotic Chernoff Lower Bound and Its Application to Community Detection in Stochastic Block Model*. The later published paper is *Rate optimal Chernoff bound and application to community detection in the stochastic block models*, EJS 14(1), 1302–1347 (2020), DOI 10.1214/20-EJS1686.

The published PDF was not accessible in this run. The explicit model-symbol comparison is therefore attributed to the inspected author preprint, not to an uninspected published page. The use of sigma_K(P) also follows the author's present instruction and is defined explicitly as the K-th singular value of the K-by-K P; this audit does not claim that its occurrence was located in the accessible preprint.

Primary reference: https://arxiv.org/pdf/1812.11269 (page 9).

## 2. Necessary versus unnecessary notation

The separate constants pi_-, c_-, c_+ and kappa were unnecessary names for familiar balance, probability-ratio and signal-conditioning restrictions. They have been removed from the current TeX. The restrictions themselves have not been deleted.

The current model uses

    p* = max_ab P_ab,
    n_a >= n/(beta K),
    p*/omega <= P_ab <= p*,
    sigma_K(P) >= c p*,

with fixed beta, omega and a fixed positive lower signal ratio. Only the community-size lower bound used by the existing proof is imposed; an upper bound from another paper is not added just to mimic its display. The constants c,C in proofs are generic except where a fixed structural lower bound is explicitly stated.

The abbreviation d=np* remains because it is used throughout the proof; it is not an independently specified parameter. U, Lambda, F, A_hat and the exact averaged R_oracle remain because they represent the established algorithm or benchmark. Proof-only G,H,Q,Omega and local weights are kept at their points of use rather than elevated to new model parameters. The row-tail proof's old local beta_i is renamed xi_i to avoid sharing the imbalance symbol. The redundant d_*=max_ij n E A_ij is replaced by d: E A_ij <= p*=d/n suffices throughout that argument.

## 3. Normalization is equivalent up to fixed constants

Write p_old for the former arbitrary scale and denote its constants by pi_old, c_old_minus, c_old_plus and kappa_old. Then

    c_old_minus p_old <= p* <= c_old_plus p_old.

One may choose beta=1/(K pi_old), omega >= c_old_plus/c_old_minus and a fixed lower signal coefficient c=kappa_old/c_old_plus. Thus the earlier class satisfies the new form. The condition n p_old^2 <= C1_old becomes n(p*)^2 <= (c_old_plus)^2 C1_old. This is not a claim that the same numerical upper bound is unchanged.

Conversely, take p_old=p*, pi_old=1/(beta K), c_old_minus=1/omega, c_old_plus=1, kappa_old=c. The new form is then a specialization of the old representation of the same fixed-comparability family. Divergence of np_old is equivalent to divergence of np*. The estimator does not use either true probability scale, so the normalization changes no computed step.

Because P is K-by-K, sigma_K(P)=sigma_min(P). This replacement changes only notation, not the signal condition. The theorem still assumes positive comparable block probabilities, a fixed nondegenerate signal ratio, fixed K, comparable community sizes and the stated upper sparsity bound. No logarithmic degree lower bound is added.

## 4. Proof notation

The Bernoulli log transform is psi(alpha), avoiding reuse of K for both community count and a transform. The optimizing tilt is alpha*, and the exact information is D_alpha evaluated at alpha*. D* and D*_{i,ab} are local proof abbreviations for these quantities. The row profiles explicitly exclude the tested index. This is algebraically the previous negative minimum of the log-affinity, not a different testing experiment. The oracle remains the actual averaged directional error, not the total-variation affinity from the reference paper.

## 5. Completed checks

- Both simple and general algorithm sections are byte-for-byte unchanged.
- Thirteen protected equation environments are byte-for-byte unchanged, including the oracle score and risk, full degree reconstruction, block fit, full row score and risk target.
- The main theorem environment changes only by the explicit p-to-p* scale normalization.
- All 95 labels and all 17 theorem/lemma/proposition/corollary environments remain, in the same order within each file.
- All seven proof inputs remain in main.tex. local_loo.tex is byte-for-byte unchanged.
- Scans of current TeX found no pi_-, c_-, c_+, sigma_min, kappa, d_* or beta_i. Indexed p_j are not mistaken for the global probability scale.
- Three PDFLaTeX passes completed. The final log has no undefined references/citations, duplicate-label warnings or overfull/underfull boxes. All fonts are embedded.
- All 25 rendered pages were visually inspected; model and sharp-tail notation were also inspected at readable scale.
- The existing algebra checker passed 45 cases, maximum absolute residual 7.105427357601002e-15. These are algebra checks, not new accuracy experiments.

The source hashes and individual preservation results are in NOTATION_CHECKS_20261026.json. This is a notation/equivalence and editorial-preservation audit. It is not a new independent certification of every inherited spectral or likelihood lemma.

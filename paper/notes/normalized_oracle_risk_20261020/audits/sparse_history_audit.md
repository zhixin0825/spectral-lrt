# Historical sparse-proof recovery audit

Read-only audit of zhixin0825/spectral-lrt, 2026-10-20. Source files were fetched through the connected GitHub API. No repository contents were changed.

## Findings

1. The earliest archived TeX contains an explicit, unconditional-in-form low-density theorem, not merely a conditional transfer proposition.
2. It covers fixed positive full-rank SBM profiles, fixed K, positive limiting proportions, d -> infinity and d = o(log n), with a W A W encoder and degree retention.
3. It claims the oracle leading error exponent, not a constant-factor risk comparison.
4. Its proof includes an actual truncated row-energy lemma and full-to-LOO bridge.
5. The original archive README simultaneously called the TeX an unreviewed historical draft, and the contemporaneous status document did not certify the sparse theorem. This is a provenance inconsistency, not a mathematical disproof of the TeX argument.
6. Later active manuscripts used a different initializer/decoder and converted the sparse discussion to a conditional transfer, then removed that discussion. I found no recorded line-by-line refutation of the old row-energy lemma or bridge.
7. No later (10–18 October) retrieved theorem removes the logarithmic-degree scale for the sharp constant-oracle theorem. Removing the entrywise P_ab lower bound on 18 October was a different issue.

## Recovered source

File: old_20261009.tex
Git path: paper/previous_draft/sbm_spectral_lrt_20261009.tex
Blob SHA: 14975a83aa5d4308b1469dc08765ee396be9fd52
Only commit changing this path: de60b64c7d7f127ae6dc1d010c840af707a080da
Commit time: 2026-10-08 23:40:17 UTC = 2026-10-09 07:40:17 Beijing
URL: https://github.com/zhixin0825/spectral-lrt/blob/de60b64c7d7f127ae6dc1d010c840af707a080da/paper/previous_draft/sbm_spectral_lrt_20261009.tex

Line map:
- 87–107: abstract and scope explicitly assert the sublogarithmic theorem.
- 125–137: fixed K, fixed positive full-rank C, positive limiting proportions; d -> infinity, d=o(log n).
- 193–218: thm:main and exponential equivalence; R_spec,ab=exp(-d I_ab+o(d)).
- 223–283: actual decoder: fixed-factor approximate k-means on sqrt(n) WU; invert empirical community centers; exact degree correction; plug-in Bernoulli LLR.
- 295–343: lem:noise, regularized concentration with arbitrary fixed polynomial failure n^{-r}.
- 438–545: independent clipped LOO coordinates and conditional MGF.
- 535–536: d=o(log n) makes n^{-r} superexponentially negligible at speed d.
- 577–608: lem:LOO-DK, perturbation acting on the same full U.
- 627–665: lem:weighted-tail, summable weight perturbations.
- 667–717: same-threshold energy recurrence.
- 720–766: lem:row-energy and fixed-row tail.
- 789–854: lem:bridge, full-to-LOO bridge.
- 856–904: center inverse removes rotation; parameter errors contribute o(d).
- 906–968: oracle Chernoff comparison and claimed completion of the main theorem.
- 1156–1187: exact limits, including distinction between exponent and risk ratio.

## Reusable mathematical inputs stated by the old proof

- E[H_R(U) 1_E] <= K R^2 exp(-c d log R).
- P(sqrt(n)||U_i|| > R, E) <= K exp(-c d log R).
- For each epsilon,H>0, choose fixed L,R large enough:
  P(||q_i O_i^T-r_i|| > epsilon d) <= exp(-H d)+O(n^{-r}).
- The non-tail full-to-LOO term is bounded only by K_{L,R} sqrt(d), explicitly at line 804.
- Center and nuisance fitting have consistency-scale error, yielding o(d) score error at line 894.

These are relevant low-degree ingredients, but alone do not establish an O(1) perturbation or bounded tilted cumulant needed for a constant-factor oracle risk theorem.

## Provenance caveat present at initial archive

At the SAME initial commit de60b64..., paper/previous_draft/README.md says:
“這是歷史草稿，不是本次重新審核並確認成立的定理版本。”
It also says:
“原稿正文未作修改。”
and:
“本次僅恢復並留檔原始碼，沒有修訂原稿 theorem／proof，也沒有宣稱已完成其證明審核。”

The contemporaneous docs/CURRENT_STATUS_20261009.md labels 1 << d_n << log n optimal LRT exponent as 未完成證明, but explicitly says finite-sample decoder failure does not automatically refute the fixed-profile asymptotic claim. This must not be turned into a claim that the old mathematical lemma was disproved.

## Active-manuscript chronology

- 508a858f2ea84f1ff85ce6741ef799518f419853, 2026-10-09 11:00:56 UTC:
  paper/current/sparse.tex introduced “Extension below logarithmic density”.
  Its cor:sparse assumes score approximation and exceptional-event bounds.
  Closing sentence: “Verifying this corollary for a specified regularizer and fitting procedure remains a separate statistical task.”
  Local copy: conditional_sparse_20261009.tex

- b0fef88c7fb9d6cddc55bd2862515d2779f43b30, 2026-10-09 16:22:11 UTC:
  sparse.tex retained the same conditional scope while notation changed.

- 93970208de0304a14601c2e7c68bd37e207f448a, 2026-10-09 17:10:55 UTC:
  end-to-end likelihood restart proof did not upgrade sparse.tex; it remained conditional.

- b288f4fd831b85f6809e1199fea44e0d695137c7, 2026-10-10 01:48:08 UTC:
  sparse.tex replaced with comments including:
  “The former conditional lower-density discussion is available in Git history.”
  “No lower-density algorithmic guarantee is claimed.”

- b7f313259cc4628d1ba22074037ff0850c6adb8c, 2026-10-10 08:56:53 UTC:
  full Bernoulli general-K theorem across densities still explicitly uses p=Omega(log n/n).
  The old sparse.tex comment remains; “across densities” means logarithmic through dense.

- 14046f7bd351454652fbf8c7dbb5d742dc078d3a, 2026-10-12:
  spectral_grouping_sufficiency_20261012.tex requires np>=M_q log n.
  Local copy: sufficiency_20261012.tex

- 84ea90a6c18aca4f9ed095a4eac09a3636629ec8, 2026-10-18:
  standalone sharp spectral theorem explicitly requires np>=c log n.
  Local copy: sharp_rate_20261018.tex

- 4fbdbf70ae4d2220f2526bffb7b04f5edf870388, 2026-10-18:
  vanishing-floor theorem removes the entrywise probability lower bound, not the density lower bound. Abstract/intro/model still retain p=Omega(log n/n).

- 17a37e5a8f950293b922214565ab76abd4c53af4:
  removed that exponent-only vanishing-floor theorem when reorganizing the paper around sharp oracle risk.

- c600075a61ec8b34cb60e1608f84a898e3dfb76e:
  restored degree-plus-spectrum general algorithm and a conditional sharp transfer. This did not reproduce or refute the old WAW row-energy proof.

## Recommended accurate historical wording

The project already had an explicit low-degree proof draft and concrete row-energy/LOO lemmas at d -> infinity, d=o(log n). It is inaccurate to describe low-degree spectral decoding as an entirely new or never-treated problem. The current mathematical task is to audit and reuse those ingredients, reconcile the changed decoder, and upgrade exponent-scale score control to a sharp constant-factor oracle comparison. The historical record does not justify silently deleting all density assumptions from the new theorem, nor does it justify discarding the old sparse route.


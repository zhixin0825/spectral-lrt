# Editorial preservation audit — 24 October 2026

Base: `8f3962e913bf32cf6c93d9325b37de4faf171b68`, repository `zhixin0825/spectral-lrt`, research branch `proof/normalized-oracle-risk-20261020`.

## Changes

The introduction and abstract now explain the spectrum-only likelihood principle before the complete degree-plus-spectrum classifier. A new Section 3 defines self-excluded spectral block counts, their factorized computation, a finite spectrum-only initializer/fit/update procedure, and its status as a motivating algorithm rather than a newly proved estimator. Section 4 retains the complete algorithm and adds exact identities for the count correction and fixed-fit LR correction. The theorem scope explicitly distinguishes no logarithmic density floor from the retained positive-profile and conditioning assumptions. A short discussion follows the theorem.

## Preserved, not re-proved

All seven proof inputs are byte-for-byte identical to their base-commit blobs:

| File | Git blob SHA |
| --- | --- |
| sharp_tail.tex | 8748751759e63a20031c16098d04cdd74716f09d |
| geometry.tex | 18873d7df29e622bd0dd5ab0429399a6ca2a90e7 |
| row_tail.tex | 06d399743270b15247940a8eabba6fd8cffbef9d |
| local_loo.tex | 360bdecb01f88a609a7968eba44605cbce165dcf |
| fitting.tex | 509b11665f82122f260e62eebdcd81491118dc38 |
| high_range.tex | 74ec413bd5058989fe4edd74390cf347ecda3bd2 |
| risk_proof.tex | 5d7299e6687b8afbff92cf83a67a1595cb21a333 |

The complete main-theorem environment is verbatim unchanged. The model, structural assumption, oracle score/risk, full reconstruction, full block fit, full row score and risk-bound equation environments are also verbatim unchanged. Their labels remain stable; numbering changes automatically because of the inserted section. The bibliography and all seven proof inputs remain in the build.

## Local build and checks

Three PDFLaTeX passes with the existing main.bbl produced a 24-page A4 PDF. The final log contains no undefined references/citations, duplicate labels, overfull boxes or underfull boxes. All rendered pages were inspected in contact sheets and the new exposition pages at readable scale. All PDF fonts are embedded. Newly added factorized-count and degree/LR bridge identities were also checked numerically on small deterministic seeded examples; those checks are algebra checks, not statistical-performance experiments.

The 24-page PDF and matching source archive were generated as conversation artifacts. Existing 21 October repository PDF/ZIP files and their build records are historical until separately replaced. This audit does not claim a new independent proof review or completion of any extension beyond the preserved theorem.

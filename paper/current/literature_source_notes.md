# Primary-source verification for manuscript references

Checked 2026-10-09; model scope and directly related sources rechecked 2026-10-10. All manuscript claims are paraphrases; no source passages copied.

| Citation key | Primary source | Verified claim / metadata |
|---|---|---|
| holland1983 | https://www.sciencedirect.com/science/article/pii/0378873383900217 | Social Networks 5(2), 109–137, 1983; original blockmodel article. |
| abbe2015 | https://arxiv.org/abs/1503.00609 | General SBM exact-recovery threshold via CH divergence, efficient recovery; FOCS 2015, 670–688. |
| zhouli2020 | https://scholars.cityu.edu.hk/en/publications/rate-optimal-chernoff-bound-and-application-to-community-detectio/ ; https://arxiv.org/pdf/1812.11269 | Zhixin Zhou and Ping Li, EJS 14(1), 1302–1347, 2020, DOI 10.1214/20-EJS1686. Cited only for study of optimal rates. The arXiv preprint has an earlier title; do not substitute that for the published title. |
| amini2013 | https://arxiv.org/pdf/1207.2340 | Amini, Chen, Bickel, Levina; AoS 41(4), 2097–2122, 2013; pseudo-likelihood computes block sums from A; iterative mixture updates. |
| gao2017 | https://www.jmlr.org/papers/v18/16-245.html ; https://www.jmlr.org/papers/volume18/16-245/16-245.pdf | Authors Gao, Ma, Anderson Y. Zhang, Harrison H. Zhou; 18(60), 1–45, 2017; weak-consistency initializer plus penalized local likelihood refinement. Their likelihood update reads incident A edges. |
| zhouamini2020 | https://jmlr.org/papers/v21/19-299.html ; https://www.jmlr.org/papers/volume21/19-299/19-299.pdf | Zhixin Zhou then Arash A. Amini, 21(40), 1–68, 2020; spectral initializer plus two pseudo-likelihood classifications; broader EM-type family. |
| abbe2020 | https://arxiv.org/pdf/1709.09565 | Authors Abbe, Fan, Kaizheng Wang, Zhong; AoS 48(3), 1452–1474, DOI 10.1214/19-AOS1854. Theorem 3.2 covers symmetric equal-sized two-block logarithmic regime with untrimmed eigenvector sign decoder and optimal error exponent. |
| zhang2024 | https://arxiv.org/html/2301.09289v3 ; https://andersonyezhang.github.io/PDF/spectral_SBM.pdf | Anderson Ye Zhang, IEEE Transactions on Information Theory 70(10), 7320–7348 (2024), DOI 10.1109/TIT.2024.3425581. Theorems are for homogeneous p/q multicommunity SBM with comparable community sizes, not general arbitrary B. Sharp exponent of trimmed U Lambda plus K-means is distinguished from information-theoretic exponent. Intro uses this restricted claim. |
| suwan2016 | https://arxiv.org/pdf/1405.6070 ; https://doi.org/10.1214/16-EJS1115 | Suwan, Dominic S. Lee, Runze Tang, Sussman, Minh Tang, Priebe; EJS 10(1), 761–782, 2016. Uses Gaussian-mixture ASE theory for empirical Bayes prior, then MCMC posterior inference. Do not call the whole method spectral-only classification. |
| zhouamini2019 | https://www.jmlr.org/papers/v20/18-170.html | Authors Zhixin Zhou and Arash A. Amini, 20(47), 1–47, 2019; data-driven regularization and spectral consistency in general bipartite settings. |
| arthur2007 | https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf ; https://research.google/pubs/k-means-the-advantages-of-careful-seeding/ | David Arthur, Sergei Vassilvitskii, SODA 2007, 1027–1035; D2 sampling and K-means objective approximation, not a general SBM recovery guarantee. |
| nemhauser1978 | https://link.springer.com/article/10.1007/BF01588971 | George L. Nemhauser, Laurence A. Wolsey, Marshall L. Fisher; Mathematical Programming 14, 265–294, 1978; monotone normalized submodular greedy guarantee. |

Project Euclid direct full-page links intermittently return a security check; their underlying published PDFs or author arXiv copies provide the text checked here.

## Proof inputs and scope checked on 2026-10-10

- Lei–Rinaldo, Theorem 5.2 (https://arxiv.org/pdf/1312.2050): operator-norm concentration for an unmodified adjacency matrix at logarithmic density, at any prescribed fixed polynomial confidence. Equal expected degrees are not assumed.
- AFWZ, Theorem 2.1 and Bernoulli Lemma 7 (https://arxiv.org/pdf/1709.09565): first-order eigenspace approximation with row concentration. The manuscript's general fixed-K application verifies the assumptions and treats the loop-free diagonal. The subsequent EM and clipped score moment estimates are the manuscript's own steps.
- Zhou–Li (https://arxiv.org/pdf/1812.11269), equations (19)–(24): block matrix P, node profiles p_kj=P_k,z_j, and the Chernoff D_alpha convention agree. Its minimax lower bounds have their own parameter-space assumptions; a binary oracle lower bound does not itself prove a global permutation-invariant minimax statement.
- Wang, Jiangzhou, arXiv:2610.04376v1 (3 October 2026), https://arxiv.org/html/2610.04376v1: recent spectrally initialized same-graph VEM preprint with a general-SBM leading Chernoff-exponent claim. Section 2.4 explicitly uses the original adjacency in refinement. It is directly relevant prior work, while the present post-compression restriction is different.

## General-density revision, 2026-10-10

- [Bandeira and van Handel, arXiv:1408.6185](https://arxiv.org/pdf/1408.6185), *Annals of Probability* 44(4), 2479–2506 (2016), DOI [10.1214/15-AOP1025](https://doi.org/10.1214/15-AOP1025): Corollary 3.12 gives the bounded-entry spectral tail; Remark 3.13 explicitly extends it to centered nonsymmetric distributions with changed constants. These support exp(-H np) confidence when p=max_ab P_ab satisfies p=Omega(log n/n).
- [AFWZ, arXiv:1709.09565](https://arxiv.org/pdf/1709.09565): Theorem 2.1 permits a mean with a small non-signal part; Lemma 7 has the actual adjustable Bernoulli tail 2 exp(-alpha n p). The current proof verifies its conditions for both positive and negative signal groups and retains the no-self-loop diagonal correction.
- [Wang, arXiv:2610.04376v1](https://arxiv.org/abs/2610.04376v1): directly re-opened the primary record; submitted October 3, 2026, author Jiangzhou Wang. Its stated procedure estimates probabilities and proportions from the same sparse graph. The manuscript treats it as a preprint and distinguishes its use of the original graph from the retained-eigenpair restriction here.
- [Zhang, arXiv:2301.09289v3](https://arxiv.org/pdf/2301.09289): directly rechecked the primary PDF. Its model uses homogeneous within/between probabilities and its Algorithm 1 trims high-degree rows, weights eigenvectors by eigenvalues, then applies K-means. The introduction retains this precise scope.

The current theorem is an independently developed deterministic replacement and spectral-likelihood argument under its own stated assumptions. The cited papers supply specified benchmark and spectral tools; their results are not claimed to prove the new full algorithm without the intervening analysis.

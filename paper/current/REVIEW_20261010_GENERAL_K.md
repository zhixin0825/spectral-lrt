# General fixed-order and density extension audit

The main theorem covers every fixed integer K >= 2. Define p=max_ab P_ab. Its density assumptions are p=Omega(log n/n) and p=o(1), with P_ab asymp p, positive community proportions, and sigma_min(N^(1/2) P N^(1/2)) asymp np. The minimum expected-degree assumption and the separate P_min, P_max, c_r, c_s, c_d notation have been removed. All probability, information, reconstruction, and objective scales use p, np, or n^2p.

The algorithm preserves global-mean growing and explicitly adds iterative global-gain component replacement, uniform-weight proposal starts, a uniform-weight current-centers refit, threshold T/sqrt(log n), and a log^2 n accepted-round cap. The deterministic stopping certificate supplies weak responsibilities for arbitrary fixed K. The growing branch alone has only its separate two-community result. No random restart is needed in the main theorem.

The floor now scales with max(||Lambda||op,1)/n. The spectral exceptional probability is exp(-H np), including a verified bounded-entry norm tail and Bennett/AFWZ row bound. Random-floor oracle moments use deterministic envelopes before factoring independent block-count moments. The sparse-score Taylor remainder is o(I), not necessarily o(1).

Independent proof review verified the pigeonhole replacement, exact hard-cost gain identity, soft/hard objective bounds, inactive cap, center matching, and variational responsibility bound. Fresh 24-graph checks cover K=3,4,6 and two degree scales; all stopping certificates pass, 7/9216 nodes are misclassified, and a collapsed endpoint triggers two accepted rounds. Historical 240-graph experiments retain their original protocol and are not attributed to the new algorithm.

The main text omits alternative Kmeans warm starts and independent-row projection branches. Original references under sources/ are unchanged.

## Notation revision verification

The model, abstract, introduction, theorem statements, proofs, conclusion, and README use the same scalar p. The adaptive floor still depends only on retained eigenpairs. Its fixed multiplier is sufficiently small relative to the probability comparison bound; no additional density parameter is introduced. The clipped-oracle proof uses the deterministic bound (1/2)min_ab P_ab without naming another extremal-probability symbol.

Static source checks and an independent notation audit precede compilation and PDF inspection.

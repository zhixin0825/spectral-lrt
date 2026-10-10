# General fixed-order and density extension audit

The main theorem now covers every fixed integer K >= 2. The density assumption is minimum expected degree >= c log n; expected degrees need not be equal and may grow faster than log n. P_n may vary in shape. Positivity ratios, positive community proportions, sparse Pmax -> 0, and uniformly conditioned rank-K signal remain stated hypotheses.

The algorithm preserves global-mean growing and explicitly adds iterative global-gain component replacement, uniform-weight proposal starts, a uniform-weight current-centers refit, threshold T/sqrt(log n), and a log^2 n accepted-round cap. The deterministic stopping certificate supplies weak responsibilities for arbitrary fixed K. The growing branch alone has only its separate two-community result. No random restart is needed in the main theorem.

The floor now scales with max(||Lambda||op,1)/n. The spectral exceptional probability is exp(-H n Pmax), including a verified bounded-entry norm tail and Bennett/AFWZ row bound. Random-floor oracle moments use deterministic envelopes before factoring independent block-count moments. The sparse-score Taylor remainder is o(I), not necessarily o(1).

Independent proof review verified the pigeonhole replacement, exact hard-cost gain identity, soft/hard objective bounds, inactive cap, center matching, and variational responsibility bound. Fresh 24-graph checks cover K=3,4,6 and two degree scales; all stopping certificates pass, 7/9216 nodes are misclassified, and a collapsed endpoint triggers two accepted rounds. Historical 240-graph experiments retain their original protocol and are not attributed to the new algorithm.

The main text omits alternative Kmeans warm starts and independent-row projection branches. Original references under sources/ are unchanged.

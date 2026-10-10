# Recovering partitions with spectral-only MCMC from random labels

MCMC recovered the true partition in many strong-signal finite experiments where random-start variational EM failed. Six independent Gibbs starts recovered 38/40 final partitions in the main strong-signal experiment, versus 6/40 for EM. A single Gibbs chain recovered 22/40. However, ordinary Gibbs, six-start Gibbs, and parallel tempering all failed on the larger balanced two-community graphs. Adding MCMC alone therefore did not make recovery reliable.

A subsequent whole-partition spectral proposal recovered all four of those larger binary endpoints in 200 sweeps. Its held-out graph results were less successful for other models. This is useful evidence for changing the proposal, not a general recovery or mixing guarantee.

## Data, information available, and target

The graph is a simple undirected SBM with known K and unknown symmetric P. We generate A and retain only the K largest-absolute eigenvalues, including their signs, and their eigenvectors U. After this step, all partition inference uses only U and the retained eigenvalues. A and the true labels are archived for evaluation and explicitly marked oracle diagnostics; they are not inputs to the inference kernels.

The implemented spectral pseudo-data are

\[
\widetilde A=U\operatorname{diag}(\lambda)U^\top,\qquad
Q_{ij}=\operatorname{clip}(\widetilde A_{ij},0,1),\quad Q_{ii}=0.
\]

For each candidate partition C, let S_ab be the sum of Q over the unordered edges in block pair (a,b), and M_ab the number of possible edges: n_a n_b between distinct groups, and n_a(n_a-1)/2 within a group. The fractional Bernoulli working likelihood is

\[
L(C,P;Q)=\prod_{a\le b}P_{ab}^{S_{ab}}(1-P_{ab})^{M_{ab}-S_{ab}}.
\]

Independent Beta(1,1) priors on P and independent uniform labels give the integrated working target

\[
\pi(C\mid Q)\propto\prod_{a\le b}\mathrm B(S_{ab}+1,M_{ab}-S_{ab}+1).
\]

S_ab can be fractional. This target is **not the exact posterior of the original binary graph**. The clipping is an explicit modeling choice; it also means this implementation reconstructs a dense Q rather than computing all block sums directly from the unmodified low-rank factors. No binomial combinatorial coefficient is added: a partition-dependent coefficient would alter the likelihood of the common observed matrix when comparing C. Empty groups are allowed, self-loops excluded, and no true class sizes or P values enter inference.

One Gibbs sweep visits every node in a random order. For each node it samples the label from the exact K conditional scores of this integrated target. Thus P is inferred jointly through integration; it is never supplied by an oracle. Posterior block means, if wanted, are (S_ab+1)/(M_ab+2).

## Main paired experiment

There are four model families, two signal levels, five independent graphs per condition, and two random initial labelings per graph: 40 graphs and 80 graph/start sets. Each initial labeling is iid uniform and independent of truth. Methods share the same graph and first initialization.

P=cB is calibrated so that the minimum ordered-pair, oracle one-node Bernoulli Chernoff information divided by log n is 0.65 or 1.35. The optimizing Chernoff tilt is fitted numerically, including the missing self-edge. These values describe the simulation signal and are not claimed as a universal recovery threshold for the working posterior. The model specification and achieved values are stored in each graph archive.

| Model | Proportions | B |
|---|---|---|
| Balanced K=2 | (0.5, 0.5) | [[5,1],[1,5]] |
| Unequal K=3 | (0.5, 0.3, 0.2) | [[6,1,1.5],[1,4,0.8],[1.5,0.8,5]] |
| Hierarchical K=4 | (0.25,0.25,0.25,0.25) | [[7,3,1,1],[3,7,1,1],[1,1,7,3],[1,1,3,7]] |
| Disassortative K=3 | (1/3,1/3,1/3) | [[1,4,3],[4,1,4],[3,4,1]] |

At n=512, MCMC and greedy ICM use 1000 sweeps; variational EM uses at most 200 iterations. EM updates mean-field responsibilities and an unknown symmetric P using the same Q and uniform label prior. Its point-P ELBO differs from the integrated posterior used by Gibbs and ICM. ICM chooses the largest conditional score instead of sampling. Parallel tempering has six replicas at beta=(1,.7,.45,.25,.12,.04). Gibbs6 has six independent iid random starts; its output is the endpoint with the largest working posterior score. These comparisons are **not matched for computation**: Gibbs6 and tempering each run six chains.

The primary metric is the **final partition**, matched to truth up to label permutation. No endpoint is selected using true labels. Best-scoring visited states are archived and summarized separately; they are not substituted into the final table. The two starts on a shared graph are not independent graph replicates, and these small counts are not precise probability estimates.

Strong signal (I/log n=1.35): each row has five graphs and ten random-start endpoints.

| Model | Variational EM | ICM | Gibbs | Gibbs6 | Parallel tempering |
|---|---:|---:|---:|---:|---:|
| Balanced K=2 | 0/10 | 3/10 | 3/10 | 8/10 | 1/10 |
| Unequal K=3 | 6/10 | 3/10 | 5/10 | 10/10 | 9/10 |
| Hierarchical K=4 | 0/10 | 0/10 | 6/10 | 10/10 | 8/10 |
| Disassortative K=3 | 0/10 | 3/10 | 8/10 | 10/10 | 10/10 |
| Total exact | 6/40 | 9/40 | 22/40 | 38/40 | 28/40 |

Spectral k-means, run separately on U diag(lambda), recovered 5/5 graphs in each strong condition. It is a reference decoder, not an initialization for MCMC. At the weaker signal, none of the six primary methods obtained exact recovery in these runs. Main weak-signal mean errors for Gibbs6 were 48.52%, 5.74%, 31.39%, and 28.89%, respectively in the model order above; spectral k-means errors were 0.35%, 34.45%, 0.59%, and 0.98%. The weak unequal case is one example where MCMC improved on this particular spectral k-means baseline without reaching exact recovery.

The full condition table is [analysis_main/final_table.md](analysis_main/final_table.md). Trajectory and best-posterior summaries are separate.

## Larger graphs, additional communities, and longer budgets

These checks are reported separately from the main experiment. All use strong signal, two graphs and two random starts per graph.

| Condition | EM | Gibbs | Gibbs6 | Parallel tempering | Spectral k-means (graphs) |
|---|---:|---:|---:|---:|---:|
| n=1024, balanced K=2, 2000 sweeps | 0/4 | 0/4 | 0/4 | 0/4 | 2/2 |
| n=1024, hierarchical K=4, 2000 sweeps | 0/4 | 2/4 | 2/4 | 2/4 | 2/2 |
| n=512, balanced K=6, 2000 sweeps | 0/4 | 4/4 | 4/4 | 4/4 | 2/2 |
| n=512, balanced K=2, 5000 sweeps | — | 1/4 | 4/4 | 2/4 | — |

The K=6 matrix has diagonal 7 and off-diagonal 1, scaled by the same calibration. The 5000-sweep binary check reuses graph seeds 0 and 1 from the main experiment and must not be counted as new graphs. Most individual binary chains still fail within that larger budget.

## Post-hoc whole-partition proposals

After observing local-chain failures, three valid MH proposals were explored. All preserve the same integrated working target, and all start with iid random C.

1. A noisy label-copy proposal from fixed auxiliary colder replicas: failed on all four larger binary endpoints.
2. A product proposal obtained from random linear projections of retained spectral coordinates: useful in the small pilot, with mixed recovery across models.
3. A product proposal based on randomly chosen spectral node prototypes: the final exploratory variant.

For the prototype proposal let X=sqrt(n)(U-mean(U)). Each proposal independently selects K ordered distinct node IDs uniformly. With those fixed prototypes, q_i(a)=0.8 softmax_a(-2||X_i-X_id(a)||^2)+0.2/K. Propose all labels independently from q, then accept with

\[
\min\{1,\exp[\ell(C')-\ell(C)]q(C)/q(C')\},\qquad \ell=\log\pi.
\]

Both proposal densities use the same prototypes. Each recorded sweep consists of a full Gibbs sweep followed by one global proposal. Prototype selection uses neither truth nor a fitted k-means partition. The strictly positive proposal floor permits every partition. Independent small-network checks confirmed detailed balance and the empirical stationary marginals for the actual implementation.

At n=512 this method used 200 sweeps, with two pilot graph seeds then three held-out graph seeds:

| Model | Pilot seeds 0,1 | Held-out seeds 2,3,4 | Combined |
|---|---:|---:|---:|
| Balanced K=2 | 4/4 | 6/6 | 10/10 |
| Unequal K=3 | 2/4 | 3/6 | 5/10 |
| Hierarchical K=4 | 3/4 | 3/6 | 6/10 |
| Disassortative K=3 | 4/4 | 3/6 | 7/10 |

On the same n=1024 confirmatory graphs it recovered 4/4 binary endpoints and 2/4 hierarchical endpoints in 200 sweeps. The proposal was developed after examining failures, and the larger graphs had already been inspected. These are exploratory algorithm-development results; the weaker held-out success in some families is retained. All candidate methods, including their failures, are archived.

## Failure diagnostics and implementation checks

In eight seed-0 conditions, truth-initialized Gibbs stayed within 5% error throughout the recorded second half of 200 sweeps; the four strong conditions stayed exactly correct. Random-start fitted block probabilities had almost indistinguishable rows (oracle one-node separation around 4e-6 to 4e-4 in units of log n). Giving accurate fixed P to a separate random-start diagnostic improved most cases, but still failed on one strong hierarchical case. This supports both flattening of P under random groups and local search barriers. A stable truth basin does not prove global optimality or fast mixing.

Independent enumeration on small graphs checked block counts, self-edge removal, empty groups, permutation invariance, exact conditionals, Gibbs detailed balance, replica swap acceptance, and monotone ICM and EM updates. Actual sampler stationary frequencies were checked against enumeration. The three global kernels received separate detailed-balance and empirical-marginal checks. JSON validation records and the final label/score audit are included alongside source. No theorem or asymptotic sampling guarantee is inferred from the recovery counts.

## Reproduction and evidence

Python 3.11.7 was used with the versions in requirements.txt. Supplemental configurations preserve source hashes; original core configurations do not preserve historical source hashes, which limits provenance verification for those runs. The final audit records this limitation. Run these commands from this directory after installing dependencies:

```sh
python independent_validation.py
python benchmark.py --n 512 --seeds 0,1,2,3,4 --starts 2 --sweeps 1000 --em-iters 200 --signal 0.65 1.35 --out results_reproduced
python benchmark.py --n 1024 --seeds 0,1 --starts 2 --sweeps 2000 --em-iters 200 --models balanced2 hierarchy4 --signal 1.35 --out results_n1024_reproduced
python run_k6.py --n 512 --seeds 0,1 --starts 2 --sweeps 2000 --em-iters 200 --models balanced6 --signal 1.35 --out results_k6_reproduced
python run_prototype_global.py --inputs results_reproduced --out results_prototype_reproduced --seeds 0 1 2 3 4 --sweeps 200
python summarize_results.py --inputs results_reproduced --out analysis_reproduced
```

The evidence archive `spectral_partition_mcmc_evidence_20261011.zip` contains all completed result directories except the compilation smoke test: graph data, retained eigenpairs, random initial labels, all chain/restart traces, EM responsibilities and P, protocol metadata, final and best-score endpoints, and failure logs. Code is stored separately in this repository. `archive_metadata.json` records the archive SHA-256 and sizes; the archive contains a per-file SHA-256 manifest. `build_evidence.py` rebuilds it from the result directories. No failed recovery is removed.

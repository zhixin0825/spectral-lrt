# Population counterexample for a finite-stage growing start

This is an exact limiting-path audit of the original global-mean growing initialization. It establishes a failure when **each intermediate stage takes one M-step**. It does **not** establish failure when every intermediate stage is fitted to convergence. In this same example two M-steps per stage succeed.

The deterministic parameter replacement used by the new decoder repairs this example in one accepted round, with one M-step per candidate replacement.

## Model

Let the five block proportions be

`pi = (4, 114, 13, 44, 25) / 200 = (.02, .57, .065, .22, .125)`.

Let `P = d B / n`, where

```text
B = [[.0071, .0914, .0119, .0129, .0132],
     [.0914, 2.7252, .2538, .2125, .2722],
     [.0119, .2538, .0285, .0276, .0300],
     [.0129, .2125, .0276, .0294, .0289],
     [.0132, .2722, .0300, .0289, .0321]]
```

Every entry is positive. The leading principal determinants of the integer matrix `10000 B` are exactly

`71, 1099496, 22194480, 49511806, 22302154`.

They are all positive, so Sylvester's criterion proves that B is positive definite and hence full rank. Choose n as a sufficiently large multiple of 200 with `n > 2.7252 d` to obtain valid block probabilities. The block expected degrees differ; each is of order d. The symbol d here is the common probability scale, not a requirement that degrees be equal.

## Working scores and limiting path

At the population profile level, every block-a node has row `B[a]`, and coordinates have weights pi. Define

`D_pi(x || y) = sum_b pi[b] (x[b] log(x[b]/y[b]) - x[b] + y[b])`.

For `P = d B / n`, pairwise differences of the sparse working profile score are exactly `-d D_pi(x || y)` up to a row-dependent constant. Candidate gains therefore use the weighted all-node sum of positive reductions in `D_pi`. All component weights are reset uniformly when a center is added, exactly as in the original growing procedure. The profile M-step is the exact responsibility-weighted average.

The calculations below use **one-based block and component indices**. Write `m_S` for the pi-weighted mean of rows whose block indices are in S.

| Stage | Added block | Profiles after its one M-step, in component order |
|---|---:|---|
| 2 | 4 | `(B_2, m_{1,3,4,5})` |
| 3 | 1 | `(B_2, m_{3,4,5}, B_1)` |
| 4 | 5 | `(B_2, m_{3,4}, B_1, B_5)` |
| 5 | 4 | `(B_2, B_4, B_1, m_{3,5}, B_4)` |

After the stage-4 M-step, block 3 now prefers component 4, but the previous center `m_{3,4}` still includes it. At stage 5, adding a new pure block-4 profile gives slightly more global gain than adding block 3. The previous mixed component then wins no block. Its soft responsibilities remain positive, but its M-step is dominated by block 4, so it converges to another copy of `B_4`. This is an exact-arithmetic effect, not a floating-point underflow artifact.

The final fitted centers duplicate block 4 and merge blocks 3 and 5. The permutation-invariant error tends to `min(pi_3, pi_5) = 13/200 = .065`. Any fixed number of additional final EM steps preserves this limiting configuration; the numerical check includes 100 extra final steps. Excluding previously selected node IDs does not prevent this path because another node of block 4 remains an eligible candidate.

These population numerical checks use the exact floored-simplex weight update with floor `1e-8`, as in the archived original implementation. The floor is inactive on every nonvanishing component. It only keeps a positive weight on the duplicate component. All limiting candidate and nonduplicate assignment inequalities are independent of this fixed positive floor because its logarithm is negligible compared with d. The new general-density engine is audited separately with its own declared weight update.

## Exact comparison certificate

Run:

```bash
python experiments/20261010_general_density/population_audit/certify_counterexample.py
```

The script uses Python `Fraction` arithmetic throughout each comparison. It bounds each logarithm by a 32-term atanh series after reducing the argument to [1, 2], and adds an explicit geometric remainder bound. Thus all assertion checks compare exact rational interval endpoints. Floating-point numbers in the saved JSON are only displays of the certified bounds.

The certified strict margins by which the selected candidate beats every other block candidate are, in stage order,

`8.942547449706262e-5, 3.913725463947909e-4, 7.663440740879990e-6, 7.982626479433350e-7`.

In the vanishing old component at stage 5, block 4 has a smaller score deficit than every other block, with certified margin at least `7.586166946867726e-4`. This establishes the soft M-step limit even when its total mass is exponentially small. Final score comparisons away from the two duplicate centers have margin at least `2.001157094528786e-3`.

`counterexample_exact_certificate.json` records every stage, gain interval, assignment comparison, and the soft-responsibility dominance check.

The exact Bernoulli profile divergence divided by d converges to this same `D_pi` when `d/n -> 0`. Because the comparisons used here are strict, the same population path persists for sufficiently sparse probabilities and sufficiently large d; for example one can choose `d = (log n)^2`. This extension concerns population profiles. A sampled-graph failure theorem additionally needs uniform control of the candidate profiles and is not claimed by this audit.

## Deterministic replacement validation

Run:

```bash
python experiments/20261010_general_density/population_audit/replacement_audit.py
```

The evaluator tests all five component removals. For each removal it inserts the available row with the largest all-node positive likelihood gain, resets the weights uniformly, and takes one exact M-step. It accepts the endpoint with the largest value of the same working mixture objective, provided this improves on the incumbent. No random restart is used.

The following objective is `L/n` minus the fixed mean self-score, so a larger value is better. It is centered only to avoid large irrelevant constants.

| d | Growing error | Growing objective | Repaired error | Repaired objective |
|---:|---:|---:|---:|---:|
| 100,000 | .065 | -2.693066321534068 | 0 | -1.169355457935237 |
| 1,000,000 | .065 | -17.505005749963967 | 0 | -1.169355457935234 |
| 100,000,000 | .065 | -1646.8183428772527 | 0 | -1.169355457935234 |
| 10,000,000,000 | .065 | -164578.15205560616 | 0 | -1.169355457935234 |
| 1,000,000,000,000 | .065 | -16457711.523328496 | 0 | -1.169355457935234 |

At `d = 10^6`, deleting component 2 and inserting block 3 is the first improving best trial. Deleting the other duplicate, component 5, gives the same optimal endpoint under a different component ordering. One round is sufficient. The next two rounds find no further improvement. At `d = 10^4`, the growing fit itself already succeeds, illustrating that finite-degree soft assignments can sometimes avoid the limiting failure.

`replacement_counterexample_results.json` contains the complete matrices, block proportions, eigenspectrum, every growing-stage profile/weight/responsibility/score trace, all replacement trials, and the additional-final-EM controls. The profile update normalizes its component-specific mass in log space, so exponentially small responsibilities cannot silently cause a component to retain an old mean.

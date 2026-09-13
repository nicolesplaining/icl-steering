# Continuation fitting helps early states but fails the diagnostic gate

September 13, 2026. The continuation-trained ridge map improves on a
prompt-only fit, but its out-of-fold error exceeds the zero-shift baseline
at 32 and 128 continuation tokens. Its mean continuation error is 0.938,
versus 0.716 for a separate scalar predictor at each position. The declared
diagnostic fails, so this candidate will not proceed to answer generation.

## Question-disjoint predictions

We used the saved paired zero-shot and ICL states for the original 128
extraction questions. A fixed shuffle split questions into four folds of 32.
For each fold, all positions from its 32 evaluation questions were excluded
from fitting on the other 96 questions. There are 620 available state pairs
in total, and the four fits use 467, 465, 462, and 466 training rows.

The primary map pools available positions 0, 1, 8, 32, and 128 with equal
weight per state. It uses uncentered ridge with penalty five and no position
feature. No penalty, strength, or normalization was selected from results.
The [plan](ltv-prefix-crossfit-plan.md) and four tests were committed as
`0f6d9d3`; the [declaration](../results/gsm8k-ltv-crossfit-v1-declaration.json)
was committed as `cad38a0` before evaluation.

## All prediction errors

Entries are normalized squared errors on out-of-fold predictions. Zero
shift has error one. The final column is the unweighted mean across the
four continuation positions, excluding the prompt. This weights each
position equally instead of letting large early target norms dominate.

| Predictor | Prompt | 1 token | 8 tokens | 32 tokens | 128 tokens | Continuation mean |
|---|---:|---:|---:|---:|---:|---:|
| Continuation ridge | 0.173186 | 0.147939 | 0.779391 | 1.513671 | 1.309460 | 0.937615 |
| Zero shift | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Prompt-only ridge | 0.159645 | 1.845578 | 4.455681 | 4.879585 | 3.091723 | 3.568142 |
| Pooled mean | 0.808344 | 0.970965 | 1.032750 | 1.095647 | 1.099540 | 1.049725 |
| Pooled scalar | 0.665021 | 0.676720 | 1.039141 | 1.266067 | 1.068047 | 1.012493 |
| Position-specific mean | 0.258494 | 0.444563 | 0.854144 | 0.942453 | 0.832130 | 0.768322 |
| Position-specific scalar | 0.148026 | 0.328464 | 0.824766 | 0.927814 | 0.784221 | 0.716316 |
| Targets permuted within position | 0.424781 | 0.614030 | 2.144403 | 3.333785 | 2.873128 | 2.241336 |

A position-specific scalar uses only its training-fold mean difference and
one fitted coefficient times the centered input state. The shuffled-target
ridge preserves each position's training-target multiset but breaks question
pairing. All controls use their own magnitudes.

## Primary direction and magnitude

| Prefix tokens | Questions | Median cosine | Median predicted / target norm |
|---:|---:|---:|---:|
| 0 | 128 | 0.917691 | 0.934835 |
| 1 | 128 | 0.958575 | 0.949694 |
| 8 | 128 | 0.587773 | 1.139512 |
| 32 | 128 | 0.293188 | 1.393952 |
| 128 | 108 | 0.321035 | 1.546215 |

All 128 questions reach the first four positions; 108 reach position 128.
The map predicts the one-token target particularly well, but later states
still show excess magnitude and weak direction. These observations do not
justify selecting only the successful position after evaluation.

The [complete result](../results/gsm8k-ltv-crossfit-v1-results.json) includes
all eight predictors, every fold and position, norms, medians, 10th/90th
percentiles, zero norms, undefined quantities, and checks. The primary's
four fold-level continuation errors are 0.936, 1.062, 0.846, and 0.969.
These folds share training questions and are not independent replications.

## Frozen gate and verification

The map fails the requirement to beat zero at every continuation position.
It passes the positive-median-cosine check. It achieves the required 10%
mean-error reduction against prompt-only ridge, pooled mean, and shuffled
targets, but fails that threshold against pooled scalar, position-specific
mean, and position-specific scalar. All criteria remain unchanged.

The [verification record](../results/gsm8k-ltv-crossfit-v1-verification.json)
records four passing tests, including perturbing excluded targets without
changing their predictions. Every ridge prediction was checked against a
separate eigensystem solution; metrics were checked with per-question dot
products. A separate invocation reproduced the saved report and all saved
prediction arrays. Model weights were unchanged; the run used CPU only.

This is method development after observing the earlier prefix diagnostic.
Question exclusion prevents direct fitting leakage, but does not make the
broader research process preregistered or this a held-out math-accuracy test.
No development or reserved answer generations were added.

## Next test

The fixed penalty of five leaves a possible regularization failure unresolved.
A separate nested cross-validation experiment can choose stronger penalties
using only each outer fold's training questions. It should retain the same
simple controls and gate, report every selected penalty and inner score,
and disclose the reused extraction pool. This does not rescue or replace
the failed fixed-penalty result. Accuracy still requires a separately
declared experiment and untouched confirmation questions.

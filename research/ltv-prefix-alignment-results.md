# The prompt-fitted map misses the ICL shift during decoding

September 13, 2026. On identical continuation prefixes, the frozen map
overpredicts the ICL shift and loses directional alignment. At every tested
continuation position, its normalized squared prediction error exceeds one,
so predicting zero shift has lower error. This supports a prompt-to-decoding
mismatch in the [failed all-token experiment](gsm8k-ltv-results.md). It does
not by itself prove why that experiment repeated or truncated.

## Same-prefix measurements

The [plan](ltv-prefix-alignment-plan.md) was fixed before the parent scores
were opened. The [collection declaration](../results/gsm8k-ltv-prefix-v1-declaration.json)
was committed as `dea717b` after the parent failure was reviewed and published.
We used only the original 128 training extraction questions, frozen model,
bank A, and frozen fits. No new fit, answer supervision, or accuracy
evaluation was performed.

Each prefix was generated greedily without steering, with an EOS-only stop
and a 128-token limit. Exactly the same raw prefix token IDs were appended
to the zero-shot and ICL prompts before two unmodified state captures.
All 128 questions reach positions 1, 8, and 32. At position 128, 108 remain;
20 had ended earlier. Unavailable positions are excluded from metrics.

The norm ratio and cosine below are medians of per-question measurements.
Normalized squared error is summed prediction error divided by summed
target energy. A zero prediction has error one. Prompt states were used to
fit the map, so their near-perfect result is in-sample.

| Prefix tokens | Questions | Target norm | Predicted norm | Predicted / target norm | Target cosine | Normalized squared error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 134.407 | 134.376 | 1.000 | 1.000000 | 9.56648e-07 |
| 1 | 128 | 91.737 | 174.539 | 1.865 | 0.616387 | 1.85699 |
| 8 | 128 | 38.213 | 125.195 | 2.852 | 0.134373 | 4.67402 |
| 32 | 128 | 38.114 | 99.607 | 2.300 | 0.028199 | 5.17258 |
| 128 | 108 | 34.952 | 90.213 | 2.347 | 0.129627 | 3.31464 |

The target shrinks after the prompt, but the map does not follow it. Its
median cosine falls from nearly one at the fitted prompt to 0.616 after
one token, 0.134 after eight, and 0.028 after 32. At position 32, median
predicted norm is 99.61 while median target norm is 38.11. The median of
per-question ratios is 2.30; it is not the ratio of those two medians.

## All declared predictors

Each cell gives median cosine followed by normalized squared error. The
full map uses the frozen ridge fit. The mean and standalone scalar retain
their own magnitudes. Scalar-norm and permuted targets are matched to the
full map's predicted norm at the same current state.

| Prefix tokens | Full map | Mean | Scalar | Scalar-norm | Permuted targets |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 / 9.56648e-07 | 0.886 / 0.254404 | 0.943 / 0.14576 | 0.943 / 0.13333 | 0.775 / 0.485626 |
| 1 | 0.616 / 1.85699 | -0.465 / 3.08863 | 0.174 / 2.14209 | 0.174 / 3.18654 | -0.308 / 4.66243 |
| 8 | 0.134 / 4.67402 | -0.053 / 4.9643 | 0.021 / 5.9975 | 0.021 / 5.54171 | -0.040 / 6.22294 |
| 32 | 0.028 / 5.17258 | 0.120 / 5.68433 | 0.237 / 8.48344 | 0.237 / 4.69958 | 0.043 / 5.23864 |
| 128 | 0.130 / 3.31464 | 0.256 / 4.09853 | 0.442 / 8.22493 | 0.442 / 2.6365 | 0.004 / 3.63566 |

All five predictors have error above one at every continuation position.
Norm matching does not establish directional quality: the standalone scalar
and its norm-matched version have the same cosine, while their errors differ.
The scalar has higher median cosine than ridge at positions 32 and 128,
but its excessive magnitude still produces large error.

## Changes on the same questions

These are medians of paired changes from each retained question's own
prompt measurement. The 128-token comparison uses the same 108 questions
at both positions. They are not differences between unrelated group medians.

| Prefix tokens | Target norm change | Full-map norm change | Norm ratio change | Cosine change |
|---:|---:|---:|---:|---:|
| 1 | -43.013732 | +36.442729 | +0.865140 | -0.383612 |
| 8 | -85.051309 | -2.003594 | +1.851487 | -0.865626 |
| 32 | -99.266142 | -29.594839 | +1.299750 | -0.971800 |
| 128 | -101.146334 | -46.277881 | +1.347375 | -0.870373 |

The [full report](../results/gsm8k-ltv-prefix-v1-results.json) includes all
five predictors' medians, 10th and 90th percentiles, paired changes, errors,
and counts. There are no zero target or prediction norms and no undefined
ratios or cosines among the included measurements.

## Verification and limits

All 32 batches completed with exit code zero. Every zero-prefix capture
reproduced the original zero-shot and ICL extraction arrays bit for bit.
The [verification record](../results/gsm8k-ltv-prefix-v1-verification.json)
binds the declaration, collection, and report hashes. Saved inputs, paired
token IDs, availability masks, fitted weights, and scalar parameters were
checked. Predictions and metrics were independently reconstructed; a
separate replay reproduced the saved report. GPU 0 ran sequentially and
was released. GPU 1 was left available.

Continuation positions are new states on the same extraction questions,
not a test of generalization to unseen questions. Prefixes come from
zero-shot generation and need not be correct or typical of ICL. This
diagnostic tests activation prediction, not whether replacing the state
with the measured target improves accuracy. A strength reduction might
reduce error, but no strength search or refit was performed here.

A subsequent continuation-trained map should first be evaluated with
question-disjoint fitting and evaluation folds. Its held-out activation
prediction should beat zero and simple mean/scalar predictors before
another answer-generation experiment. The separate
[prompt-only result](ltv-first-token-interpretation.md) also needs fixed-token
and matched prompt-only controls. Neither path changes the failed parent
candidate or counts as independent positive confirmation.

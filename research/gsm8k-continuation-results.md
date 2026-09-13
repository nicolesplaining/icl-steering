# Continuation steering improves development accuracy but ties shuffled targets

September 13, 2026. Regularized continuation steering scores 115/128 after
blinded review, versus 93/128 zero-shot, with no truncations. It matches
ICL-A's 115/128 and is one answer below ICL-B. However, the shuffled-target
map also scores 115/128. The sole candidate fails the frozen requirement to
strictly beat every mean/scalar/shuffled control. No reserved answers were
generated, and no control replaces the primary candidate.

This is a positive development signal over zero-shot, not independently
confirmed ICL transfer. Correct source-to-target pairing has not been shown
to improve answer accuracy. The comparison with the same map applied only
at the prompt is also inconclusive: 115 versus 111, with nine wins and five
losses, a gain of 3.125 percentage points and a 95% interval of
[-2.34375, 8.59375].

## Setup and all conditions

The [protocol](gsm8k-continuation-protocol.md) was declared before fitting.
The frozen Qwen2.5-Math-7B model used the same prompts, two eight-example
banks, greedy decoding, 1,024-token answer limit, and 4,096-token context
limit as the previous experiment. Six baseline conditions were inherited
exactly. The 128 development questions have informed earlier experiments;
the 256 reserved questions remain untouched.

The map uses 620 paired states from 128 separate extraction questions at
prefix lengths 0, 1, 8, 32, and 128. Each pair uses identical zero-shot-generated
prefix tokens in the zero-shot and ICL contexts. The uncentered ridge penalty
is 0.1 times the mean squared input-state norm, selected in the preceding
[nested activation crossfit](ltv-prefix-regularization-results.md). It adds
the predicted shift after final normalization, immediately before the LM
head, at prefix lengths 0 through 128 inclusive. Later states are unchanged.
The fitted map is recomputed on the current state, with strength one.

The shuffled control permutes target shifts within each measured position.
It uses its own nested-selected penalty scale of one, ten times the primary
penalty. This is not an equal-penalty permutation experiment. It preserves
position-specific target distributions while breaking individual pairings.
Neither its tie nor its activation fit establishes equivalence of mechanisms.

All counts below are out of 128. Both accuracy columns reject truncated
answers. Only unparsed responses received blinded transcription; parsed
grades remain fixed.

| Condition | Explicit parser | Audited | Unparsed | Truncated |
|---|---:|---:|---:|---:|
| Zero-shot | 78 | 93 | 23 | 3 |
| ICL-A | 114 | 115 | 1 | 0 |
| ICL-B | 115 | 116 | 2 | 0 |
| First | 97 | 108 | 11 | 0 |
| Step by step | 105 | 110 | 7 | 0 |
| Old map, prompt only | 105 | 113 | 11 | 0 |
| Regularized map, first 129 tokens | 96 | 115 | 20 | 0 |
| Same regularized map, prompt only | 103 | 111 | 10 | 0 |
| Pooled mean | 67 | 89 | 46 | 0 |
| Pooled affine scalar | 63 | 78 | 55 | 0 |
| Position-specific mean | 93 | 108 | 21 | 0 |
| Position-specific affine scalar | 29 | 41 | 89 | 51 |
| Shuffled-target map | 99 | 115 | 19 | 3 |

The candidate's audited accuracy is 89.84%, versus 72.66% zero-shot. Its
17.1875-point gain has an unadjusted paired 95% interval of [10.15625, 24.21875],
24 wins and two losses. The Holm-adjusted exact p-value is 0.0000944138.
This is conditional on the reused development sample and does not correct
for the broader sequence of experiments that led to this method.

The explicit-parser metric tells a different story about output formatting:
the candidate scores 96/128, below the same-map prompt-only control's 103/128,
step by step's 105/128, and both ICL banks. Blinded transcription recovers
19 completed correct answers for the candidate and eight for the same-map
prompt-only control. The audited four-answer continuation advantage therefore
does not appear under the original explicit parser. Both metrics were
specified before generation and are reported without changing either parser.

## Paired candidate comparisons

Every contrast is the candidate minus the named condition. Gains and 95%
intervals are percentage points. Intervals use 10,000 paired bootstrap draws
with seed 907 and are unadjusted. Exact two-sided McNemar tests receive Holm
adjustment across 12 contrasts, separately for each metric.

### Audited

| Comparison | Gain | 95% interval | Wins | Losses | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | +17.19 | [10.16, 24.22] | 24 | 2 | 1.04904e-05 | 9.44138e-05 |
| ICL-A | +0.00 | [-6.25, 5.47] | 7 | 7 | 1 | 1 |
| ICL-B | -0.78 | [-7.03, 5.47] | 8 | 9 | 1 | 1 |
| First | +5.47 | [0.00, 11.72] | 11 | 4 | 0.118469 | 0.947754 |
| Step by step | +3.91 | [-2.34, 10.94] | 12 | 7 | 0.359283 | 1 |
| Old map, prompt only | +1.56 | [-3.91, 7.03] | 8 | 6 | 0.790527 | 1 |
| Same regularized map, prompt only | +3.12 | [-2.34, 8.59] | 9 | 5 | 0.42395 | 1 |
| Pooled mean | +20.31 | [12.50, 28.91] | 30 | 4 | 6.16489e-06 | 6.16489e-05 |
| Pooled affine scalar | +28.91 | [20.31, 37.50] | 40 | 3 | 3.02134e-09 | 3.32348e-08 |
| Position-specific mean | +5.47 | [-0.78, 12.50] | 13 | 6 | 0.167068 | 1 |
| Position-specific affine scalar | +57.81 | [48.44, 67.19] | 77 | 3 | 1.41284e-19 | 1.69541e-18 |
| Shuffled-target map | +0.00 | [-5.47, 5.47] | 7 | 7 | 1 | 1 |

### Explicit parser

| Comparison | Gain | 95% interval | Wins | Losses | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | +14.06 | [3.91, 23.44] | 31 | 13 | 0.00955988 | 0.0669192 |
| ICL-A | -14.06 | [-22.66, -5.47] | 8 | 26 | 0.00293506 | 0.0234804 |
| ICL-B | -14.84 | [-23.44, -6.25] | 8 | 27 | 0.00187823 | 0.016904 |
| First | -0.78 | [-9.38, 7.81] | 15 | 16 | 1 | 1 |
| Step by step | -7.03 | [-15.62, 1.56] | 13 | 22 | 0.175465 | 1 |
| Old map, prompt only | -7.03 | [-16.41, 1.56] | 13 | 22 | 0.175465 | 1 |
| Same regularized map, prompt only | -5.47 | [-14.06, 3.12] | 13 | 20 | 0.296206 | 1 |
| Pooled mean | +22.66 | [12.50, 32.81] | 41 | 12 | 8.17133e-05 | 0.000817133 |
| Pooled affine scalar | +25.78 | [15.61, 35.94] | 44 | 11 | 8.69937e-06 | 9.56931e-05 |
| Position-specific mean | +2.34 | [-6.25, 10.94] | 17 | 14 | 0.7201 | 1 |
| Position-specific affine scalar | +52.34 | [40.62, 63.28] | 78 | 11 | 1.36738e-13 | 1.64086e-12 |
| Shuffled-target map | -2.34 | [-10.94, 6.25] | 14 | 17 | 0.7201 | 1 |

## What remains unresolved

The candidate improves on pooled mean and scalar controls, but the
position-specific mean is only seven answers behind, with an interval
including zero for that difference. The shuffled map ties the candidate
with seven wins and seven losses. These results leave open whether the
useful part is a shared, state-dependent change to output behavior rather
than a question-specific ICL transformation. The independently chosen
regularization strengths further limit a pairing-specific conclusion.

The candidate and same-map prompt-only control had exactly matching initial
states, shifts, applied states, and first generated tokens on all 128
questions. Their later intervention is the controlled difference, but its
small audited advantage is not reliably distinguished from zero. The
candidate also has no established advantage over First, step by step, or
the old prompt-only map after the declared multiple-comparison adjustment.

The position-specific affine scalar, which looked competitive on saved
activation targets, truncates 51/128 answers and scores only 41/128 after
review. The pooled scalar scores 78/128 despite no truncations. Activation
prediction quality alone remains a poor basis for claiming math improvement.
The regularized candidate avoids the old map's collapse, but several things
changed together: training states, penalty, and intervention duration. This
experiment does not isolate which change caused that improvement.

The [output-head interpretation](ltv-output-head-interpretation.md) still
applies while the linear map is active. Accuracy alone does not prove that
an internal reasoning procedure was transferred.

The next useful question is whether the map's gain survives controls that
separate shared position-dependent shifts from correctly paired residuals,
with regularization handled explicitly. That requires a new declared
development experiment. This failed gate does not authorize reserved-set
generation or promotion of the shuffled control.

## Verification and provenance

- All 896 new generations completed with exit code zero, producing 1,664
  total rows. Physical GPU 0 was used sequentially; GPU 1 was left available.
- The complete [runtime audit](../results/gsm8k-continuation-v1-runtime-check.json)
  replayed 224 batches, 98,556 saved state vectors and 24,639 LM-head calls.
  Vector counts include padded finished rows. Fits, predicted shifts,
  applied BF16 states, and the 129-token cutoff passed independent checks.
- The [first-token check](../results/gsm8k-continuation-v1-first-token-check.json)
  verified all 128 same-map prompt-only comparisons before answer scoring.
- The [packet lock](../results/gsm8k-continuation-v1-review-lock.json) was
  committed as `d32efd0` before new annotation. All 54 inherited annotations
  remain unchanged. All 232 new responses were individually reviewed with
  condition labels and reference answers hidden. Lossless repetition display
  was verified against all 286 original packet responses.
- Complete [annotations](../results/gsm8k-continuation-v1-annotations.json)
  and their [lock](../results/gsm8k-continuation-v1-annotation-lock.json) were
  committed and pushed as `0fc80f7` before scoring. No generated code was
  executed and no missing numeric answer was calculated during review.
- The separate [recount](../results/gsm8k-continuation-v1-recount.json)
  verified all 13 condition counts, 24 paired contrasts, wins, losses, exact
  p-values, Holm adjustments, and new gates. Bootstrap intervals were
  produced by the frozen reporter and were not independently recomputed.
- The [machine-readable result](../results/gsm8k-continuation-v1-results.json)
  retains the failed selection and every comparison. Raw responses, fitted
  maps, state tensors and model weights remain outside Git.

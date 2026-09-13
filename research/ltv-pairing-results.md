# The pairing advantage depends on regularization

September 13, 2026. At penalty scale 0.1, correctly paired ICL targets score
115/128 after blinded review, compared with 105/128 for both shuffled targets
and position-average targets. At scale one, the corresponding scores are
113, 115, and 110. Neither matched-penalty pairing advantage survives the
seven-comparison Holm adjustment. This diagnostic selects no candidate and
authorizes no reserved-set generation.

The clearest declared contrast is within the shuffled control: increasing
the penalty raises audited accuracy from 105 to 115, with 11 wins and one
loss. The adjusted exact p-value is 0.0444336. This helps explain why the
previous real-low versus shuffled-high comparison tied. It does not prove
that correctly paired ICL shifts are unnecessary, or that the two maps use
the same mechanism.

## Fixed comparison

The [protocol](ltv-pairing-protocol.md) was declared before these four new
conditions were generated. It uses the same 620 extraction states, frozen
Qwen2.5-Math-7B model, prompts, eight-example ICL bank, and 128 development
questions as the [continuation run](gsm8k-continuation-results.md). All 1,664
parent output rows were inherited exactly; four conditions added 512 rows.

For each of the five measured prefix positions, the shared-target condition
replaces individual ICL-minus-zero shifts with their position average.
It then fits the same uncentered ridge map to those targets. Its predictions
remain functions of the current normalized activation, with exact linear-map
rank at most five. It does not receive the current prefix index as a
separate feature. This differs from the earlier position-specific mean,
which directly interpolates vectors using token position.

All three target constructions are compared at both previously selected
penalty scales: 0.1 and one times the mean squared input-state norm. The
shuffled conditions use the same single within-position permutation, seed
2390. No new strength or penalty search was performed. All six maps
intervene after final normalization for the first 129 generated tokens,
then leave the remaining states unchanged. The answer budget is 1,024 tokens.

| Targets | Scale 0.1 audited | Scale 1 audited | Scale 0.1 parser | Scale 1 parser |
|---|---:|---:|---:|---:|
| Correctly paired | 115 | 113 | 96 | 101 |
| Shuffled within position | 105 | 115 | 96 | 99 |
| Position averages only | 105 | 110 | 90 | 90 |

At scale 0.1, paired versus shared targets has 15 wins and five losses,
a gain of 7.8125 percentage points, unadjusted 95% interval [0.78125, 14.84375],
and adjusted p = 0.206947. Paired versus shuffled targets has 14 wins and
four losses, the same gain, interval [1.5625, 14.84375], and adjusted
p = 0.185303. These are suggestive development effects, not confirmed pairing
benefits. Under the explicit parser, paired and shuffled targets tie at 96.

At scale one, correct pairings beat shared averages by three answers and
lose to shuffled targets by two. Both intervals include zero. The actual
ICL baselines remain 115 and 116 after review, versus 93 zero-shot.
No equivalence or noninferiority test was declared, so a nonsignificant
difference must not be described as equivalence.

## Every condition

All counts are out of the same 128 development questions. Both accuracy
columns exclude truncated responses. Only unparsed answers received blinded
transcription; parsed grades remain fixed.

| Condition | Explicit parser | Audited | Unparsed | Truncated |
|---|---:|---:|---:|---:|
| Zero-shot | 78 | 93 | 23 | 3 |
| ICL-A | 114 | 115 | 1 | 0 |
| ICL-B | 115 | 116 | 2 | 0 |
| First | 97 | 108 | 11 | 0 |
| Step by step | 105 | 110 | 7 | 0 |
| Old map, prompt only | 105 | 113 | 11 | 0 |
| Paired targets, scale 0.1 | 96 | 115 | 20 | 0 |
| Paired scale 0.1, prompt only | 103 | 111 | 10 | 0 |
| Pooled mean | 67 | 89 | 46 | 0 |
| Pooled affine scalar | 63 | 78 | 55 | 0 |
| Position-specific mean | 93 | 108 | 21 | 0 |
| Position-specific affine scalar | 29 | 41 | 89 | 51 |
| Shuffled targets, scale 1 | 99 | 115 | 19 | 3 |
| Shared targets, scale 0.1 | 90 | 105 | 22 | 3 |
| Shuffled targets, scale 0.1 | 96 | 105 | 15 | 4 |
| Paired targets, scale 1 | 101 | 113 | 16 | 0 |
| Shared targets, scale 1 | 90 | 110 | 27 | 2 |

## All declared paired contrasts

Gains and 95% intervals are percentage points, left condition minus right.
Intervals use 10,000 paired bootstrap draws with seed 907 and are unadjusted.
Exact two-sided McNemar p-values receive Holm adjustment across seven
contrasts separately for each metric. All comparisons are on reused
development questions that informed this diagnostic's design.

### Audited

| Comparison | Gain | 95% interval | Wins | Losses | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Paired targets, scale 0.1 minus Shared targets, scale 0.1 | +7.81 | [0.78, 14.84] | 15 | 5 | 0.0413895 | 0.206947 |
| Paired targets, scale 0.1 minus Shuffled targets, scale 0.1 | +7.81 | [1.56, 14.84] | 14 | 4 | 0.0308838 | 0.185303 |
| Paired targets, scale 1 minus Shared targets, scale 1 | +2.34 | [-2.34, 7.03] | 7 | 4 | 0.548828 | 1 |
| Paired targets, scale 1 minus Shuffled targets, scale 1 | -1.56 | [-6.25, 3.12] | 4 | 6 | 0.753906 | 1 |
| Paired targets, scale 0.1 minus Paired targets, scale 1 | +1.56 | [-3.91, 7.03] | 7 | 5 | 0.774414 | 1 |
| Shuffled targets, scale 0.1 minus Shuffled targets, scale 1 | -7.81 | [-13.28, -3.12] | 1 | 11 | 0.00634766 | 0.0444336 |
| Shared targets, scale 0.1 minus Shared targets, scale 1 | -3.91 | [-8.59, 0.78] | 2 | 7 | 0.179688 | 0.71875 |

### Explicit parser

| Comparison | Gain | 95% interval | Wins | Losses | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Paired targets, scale 0.1 minus Shared targets, scale 0.1 | +4.69 | [-4.69, 14.06] | 20 | 14 | 0.391528 | 1 |
| Paired targets, scale 0.1 minus Shuffled targets, scale 0.1 | +0.00 | [-9.38, 9.38] | 17 | 17 | 1 | 1 |
| Paired targets, scale 1 minus Shared targets, scale 1 | +8.59 | [1.56, 15.62] | 17 | 6 | 0.0346897 | 0.242828 |
| Paired targets, scale 1 minus Shuffled targets, scale 1 | +1.56 | [-4.69, 7.81] | 10 | 8 | 0.814529 | 1 |
| Paired targets, scale 0.1 minus Paired targets, scale 1 | -3.91 | [-11.72, 3.91] | 10 | 15 | 0.424356 | 1 |
| Shuffled targets, scale 0.1 minus Shuffled targets, scale 1 | -2.34 | [-9.38, 4.69] | 8 | 11 | 0.647606 | 1 |
| Shared targets, scale 0.1 minus Shared targets, scale 1 | +0.00 | [-5.47, 5.47] | 6 | 6 | 1 | 1 |

## Interpretation and limits

The previous tie involved two factors: target pairing and regularization.
Matching the penalty reveals a ten-answer audited pairing advantage at the
lower scale, but no comparable advantage at the higher scale. Conversely,
stronger regularization helps the shuffled map by ten answers and the shared
map by five, while changing the correctly paired map by only two. These
patterns are consistent with regularization suppressing harmful variation
in the control fits, but they do not identify a causal mechanism within the
model. Only one fixed target permutation was tested.

The shared-average maps improve descriptively over zero-shot, but do not
match the full low-penalty map's audited count. This weakens a claim that
five average shifts alone reproduce the entire observed gain at that
penalty. It also leaves the need for question-specific target information
unproven after the declared correction. The explicit-parser and audited
metrics disagree about several contrasts; both are retained.

This development sample has now supported several stages of method design.
A fresh development evaluation with a fixed method is more informative than
further selection on these same 128 questions. This diagnostic is not a
selection stage: the original candidate's failed gate remains failed, and
none of these controls is promoted. A subsequent candidate needs its own
prior declaration and development criteria. The 256 reserved questions
remain untouched.

## Verification

- All 512 new generations completed with exit code zero. Physical GPU 0
  was used sequentially, leaving GPU 1 available. No GPU process remains
  from this completed run.
- Both inherited maps were reconstructed exactly before generation. The
  real-data fit audit independently checked the shared map's five-column
  factorization and the identity W_paired = W_shared + W_residual at both
  penalties. See the [fit declaration](../results/gsm8k-pairing-v1-fit-declaration.json).
- The complete [runtime audit](../results/gsm8k-pairing-v1-runtime-check.json)
  replayed 128 batches, 66,048 saved state vectors and 16,512 actual LM-head
  calls, checking predicted shifts, BF16 applied states, and cutoff accounting.
  Counts include padded finished rows.
- The [packet lock](../results/gsm8k-pairing-v1-review-lock.json) was committed
  as `8638ff4` before new annotation. All 286 inherited records are unchanged.
  All 49 new unparsed responses were individually reviewed with conditions
  and reference answers hidden. Lossless display was verified for all 335.
- Complete [annotations](../results/gsm8k-pairing-v1-annotations.json) and
  their [lock](../results/gsm8k-pairing-v1-annotation-lock.json) were committed
  and pushed as `4029777` before scoring. Arithmetic mistakes, unfilled
  placeholders and unfinished answers were preserved; no code was executed
  and no missing answer was calculated during review.
- The independent [recount](../results/gsm8k-pairing-v1-recount.json) verified
  all 17 condition counts and 14 contrasts, including wins, losses, exact
  tests, Holm adjustment, and the absence of candidate selection.
- The separate [bootstrap check](../results/gsm8k-pairing-v1-bootstrap-check.json)
  reproduced all 14 intervals with independent count-weighted aggregation
  and explicit interpolation, using the declared PCG64 draws.
- The [complete result](../results/gsm8k-pairing-v1-results.json) preserves
  every reported condition and comparison. Raw responses, model weights,
  fitted maps and state tensors remain outside Git.

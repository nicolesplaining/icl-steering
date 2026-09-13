# Controls do not establish a need for the full question-specific map

September 13, 2026. The fixed ridge scores 107/128 after blinded answer
review. Its scalar control scores 106/128 when supplied with the same
per-question vector norms. This weakens the claim that the learned full-map
directions are needed for the development gain. It does not establish
equivalence, and the scalar control still depends on ridge for its norm.

The raw mean scores 96/128; that positive ridge contrast survives the declared
Holm adjustment. First scores 108/128 and step-by-step scores 110/128. The
original text-cue gate remains failed, and no confirmation questions were run.

## Completed development comparisons

All sixteen conditions use the same 128 development questions. Seven output
sets were inherited exactly; nine controls were added after observing the
candidate results. No control was separately tuned. The model weights remain
frozen. The main setting is block 13, strength 0.5, final prompt token only.
Only the old mean control uses block 7 and all-token injection.

| Condition | Explicit parser | Audited | Unparsed | Truncated |
|---|---:|---:|---:|---:|
| Zero-shot | 78 | 93 | 23 | 3 |
| ICL-A | 114 | 115 | 1 | 0 |
| ICL-B | 115 | 116 | 2 | 0 |
| First | 97 | 108 | 11 | 0 |
| Step by step | 105 | 110 | 7 | 0 |
| Old mean, block 7, all tokens | 89 | 94 | 14 | 1 |
| Fixed ridge | 97 | 107 | 13 | 3 |
| Raw mean | 84 | 96 | 17 | 4 |
| Norm-matched mean | 87 | 98 | 18 | 4 |
| Norm-matched scalar | 96 | 106 | 14 | 2 |
| Shuffled extraction targets | 85 | 100 | 23 | 3 |
| Rotated demonstration pairs | 94 | 103 | 16 | 2 |
| Reversed ridge | 71 | 87 | 25 | 2 |
| Random 31 | 85 | 100 | 23 | 1 |
| Random 59 | 74 | 86 | 21 | 1 |
| Random 83 | 82 | 95 | 22 | 2 |

Only unparsed responses received blinded transcription. Explicit-parser
grades stayed fixed, and truncated responses receive no credit in either
accuracy column. These are counts out of 128, not percentages.

## Audited paired comparisons

Every row is ridge minus the named condition. Gains and intervals are
percentage points. Intervals use 10,000 paired bootstrap resamples with seed
907 and are exploratory and unadjusted. The exact two-sided McNemar tests
receive Holm adjustment across all fifteen contrasts, separately for each
metric. Discrete paired data can yield a bootstrap interval excluding zero
while the exact test does not reject.

| Comparison | Gain | 95% interval | Wins | Losses | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | +10.94 | [3.91, 17.97] | 18 | 4 | 0.004344 | 0.052122 |
| ICL-A | -6.25 | [-12.50, 0.00] | 4 | 12 | 0.076813 | 0.460876 |
| ICL-B | -7.03 | [-13.28, -0.78] | 4 | 13 | 0.049042 | 0.392334 |
| First | -0.78 | [-7.81, 6.25] | 10 | 11 | 1.000000 | 1.000000 |
| Step by step | -2.34 | [-9.38, 4.69] | 8 | 11 | 0.647606 | 1.000000 |
| Old mean, block 7, all tokens | +10.16 | [3.91, 17.19] | 16 | 3 | 0.004425 | 0.052122 |
| Raw mean | +8.59 | [3.91, 14.06] | 12 | 1 | 0.003418 | 0.044434 |
| Norm-matched mean | +7.03 | [1.56, 12.50] | 11 | 2 | 0.022461 | 0.202148 |
| Norm-matched scalar | +0.78 | [-3.12, 4.69] | 4 | 3 | 1.000000 | 1.000000 |
| Shuffled extraction targets | +5.47 | [0.78, 10.94] | 9 | 2 | 0.065430 | 0.458008 |
| Rotated demonstration pairs | +3.12 | [0.78, 6.25] | 4 | 0 | 0.125000 | 0.625000 |
| Reversed ridge | +15.62 | [7.81, 24.22] | 25 | 5 | 0.000325 | 0.004549 |
| Random 31 | +5.47 | [-0.78, 11.72] | 12 | 5 | 0.143463 | 0.625000 |
| Random 59 | +16.41 | [8.59, 24.22] | 25 | 4 | 0.000104 | 0.001556 |
| Random 83 | +9.38 | [3.12, 16.41] | 16 | 4 | 0.011818 | 0.118179 |

## Explicit-parser paired comparisons

Every row is ridge minus the named condition. Gains and intervals are
percentage points. Intervals use 10,000 paired bootstrap resamples with seed
907 and are exploratory and unadjusted. The exact two-sided McNemar tests
receive Holm adjustment across all fifteen contrasts, separately for each
metric. Discrete paired data can yield a bootstrap interval excluding zero
while the exact test does not reject.

| Comparison | Gain | 95% interval | Wins | Losses | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | +14.84 | [7.03, 22.66] | 24 | 5 | 0.000546 | 0.006936 |
| ICL-A | -13.28 | [-21.09, -5.47] | 5 | 22 | 0.001514 | 0.016651 |
| ICL-B | -14.06 | [-21.88, -7.03] | 4 | 22 | 0.000534 | 0.006936 |
| First | +0.00 | [-7.81, 7.81] | 14 | 14 | 1.000000 | 1.000000 |
| Step by step | -6.25 | [-14.84, 2.34] | 11 | 19 | 0.200488 | 0.843188 |
| Old mean, block 7, all tokens | +6.25 | [-1.56, 14.06] | 17 | 9 | 0.168638 | 0.843188 |
| Raw mean | +10.16 | [3.91, 16.41] | 15 | 2 | 0.002350 | 0.021149 |
| Norm-matched mean | +7.81 | [2.34, 13.28] | 12 | 2 | 0.012939 | 0.090576 |
| Norm-matched scalar | +0.78 | [-3.12, 4.69] | 4 | 3 | 1.000000 | 1.000000 |
| Shuffled extraction targets | +9.38 | [3.91, 14.84] | 13 | 1 | 0.001831 | 0.018311 |
| Rotated demonstration pairs | +2.34 | [-0.78, 6.25] | 4 | 1 | 0.375000 | 1.000000 |
| Reversed ridge | +20.31 | [11.72, 29.69] | 33 | 7 | 0.000042 | 0.000634 |
| Random 31 | +9.38 | [2.34, 17.19] | 18 | 6 | 0.022656 | 0.135935 |
| Random 59 | +17.97 | [9.38, 26.56] | 29 | 6 | 0.000117 | 0.001636 |
| Random 83 | +11.72 | [4.69, 19.53] | 20 | 5 | 0.004077 | 0.032619 |

## Interpretation

The scalar direction is the extraction mean difference plus one fitted
coefficient times the centered zero-shot query activation. It has no learned
cross-coordinate matrix. Before injection it is normalized to the full
ridge prediction length for that question. Thus this control tests whether
the full map is needed to choose direction at a matched magnitude; it does
not test a standalone replacement that avoids computing the ridge map.

Ridge wins four questions and loses three against this scalar control.
The audited difference is +0.78 points, with interval [-3.12, 4.69]. The
experiment supplies no clear evidence of an advantage over that simpler
direction rule. The matched mean is lower at 98/128, but its adjusted
comparison also does not reject. Neither result proves equality.

The shuffled-target map scores 100/128 and rotated-pair map scores 103/128.
Their exact adjusted comparisons do not establish ridge superiority. Random
directions vary substantially, from 86 to 100 correct. Reporting only the
weakest random direction would overstate the evidence.

For audited accuracy, the only Holm p-values below 0.05 are raw mean, reversed
ridge, and random seed 59. The zero-shot comparison is p=0.052122 after this
family adjustment. The positive raw-mean contrast remains useful development
evidence, but it does not verify query association or meet the text-cue gate.

## Verification and provenance

- The [protocol](gsm8k-fixed-controls-protocol.md) and
  [declaration](../results/gsm8k-fixed-controls-v1-declaration.json) were fixed
  before the new control generations. This was explicitly a supplementary
  diagnostic after observing the candidate, not a preregistered confirmation.
- The process exited with code zero after 2,048 rows and released GPU 0.
  GPU 1 was left available. No reserved questions were generated.
- An independent [runtime replay](../results/gsm8k-fixed-controls-v1-runtime-check.json)
  verified every one of the 1,408 intervention vector hashes and norms, plus
  the frozen inputs and unchanged inherited generation rows.
- The shared packet was [locked](../results/gsm8k-fixed-controls-v1-review-lock.json)
  in commit `0116d30`. All 56 inherited annotations were retained exactly.
  The 70 new transcriptions completed all 126 unique unparsed responses.
- Full [annotations](../results/gsm8k-fixed-controls-v1-annotations.json) and
  their [lock](../results/gsm8k-fixed-controls-v1-annotation-lock.json) were
  committed as `e038105` before the diagnostic report was run.
- An independent [recount](../results/gsm8k-fixed-controls-v1-recount.json)
  checked all condition counts and all 30 paired contrasts, including wins,
  losses, exact p-values, and Holm adjustment. The frozen reporting function
  produced the bootstrap intervals.

The [machine-readable report](../results/gsm8k-fixed-controls-v1-results.json)
preserves the original ineligible selection. Raw generations and fitted
activation arrays remain ignored.

## Next experiment

A wider search over the same internal-block ridge settings is not justified
by these controls. The next implementation to examine is the LTV paper's
state-dependent mapping during decoding, with extraction and injection at
the same final normalized activation. The
[related-work audit](conditional-steering-related-work.md) documents why
copying the released wrapper unchanged would not establish that experiment.

This would be a different intervention, requiring its own declared settings,
matched scalar and mean controls, and a validation gate before any untouched
confirmation sample. It does not reopen or relabel this failed candidate.

# Applying the prompt-fitted map throughout decoding fails

September 13, 2026. The final-normalized-state map scores 25/128 after blinded
review, versus 93/128 zero-shot. It hits the 1,024-token limit on 102/128
questions. This candidate fails every new gate. The inherited ICL screen
still passes, and all 256 reserved questions remain untouched.

The same map applied only at the final prompt token scores 113/128 with no
truncations. That is a useful development observation, but this condition
was a control. It does not replace the failed candidate or establish an
independent positive result.

## All development conditions

All counts are out of 128 identical development questions. The five
baselines were inherited exactly. The six interventions use frozen weights,
strength one, and the final normalized state immediately before the LM head.
The map was fitted on 128 separate training questions with eight ICL examples
and an uncentered ridge penalty of five. The [protocol](gsm8k-ltv-protocol.md)
was fixed before extraction and generation.

| Condition | Explicit parser | Audited | Unparsed | Truncated |
|---|---:|---:|---:|---:|
| Zero-shot | 78 | 93 | 23 | 3 |
| ICL-A | 114 | 115 | 1 | 0 |
| ICL-B | 115 | 116 | 2 | 0 |
| First | 97 | 108 | 11 | 0 |
| Step by step | 105 | 110 | 7 | 0 |
| Full map, every token | 23 | 25 | 103 | 102 |
| Raw mean, every token | 23 | 34 | 102 | 86 |
| Scalar, every token | 0 | 0 | 128 | 128 |
| Norm-matched scalar, every token | 0 | 0 | 128 | 128 |
| Norm-matched shuffled targets, every token | 13 | 16 | 112 | 108 |
| Full map, prompt only | 105 | 113 | 11 | 0 |

Only unparsed responses received blinded transcription. Parsed grades stay
fixed; truncated responses receive no credit in either accuracy column.
Both scalar controls repeat without completing any of the 128 answers.
The raw mean and shuffled-target controls also suffer frequent truncation.

## Paired candidate comparisons

Every contrast below is the all-token full map minus the named condition.
Gains and 95% intervals are percentage points. Intervals use the declared
10,000 paired bootstrap draws with seed 907 and are unadjusted. Exact
two-sided McNemar tests receive Holm adjustment across ten contrasts
separately for each metric. These are reused development questions.

### Audited

| Comparison | Gain | 95% interval | Wins | Losses | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | -53.12 | [-63.28, -42.19] | 6 | 74 | 5.39665e-16 | 2.69832e-15 |
| ICL-A | -70.31 | [-78.12, -61.72] | 1 | 91 | 3.75624e-26 | 3.75624e-25 |
| ICL-B | -71.09 | [-79.69, -62.50] | 2 | 93 | 2.30272e-25 | 2.07244e-24 |
| First | -64.84 | [-74.22, -55.47] | 3 | 86 | 3.79889e-22 | 2.65922e-21 |
| Step by step | -66.41 | [-75.78, -57.03] | 4 | 89 | 6.16725e-22 | 3.70035e-21 |
| Raw mean, every token | -7.03 | [-15.62, 1.56] | 11 | 20 | 0.149613 | 0.299226 |
| Scalar, every token | +19.53 | [12.50, 26.56] | 25 | 0 | 5.96046e-08 | 2.38419e-07 |
| Norm-matched scalar, every token | +19.53 | [12.50, 26.56] | 25 | 0 | 5.96046e-08 | 2.38419e-07 |
| Norm-matched shuffled targets, every token | +7.03 | [-1.56, 15.62] | 21 | 12 | 0.162756 | 0.299226 |
| Full map, prompt only | -68.75 | [-77.34, -60.16] | 2 | 90 | 1.72827e-24 | 1.38262e-23 |

### Explicit parser

| Comparison | Gain | 95% interval | Wins | Losses | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | -42.97 | [-52.34, -32.81] | 5 | 60 | 4.86956e-13 | 2.43478e-12 |
| ICL-A | -71.09 | [-78.91, -63.28] | 0 | 91 | 8.07794e-28 | 8.07794e-27 |
| ICL-B | -71.88 | [-79.69, -63.28] | 1 | 93 | 9.59255e-27 | 8.63329e-26 |
| First | -57.81 | [-67.19, -48.44] | 3 | 77 | 1.41284e-19 | 8.47705e-19 |
| Step by step | -64.06 | [-73.44, -54.69] | 3 | 85 | 7.34465e-22 | 5.14126e-21 |
| Raw mean, every token | +0.00 | [-7.81, 7.81] | 13 | 13 | 1 | 1 |
| Scalar, every token | +17.97 | [11.72, 25.00] | 23 | 0 | 2.38419e-07 | 9.53674e-07 |
| Norm-matched scalar, every token | +17.97 | [11.72, 25.00] | 23 | 0 | 2.38419e-07 | 9.53674e-07 |
| Norm-matched shuffled targets, every token | +7.81 | [-0.78, 16.41] | 21 | 11 | 0.110184 | 0.220368 |
| Full map, prompt only | -64.06 | [-72.66, -55.47] | 1 | 83 | 8.78879e-24 | 7.03104e-23 |

## What this establishes

Repeated application is damaging in this setup. The primary candidate
scores 19.53% with 79.69% truncation. It loses 74 questions and wins six
against zero-shot. Its audited loss is 53.12 percentage points, with a
95% interval from 42.19 to 63.28 points below zero-shot. The implemented
intervention was independently verified at the actual LM-head input, so
the result cannot be dismissed as an unverified hook placement.

Prompt-only injection scores 88.28%, compared with 72.66% zero-shot, 85.94%
step by step, and 89.84% or 90.62% with actual ICL. Its exploratory paired
gain over zero-shot is 15.62 points, interval [7.03, 24.22], with 26 wins and
six losses. The three-question advantage over step by step has not been
established as reliable. No prompt-only mean, scalar, or shuffled-target
control was generated in this run. Those all-token controls cannot establish
that the full map is needed for prompt-only success.

The [earlier decoding-state diagnostic](ltv-decoding-state-shift.md) showed
a sharp change after the first generated token, but that alone does not
prove the cause of failure. The [identical-prefix diagnostic plan](ltv-prefix-alignment-plan.md)
was frozen before scores were opened. It will measure the actual ICL-minus-zero
target at fixed continuation positions on the original extraction questions,
without new fitting or reference-answer supervision. It does not authorize
a replacement accuracy candidate.

The [output-head interpretation](ltv-output-head-interpretation.md) also
limits a reasoning-transfer claim. In exact arithmetic, this final-state
linear map is equivalent to a fixed low-rank output-head change. Any later
positive accuracy result must still be distinguished from evidence about
the internal reasoning process.

## Verification and provenance

- Generation completed all 1,408 rows with exit code zero. No reserved
  outputs were generated. GPU 0 was used sequentially; GPU 1 was left available.
- The complete [trajectory audit](../results/gsm8k-ltv-v1-runtime-check.json)
  replayed 192 new batches, 768 new rows, 652,444 saved state vectors and
  163,111 LM-head calls. State-vector counts include padded finished rows.
- The [review packet lock](../results/gsm8k-ltv-v1-review-lock.json) was
  committed as `ac8eead` before new annotation. All 44 inherited records
  remain unchanged.
- All 573 [annotations](../results/gsm8k-ltv-v1-annotations.json) and their
  [lock](../results/gsm8k-ltv-v1-annotation-lock.json) were committed and
  pushed as `1cca5c5` before scoring. Of 529 new responses, 348 were individually
  reviewed and 181 exactly matched five manually reviewed answerless responses.
  The [match log](../results/gsm8k-ltv-v1-review-exact-matches.json) records
  each full-text equality. Repetition compression was lossless for all 573.
- A separate [recount](../results/gsm8k-ltv-v1-recount.json) checked every
  condition count, all 20 contrasts, wins, losses, exact p-values, Holm
  adjustment, and the new candidate gates. Bootstrap intervals come from
  the frozen reporter and were not independently recomputed.
- The [machine-readable result](../results/gsm8k-ltv-v1-results.json)
  preserves the failed selection and all comparisons. Raw generations,
  model weights, fitted tensors, and state arrays remain ignored.

This is a negative result for the declared math adaptation of Linear Task
Vectors. It is not a replication of the paper's classification benchmarks
or evidence that all ICL steering methods fail.

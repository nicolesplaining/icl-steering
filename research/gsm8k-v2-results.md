# GSM8K: ICL helps, but the mean direction falls short

September 13, 2026. The frozen Qwen2.5-Math-7B experiment and all three
supplementary prefix controls are complete. Real eight-example ICL improves
answer accuracy after review. The selected constant activation shift does
not establish a reliable mathematical improvement over zero-shot and trails
the step-by-step text prompt.

## Test results

Every condition uses the same 256 held-out GSM8K questions. The primary
metric uses the declared explicit-answer parser. The supplementary audit
reviews unparsed answers only, keeps parsed grades fixed, and gives no credit
to length-truncated generations.

| Condition | Primary correct | Audited correct | Unparsed | Truncated |
|---|---:|---:|---:|---:|
| Zero-shot | 173 | 197 | 39 | 1 |
| ICL-A | 231 | 231 | 0 | 1 |
| ICL-B | 227 | 227 | 0 | 0 |
| First | 198 | 213 | 23 | 1 |
| Step by step | 208 | 221 | 18 | 0 |
| Selected steering | 198 | 205 | 22 | 0 |
| Reversed steering | 162 | 189 | 47 | 1 |
| Random 31 | 161 | 185 | 42 | 3 |
| Random 59 | 173 | 192 | 37 | 4 |
| Random 83 | 180 | 196 | 33 | 1 |
| Rotated example pairs | 187 | 201 | 30 | 1 |
| Shuffled example tokens | 175 | 196 | 38 | 2 |
| Repeated filler | 165 | 184 | 41 | 4 |

The two ICL banks improve audited accuracy over zero-shot by 13.28125 and
11.71875 percentage points. Their paired-bootstrap 95% intervals are
[8.203125, 18.359375] and [6.640625, 16.796875] points. They use disjoint
demonstrations but share the test questions, so these are not independent
dataset replications.

The primary steering gain is 25 questions, or 9.765625 points. The audit
recovers 24 correct zero-shot answers and seven correct steering answers,
reducing that gap to eight questions, or 3.125 points. Its interval includes
zero. The point estimate recovers about 24% of ICL-A's audited gain, but this
ratio does not establish a positive effect.

| Audited comparison, steering minus control | Gain, percentage points | 95% interval | Wins | Losses |
|---|---:|---:|---:|---:|
| Zero-shot | +3.125 | [-0.78125, 7.03125] | 17 | 9 |
| First | -3.125 | [-8.203125, 1.953125] | 18 | 26 |
| Step by step | -6.25 | [-10.9375, -1.5625] | 12 | 28 |
| Reversed steering | +6.25 | [1.953125, 10.546875] | 25 | 9 |
| Random 31 | +7.8125 | [3.515625, 12.109375] | 27 | 7 |
| Random 59 | +5.078125 | [0.78125, 9.375] | 22 | 9 |
| Random 83 | +3.515625 | [-0.390625, 7.421875] | 18 | 9 |
| Rotated example pairs | +1.5625 | [-1.5625, 4.6875] | 10 | 6 |
| Shuffled example tokens | +3.515625 | [-0.78125, 8.203125] | 22 | 13 |
| Repeated filler | +8.203125 | [3.125, 13.28125] | 32 | 11 |

All intervals use 10,000 paired resamples, seed 907, and are exploratory and
unadjusted. Real steering still beats reversal, two random directions, and
filler under this metric. It does not clearly outperform rotated example
pairs, shuffled tokens, or the third random direction. Beating harmful or
unnatural controls is insufficient to establish a useful ICL intervention.

## What was fixed before the test

The model is `Qwen/Qwen2.5-Math-7B` at revision
`b101308fe89651ea5ce025f25317fea6fc07e96e`. Weights remain frozen.
Extraction uses 128 training questions, validation uses 64 other training
questions, and the test uses 256 previously unused official test questions.
The two fixed support banks each contain eight train-only pseudo-solutions.
Query instructions and the final `A:` suffix are matched across prompts.
Greedy decoding uses a 1,024-token output cap and a 4,096-token context cap.

The declared 24-setting validation sweep selected block index 7, strength
0.5, with injection at the last prefill position and each decoded token.
Selection and directions were locked before test generation. Prefix controls
use the same selected parameters and direction norm, without further tuning.
The [configuration](../configs/gsm8k_steering.json),
[prefix protocol](gsm8k-prefix-test-protocol.md), and
[earlier audit](gsm8k-audit.md) record the design and failed settings.

The shared review packet contains 268 unique unparsed question/response
pairs from all thirteen conditions. It withholds condition names, reference
answers, and original grades. One assistant reviewer transcribed the stated
numbers, preserving wrong answers and leaving absent or unresolved answers
unanswered. It did not compute missing totals or execute generated code.
Output style can still suggest a condition. This is not an independent
semantic evaluation of all generated solutions.

All annotations were saved and committed in `a38e70a` before scoring.
The [annotations](../results/gsm8k-v2-test-annotations.json) and
[review lock](../results/gsm8k-v2-test-review-lock.json) record that boundary.
The [full result](../results/gsm8k-v2-test-answer-audit.json) contains every
comparison, annotation, and input hash. Raw outputs and the blinded packet
are retained in the ignored run directories on the laptop and server.

## Implication for the next experiment

This closes the constant-mean experiment without a convincing positive
steering result. It should not be rescued by selecting another layer or
strength on these test questions. The strong full-ICL contrast makes this
setup worth retaining.

The extraction-only [conditional geometry check](gsm8k-conditional-geometry.md)
shows that averaging discards predictable changes that depend on the query.
A fresh experiment could test whether those changes improve answers through
a query-dependent linear intervention. It would need new validation and
confirmation questions, a constant-mean comparison, the same text controls,
and an answer audit. Control-prefix changes are also predictable, so better
activation reconstruction alone cannot establish the hypothesis. A linear
map would test a different claim from one universal additive direction.

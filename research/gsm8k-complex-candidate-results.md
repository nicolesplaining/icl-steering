# Fixed continuation candidate on fresh complex questions

September 13, 2026. The candidate failed the declared development gate. Audited accuracy was 224/256, versus 216/256 zero-shot: +3.125 percentage points, paired 95% interval [-1.171875, 7.421875]. It scored one more answer than the identical map applied only at the prompt, two fewer than the First cue, and three fewer than the step-by-step cue. No reserved generation is authorized or was performed.

## Setup and provenance

This run followed the [passing ICL screen](gsm8k-complex-screen-results.md) on its same 256 fresh, complex training questions. It reused the original 620 extraction states from 128 separate training questions and their frozen maps without refitting. The sole candidate applies the correctly paired map at penalty scale 0.1 after final normalization for the first 129 generated tokens. Model weights remain frozen. The model, demonstrations, decoding, and all 17 conditions were fixed in the [protocol](gsm8k-complex-candidate-protocol.md) before screen accuracy was inspected; the [run declaration](../results/gsm8k-complex-candidate-v1-declaration.json) preceded new generation.

All 3,072 new outputs completed successfully, alongside 1,280 unchanged baseline outputs. The [full runtime audit](../results/gsm8k-complex-candidate-v1-runtime.json) replayed 330,376 saved state vectors and verified 82,594 output-head calls across all 768 batches. All 256 candidate/prompt-only initial states, shifts, applied states, and first tokens matched. This verifies implementation, not effectiveness.

The [packet lock](../results/gsm8k-complex-candidate-v1-review-lock.json) preceded answer review. All 279 new unparsed responses were individually reviewed, and all 39 inherited annotations were preserved exactly. The [annotations](../results/gsm8k-complex-candidate-v1-annotations.json) and [annotation lock](../results/gsm8k-complex-candidate-v1-annotation-lock.json) were committed in `3990986` before scoring. Review transcribed stated answers without solving missing calculations, correcting explicit numbers, or executing code. Parsed grades stayed fixed; truncated responses receive no completed-answer credit.

The [independent recount](../results/gsm8k-complex-candidate-v1-recount.json) matched all counts, 32 paired contrasts, intervals, exact McNemar tests, Holm adjustments, and selection gates. It uses saved parses and the same declared PCG64 draws; it does not independently reimplement parsing or repeat the runtime audit. This is development evidence on an ICL-screened population, not independent confirmation.

## All condition counts

Each condition has 256 questions. Primary uses the explicit answer parser. Audited additionally credits an explicit answer found during frozen blind review.

| Condition | Primary correct | Audited correct | Audited % | Unparsed | Truncated |
|---|---:|---:|---:|---:|---:|
| zero | 210 | 216 | 84.38 | 10 | 0 |
| icl_a | 227 | 230 | 89.84 | 6 | 1 |
| icl_b | 227 | 229 | 89.45 | 4 | 0 |
| first | 221 | 226 | 88.28 | 6 | 0 |
| cot | 218 | 227 | 88.67 | 13 | 0 |
| steered | 219 | 224 | 87.50 | 11 | 1 |
| regularized_prefill | 217 | 223 | 87.11 | 9 | 1 |
| shared_low | 213 | 222 | 86.72 | 16 | 2 |
| permuted_low | 198 | 208 | 81.25 | 19 | 3 |
| permuted | 202 | 209 | 81.64 | 13 | 3 |
| shared_high | 206 | 216 | 84.38 | 18 | 3 |
| real_high | 212 | 222 | 86.72 | 16 | 2 |
| prefill | 213 | 220 | 85.94 | 8 | 1 |
| mean | 198 | 216 | 84.38 | 30 | 0 |
| scalar | 189 | 204 | 79.69 | 40 | 0 |
| position_mean | 206 | 217 | 84.77 | 19 | 1 |
| position_scalar | 107 | 130 | 50.78 | 124 | 40 |

## All candidate comparisons

Differences are candidate minus comparator, in percentage points. Each interval uses 10,000 paired bootstrap draws with seed 907. McNemar tests are exact and two-sided. Holm adjustment covers 16 contrasts separately for each metric; intervals are unadjusted. Wins/losses count questions where only one condition answered correctly.

### Primary

| Comparator | Gain pp | 95% interval pp | Wins/losses | McNemar p | Holm p |
|---|---:|---|---:|---:|---:|
| zero | +3.5156 | [-1.1719, +8.5938] | 24/15 | 0.1996 | 1 |
| icl_a | -3.1250 | [-7.0312, +0.7812] | 10/18 | 0.1849 | 1 |
| icl_b | -3.1250 | [-7.0312, +0.7812] | 9/17 | 0.1686 | 1 |
| first | -0.7812 | [-5.4688, +3.9062] | 16/18 | 0.8642 | 1 |
| cot | +0.3906 | [-4.2969, +5.0781] | 19/18 | 1 | 1 |
| regularized_prefill | +0.7812 | [-3.5156, +5.0781] | 16/14 | 0.8555 | 1 |
| shared_low | +2.3438 | [-2.3438, +6.6406] | 21/15 | 0.405 | 1 |
| permuted_low | +8.2031 | [+3.1250, +13.2812] | 33/12 | 0.002459 | 0.03442 |
| permuted | +6.6406 | [+1.9531, +11.3281] | 28/11 | 0.009475 | 0.1137 |
| shared_high | +5.0781 | [+0.3906, +9.7656] | 26/13 | 0.05325 | 0.5858 |
| real_high | +2.7344 | [-1.9531, +7.0410] | 21/14 | 0.3105 | 1 |
| prefill | +2.3438 | [-2.3438, +7.0312] | 22/16 | 0.4177 | 1 |
| mean | +8.2031 | [+2.7344, +13.6719] | 37/16 | 0.005486 | 0.07132 |
| scalar | +11.7188 | [+6.2500, +17.1875] | 42/12 | 5.209e-05 | 0.0007814 |
| position_mean | +5.0781 | [+0.3906, +9.7656] | 26/13 | 0.05325 | 0.5858 |
| position_scalar | +43.7500 | [+37.1094, +50.3906] | 119/7 | 2.107e-27 | 3.371e-26 |

### Audited

| Comparator | Gain pp | 95% interval pp | Wins/losses | McNemar p | Holm p |
|---|---:|---|---:|---:|---:|
| zero | +3.1250 | [-1.1719, +7.4219] | 20/12 | 0.2153 | 1 |
| icl_a | -2.3438 | [-5.8594, +1.1719] | 8/14 | 0.2863 | 1 |
| icl_b | -1.9531 | [-5.4688, +1.5625] | 7/12 | 0.3593 | 1 |
| first | -0.7812 | [-4.6875, +3.1250] | 13/15 | 0.8506 | 1 |
| cot | -1.1719 | [-4.6875, +2.3438] | 10/13 | 0.6776 | 1 |
| regularized_prefill | +0.3906 | [-3.1250, +4.2969] | 12/11 | 1 | 1 |
| shared_low | +0.7812 | [-3.1250, +4.6875] | 14/12 | 0.845 | 1 |
| permuted_low | +6.2500 | [+1.5625, +10.9375] | 26/10 | 0.01133 | 0.1586 |
| permuted | +5.8594 | [+1.5625, +10.5469] | 25/10 | 0.01667 | 0.2168 |
| shared_high | +3.1250 | [-1.1719, +7.4219] | 19/11 | 0.2005 | 1 |
| real_high | +0.7812 | [-3.1250, +4.6875] | 14/12 | 0.845 | 1 |
| prefill | +1.5625 | [-2.3438, +5.8594] | 16/12 | 0.5716 | 1 |
| mean | +3.1250 | [-1.1719, +7.4219] | 21/13 | 0.2295 | 1 |
| scalar | +7.8125 | [+3.1250, +12.8906] | 30/10 | 0.002221 | 0.03332 |
| position_mean | +2.7344 | [-1.1719, +6.6406] | 17/10 | 0.2478 | 1 |
| position_scalar | +36.7188 | [+30.0781, +43.3594] | 101/7 | 1.844e-22 | 2.95e-21 |

## Declared gate

| Requirement | Result |
|---|---|
| Prior ICL screen passes | Pass |
| Audited zero-shot gain at least 3 pp | Pass: +3.125 pp |
| Audited zero-shot interval strictly above zero | Fail: lower bound -1.171875 pp |
| Strictly beats all eight null controls on audited counts | Pass: strongest null, shared_low, scores 222 vs 224 |
| Strictly beats identical-map prompt-only control | Pass: 224 vs 223 |
| Matches or exceeds First, CoT, and old prefill | Fail: First 226 and CoT 227 exceed 224; old prefill scores 220 |
| Primary score at least zero-shot | Pass: 219 vs 210 |
| Candidate truncation at most 5% | Pass: 1/256, or 0.390625% |

## Interpretation and next step

The strongest task-relevant positive comparison is against the matched-penalty shuffled targets: primary +8.203125 pp, Holm p=0.034425. Its audited counterpart is +6.25 pp, but Holm p=0.158634. The candidate is only two audited answers ahead of shared targets at the same penalty and one ahead of prompt-only use of the identical map; both intervals include zero. These results give a limited development signal against one shuffled control, not evidence that question-dependent continuation steering is necessary or better than a textual cue.

The two scalar controls are reliably worse on audited scores after adjustment, and the position-specific scalar truncates 40/256 responses. Beating these weak controls does not resolve the missing benefit over zero-shot, shared targets, prompt-only application, or the text cues. The large primary/audited gap for several controls also makes answer formatting material to interpretation.

The candidate will not advance to any reserved partition, and no alternative condition will be promoted from this run. The next proposed study is a Qwen adaptation of the [released original and complex demonstration banks](gsm8k-published-complex-prompts.md), with matched zero-shot framing. Their prompt lengths fit the existing context budget. New demonstrations require new, separate extraction data and a newly declared map. Evaluation identities and success criteria must be frozen before inference; all current reservations remain reserved for their original experiments. No such follow-up has launched at this report.

Full machine-readable [scores](../results/gsm8k-complex-candidate-v1-results.json) and [selection](../results/gsm8k-complex-candidate-v1-selection.json) accompany this report.

# ICL helps on the fresh complex-training screen

September 13, 2026. Both unchanged ICL banks pass the declared screen on
256 fresh training questions with at least four reference calculation
annotations. Audited completed accuracy is 230/256 for ICL-A and 229/256 for
ICL-B, versus 216/256 zero-shot. The gains are modest, and ICL-B only just
clears the five-point threshold. This qualifies the fixed steering candidate
for development evaluation; it is not a positive steering result.

## Every condition

| Condition | Parser correct | Audited correct | Audited accuracy | Unparsed | Truncated |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 210 | 216 | 84.38% | 10 | 0 |
| ICL-A | 227 | 230 | 89.84% | 6 | 1 |
| ICL-B | 227 | 229 | 89.45% | 4 | 0 |
| First | 221 | 226 | 88.28% | 6 | 0 |
| Step by step | 218 | 227 | 88.67% | 13 | 0 |

All counts are out of 256. Parsed grades remain fixed. Only unparsed
responses received blinded transcription, and truncated answers receive no
completed-answer credit. Both ICL banks also exceed zero-shot under the
explicit parser by 17 answers, or 6.64 percentage points.

## All reported paired comparisons

Gains and intervals below are percentage points on audited accuracy. These
are the declared unadjusted, exploratory 95% paired-bootstrap intervals,
with 10,000 resamples and seed 907. They are not simultaneous intervals.

| Comparison | Gain | 95% interval | Wins | Losses |
|---|---:|---:|---:|---:|
| ICL-A minus Zero-shot | +5.47 | [1.56, 9.38] | 21 | 7 |
| ICL-A minus First | +1.56 | [-1.95, 5.47] | 14 | 10 |
| ICL-A minus Step by step | +1.17 | [-2.34, 4.69] | 11 | 8 |
| ICL-B minus Zero-shot | +5.08 | [0.78, 9.38] | 23 | 10 |
| ICL-B minus First | +1.17 | [-2.73, 5.08] | 15 | 12 |
| ICL-B minus Step by step | +0.78 | [-2.73, 4.30] | 12 | 10 |
| First minus Zero-shot | +3.91 | [-0.78, 8.59] | 24 | 14 |
| Step by step minus Zero-shot | +4.30 | [0.00, 8.59] | 22 | 11 |

Both ICL gains exceed five points and both zero-shot comparison intervals
have strictly positive lower bounds. Every condition truncates below 5%.
All five screen checks pass. The ICL-versus-text-cue intervals include zero,
so these data do not establish superiority over First or step by step.

## Design and verification

The [protocol](gsm8k-complex-screen-protocol.md) fixes the threshold, seed
3401, 256-question screen, new 512-question reservation, unchanged support
banks, model revision, prompts and inference settings. The
[inventory](gsm8k-fresh-pool-audit.md) excludes the union of prior local and
server records, plus both earlier reservations. The reference annotation
count defines the population; no question was selected by model success or
failure. This is a new exploratory follow-up, not a replication of a claim
that this metadata criterion must make ICL help.

All 1,280 generations completed with exit code zero at 08:43:43 UTC.
Physical GPU 0 was used sequentially. The supervisor and worker have exited.
The [sample declaration](../results/gsm8k-complex-screen-v1-declaration.json)
was committed before inference. The 39-item blind packet was frozen in
commit `f975aec`; all 39 responses were individually reviewed and the complete
annotations committed in `1fb9f11` before any accuracy was opened.

Unresolved expressions and separate component quantities without a requested
combined total remain null. Explicitly stated wrong numbers were preserved.
No generated code was executed or missing arithmetic completed. This remains
a single-reviewer transcription audit; it is not a regrade of parsed answers.

The [independent recount](../results/gsm8k-complex-screen-v1-recount.json)
rebuilds the packet, checks exact rational answers and both accuracy counts,
verifies all eight reported paired intervals, and recomputes every gate.
It imports no production scorer. It shares the declared PCG64 draws while
using separate count-weighted resampling and percentile interpolation.

## Next step

The [candidate and gates](gsm8k-complex-candidate-protocol.md) and
[implementation](gsm8k-complex-candidate-implementation.md) were fixed before
these scores were opened. Evaluate the unchanged regularized continuation
map with all 17 declared conditions on these development questions. It must
beat the stronger shuffled map and the same-map prompt-only control, among
other requirements. No map or parameter is selected from this screen.

All reservations remain untouched. A candidate development pass would still
require separately declared independent confirmation. The earlier failed
experiments retain their original outcomes.

Complete machine-readable scores are in the
[screen result](../results/gsm8k-complex-screen-v1-results.json).

# ICL steering

[GSM8K results](research/gsm8k-v2-results.md): full ICL improves audited
accuracy, but the selected mean activation direction falls short.

The [conditional follow-up](research/gsm8k-conditional-results.md) failed
validation: its best question-specific shift scored 90/96, versus 89/96
zero-shot. Actual ICL also failed the required gain on this sample.
The held-out test was not run.

The [fixed development screen](research/gsm8k-test-development-results.md)
replicated the ICL benefit: 115/128 and 116/128 versus 93/128 zero-shot after
review. The [fixed steering candidate](research/gsm8k-fixed-candidate-results.md)
scored 107/128 versus 96/128 for its raw mean, a positive development signal.
It missed the text-cue gate, so confirmation remains stopped. The
[completed controls](research/gsm8k-fixed-controls-results.md) weaken the
case for the full map: a scalar control supplied with ridge's per-question
norms scores 106/128. The raw-mean advantage survives adjustment, but the
controls do not establish a need for ridge's learned directions.

The [final-state LTV run](research/gsm8k-ltv-results.md) failed: applying the
map at every decoding step scores 25/128, with 102 truncated answers.
Prompt-only application scores 113/128 with no truncations, but it was a
control on reused development questions. It needs matched prompt-only
controls and independent confirmation before a positive claim. The declared
candidate remains ineligible; no confirmation questions were generated.
An [early state diagnostic](research/ltv-decoding-state-shift.md) finds that
decoding activations move away from the prompt extraction span; this is
not an accuracy result.
The [identical-prefix check](research/ltv-prefix-alignment-results.md)
confirms poor prediction of the actual ICL shift during decoding. At all
four tested continuation positions, predicting zero shift has lower squared
error. The [prompt-only interpretation](research/ltv-first-token-interpretation.md)
shows that its effect enters later decoding through the first generated token.

The [continuation crossfit](research/ltv-prefix-crossfit-results.md) failed
with penalty five. [Nested regularization](research/ltv-prefix-regularization-results.md)
passes the activation diagnostic: mean continuation error is 0.639 versus
0.716 for the strongest scalar control. The margin is modest and concentrated
early in the answer. This warrants an accuracy experiment, not a positive
steering claim; no reserved answers have been generated.

The [continuation accuracy run](research/gsm8k-continuation-results.md)
scores 115/128 after blinded review, versus 93/128 zero-shot, with no
truncations. It matches ICL-A, but the shuffled-target map also scores
115/128. The candidate therefore fails its control gate. Its four-answer
advantage over the same map at the prompt only is not established as
reliable. Explicit-parser accuracy is 96/128. All 232 new unparsed responses
were individually reviewed and committed before scoring; the independent
recount passed. The 256 reserved questions remain untouched.

The [pairing diagnostic](research/ltv-pairing-results.md) is complete.
At penalty scale 0.1, paired targets score 115/128 versus 105/128 for both
shuffled and shared-average targets. At scale one the scores are 113, 115,
and 110. Neither pairing advantage survives the declared adjustment.
Stronger regularization improves the shuffled control by ten answers.
All new annotations were committed before scoring; full trace, recount and
bootstrap checks passed. No candidate was selected or reserved answer generated.

A [fresh training screen](research/gsm8k-complex-screen-results.md) passes
on 256 questions with at least four reference calculation annotations:
ICL-A scores 230/256 and ICL-B 229/256 after review, versus 216/256 zero-shot.
The [local/server inventory](research/gsm8k-fresh-pool-audit.md) excludes
every recorded prior question and both earlier reservations. Five unchanged
baseline conditions establish a modest ICL gain on this population before
new steering evaluation. The [sample declaration](results/gsm8k-complex-screen-v1-declaration.json)
also fixes a separate 512-question reservation. All 39 annotations were committed
before scoring, and the independent count and interval replay passed.
The conditional [candidate plan](research/gsm8k-complex-candidate-protocol.md)
retains all 17 prior conditions and requires superiority over the stronger
shuffled control and same-map prompt-only control. Its gates are fixed before
the screen's accuracy is opened; no steering stage starts unless the screen passes.
The [conditional implementation](research/gsm8k-complex-candidate-implementation.md)
has passed saved-map reconstruction and synthetic execution tests. The reviewed
screen now qualifies this fixed candidate for development evaluation.

See the [experiment instructions](replication/README.md) and
[literature review](research/icl-task-review.md).

The [related-work audit](research/conditional-steering-related-work.md)
identifies Linear Task Vectors as direct precedent for the question-specific
map and records differences between its paper, released code, and our setup.

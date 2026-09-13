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

See the [experiment instructions](replication/README.md) and
[literature review](research/icl-task-review.md).

The [related-work audit](research/conditional-steering-related-work.md)
identifies Linear Task Vectors as direct precedent for the question-specific
map and records differences between its paper, released code, and our setup.

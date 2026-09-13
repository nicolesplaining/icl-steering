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
It missed the text-cue gate, so confirmation remains stopped. Further controls
are [running on the same development sample](research/gsm8k-fixed-controls-protocol.md)
to interpret the gain.

See the [experiment instructions](replication/README.md) and
[literature review](research/icl-task-review.md).

The [related-work audit](research/conditional-steering-related-work.md)
identifies Linear Task Vectors as direct precedent for the question-specific
map and records differences between its paper, released code, and our setup.

# ICL steering

[GSM8K results](research/gsm8k-v2-results.md): full ICL improves audited
accuracy, but the selected mean activation direction falls short.

The [conditional follow-up](research/gsm8k-conditional-results.md) failed
validation: its best question-specific shift scored 90/96, versus 89/96
zero-shot. Actual ICL also failed the required gain on this sample.
The held-out test was not run.

The [fixed development screen](research/gsm8k-test-development-results.md)
replicated the ICL benefit: 115/128 and 116/128 versus 93/128 zero-shot after
review. A [single fixed steering candidate](research/gsm8k-fixed-candidate-protocol.md)
will now be compared with the matched mean and strong text cues.

See the [experiment instructions](replication/README.md) and
[literature review](research/icl-task-review.md).

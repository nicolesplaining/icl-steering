# ICL steering

[GSM8K results](research/gsm8k-v2-results.md): full ICL improves audited
accuracy, but the selected mean activation direction falls short.

The [conditional follow-up](research/gsm8k-conditional-results.md) failed
validation: its best question-specific shift scored 90/96, versus 89/96
zero-shot. Actual ICL also failed the required gain on this sample.
The held-out test was not run.

A [fixed development screen](research/gsm8k-test-development-protocol.md)
now checks both ICL banks on 128 unused official test questions before more
steering. Its [sample and stopping rule](results/gsm8k-test-development-v1-declaration.json)
were committed before generation. Results are pending.

See the [experiment instructions](replication/README.md) and
[literature review](research/icl-task-review.md).

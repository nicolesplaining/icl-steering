# Published-bank screen implementation

September 13, 2026. The [runner](../analysis/gsm8k_published_screen.py) implements
the fixed three-condition [protocol](gsm8k-published-screen-protocol.md).
Preparation replays the reconciled exclusions, freezes all three partitions
and their prompts, and verifies tokenizer/config bytes against the earlier
context-budget inspection. Screen generation has no extraction or reserved
stage. A committed manifest declaration is required before model loading.

The [backend](../analysis/published_screen_backend.py) searches only generated
text for the declared question markers. Full generated-text decoding preserves
true line starts even with long indentation. Each sequence stops independently;
padding after an earlier boundary stop is excluded from its saved tokens.
Only text before that boundary reaches the unchanged explicit-answer parser.
Old experiments and their parsers are not modified.

Batches are saved atomically and validated against their original question
order, prompt bytes, token decoding, termination, and grades. Resume preserves
completed batches. The final export contains all 1,536 development responses.
Scoring requires a complete blind-review freeze and reports all six declared
contrasts. Only the preselected complex bank can pass the screen; a pass does
not authorize steering or reserved generation.

All nine tests passed on the server CPU in 1.947 seconds, without model
weights or GPU use. They cover new-question boundaries, inline mentions and
long indentation, padding accounting, malformed early termination, disjoint
sampling without a complexity filter, exact prompt construction, wrong-GPU
and missing-declaration guards, a full synthetic 1,536-response run, unchanged
resume, corrupt split rejection, frozen-review scoring, tied-screen failure,
and hand-computed exact-test/Holm checks. The actual Transformers stopping
callback was exercised on two scripted sequences that finish differently.

These tests establish runner behavior on synthetic data. Real preparation,
generation, frozen answer review, and independent score recount remain
necessary for an experimental result.

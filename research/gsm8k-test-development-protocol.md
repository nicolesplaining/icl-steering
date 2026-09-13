# One fixed test-distribution ICL screen

Declared September 13, 2026, after conditional v1 failed validation. That
run remains failed and its 512 reserved questions remain untouched. The
earlier 256-question official-test experiment showed a substantial ICL
benefit; the new 96-question training-split validation did not. This motivates
checking the evaluation distribution before spending more compute on steering.
It does not establish why the two samples differ.

Use the same frozen Qwen2.5-Math-7B revision, two eight-example support banks,
matched prompts, greedy decoding, 1,024-token limit, and stopping rules.
Exclude all questions in the earlier corrected experiment, all conditional-v1
validation and reserved test questions, both support banks, and the first 128
rows of both original source splits. Exclude normalized-text duplicates too.
From the remaining official test questions, shuffle once with Python seed
1701, take 128 development questions, then reserve 256 for a possible later
confirmation. Save sorted identities and rendered prompts before generation.
There are 423 eligible questions before this allocation.

The 128 questions are development data, despite originating in the official
test split. Do not report them as held-out confirmation after using their
results to decide on steering. No examples are selected by zero-shot failure,
ICL success, reference answer, length, or difficulty. Keep the 256 reservation
unused during screening; never draw a replacement sample if this one fails.

Generate five conditions on every development question: zero-shot, ICL-A,
ICL-B, First, and step-by-step. Finish all five before opening any scores.
Export one shuffled, deduplicated packet of all unparsed answers with condition
and reference answer hidden. Transcribe stated answers without solving missing
calculations or executing generated code. Preserve wrong answers; leave absent
or unresolved answers null. Freeze and commit every annotation before scoring.
Parsed grades remain fixed and truncated responses receive no credit.

Both ICL banks must improve audited completed accuracy over zero-shot by at
least five percentage points, and each paired-bootstrap 95% interval must have
a lower bound strictly above zero. All five conditions must truncate at most
5%. Use 10,000 paired resamples with seed 907. Report explicit-parser accuracy,
audited accuracy, truncation, paired wins/losses, and all gate outcomes. These
are exploratory, unadjusted screening intervals, not confirmatory evidence.

If any gate fails, stop this fixed-bank setup without another sample or seed.
Do not change the failed conditional-v1 selection or run its test. If the
screen passes, the next candidate is fixed now: the existing extraction-fitted
ridge map at block 13, strength 0.5, prefill only. Do not run another parameter
grid. Compare it on these development questions with zero-shot, its raw mean,
First, and step-by-step before considering confirmation. It must gain at least
three points over zero-shot, strictly beat the raw mean, match or exceed both
text cues, and truncate at most 5%. Those later generation and confirmation
stages require a separate frozen implementation; this screen has no test or
steering entry point. A passing screen alone does not fulfill the steering goal.

If that fixed candidate passes development, use the reserved 256 only after
locking all sixteen controls listed in the conditional-v1 protocol, including
permuted-query, norm-matched mean, scalar, rotated-pair, reverse, and random
controls. Do not use confirmation results to tune the intervention. Retain all
prior negative results and disclose the sequence of exploratory follow-ups.

Run sequentially on physical GPU 0 only. Leave GPU 1 for the other project.
Raw outputs and activation arrays remain ignored. Nicole Ma is the sole Git
author and committer.

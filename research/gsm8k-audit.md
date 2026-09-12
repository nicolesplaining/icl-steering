# GSM8K experiment audit

September 6, 2026. No positive activation-steering result has been verified.
The earlier statement that we reproduced a strong, clean ICL effect was too
strong. The code audit changes the next experiment.

## Evidence from saved outputs

The saved 128-question zero-shot run was scored as 57/128, or 44.5%.
One rejected response ends with `The final answer is \(\boxed{18}\) dollars.`,
which is the correct answer. The legacy regex captures a trailing backslash,
then fails to convert the capture to a number. Other valid boxes lack its
required answer phrase.

Counting only plain numeric boxes, with the old score for responses without
one, gives 88/128, or 68.8%. The documented parser in
`replication/gsm8k_protocol.py` gives 101/128, or 78.9%, by accepting balanced
numeric boxes and explicit final-answer markers. It stops before a subsequent
generated question and never sees the gold answer while parsing.

These are scores on the saved text, not completed-generation accuracy. The
old run did not retain token counts or finish reasons. The corresponding ICL
generations were also not saved, so subtracting either corrected baseline from
the old 82.8% ICL score would mix grading rules. We cannot recover a fair ICL
contrast from these artifacts. The per-question audit and source hash are in
[`results/gsm8k-parser-audit.json`](../results/gsm8k-parser-audit.json).

## Additional confounds

- Zero-shot ends in `Answer: Let's think step by step.` and requests boxes.
  ICL ends in `A:` with demonstrations of a plain numeric answer.
  The v1 activation difference therefore includes changes to the target token,
  instructions, and answer style.
- The local zero-shot string also contains two literal backslashes before
  `boxed`, unlike the upstream prompt.
- The local wrapper generates several copies of each identical prompt with
  greedy decoding. These are not independent support draws for majority vote.
  The [released script](https://github.com/mlbio-epfl/joint-inference/blob/d45cac5/unsupervised_icl_llm_gsm8k.py)
  samples a new support sequence for each repeat.
- Our unsupervised wrapper makes another evaluation pass after each
  refinement, uses fewer turns, and caps outputs at 512 tokens rather than the
  release's 1,024. Its supervised examples also differ from the release's
  fixed eight Qwen examples. Calling it an exact replication was incorrect.
- The v1 steering sweep tries layers and strengths on the same 32 evaluation
  questions. Any selected best score is a validation result, not a final test.
  It stores no generated text or termination information.
- Centered rank-one variance is not a test of a common mean shift. Identical
  shifts would have zero centered variance. The corrected geometry report
  also measures uncentered energy along the mean and agreement across banks.

## Corrected experiment

`configs/gsm8k_steering.json` defines the next run before inference. It retains
Qwen2.5-Math-7B and the saved train-only support bank. This tests a fixed
eight-example ICL contrast; it does not claim to replicate unsupervised
joint inference.

Use two disjoint banks of eight support solutions, sampled once with seed
777. All zero/ICL query prompts have identical instructions and end in `A:`.
Only the demonstrations differ. There is one greedy generation per condition
and question, with a shared 1,024-token cap. Prompt plus output must fit the
4,096-token context without silent truncation.

The first 128 train examples and first 128 test examples are reserved because
they contributed to earlier work. Select 128 extraction and 64 validation
questions from the remaining train split, then 256 final test questions from
the remaining test split. Check text overlap in addition to index overlap.
Pin the model revision, resolve and record a dataset revision during prepare,
and save data hashes, split indices, support banks, and every prompt.

First check actual ICL. Both support banks must beat zero-shot by at least
five percentage points in validation completed-answer accuracy. Truncation
must be at most 5% in each baseline. Otherwise stop before fitting a direction
or opening final test results. A failed gate needs a new documented experiment,
not a reinterpretation of the old score.

Fit the mean ICL-A minus zero activation difference using extraction questions
only. Validate layers 7, 13, 20, and 27, positive multiples 0.25, 0.5, and 1 of
the fitted shift, and injection either at the last prefill position only or at
that position plus each decoded token. Select maximum completed accuracy;
ties choose smaller strength, shallower layer, then scope alphabetically.
Require a validation improvement of at least three points before final testing.

Freeze selection and direction hashes before opening final test results.
Compare the selected intervention with zero-shot, both actual ICL banks,
`First,` and step-by-step text prefixes, the reversed direction, and three
norm-matched random directions with seeds 31, 59, and 83. Save every prompt,
generated token ID, answer, and finish reason as batches complete. Model
weights stay frozen. Report paired differences and uncertainty, including
negative results. Individual bootstrap intervals are exploratory and
unadjusted; a small favorable comparison is not sufficient to declare success.

If another configuration is tried after seeing these final test results, its
new confirmation set must exclude them. A useful scientific result must
survive these controls rather than merely be the best value in a sweep.

The original H100 became unreachable before its completed v1 statistics could
be retrieved. Those statistics remain unavailable.

On September 13, access to a replacement machine succeeded through the
laptop's configured SOCKS proxy. All 53 regression tests passed there, and
`gsm8k-steering-v2` started with the configuration above. Its source is commit
`929df44`; the saved support artifact has the same SHA-256 recorded in the
parser audit. Code, model cache, and run outputs are on the mounted persistent
volume. Validation is running; no corrected ICL or steering result has yet
been established.

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
volume. Steering validation is running; no positive steering result has yet
been established.

### Initial validation and answer audit

On all 64 validation questions, the prespecified explicit-answer scores are
48/64 for zero-shot, 59/64 for ICL-A, and 58/64 for ICL-B, with no truncated
outputs in these conditions. These gains are partly answer-format effects:
10 zero-shot responses and one response in each ICL bank are unparsed.

Inspection of all 12 unparsed responses recovers eight correct zero-shot
answers and the correct ICL answer in each bank. The rejected responses
include plain concluding sentences without an answer marker, prose inside a
box, and `12 liters per 100 km`, where 100 is part of the unit. The audit
preserves erroneous stated answers and never executes generated code.

The resulting scores are 56/64 for zero-shot, 60/64 for ICL-A, and 59/64 for
ICL-B. These are exploratory corrections to unparsed answers only; parsed
answers retain their original grades. The ICL gains narrow to 6.25 and 4.69
percentage points. They do not establish a robust mathematical advantage.
The frozen primary metric and selection rule remain unchanged. Any steering
claim also needs an answer audit and comparison with the text-prefix controls.
See the per-response hashes, transcribed answers, and paired estimates in
[`results/gsm8k-v2-validation-answer-audit.json`](../results/gsm8k-v2-validation-answer-audit.json).

The same inspection of all unparsed text-prefix responses gives 57/64 for
`First,` and 55/64 for `Let's think step by step.`, compared with 51/64 each
under the primary metric. The step-by-step condition has one truncated
response, which receives no completed-answer credit. These controls further
limit what can be inferred from the small validation ICL gain. The
[supplementary test-audit protocol](gsm8k-answer-audit-protocol.md) was declared
before final test inference; it withholds condition names and reference
answers from the review packet and applies the same rules to all ten final
test conditions. The [timestamped declaration](../results/gsm8k-v2-audit-declaration.json)
records zero test outputs and no test lock at declaration, with the audit code
and protocol hashes.

### Completed selection and validation audit

The completed 24-setting sweep selected block 7, strength 0.5, with injection
at the last prefill position and every decoded token. Its primary completed
score is 55/64 versus 48/64 for zero-shot, with no truncated responses.
Selection and direction hashes were locked before the 256-question test
started. The final test is now running on GPU 0; the three supplementary
prefix controls will run afterward on the same GPU.

The apparent validation gain does not survive answer reading at its original
size. Reviewing all five unparsed selected-condition answers recovers three
correct answers. One response states an incorrect fuel-consumption value of
833, which the audit preserves. Another concludes that a shopper lacks enough
money and states no numeric amount remaining; the audit does not invent one.

| Validation condition | Primary correct | Audited correct |
|---|---:|---:|
| Zero-shot | 48/64 | 56/64 |
| Selected steering | 55/64 | 58/64 |
| ICL-A | 59/64 | 60/64 |
| ICL-B | 58/64 | 59/64 |
| First | 51/64 | 57/64 |
| Step by step | 51/64 | 55/64 |

The audited steering gain over zero-shot is 3.125 percentage points, with
five wins and three losses. Its exploratory paired-bootstrap 95% interval
is [-4.6875, 12.5] points. The advantage over `First,` is one question and
also has an interval spanning zero. These are selected validation estimates,
not independent confirmation. The review reuses 26 baseline annotations and
adds four newly reviewed unique responses; one selected response duplicates
a previously reviewed response. This audit leaves the frozen selection and
test unchanged. The full sweep, all 30 annotations, and paired estimates
are in [`results/gsm8k-v2-selected-validation-audit.json`](../results/gsm8k-v2-selected-validation-audit.json).

Large interventions at deep blocks cause clear failures. At strength 1 with
injection through decoding, blocks 13, 20, and 27 truncate 20, 45, and 42 of
64 responses, respectively. Their completed scores are 30/64, 3/64, and
10/64. These failed settings are retained in the saved sweep. They do not
justify stopping the smaller selected intervention's held-out test.

### Activation geometry

Across the 128 extraction questions, projection onto the mean direction retains
72.2% to 88.4% of uncentered squared activation-difference energy at the four
tested layers. Split-half mean directions have cosine similarity of
0.992 to 0.998; the two disjoint support banks agree at 0.879 to 0.968.
These measurements support a stable common activation shift for these
prompts. They do not show that the shift carries useful mathematical
information: prompt length and output format remain possible explanations.
Causal accuracy tests are still running. The measurements and input hashes
are in [`results/gsm8k-v2-geometry.json`](../results/gsm8k-v2-geometry.json).

An extraction-only control experiment changes the 1,922 demonstration tokens
while preserving the query positions. It exactly reproduces the original
mean directions on a second H100, with zero maximum absolute replay error.
All control prompts, including rotated solution pairings, have the same total
token counts as the real ICL prompts.

| Block index | Rotated pairings, cosine to ICL | Shuffled tokens, cosine to ICL | Repeated filler, cosine to ICL |
|---|---:|---:|---:|
| 7 | 0.978 | 0.036 | -0.026 |
| 13 | 0.969 | 0.343 | 0.124 |
| 20 | 0.817 | 0.331 | 0.237 |
| 27 | 0.632 | 0.384 | 0.470 |

Block indices are zero based. The orientation varies with the prefix's
structure, but a common shift is not specific to worked examples. At block 7,
the mean direction retains 96.2% of the filler difference energy and 87.4%
of the shuffled-token difference energy, compared with 88.4% for real ICL.
Rotating solutions barely changes the early-layer direction. These controls
therefore weaken any interpretation of high shared energy alone as evidence
for useful ICL. They do not settle the causal accuracy question.

The measurements are in
[`results/gsm8k-v2-prefix-geometry.json`](../results/gsm8k-v2-prefix-geometry.json).
The [supplementary test protocol](gsm8k-prefix-test-protocol.md) adds these
three directions at the primary selection's layer, strength, and scope, with
matched norms and no additional tuning. It expands the answer-audit packet
to all thirteen conditions and preserves the original review rubric.

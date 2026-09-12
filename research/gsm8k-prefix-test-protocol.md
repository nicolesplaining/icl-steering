# Fixed prefix-direction test controls

Declared September 13, 2026, after extraction diagnostics and before the
primary GSM8K run's final test. This adds three supplementary conditions to
the existing ten-condition test. Selection still uses the original validation
rule. No prefix-control accuracy is used to choose parameters.

Use the control activations from `gsm8k-prefix-controls-v1`, on the same 128
extraction questions as the real ICL mean. The controls rotate solutions
across the eight support questions, shuffle the demonstration tokens, or
replace those tokens with repetitions of the single token for " the".
All three control prompts have exactly the same token length as their real
ICL counterparts in this run. Shuffling and filler leave the instruction
header and query token IDs at their exact original positions.

When the primary run saves an eligible selection and its test lock, use that
layer, positive strength, and injection scope for every control. For each
control, subtract the zero-shot activation from its activation for each
extraction question, then average. Scale this mean to the norm of the real
ICL mean at the selected layer. Do not tune a separate control strength.

Evaluate `prefix_rotated_pairs`, `prefix_token_shuffle`, and
`prefix_length_filler` on all 256 locked test questions, with the zero-shot
query prompt and the same decoding, stopping, and grading code as the primary
run. Keep outputs in a separate run directory. If primary validation fails,
skip these tests. The supplementary job must wait for the primary test lock
before loading its inference model or generating any test answer.

After both runs finish, combine their saved rows for the answer-review
packet. Apply the existing supplementary answer-audit rubric to all thirteen
conditions, with one shared annotation for identical question/response
pairs. The packet withholds conditions and reference labels. Save all
annotations before scoring. The original audit code and its annotation
rules remain fixed; record the source hashes of both input collections.

In addition to the original comparisons, report real steering versus each
prefix control on both primary completed-answer accuracy and audited
completed-answer accuracy. Use the existing paired bootstrap implementation,
10,000 resamples, and seed 907. These additional comparisons are exploratory
and unadjusted. They cannot be used to retune the intervention on these test
questions.

The shuffled and filler contexts are unnatural controls. Beating them alone
does not establish useful ICL. Rotating solutions changes both pairing and
recency, so a difference from that control is not a pure test of reasoning.
The original text-prefix controls and the answer audit remain necessary for
interpreting a mathematical improvement.

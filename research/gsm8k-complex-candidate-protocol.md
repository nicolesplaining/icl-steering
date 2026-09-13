# Fixed continuation candidate on fresh complex questions

Declared September 13, 2026, while the five-condition ICL screen is still
running and before its accuracy or any new steering answers are inspected.
This plan is conditional on every gate in the
[screen protocol](gsm8k-complex-screen-protocol.md) passing after frozen
blinded review and independent recount. Failure of that screen stops this
follow-up without steering generation or a replacement sample.

## Fixed method and questions

Use the screen's exact 256 development questions, five baseline output sets,
two support banks, frozen Qwen2.5-Math-7B revision, prompts, greedy decoding,
1,024-token budget, 4,096-token context limit, and batches of four. Inherit
the reviewed baseline outputs and their annotations exactly. Preserve the
screen's new 512-question reservation and both earlier reservations.

The sole candidate remains `steered`: the correctly paired continuation map
at penalty scale 0.1, strength one, immediately after final normalization,
active at prefix lengths zero through 128 inclusive. Its extraction inputs
are the original 620 states from 128 training questions. Do not refit, add
extraction data, change penalties, tune strength, or alter support examples.

Reuse the original maps byte for byte from these archives:

- `gsm8k-pairing-v1/maps.npz`, SHA-256
  `ee9c4fa68737abe5671c54fa740f41fc6a2f8a0aec1e5c7668b4b641593a9137`.
- `gsm8k-continuation-v1/maps.npz`, SHA-256
  `512875b4246793687eb5227d7c774bb26220ab3ac7cfff33f7971b46bd3633d7`.
- `gsm8k-ltv-v1/maps.npz`, SHA-256
  `85a10858a81a53b94bb814e8a4705f38d34f4a036b4b1fa0303784e0a41342b3`.

The pairing archive supplies the six target/penalty combinations. The
continuation archive supplies its original pooled and position-specific
controls and the identical real map for prompt-only application. The old LTV
archive supplies only its original penalty-five prompt-only control.

## Conditions and execution

Retain all 17 conditions from the completed pairing diagnostic. The five
screen baselines are `zero`, `icl_a`, `icl_b`, `first`, and `cot`. Generate
12 new conditions, in this fixed order:

1. `steered`: correctly paired targets, penalty scale 0.1.
2. `regularized_prefill`: the identical map at the prompt only.
3. `shared_low`: position-average targets, scale 0.1.
4. `permuted_low`: the original within-position permutation, scale 0.1.
5. `permuted`: that same permutation, scale one.
6. `shared_high`: position-average targets, scale one.
7. `real_high`: correctly paired targets, scale one.
8. `prefill`: the original prompt-fitted penalty-five map, prompt only.
9. `mean`: the original pooled mean shift.
10. `scalar`: the original pooled affine scalar.
11. `position_mean`: the original interpolated position-specific mean.
12. `position_scalar`: the original interpolated position-specific scalar.

Except for the two prompt-only controls, apply interventions for the first
129 generated tokens and then return the unchanged normalized state. Retain
the previously audited definitions, prefix interpolation, NumPy float64
prediction, BF16 cast/add sequence, and final-normalization hook boundary.
Weights and cached keys and values remain unchanged. Do not promote any
control, including `real_high`, after seeing its results.

There are 3,072 new outputs and 4,352 total development outputs. Preserve
the original batches on resume. Save states, predictions, applied vectors,
prefix positions, and actual LM-head checks for independent replay. Require
exact agreement between the candidate and same-map prompt-only condition at
their initial prompt states, shifts, and first generated tokens. Require a
complete runtime replay before scoring. A new frozen implementation and
input declaration are still required before this generation stage can run.

## Review, reporting, and development gate

Finish every condition before opening aggregate scores. Export and freeze
one shuffled, deduplicated blind packet. Preserve inherited annotations by
response hash. Individually transcribe new unparsed responses without
solving missing calculations, correcting wrong answers, or executing code.
Freeze and commit every annotation before scoring. Parsed grades stay fixed
and truncated responses receive no completed-answer credit.

Report all condition counts, unparsed and truncated counts, and the 16
candidate-minus-comparator contrasts for both the explicit-parser and audited
metrics. Use 10,000 paired bootstrap draws with seed 907 and exact two-sided
McNemar tests. Adjust p-values with Holm across 16 contrasts separately for
each metric. The intervals are unadjusted and the evaluation is development,
not independent confirmation of a selected method.

The candidate passes development only when every condition below holds:

- The reviewed ICL screen passed.
- Audited accuracy improves over zero-shot by at least three percentage
  points, with a paired 95% interval whose lower bound is strictly positive.
- Audited accuracy strictly exceeds all eight null controls: `mean`,
  `scalar`, `position_mean`, `position_scalar`, `permuted`, `shared_low`,
  `permuted_low`, and `shared_high`.
- Audited accuracy strictly exceeds `regularized_prefill`.
- Audited accuracy matches or exceeds `first`, `cot`, and the old `prefill`.
- Explicit-parser completed accuracy is at least zero-shot accuracy.
- The candidate truncates at most 5% of its outputs.

These gates preserve the earlier continuation requirements and add the
matched-penalty controls, a positive zero-shot interval, and a parser-direction
check. They do not reopen or change the earlier failed gate. A tie with the
stronger shuffled map still fails. Pointwise development superiority alone
does not establish a reliable pairing effect.

If a gate fails, report the failure and leave all reservations untouched.
If all gates pass, separately implement and declare confirmation on all
512 newly reserved questions. Retain the 17 conditions and add a reversed
map, norm-matched mean, and three fixed random-direction controls. Freeze
their definitions, all confirmation contrasts and statistical criteria
before any reserved generation. Confirmation must establish an audited gain
over zero-shot, the mean/scalar/shared/shuffled controls, and the same-map
prompt-only control. A screen pass, a development win, or a helpful control
does not complete the steering goal.

Run sequentially on physical GPU 0 only. Keep raw outputs, activations and
maps ignored, and leave GPU 1 for the other project. Nicole Ma is the sole
Git author and committer.

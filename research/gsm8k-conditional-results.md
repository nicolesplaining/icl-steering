# Conditional GSM8K steering failed validation

September 13, 2026. The best query-dependent intervention scored 90/96 after
answer review, versus 89/96 for zero-shot and 88/96 for its matched mean.
Actual ICL scored 89/96 and 87/96. Neither the ICL prerequisite nor the
minimum steering gain passed. The declared 512-question test was not run.

## Complete validation results

All 22 conditions use the same 96 fresh questions from the official training
split. The best ridge candidate was selected by audited completed accuracy,
with the tie rule declared before generation. Counts below are out of 96.

| Condition | Explicit parser | Audited | Unparsed | Truncated |
|---|---:|---:|---:|---:|
| zero | 81 | 89 | 9 | 0 |
| icl_a | 88 | 89 | 2 | 0 |
| icl_b | 87 | 87 | 1 | 1 |
| first | 81 | 88 | 8 | 0 |
| cot | 84 | 87 | 6 | 0 |
| legacy_mean | 82 | 87 | 7 | 2 |
| ridge_l7_a0.5_prefill | 82 | 89 | 9 | 1 |
| mean_l7_a0.5_prefill | 80 | 87 | 9 | 1 |
| ridge_l7_a0.5_all | 83 | 87 | 6 | 2 |
| mean_l7_a0.5_all | 82 | 87 | 7 | 2 |
| ridge_l7_a1_prefill | 82 | 88 | 8 | 1 |
| mean_l7_a1_prefill | 83 | 87 | 5 | 1 |
| ridge_l7_a1_all | 77 | 82 | 6 | 0 |
| mean_l7_a1_all | 80 | 85 | 7 | 0 |
| ridge_l13_a0.5_prefill | 85 | 90 | 5 | 0 |
| mean_l13_a0.5_prefill | 84 | 88 | 5 | 1 |
| ridge_l13_a0.5_all | 80 | 85 | 6 | 0 |
| mean_l13_a0.5_all | 83 | 87 | 5 | 1 |
| ridge_l13_a1_prefill | 82 | 89 | 7 | 0 |
| mean_l13_a1_prefill | 84 | 88 | 5 | 0 |
| ridge_l13_a1_all | 34 | 35 | 52 | 50 |
| mean_l13_a1_all | 52 | 53 | 30 | 31 |

The selected setting is block 13, strength 0.5, final prompt token only.
Its audited gain over zero-shot is one question, or 1.04 percentage points,
with three paired wins and two losses. The exploratory 95% paired-bootstrap
interval is approximately [-3.13, 5.23] points. This is a selected validation
estimate, not an independent confirmation.

## Gate outcomes

- ICL-A must gain at least 5 points over zero-shot: failed, 0 points.
- ICL-B must gain at least 5 points: failed, -2.08 points.
- Selected ridge must gain at least 3 points: failed, +1.04 points.
- Selected ridge must strictly beat its same-setting mean: passed, 90 versus 88.
- Selected ridge must match both text cues and the old mean: passed, versus 88, 87, and 87.
- Zero-shot, both ICL banks, and the selected ridge must truncate at most 5%: passed.

The frozen selection implementation returned `eligible: false`. A separate
CPU check verified that the test entry point rejects this selection before
loading a model. No alternative candidate was substituted.

## What failed and what remains uncertain

Full-strength block-13 steering on every token causes severe repetition:
50/96 ridge responses and 31/96 mean responses reach the length limit.
The corresponding prefill-only settings have no truncations. This points
to a scope-dependent generation failure. Exact vector hashes and norms
were replayed for every steering row, and the frozen inputs passed their
integrity checks; there is no evidence here of a vector-routing error.

The new zero-shot audited score is 92.71%, compared with 76.95% on the
earlier 256-question official test sample. Demonstrations, model revision,
query template, and decoding settings are unchanged. The samples come
from different source splits, so their scores do not identify the cause of
the difference. This run does establish that the hoped-for ICL gain is absent
on this validation sample. It does not establish a general absence of ICL
benefits, nor a useful conditional steering effect.

The next screening decision should check ICL before running another steering
grid. Sampling another set until a gain appears would overstate evidence; any
new sample and stopping rule must be declared and every screening result
reported. The unused confirmation questions from this failed run remain
reserved and were not used to rescue its selection.

## Review and provenance

The [protocol](gsm8k-conditional-protocol.md) and
[declaration](../results/gsm8k-conditional-v1-declaration.json) preceded new
evaluation. The shared blinded packet contained 157 distinct unparsed
responses across 2,112 generations. All annotations were saved and pushed
in `53fe890` before scores or condition mappings were inspected. Wrong
stated numbers were preserved, absent or unresolved answers left null, and
generated code was never executed. Parsed grades remain fixed and truncated
responses receive no credit. This single-reviewer procedure does not audit
the semantics of already-parsed solutions.

Unlike the earlier v2 supplementary audit, this experiment explicitly uses
audited accuracy for selection. The inherited audit helper contains an old
supplementary-only description; the new protocol governs this run. See the
[complete metrics and selection](../results/gsm8k-conditional-v1-validation.json),
[annotations](../results/gsm8k-conditional-v1-validation-annotations.json), and
[annotation freeze](../results/gsm8k-conditional-v1-validation-annotation-lock.json).
Raw generations, fitted maps, and query activations remain in ignored run
directories on the laptop and node as applicable. No model weights changed.

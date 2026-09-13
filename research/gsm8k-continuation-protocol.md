# Test regularized continuation steering on math answers

September 13, 2026. The [nested activation crossfit](ltv-prefix-regularization-results.md)
passed its diagnostic gate, with modest gains concentrated early in the
answer. This is a new development experiment, not confirmation of the failed
all-token prompt-fitted map. No new answer scores have been inspected.

Use the same frozen Qwen2.5-Math-7B revision, GSM8K revision, two eight-example
banks, prompt strings, batch size four, greedy decoding, 1,024-token answer
budget, and 4,096-token context limit as `gsm8k-ltv-v1`. Reuse its 128
development questions and the exact zero-shot, ICL-A, ICL-B, First,
step-by-step, and original prompt-only output rows. Keep the 256 reserved
questions untouched. These development questions have informed prior work.

Fit on the saved 620 available paired states from the original 128 training
extraction questions at prefix lengths 0, 1, 8, 32, and 128. Each available
state has equal weight. Use uncentered ridge with penalty 0.1 times the mean
squared zero-shot state norm. All four nested folds independently selected
that rule. Fit the shuffled-target ridge with scale one, the rule selected
in all four shuffled-control folds. Shuffle targets within position using
NumPy seed 2390. Do not choose a new rule from development answers.

Use NumPy float64, reject solve residuals above 1e-8, and freeze the fitted
arrays before generation. Also fit a pooled mean and affine scalar, and a
separate mean and affine scalar at each measured position. A scalar fit is
mean(D) + c times h minus mean(X), with c from centered least squares. For
intermediate integer prefix lengths, linearly interpolate position-specific
mean vectors or affine bias vectors and scalar coefficients between adjacent
measured positions. There is no norm matching or strength selection.

The primary adds the regularized prediction with strength one immediately
after final normalization and before the LM head, for prefix lengths zero
through 128 inclusive. Thus it may modify the first 129 generated tokens.
After that, the hook returns the original normalized state unchanged. The
full answer budget remains 1,024 tokens. This bounded scope avoids extending
the intervention beyond the measured prefix range. It does not establish
that intervened prefixes match the zero-shot prefixes used for training.

Generate seven new conditions, in this order:

- `steered`: the regularized full map for prefix lengths 0–128.
- `regularized_prefill`: the identical full map only at prefix length zero.
- `mean`: the pooled mean for prefix lengths 0–128.
- `scalar`: the pooled affine scalar for prefix lengths 0–128.
- `position_mean`: the interpolated position-specific mean for that range.
- `position_scalar`: the interpolated position-specific affine scalar.
- `permuted`: the independently regularized shuffled-target map for that range.

Together with the six inherited conditions, there are 13 conditions and
1,664 development rows. The original prompt-only condition retains its name
`prefill` and its original penalty-five map. It is distinct from the new
`regularized_prefill` control. Do not promote a control to primary afterward.

Save unmodified states, predicted shifts, applied states, prefix indices,
and actual LM-head checks for independent replay. Preserve original groups
of four and immutable batch records on resume. Reject nonfinite predictions,
changed fits, missing calls, or interventions after the declared cutoff.
No model weights or cached keys and values are modified by the hook.

After all conditions finish, export one shuffled blind review packet.
Preserve all inherited annotations by response hash. Transcribe only stated
answers without correcting arithmetic or executing generated code. Commit
complete annotations before scoring. Parsed grades remain fixed and truncated
responses receive no credit. Report every condition, parse count, truncation
count, and all 12 primary contrasts for both metrics. Use 10,000 paired
bootstrap draws with seed 907 and exact McNemar tests with Holm adjustment
across 12 contrasts separately for each metric.

The sole candidate passes development only if the inherited ICL screen
passes, audited accuracy gains at least three percentage points over zero,
strictly exceeds all five mean/scalar/shuffled controls, strictly exceeds
`regularized_prefill`, matches or exceeds First, step-by-step, and the old
`prefill`, and truncates at most 5% of responses. The strict comparison to
the same map's prompt-only output tests whether the continuation component
adds anything beyond first-token selection. Freeze this gate before fitting.

If any gate fails, publish the failure and do not generate reserved answers.
If it passes, separately declare and implement confirmation on all 256
reserved questions, retaining these 13 conditions and adding a reversed
map, a norm-matched mean, and three fixed random-direction controls. Positive
confirmation must establish an audited gain over zero, mean/scalar controls,
and the same map's prompt-only condition. Development or activation metrics
alone cannot complete the research goal. All confirmation contrasts and
limitations must be reported even if unfavorable.

Run sequentially on physical GPU 0 only. Leave GPU 1 for the other project.
Nicole Ma is the sole Git author and committer. Do not commit AGENTS.md,
credentials, raw generations, model weights, fitted arrays, or activations.

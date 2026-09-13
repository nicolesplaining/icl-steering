# Check the target shift on identical continuation prefixes

September 13, 2026. This follow-up is specified while the fixed LTV
validation is still running, before opening its scores. It follows the
[observed change in decoding states](ltv-decoding-state-shift.md).
No follow-up GPU job has started. Run this diagnostic only if the current
candidate fails its declared gates, after completing its review and report.
It does not change those gates or authorize another validation candidate.

The existing span diagnostic cannot tell whether the predicted shift points
toward an actual ICL state. Measure that target directly on the original
128 training extraction questions. Reuse the frozen model revision, bank A,
prompt strings, and fitted maps from `gsm8k-ltv-v1`. Do not use development
or reserved questions, reference answers, or correctness to choose prefixes.

For each extraction question, generate one unsteered zero-shot continuation
with greedy decoding, a maximum of 128 new tokens, and the model's declared
EOS tokens. Preserve original groups of four. Keep raw token IDs before the
first EOS, without decoding and retokenizing them. At prefix lengths
0, 1, 8, 32, and 128, append the identical prefix IDs to both the zero-shot
and ICL-A tokenized prompts. Exclude a question at a position if it has
fewer non-EOS continuation tokens. Report the resulting count at every
position. Preserve all four rows during forward passes; exclude unavailable
positions only from the measurements.

Capture the last token's final normalized state in two unmodified forward
passes, with attention-mask-derived position IDs and no intervention.
Define the measured target as ICL state minus zero-shot state. Evaluate the
existing frozen full map, raw mean, standalone scalar, norm-matched scalar,
and norm-matched permuted map on that same zero-shot state. Do not refit,
change strength, or select a position from the resulting measurements.

For every fixed position and predictor, report target and predicted norms,
their ratio, cosine with the target, and summed squared prediction error
divided by summed squared target norm. Report medians and 10th/90th
percentiles for per-question quantities. Report zero target or prediction
norms explicitly and exclude undefined ratios or cosines with their counts;
reject nonfinite states. Also report changes from the prompt-final position
on the same retained questions, so early endings do not change the cohort
used for that comparison.

Before interpreting results, verify that the zero-prefix captures reproduce
the original extraction arrays under the same batching and runtime. An
unexplained mismatch is a capture failure to investigate. Verify prefix IDs
are identical across paired contexts, fit and input hashes are unchanged,
and the question IDs belong only to the original extraction split. Test
these checks with a small model before using the H100. Save a code and input
declaration before collecting data, then independently replay the reported
prediction errors from the saved arrays.

The prompt measurements are in-sample for the existing fit. Continuation
states are new positions on the same extraction questions, not an independent
test of question generalization. Prefixes come from the zero-shot model;
they need not be correct or typical of ICL generation. This measures an
activation relationship under fixed prefixes, not accuracy or the causal
benefit of steering. Even good alignment would still require a separately
declared intervention experiment and held-out confirmation.

Run sequentially on physical GPU 0 after it is free. Keep token IDs and
activation arrays ignored. Publish aggregate measurements and failures with
Nicole Ma as the sole commit author and committer.

The [numerical helpers](../analysis/ltv_prefix_metrics.py) preserve paired
token IDs and compute the declared aggregate metrics. The
[collector](../analysis/ltv_prefix_collect.py) requires a completed, reviewed
parent failure and a declaration before loading the model. Eight CPU tests
pass using synthetic arrays and a small random Qwen2 model. They cover EOS,
early endings and full 128-token prefixes, original-state reproduction,
left-padding positions, saved-array tampering, context limits, undefined
quantities, and comparisons on the same retained questions.

Independent replay and report assembly still need implementation before
collecting real-model data. No diagnostic data has been collected.

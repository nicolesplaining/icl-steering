# Decode with a final-state linear ICL map

Declared after the internal-block controls: ridge 107/128, norm-matched
scalar 106/128, raw mean 96/128, and step-by-step 110/128. That candidate
remains ineligible. This is a different intervention informed by
[Linear Task Vectors](https://arxiv.org/html/2605.20730v1), not an independent
confirmation or a new claim of method novelty.

Use the same frozen Qwen2.5-Math-7B model, dataset revisions, prompts, two
eight-example banks, greedy decoding, batch size four, 1,024-token answer
budget, and 4,096-token context limit as the fixed development screen.
Retain its 128 development questions and five completed baseline output
sets exactly. These questions have informed earlier method choices.
Retain all prior annotations by response hash wherever they recur.

Extract final normalized last-token states on the original 128 training
extraction questions, with and without bank A. No extraction question is a
development or reserved question. Fit only prompt-final states, not answer
tokens. Let X contain zero-shot row states and D the corresponding
ICL-minus-zero-shot differences. With lambda=5, fit the uncentered ridge
map in dual form: A=(XX^T+5I)^(-1)D and delta(h)=h X^T A. Use NumPy
float64 and reject a relative solve residual above 1e-8. The query's gold
answer never enters fitting. Freeze extraction arrays and fitted maps before
development generation.

At inference, add delta(h) with strength one to the last token's final
normalized state, immediately before the language-model head. Recompute it
at every generation step from that step's unmodified normalized state.
Final-layer cached keys and values are not modified. This follows the
paper's stated decoding extension at a matched extraction/injection site;
it differs from its released prefill-only decoder-block hook. It is also a
math adaptation, using 128 extraction queries and eight demonstrations.
Prompt-only fitting may extrapolate poorly to generated reasoning states.

Evaluate one fixed candidate and five controls, in this order:

- `steered`: the full map at every step.
- `mean`: the raw extraction mean difference at every step.
- `scalar`: a standalone affine scalar predictor, mean(D) plus c times
  h-mean(X), where c minimizes centered extraction squared error. Its own
  magnitude is used; it does not compute or borrow a ridge norm.
- `scalar_norm`: that scalar direction, matched at each step to the real
  map's norm evaluated on the same current state.
- `permuted`: the same ridge fitting rule after permuting extraction D rows
  once with NumPy seed 1901, matched to the real map's norm at each step.
- `prefill`: the real map applied only at the final prompt token.

This gives eleven conditions including the five inherited baselines, or
1,408 development rows. No other strength, layer, penalty, or scope is
eligible for selection. Use original groups of four questions even on
resume. Save each completed batch atomically. Save normalized input states
and proposed shifts for independent trajectory replay. Check that the LM
head actually receives the modified states and save those states too. Keep these arrays
and all raw generations ignored. No clipping or adaptive strength reduction
is allowed after seeing outputs. Nonfinite shifts are an implementation
failure, not grounds to silently skip steering.

Finish all conditions before exporting one shuffled, deduplicated blind
review packet. Preserve inherited annotations. Transcribe only stated
answers on new unparsed responses, without correcting calculations or
executing code. Commit the complete annotations before opening scores.
Parsed grades remain fixed and truncated responses receive no credit.

The only candidate is `steered`. To pass development, it must gain at least
three percentage points over zero-shot, strictly exceed mean, scalar,
scalar_norm, and permuted audited accuracy, match or exceed prefill and
both text cues, and have at most five percent truncated responses. The
inherited ICL-A and ICL-B gains must still satisfy their original screen
gates. Report all conditions and all ten candidate contrasts, using
10,000 paired bootstrap draws with seed 907 and exact McNemar tests with
Holm adjustment across the ten contrasts separately for each metric.

If any gate fails, stop this candidate and leave the 256 reserved questions
untouched. Do not rename a control as the candidate or relax the rule.
The development runner has no confirmation entry point. A passing candidate
would require a separate frozen confirmation runner and declaration before
generating any reserved output, with the same candidate and all eleven
conditions, plus reversed, mean-norm-matched, and three random directions.
Confirmation must independently establish a positive audited gain over
zero-shot and the mean and scalar controls; development success alone cannot
complete the research goal.

Run sequentially on physical GPU 0 only, leaving GPU 1 for the other project.
Nicole Ma is the sole Git author and committer. Do not commit AGENTS.md,
credentials, weights, activation arrays, or raw model generations.

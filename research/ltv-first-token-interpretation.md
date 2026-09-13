# Prompt-only final-state steering chooses the first token

September 13, 2026. The prompt-only control in the
[completed LTV run](gsm8k-ltv-results.md) scored 113/128. Its intervention
operates after the decoder has built the prompt's keys and values. The
hook clones the final normalized output and changes the last state passed
to the language-model head. It leaves the prompt cache unchanged and skips
every later call.

Consequently, its only route into subsequent greedy decoding is the first
generated token. Given the same first token, prompt cache, and deterministic
runtime, subsequent computation is the ordinary zero-shot computation.
This differs from an internal-block prompt intervention, which can change
downstream keys and values that later tokens attend to.

A [descriptive check](../results/gsm8k-ltv-v1-first-token.json), performed
after the complete score report, finds that prompt-only steering selects
just three distinct first-token IDs across the 128 questions. The resulting
first words are First, on 70 questions, Let's on 50, Let on two, and To on six.
Let's and Let share the first token. Seven questions have the same first
token as zero-shot, and all seven have exactly identical complete generated
token sequences. This uses saved token IDs; it is not a new GPU replay.

The 113/128 score therefore does not demonstrate that a hidden reasoning
state persists after prompt-only injection. It suggests a useful rule for
choosing an opening token from ICL-derived activations. Whether that rule
beats simpler choices remains untested. The existing First cue includes
punctuation in the prompt and is not a controlled replacement of exactly
one generated token.

Any follow-up should compare fixed opening tokens and prompt-only mean,
scalar, and shuffled-target maps. A forced-token replay can check that the
selected first token reproduces the continuation. Such a replay would be a
mechanism check, not evidence that the fitted map is necessary. New matched
controls on the reused development questions and a separately declared
held-out test are still needed before a positive steering claim.

The all-token candidate remains failed. This observation does not change
its gates, and the separate identical-prefix diagnostic proceeds as declared.

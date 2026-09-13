# Published demonstration banks on fresh GSM8K questions

September 13, 2026. Declare this screen after the previous fixed continuation
candidate failed, before selecting these questions or generating any outputs.
The purpose is to establish an ICL benefit using the actual released prompt
banks before another activation-steering study. A screen pass alone does not
complete the steering goal or authorize reserved generation.

## Paper adaptation and fixed prompts

Use the eight original and eight complex demonstrations released by Fu et al.
in Complexity-Based Prompting at upstream commit
`378f7a88fb4c6a3fdb294f0dcf8a702420f76724`. Their reported comparison is complex
versus original demonstrations on older OpenAI models. Our added zero-shot
control tests whether ICL helps Qwen2.5-Math. Using unused training questions
also differs from their benchmark test evaluation. This is a Qwen adaptation
of their prompt assets, not reproduction of their historical scores.
See the [source inspection](gsm8k-published-complex-prompts.md).

Keep both files byte for byte:

- Original: SHA-256 `db8c84b10707a16cc2f61e192e1e55452393510470a3d5790db86188a42ae4e0`.
- Complex: SHA-256 `d20a8061b484da2d93d10152727c5c137f7bec76f5f4f9e1c94f17cf33051aee`.

Use this exact query template, including the final newline:

```text
Question: {question}
Let's think step by step
```

The three conditions, generated in order, are `zero`, `icl_original`, and
`icl_complex`. Zero uses only the query. The other two prepend the unchanged
published file followed by one additional newline. Do not add a chat template,
system message, special tokens, answer-format instruction, or other cue.
`icl_complex` is the sole prespecified target for any later steering study.
Do not promote the original bank if the complex bank fails this screen.

## Data and execution

Use the [reconciled inventory](gsm8k-published-inventory.md), which contains
6,282 unused training questions after recorded laptop/server uses, both
published banks, the first 128 rows of each split, cross-split normalized
duplicates, and all three earlier reservations are excluded. Replay the
exclusions against the pinned dataset before preparation. Reject insufficient
data or changed inputs; do not relax exclusions or replace selected questions.

Use all eligible training questions without a solution-length or difficulty
filter. Start in dataset-index order, shuffle once with Python's
`random.Random(3402)`, assign the first 256 to extraction, the next 512 to
screen development, and the next 512 to a new reservation. Sort each selected
partition by numeric dataset index for execution. All partitions and prompt
texts must be frozen and committed through a hashed declaration before any
screen generation. Extraction and reservation are selected but receive no
model inference in this screen. The three earlier reservations stay reserved
for their original experiments.

The [configuration](../configs/gsm8k_published_screen.json) fixes
Qwen2.5-Math-7B revision `b101308fe89651ea5ce025f25317fea6fc07e96e` and GSM8K
revision `740312add88f781978c0658806c59bc2815b9866`. Use frozen BF16 weights,
greedy decoding, batches of four, a 4,096-token total context and up to 1,024
new tokens. Verify prompt lengths with the pinned tokenizer. No prompt or
demonstration truncation is permitted. There are 1,536 screen outputs.

Stop each sequence independently at EOS or the beginning of a subsequent
question. The new runner must recognize case-insensitive line-start markers
`Question:`, `Q:`, `Problem:`, `Human:`, `User:`, and `Given the following
question,` or `Given the following problem,`, also allowing a space in place
of the final comma and horizontal whitespace before a marker. A marker at
the start of generated text also ends the response. Apply this boundary only
to generated tokens, never to the demonstration prefix. Save raw generated
tokens and text, the cut query response, and the stop reason. Grade only the
query response. Reaching the token limit without EOS or a question boundary
counts as truncation and receives no completed-answer credit.

Implement and test the boundary handling separately from previous runners;
do not change prior frozen parsing or results. Preserve original batches on
resume. Verify complete condition/question identities, prompts, token counts,
stop reasons, and grades before exporting the review packet. Run sequentially
on physical GPU 0 only, leaving GPU 1 for the other project.

## Review, comparisons, and gate

Complete all three conditions before opening aggregate scores. Freeze and
commit a shuffled, deduplicated packet of unparsed responses, blinded to
condition and gold answer. Individually transcribe stated numbers without
solving missing calculations, correcting explicit answers, or executing code.
Commit the complete annotations and hashes before scoring. Primary grades use
the existing explicit-answer parser on the cut query response; audited grades
add only the frozen review of unparsed responses. Report both, along with
unparsed and truncated counts.

Report three fixed contrasts for each metric: complex minus zero, original
minus zero, and complex minus original. Use 10,000 paired bootstrap draws with
seed 907 and exact two-sided McNemar tests. Adjust the three p-values with Holm
separately for each metric. Intervals are unadjusted. Require an independent
recount of identities, counts, comparisons, and every gate before interpreting
the screen.

The screen passes only if all of these hold:

- All three conditions contain exactly the declared 512 development questions.
- Audited complex-minus-zero gain is at least five percentage points.
- That gain's paired 95% interval has a strictly positive lower bound.
- Its audited McNemar p-value after the declared three-comparison Holm
  adjustment is strictly below 0.05.
- Complex primary completed accuracy is at least zero-shot primary accuracy.
- Every condition's truncation rate is at most 5%.

Complex-versus-original measures the paper's demonstration-complexity claim,
but is not a gate for the separate question of whether ICL helps at all.
Report a null or negative complexity comparison plainly; do not call it a
replicated complexity benefit. The original bank never becomes a fallback
target based on its score.

If the screen fails, report it and do not generate extraction or reserved
answers from this protocol. If it passes, separately declare and implement
an activation-steering study using the new extraction partition and the fixed
complex bank. Do not reuse the earlier maps as if they were fitted to these
demonstrations. Freeze intervention definitions, tuning rules, controls, and
success criteria before any steering accuracy is observed. Confirmation still
requires its own declaration and untouched reserved questions.

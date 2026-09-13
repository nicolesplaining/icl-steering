# A fixed ICL screen on longer reference calculations

September 13, 2026. The earlier random training sample did not benefit from
the existing ICL banks. The later official-test development sample did, but
has now informed several unsuccessful steering methods. This screen asks
whether those unchanged banks help on a fresh training population defined
before generation. It does not replace either failed experiment or release
either of their reservations.

Use the union of the separately verified local and server inventories in
`results/gsm8k-fresh-inventory-union-v1.json`. Require all input hashes and
the complete pool to match. There are 2,519 eligible training questions
with at least four complete `<<...>>` reference calculation annotations.
This annotation count is a metadata proxy, not demonstrated model difficulty.
The choice of threshold follows the published inventory but precedes all
model outcomes on this population. This is exploratory follow-up design.

In source-index order, retain exactly the questions meeting that threshold.
Shuffle once with Python seed 3401. Take 256 for the screen and the next 512
for a new, separate reservation. Save each allocation in source-index order.
Never replace questions, change the threshold or seed, or draw another screen
because of poor outcomes. The old 512 and 256 reservations stay excluded.
The remaining official-test questions are not used. No fresh answers are
used for extraction or fitting.

Use frozen Qwen2.5-Math-7B revision
`b101308fe89651ea5ce025f25317fea6fc07e96e` and the same two eight-example
support banks, matched prompts, greedy decoding, 1,024-token answer budget,
4,096-token context budget, batch size four, and stopping and parsing rules.
Refuse prompt overflow rather than truncating or replacing a question.
Prepare all prompts before inference. Run five conditions on every screen
question: zero-shot, ICL-A, ICL-B, First, and step by step.

Complete all 1,280 outputs before opening any aggregate accuracy. Shuffle
and deduplicate all unparsed responses into a condition-blind review packet.
Freeze the packet before review. Transcribe only stated numerical answers;
do not finish calculations, solve questions, execute generated code, or
substitute reference answers. Preserve wrong answers and leave missing or
unresolved answers null. Commit complete annotations before scoring. Parsed
grades stay fixed, and truncated responses receive no completed-answer credit.

Both ICL banks must improve audited completed accuracy over zero-shot by at
least five percentage points. Each paired 95% bootstrap interval must have
a lower bound strictly above zero. Every condition must truncate at most 5%.
Use 10,000 paired resamples with seed 907. Report explicit-parser accuracy,
audited accuracy, unparsed counts, truncation, paired wins and losses, and
each gate outcome. These are unadjusted exploratory screening intervals,
not confirmation of steering. A failed gate stops this declared screen.

If the screen passes, the only next steering candidate is the existing
correctly paired continuation map at penalty scale 0.1, strength one, after
final normalization, active for the first 129 generated tokens. Keep its
original 620 extraction states and weights fixed. Its parent pairing archive
is `runs/gsm8k-pairing-v1`, with maps SHA-256
`ee9c4fa68737abe5671c54fa740f41fc6a2f8a0aec1e5c7668b4b641593a9137`.
Do not select a different map from screen results. The previously failed
candidate remains failed on its original development evaluation.

Before any new steering generation, separately freeze that implementation,
its controls and gates. Include both matched-penalty shared and shuffled
maps, the stronger shuffled control, the same-map prompt-only control, and
both text cues. Do not refit maps, retune penalties, or promote a control.
The new 512-question reservation can only be considered after a separately
declared candidate evaluation passes. This screen has no steering or reserved
generation entry point; passing it alone does not satisfy the research goal.

Run sequentially on physical GPU 0 only. Leave GPU 1 available to the other
project. Keep raw outputs, fitted maps, and activations ignored. Nicole Ma is
the sole Git author and committer.

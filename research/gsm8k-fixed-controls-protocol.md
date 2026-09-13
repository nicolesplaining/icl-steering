# Diagnose the fixed candidate's development gain

Declared after observing fixed-candidate scores of 107/128, zero-shot 93/128,
and raw mean 96/128. The candidate failed its text-cue gate. That selection
remains failed, and neither its 256 reserved questions nor the earlier
conditional run's 512 reserved questions will be generated here.

This is a supplementary diagnostic on the same 128 development questions,
not a confirmation, a new sample, or a new parameter search. Its purpose is
to distinguish query-associated activation information from magnitude and
shared-direction effects. The current positive comparison with the raw mean
cannot make that distinction by itself.

Keep the candidate fixed at block 13, strength 0.5, prefill only. Reuse the
seven completed conditions and all 56 frozen annotations exactly. Copy the
existing fitted maps and unmodified zero-shot query activation cache without
refitting or extracting any new question. Keep all prompt, model, decoding,
batching, and stopping settings fixed.

Generate the nine remaining controls from the previously specified sixteen
conditions, on the same development questions:

- The old v2 mean at block 7, strength 0.5, every token.
- The block-13 mean orientation scaled per question to the ridge vector norm.
- The scalar affine predictor, matched to that norm.
- The ridge map fitted after permuting extraction target rows, norm-matched.
- The ridge map fitted to rotated demonstration-pair differences, norm-matched.
- The negative of the real ridge vector.
- Three fixed random orientations, seeds 31, 59, and 83, norm-matched.

The permutation acts on the 128 extraction target rows before fitting. It
does not permute the development questions. The maps and permutation were
already fixed in conditional v1. Use the original raw means for the two
unmodified mean controls, and the real candidate's per-question norm for the
listed matched controls. No control receives independent tuning. This adds
1,152 generations to the 896 existing rows, for 2,048 rows across 16 conditions.

Complete all controls before opening their scores. Export a shared shuffled,
deduplicated packet of unparsed responses, with conditions and references
hidden. Carry forward the 56 existing annotations by exact response hash,
without revisions. Review the remaining responses under the same literal
transcription rubric and commit the merged annotations before scoring.
Parsed grades stay fixed; truncated responses receive no credit. Do not
execute generated code or solve missing calculations.

Report every condition's explicit-parser and audited completed counts,
truncation, and all fifteen paired comparisons of the fixed ridge with the
other conditions. Use 10,000 paired bootstrap resamples and seed 907 for
exploratory unadjusted 95% intervals. Also report two-sided exact McNemar
tests and Holm-adjusted p-values over all fifteen comparisons, separately
for each metric. The reported family includes the text and actual-ICL
comparisons; do not select favorable comparisons after seeing results.

Query association is unsupported if shuffled-target or norm-matched simple
controls explain the gain comparably well. Even an advantage over those
controls would remain an exploratory mechanism signal on development data,
not a passed text gate or independent confirmation. This diagnostic has no
selection or confirmation stage, and cannot mark the original candidate
eligible. Preserve the negative gate outcome alongside any positive contrasts.

All inference runs sequentially on physical GPU 0, leaving GPU 1 available.
Raw generations, fitted maps, and activations stay ignored. Nicole Ma remains
the sole Git author and committer.

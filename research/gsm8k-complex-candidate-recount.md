# Independent candidate score recount

September 13, 2026. The candidate is generating under declaration
`ff0366d86c536a045662db8811f3093dcf6fb442`. No candidate accuracy has been
opened. The first real batch passed replay of 516 saved state vectors and
129 LM-head calls. That limited check does not replace the required complete
trajectory audit.

The subsequent [two-condition runtime audit](../results/gsm8k-complex-candidate-v1-pair-runtime.json)
passed all 512 candidate and same-map prompt-only responses in 128 batches.
It replayed 33,280 saved state vectors and verified 8,320 LM-head calls.
For all 256 questions, the initial normalized states, intervention vectors,
applied states, prompt lengths, and first generated tokens matched exactly.
The CPU audit used a separate snapshot of immutable batches while GPU 0
continued the remaining controls. The recorded `complete: false` correctly
indicates that this covers two conditions, not the entire experiment.
No accuracy was opened.

[The separate recount](../analysis/complex_candidate_score_recount.py)
checks all 4,352 development rows across 17 conditions. It reconstructs the
blind review packet, verifies exact inheritance of the 1,280 baseline rows
and their annotations, and compares rational answers with the pinned gold
values. Truncations receive no completed-answer credit.

For both the parser and reviewed metrics, the recount independently checks
all 16 candidate comparisons, including paired wins and losses, gains,
10,000-draw bootstrap intervals, exact two-sided McNemar probabilities, and
Holm adjustments across the 16-comparison family. It recomputes every frozen
candidate gate without importing the production scorer or gate function.
Passing development still leaves reserved generation unauthorized.

The recount checks declaration, annotation, and complete runtime-audit
provenance. It does not repeat the runtime reconstruction or the prior ICL
screen audit. It uses stored explicit parses, and its separate count-weighted
bootstrap implementation shares the declared PCG64 draws with production.
The original independent screen recount and complete runtime audit remain
separate requirements.

Ten CPU tests passed in 8.36 seconds. A hand-specified synthetic candidate
has 249 parser-correct and 250 reviewed-correct answers; each comparator has
239 and 240. The resulting ten wins and zero losses give an exact p-value
of 1/512 and a Holm-adjusted value of 1/32. Tests reject altered counts,
intervals, exact probabilities, adjusted probabilities, gates, inherited
annotations, absent annotation freezes, and incomplete runtime audits. A
separate tied-candidate fixture correctly fails the development gates.

The checker adds no generation, map-fitting, or selection changes. Run it
after all outputs, the complete trace audit, frozen review, and production
scoring exist:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.complex_candidate_score_recount \
  --run runs/gsm8k-complex-candidate-v1 \
  --output runs/gsm8k-complex-candidate-v1/score-recount.json
```

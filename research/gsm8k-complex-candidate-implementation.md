# Conditional candidate implementation

September 13, 2026. The fixed candidate runner is implemented and tested
before the real ICL screen's accuracy is opened. It cannot prepare a steering
run from an unfinished or failed screen. A passing screen must also have a
matching independent recount and frozen annotations. The real candidate
preparation and generation had not yet run at that implementation checkpoint.

The runner reuses all three declared map archives byte for byte and checks
that the candidate and same-map prompt-only control have identical matrices.
The [map check](../results/gsm8k-complex-map-check-v1.json) independently
reconstructs the original fits and verifies all 12 intervention conditions
at 82 condition/position combinations, with four states per check. Predictions
agree in float64 tolerance and exactly after the stored float32 conversion.
The old prompt map's solve residual is 3.6652e-14. No new fit is saved and no
new model answer is produced by that check.

Five CPU tests passed in 11.57 seconds. They cover routing, changed-map
rejection, unfinished and failed screen guards, declarations before backend
loading, source-input tampering, and the full synthetic generation pipeline.
The latter produces 3,072 new rows and inherits 1,280 baseline rows. Resume
adds no new calls. Independent trace replay checks 8,192 synthetic state
vectors and matches all 256 candidate/prompt-only first tokens. Deliberately
changing one first token is rejected. Scoring requires frozen annotations,
and the synthetic tied conditions correctly fail the candidate gate.

The runtime audit binds its result to the exact generation-row digest and
checks actual LM-head inputs, saved shifts, applied states, cutoff accounting,
and the candidate/prompt-only initial states and first tokens. Its independent
predictions come from reconstruction of the original extraction fits rather
than the new routing function.

The [protocol and gates](gsm8k-complex-candidate-protocol.md) remain unchanged.
A passing screen does not automatically launch steering, and a development
pass does not authorize reserved generation. Input and implementation
declarations are required before their respective generation stages.

## Completed generation and runtime audit

Generation subsequently finished at 11:03:03 UTC on September 13, with exit
code zero. All 12 new conditions have 256 responses, giving 3,072 new and
4,352 total responses including the five inherited baselines. The worker and
supervisor exited and released GPU 0. No reserved questions were generated.

The [complete runtime replay](../results/gsm8k-complex-candidate-v1-runtime.json)
passed at 11:06:31 UTC. It checked all 768 new batches, replayed 330,376 saved
state vectors, and verified 82,594 LM-head calls. Candidate and same-map
prompt-only initial states, shifts, applied states, and first generated tokens
matched on all 256 questions. The audit report matches the frozen manifest,
fit, checker source, and complete generation rows. This verifies execution of
the declared intervention, not its accuracy or scientific benefit.

The [review lock](../results/gsm8k-complex-candidate-v1-review-lock.json) was
committed before reading the new answers. Its packet contains 318 unique
unparsed responses, including 39 unchanged inherited annotations and 279 new
responses requiring individual review. The review lock's row digest uses
condition/problem-ID sorted order, as does the packet; the runtime report's
row digest uses the runner's collection order. Both were checked against the
same complete generation file. All 279 new responses were subsequently
reviewed and committed before scoring. The [completed evaluation](gsm8k-complex-candidate-results.md)
failed the development gate; its independent score recount passed.

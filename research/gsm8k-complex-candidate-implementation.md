# Conditional candidate implementation

September 13, 2026. The fixed candidate runner is implemented and tested
before the real ICL screen's accuracy is opened. It cannot prepare a steering
run from an unfinished or failed screen. A passing screen must also have a
matching independent recount and frozen annotations. The real candidate
preparation and generation have not been run.

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
than the new routing function. Real traces still require their own completed
audit if this conditional experiment runs.

The [protocol and gates](gsm8k-complex-candidate-protocol.md) remain unchanged.
A passing screen does not automatically launch steering, and a development
pass does not authorize reserved generation. Input and implementation
declarations are required before their respective generation stages.

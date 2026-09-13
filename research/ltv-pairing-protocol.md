# Separate shared shifts from ICL pairings

September 13, 2026. The [continuation candidate](gsm8k-continuation-results.md)
scores 115/128 after blinded review, tying its shuffled-target control. The
real and shuffled maps used different nested-selected penalties. This new
development diagnostic separates that regularization difference from the
contribution of individual ICL pairings. It does not reopen the failed gate,
select a new candidate, or authorize reserved answers.

Use the exact 620 extraction states from the completed continuation run.
For each measured prefix position j, compute the average target shift mu_j
across available extraction questions. Replace every target D_i at position
j by mu_j to form the shared-target matrix M. Fit the same uncentered ridge
map to D, M, and the previously fixed within-position permutation of D,
seed 2390, at both penalty scales 0.1 and one times mean squared input norm.
These are the two already selected penalties, not a new search.

The shared-target map receives the current normalized activation at inference.
It does not receive the prefix index as a separate predictor. Its targets
contain only five average shifts, so its exact linear-map rank is at most
five. It differs from the earlier position-specific mean control, which
interpolated average vectors directly from a known token position. The
shared map retains state dependence through its regression coefficients.

At each fixed penalty, ridge linearity gives W_D = W_M + W_(D-M). Test this
identity and independently replay fits before generation. Reconstruct the
old real scale-0.1 and shuffled scale-one maps exactly from the same inputs;
reuse their completed outputs rather than generating them again. A shared
map tie would not prove that residual information never matters; a gain for
the real map would still need independent confirmation.

Keep all 13 previous conditions and their exact 1,664 output rows. Generate
only these four new conditions, sequentially in this order:

- `shared_low`: shared targets, penalty scale 0.1.
- `permuted_low`: the original within-position permutation, scale 0.1.
- `real_high`: correctly paired targets, scale one.
- `shared_high`: shared targets, scale one.

All four use the same Qwen2.5-Math-7B revision, prompts, zero-shot input,
strength one, final-normalization intervention site, greedy decoding,
1,024-token answer budget, 4,096-token context limit and batches of four.
Intervene at prefix lengths zero through 128 inclusive, then return the
unmodified state. Save the same states, shifts, applied BF16 vectors and
actual LM-head checks for independent replay. Model weights and cached
keys and values remain unchanged. Use physical GPU 0 only and leave GPU 1
for the other project. There are 512 new generations and 2,176 total rows.

Freeze source, input hashes, full fits and declarations before generation.
Require complete trace replay after all 128 new batches finish. Export a
shuffled blind packet, preserve inherited annotations by response hash,
individually review new unparsed responses, and commit all annotations before
opening scores. Transcribe stated numbers without correcting arithmetic or
executing generated code. Missing answers stay null. Parsed grades stay
fixed; truncated answers receive no credit in both reported metrics.

Report every condition count and these seven paired contrasts for both the
explicit-parser and audited metrics:

1. `steered` minus `shared_low`.
2. `steered` minus `permuted_low`.
3. `real_high` minus `shared_high`.
4. `real_high` minus `permuted`.
5. `steered` minus `real_high`.
6. `permuted_low` minus `permuted`.
7. `shared_low` minus `shared_high`.

Use 10,000 paired bootstrap draws, seed 907, and exact two-sided McNemar tests
with Holm adjustment across seven contrasts separately for each metric.
Report unadjusted intervals and both wins and losses. No best-condition
selection, stopping threshold, or automatic confirmation transition is
part of this diagnostic. The 128 development questions are reused and have
informed this design; these comparisons are exploratory with respect to the
larger experiment sequence. Keep all 256 reserved questions untouched.

Nicole Ma is the sole Git author and committer. Exclude AGENTS.md,
credentials, raw generations, fitted maps, model weights and activations
from commits. Document the outcome even if no pairing-specific benefit
appears. A later candidate requires its own declared development plan.

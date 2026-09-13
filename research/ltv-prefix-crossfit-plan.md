# Test a continuation-trained map on excluded questions

September 13, 2026. This diagnostic follows the published
[prompt-to-decoding mismatch](ltv-prefix-alignment-results.md). It is a new
fit, proposed after observing that mismatch. It does not alter the failed
all-token candidate or its untouched reserved questions.

Use the saved identical-prefix states from `gsm8k-ltv-prefix-v1`. No further
model generation is needed. Retain all 128 extraction questions and the
available positions 0, 1, 8, 32, and 128. The inputs contain no query answer
supervision. The target is the final normalized ICL state minus the zero-shot
state for the identical prefix.

Shuffle the 128 questions once with NumPy seed 2713 and divide the shuffled
order into four contiguous groups of 32. Each group is an evaluation fold;
fit on the other 96 questions. Every state from a question stays in the same
fold. Aggregate only predictions made while that question was excluded from
fitting. Report each fold as well as all out-of-fold predictions.

The sole primary predictor is an uncentered linear ridge map with penalty
five, fit on all available question-position rows from the training fold.
Each available row has equal weight. There is no position feature, centering,
strength search, normalization, clipping, or penalty selection. Fit and
predict in NumPy float64, rejecting nonfinite values or relative solve
residual above 1e-8. This uses the same fitting equation as the previous map
but includes continuation states. Unequal availability means questions with
early EOS supply four rows rather than five; report the row counts.

Compute these fixed controls using only each fold's training questions:

- Zero shift.
- Ridge fitted only on the prompt-final position, penalty five.
- One pooled mean target.
- One pooled affine scalar, mean(D) + c times h minus mean(X), with c chosen
  by centered least squares over all training rows.
- A separate mean target at each of the five positions.
- A separate affine scalar at each position.
- A pooled ridge map after independently permuting target rows within each
  position, using NumPy seed 1901 plus the fold number. Keep input rows in
  their original order and preserve each position's target multiset. Use
  its own predicted magnitude.

Report normalized squared error, median cosine, median predicted-to-target
norm ratio, 10th/90th percentiles, zero norms, undefined quantities, and
question counts for every position and predictor. Normalized error is the
sum of squared residuals divided by the sum of squared target norms.
Also report the unweighted average of the four continuation-position
normalized errors. This prevents the large early target from dominating
the comparison. Do not interpret fold variation as independent replications.

The diagnostic passes only if the primary has error below one and positive
median cosine at each continuation position, and its average continuation
error is at least 10% below every nonzero control's average continuation
error. The position-specific mean and scalar are included in that rule.
Prompt-position performance is reported, but is not a gate for a future
continuation component. All controls and failed criteria stay in the report.
These thresholds decide whether another accuracy experiment is warranted;
they do not establish a positive math result or causal ICL transfer.

Commit the code, tests, and this plan before preparing the run. Then commit
a declaration binding the manifest, source, input files, and fold assignment
before evaluation. Check ridge predictions against a separate eigensystem
solution and check reporting calculations using per-question dot products.
Save out-of-fold arrays locally on the server, ignored by Git. No model
weights, raw generations, or activation tensors may be committed.

If this diagnostic passes, freeze a full extraction fit and separately
declare an accuracy experiment with matching controls and a validation gate
before using any reserved questions. If it fails, publish the failure and
do not launch that continuation-map candidate. A positive activation metric
alone cannot complete the research goal.

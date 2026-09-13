# Select ridge regularization inside the training folds

September 13, 2026. The [fixed-penalty crossfit](ltv-prefix-crossfit-results.md)
failed its gate. This new diagnostic tests whether regularization accounts
for its poor later-position generalization. It follows inspection of those
results and is further development on the same extraction pool. It does not
revise the previous result or consume reserved math questions.

Retain the exact four outer question folds, paired states, available
positions, equal weight per available state, and uncentered ridge equation.
Only the penalty rule is selected. Candidate rules, in tie-breaking order,
are penalty 5 and training-state mean squared norm multiplied by 0.001,
0.01, 0.1, 1, or 10. Compute the scale using training states only. There is
no centering, scalar prior, strength change, position feature, or extra grid.

Within each outer training set of 96 questions, shuffle its sorted question
indices with NumPy seed 2714 plus the outer fold number. Divide into three
inner groups of 32. Fit on 64 questions and predict the excluded 32, keeping
all of a question's states together. For each penalty rule, pool the inner
out-of-fold squared errors and target energies separately by position, then
average the four continuation-position normalized errors. Select the rule
with the smallest average, breaking exact ties by the listed order. Refit
on all 96 outer training questions and predict the excluded outer 32.
Outer target values must not enter penalty selection or fitting.

Tune the shuffled-target control independently over the same rules. Permute
training targets within each position with seed 1901 + 100 times the outer
fold number + the inner fold number. For the outer refit, replace the inner
fold number with 99. Evaluate shuffled fits against the true excluded
targets. This gives the control its own best inner-selected regularization,
rather than comparing a tuned map with an underregularized control.

Retain all eight predictions from the fixed-penalty crossfit as reported
comparisons. The new primary is the nested-selected ridge, and the new
shuffled control is also reported. Publish every inner rule score, actual
selected outer penalty, outer fold result, and pooled out-of-fold metric.
No choice is made from outer scores. Folds share training data, and these
reused extraction questions are not independent research confirmation.

Keep the same diagnostic gate: error below one and positive median cosine
at every continuation position, and mean continuation error at least 10%
below each nonzero control. Include the independently tuned shuffled control
and all original controls in that comparison. Prompt-position performance
is reported but not gated. If any criterion fails, do not launch this
candidate in answer generation. A pass only warrants a separate accuracy
design; it cannot complete the research goal.

Run on CPU in NumPy float64. Check selected outer fits against a direct
linear solve with relative residual at most 1e-8, and check predictions from
the eigensystem implementation against that solution. Use the previously
verified reporting calculations and replay saved results separately. Test
that perturbing outer evaluation targets leaves penalty selection and
predictions unchanged, and check the spectral predictions against direct
solves across the whole penalty grid on synthetic data.

Commit code, tests, and this plan before preparation. Commit the input,
fold, and source declaration before evaluation. Keep raw state and prediction
arrays ignored. No new GPU generation or query reference answers are used.

# Regularization passes the activation gate, with weak late-position gains

September 13, 2026. Nested-selected ridge has mean continuation prediction
error 0.639, compared with 0.716 for the strongest scalar control, a 10.8%
reduction. It passes the unchanged activation diagnostic gate. This is
evidence for better prediction of ICL activation differences on excluded
questions, not yet evidence that injecting those predictions improves math.

The margin is modest. At 32 tokens, error is 0.976, only 2.4% below the
zero-shift baseline. The position-specific scalar is better at both 32 and
128 tokens. Two of four folds have error above one at 32 tokens. The pass
comes from the declared pooled criteria, not uniform superiority.

## What was selected and what remained excluded

This follows the [failed fixed-penalty fit](ltv-prefix-crossfit-results.md).
The exact same 128 extraction questions, 620 available paired states, and
four outer folds were retained. Each outer fold excludes 32 questions from
both fitting and penalty selection. Within the remaining 96, three inner
folds compare six penalty rules: five, or the training-state mean squared
norm multiplied by 0.001, 0.01, 0.1, 1, or 10.

Selection minimizes the unweighted mean of four continuation-position
normalized errors pooled across inner excluded questions. Outer targets
never enter that choice. The shuffled-target ridge has its own independent
penalty selection, using targets shuffled within training positions. Its
evaluation targets remain the true excluded targets.

All four real-map folds choose scale 0.1, yielding penalties around 5,000.
All four shuffled-target folds choose scale 1, around 50,000. No rule was
chosen from outer scores. The [plan](ltv-prefix-regularization-plan.md) and
tests were committed as `d438d66`; the input and source
[declaration](../results/gsm8k-ltv-regularization-v1-declaration.json) was
committed as `ee63e19` before evaluation.

| Outer fold | Real-map rule | Actual penalty | Shuffled rule | Actual penalty | Real-map outer continuation error |
|---:|---|---:|---|---:|---:|
| 0 | scaled_0.1 | 4944.133 | scaled_1 | 49441.328 | 0.640274 |
| 1 | scaled_0.1 | 5012.586 | scaled_1 | 50125.865 | 0.699601 |
| 2 | scaled_0.1 | 5102.793 | scaled_1 | 51027.932 | 0.617703 |
| 3 | scaled_0.1 | 5032.153 | scaled_1 | 50321.532 | 0.620704 |

## All out-of-fold prediction errors

Values are summed squared prediction errors divided by summed squared
target norms. Zero shift scores one. The final column averages the four
continuation positions equally, excluding the prompt. All controls from
the previous crossfit are retained exactly; the tuned shuffled control is
additional. Counts are 128 through position 32 and 108 at position 128.

| Predictor | Prompt | 1 token | 8 tokens | 32 tokens | 128 tokens | Continuation mean |
|---|---:|---:|---:|---:|---:|---:|
| Nested-selected ridge | 0.166984 | 0.182078 | 0.562392 | 0.976116 | 0.835782 | 0.639092 |
| Nested-selected shuffled targets | 0.253176 | 0.426420 | 0.940246 | 1.068327 | 0.974419 | 0.852353 |
| Fixed penalty-five ridge | 0.173186 | 0.147939 | 0.779391 | 1.513671 | 1.309460 | 0.937615 |
| Zero shift | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Prompt-only ridge | 0.159645 | 1.845578 | 4.455681 | 4.879585 | 3.091723 | 3.568142 |
| Pooled mean | 0.808344 | 0.970965 | 1.032750 | 1.095647 | 1.099540 | 1.049725 |
| Pooled scalar | 0.665021 | 0.676720 | 1.039141 | 1.266067 | 1.068047 | 1.012493 |
| Position-specific mean | 0.258494 | 0.444563 | 0.854144 | 0.942453 | 0.832130 | 0.768322 |
| Position-specific scalar | 0.148026 | 0.328464 | 0.824766 | 0.927814 | 0.784221 | 0.716316 |
| Fixed penalty-five shuffled targets | 0.424781 | 0.614030 | 2.144403 | 3.333785 | 2.873128 | 2.241336 |

## Primary direction and magnitude

| Prefix tokens | Median cosine | Median predicted / target norm |
|---:|---:|---:|
| 0 | 0.922465 | 0.888653 |
| 1 | 0.946417 | 0.883441 |
| 8 | 0.655327 | 0.802932 |
| 32 | 0.383884 | 0.872210 |
| 128 | 0.418377 | 0.977185 |

Regularization reduces the excess magnitude of the penalty-five fit and
improves its late-position error. The learned map has its clearest advantage
at positions one and eight. The average gain over position-specific scalar
does not imply an advantage at every position. The latter has error 0.928
versus 0.976 at position 32, and 0.784 versus 0.836 at position 128.

## Gate, verification, and limits

The declared gate passes: pooled error is below one and median cosine is
positive at all four continuation positions; mean continuation error is at
least 10% below each nonzero control. These are development thresholds
without a claim of statistical significance. The fold training sets overlap,
and this experiment was designed after seeing earlier results on the same
extraction pool. Nested selection prevents direct use of outer targets but
does not erase that research history.

The [full result](../results/gsm8k-ltv-regularization-v1-results.json) reports
all six inner rule scores for both maps in each outer fold, every chosen
penalty, each outer result, and all pooled metrics. The
[verification record](../results/gsm8k-ltv-regularization-v1-verification.json)
records three passing tests, including perturbing outer targets without
changing selected penalties or predictions. Selected predictions match a
separate direct solve, with residuals below 1e-8. A separate invocation
reproduced the saved report and every saved prediction array.

The model weights remain frozen. This stage used CPU only, with no new
development or reserved answer generation. Original state and prediction
arrays remain ignored. No positive steering accuracy has been verified.

## Next accuracy experiment

The diagnostic now warrants a separately declared accuracy experiment. It
must retain strong position-specific controls and isolate any benefit beyond
prompt-only token selection. The [first-token interpretation](ltv-first-token-interpretation.md)
still applies at the final normalization site. It should also explicitly
address the fact that training prefixes came from zero-shot generation,
while steering can produce different prefixes, and that measured positions
extend only through 128 tokens.

A full extraction fit, intervention scope, control construction, validation
gate, and review procedure must be frozen before generating new answers.
All 256 reserved questions stay untouched until a candidate passes a
separate development gate and a confirmation protocol is declared. The
activation result alone does not complete the research goal.

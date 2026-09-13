# The prompt-fitted map sees different states during decoding

September 13, 2026. This is a descriptive snapshot taken while LTV validation
was running, before opening its answer scores. The
[analysis code](../analysis/ltv_state_shift.py), committed as `bf82f60` before
the calculation, fixes the first 32 development questions in their existing
order and token positions 0, 1, 8, 32, and 128. It uses only the full-map
condition. The [saved result](../results/gsm8k-ltv-v1-state-shift.json) includes
source-file hashes and within-cohort 10th and 90th percentiles.

The fitted map uses 128 prompt-final extraction states of width 3,584.
Their uncentered row span has rank 128. For each saved state h, the span
statistic is the squared length of the component orthogonal to that span,
divided by the squared length of h. This measures distance from a linear
span, not a probability of being outside the training distribution.

| Generated prefix tokens | Questions | Median energy outside extraction span | Median shift/state norm | Median shift-state cosine |
|---:|---:|---:|---:|---:|
| 0, prompt final | 32 | 3.63% | 0.553 | -0.117 |
| 1 | 32 | 16.03% | 0.561 | -0.818 |
| 8 | 32 | 20.70% | 0.553 | -0.695 |
| 32 | 32 | 23.58% | 0.516 | -0.612 |
| 128 | 30 | 22.00% | 0.465 | -0.598 |

At the first decoding position, the median paired increase in outside-span
energy is 11.72 percentage points relative to each question's own prompt
state. The median paired change in shift-state cosine is -0.669. The
relative shift norm stays near its prompt value at that position. The
change is therefore not simply an explosion in relative shift magnitude.

These observations support examining whether a prompt-only fitted map
transfers to continuation states. They do not prove a harmful intervention.
The states follow prefixes generated under steering, so ordinary changes
during decoding and effects of those steered prefixes are not separated.
The current state itself is unmodified before its shift is calculated.

Later positions exclude questions that have ended. The last recorded token
decision is excluded because it can be EOS or padding after a custom stop.
The paired changes in the JSON compare retained states with their own
prompt states. No correctness fields, parser outcomes, finish reasons,
answer texts, or reference answers enter the calculation.

This diagnostic changes no intervention, gate, or sample. All conditions
still need to finish and undergo the declared blinded review. If the
candidate fails, a useful next diagnostic would measure actual ICL-minus-zero
differences on identical continuation prefixes from extraction-only
questions. That would test the missing target relationship directly; this
span analysis cannot establish it.

The two geometry tests cover in-span and orthogonal states, a known
antiparallel shift, and rejection of undefined zero/nonfinite inputs. The
frozen inference code and fitted arrays were not changed.

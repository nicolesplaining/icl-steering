# A positive steering signal, with a failed text-cue gate

September 13, 2026. The fixed query-dependent ridge intervention scores
107/128 after blinded answer review, compared with 93/128 zero-shot and
96/128 for the same-setting constant mean. Both exploratory paired intervals
exclude zero. It falls short of First at 108/128 and step-by-step at 110/128,
so the declared confirmation gate fails. No confirmation questions were run.

This is the first clear development signal here that the conditional shift
outperforms its raw mean. It is not yet evidence that query association, rather
than magnitude or another shared activation effect, caused the difference.
The intervention is a linear map producing a different vector for each
question, not one universal direction. The model weights stay frozen.

The subsequent [supplementary controls](gsm8k-fixed-controls-results.md)
score 106/128 for a scalar control given ridge's per-question norms, compared
with ridge's 107/128. The raw-mean contrast survives adjustment, but a need
for the learned full-map directions is not established. The failed gate
remains unchanged.

## Development results

All conditions use the same 128 questions. The five baseline output sets
and their annotations were preserved from the completed ICL screen. Only
the fixed ridge candidate and matched raw mean were newly generated.

| Condition | Explicit parser | Audited | Unparsed | Truncated |
|---|---:|---:|---:|---:|
| Zero-shot | 78 | 93 | 23 | 3 |
| ICL-A | 114 | 115 | 1 | 0 |
| ICL-B | 115 | 116 | 2 | 0 |
| First | 97 | 108 | 11 | 0 |
| Step by step | 105 | 110 | 7 | 0 |
| Fixed ridge | 97 | 107 | 13 | 3 |
| Matched raw mean | 84 | 96 | 17 | 4 |

| Audited comparison, ridge minus control | Gain, points | 95% interval | Wins | Losses |
|---|---:|---:|---:|---:|
| Zero-shot | +10.94 | [3.91, 17.97] | 18 | 4 |
| Raw mean | +8.59 | [3.91, 14.06] | 12 | 1 |
| First | -0.78 | [-7.81, 6.25] | 10 | 11 |
| Step by step | -2.34 | [-9.38, 4.69] | 8 | 11 |
| ICL-A | -6.25 | [-12.50, 0.00] | 4 | 12 |
| ICL-B | -7.03 | [-13.28, -0.78] | 4 | 13 |

The primary-parser gap over zero-shot is 19 questions. Review rescues 15
zero-shot answers and ten ridge answers, leaving a 14-question audited gap.
The raw-mean comparison also survives review: 13 questions before review,
11 after. Truncation is low and is scored as failure in both metrics.

The candidate passes the minimum three-point zero-shot gain, strict raw-mean
advantage, and truncation gates. It fails the requirement to match both text
cues. Their paired intervals overlap zero, but the frozen gate uses point
estimates; that rule has not been relaxed after observing the result. The
selection artifact records `eligible: false`.

## Limits and next diagnostic

The candidate was fixed at block 13, strength 0.5, prefill only before these
development scores were opened. Its map was fitted on the old 128 extraction
questions, using activation differences rather than mathematical answer
labels. There was no new parameter grid. Nevertheless, the development sample
was first screened for an ICL benefit, and these comparisons are exploratory,
with unadjusted 10,000-resample paired intervals using seed 907. They are not
held-out confirmation.

Magnitude-matched mean, scalar, shuffled-target, rotated-pair, reverse, and
random controls have not yet been evaluated on this sample. Those comparisons
are needed to interpret why the ridge beats the raw mean. A supplementary
diagnostic can reuse this completed development sample and fixed parameters;
it cannot change the failed gate or count as an independent replication.
The 256 reserved questions and the earlier failed run's 512-question
reservation remain untouched.

## Verification

The [protocol](gsm8k-fixed-candidate-protocol.md) was pushed in `cc68aee`,
before the screen's scores were opened. The
[input declaration](../results/gsm8k-fixed-candidate-v1-declaration.json)
preceded steering generation. Preparation replayed the passing screen audit
and checked exact baseline-file and fitted-map identity. All 896 rows passed
prompt, identity, parameter, grade, and token-accounting checks. An independent
NumPy calculation reproduced all 256 saved intervention-vector hashes and
norms. The process exited successfully and released GPU 0.

The combined blinded packet had 56 unique unparsed responses. The 44 inherited
annotations stayed unchanged; 12 new responses were transcribed without
condition labels or reference answers. The merged annotations were pushed in
`3309945` before scoring. Wrong stated numbers were preserved, missing answers
were not solved, and generated code was not executed. Parsed grades remain
fixed, so this single-reviewer process is not a full semantic solution audit.

See [complete metrics and gates](../results/gsm8k-fixed-candidate-v1-results.json),
[annotations](../results/gsm8k-fixed-candidate-v1-annotations.json), and the
[annotation freeze](../results/gsm8k-fixed-candidate-v1-annotation-lock.json).

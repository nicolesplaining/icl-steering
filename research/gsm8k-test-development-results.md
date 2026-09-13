# ICL passes the fixed development screen

September 13, 2026. On the predeclared 128-question development sample,
both frozen eight-example banks improve audited completed accuracy over
zero-shot. This replicates an ICL benefit on new official-test questions.
These questions now serve as development data, not held-out confirmation.
No steering result is available from this screen.

| Condition | Explicit parser | Audited | Unparsed | Truncated |
|---|---:|---:|---:|---:|
| Zero-shot | 78 | 93 | 23 | 3 |
| ICL-A | 114 | 115 | 1 | 0 |
| ICL-B | 115 | 116 | 2 | 0 |
| First | 97 | 108 | 11 | 0 |
| Step by step | 105 | 110 | 7 | 0 |

All counts are out of 128. After answer review, ICL-A improves on zero-shot
by 17.19 percentage points, with a paired-bootstrap 95% interval of
[10.16, 25.00] points and 25 wins versus three losses. ICL-B improves by
17.97 points, with interval [10.94, 25.78] and 25 wins versus two losses.
Both pass the predeclared minimum five-point gain and positive-lower-bound
gates. Truncation is below 5% for all five conditions.

The text cues remain strong controls. Step-by-step scores 110/128 and First
scores 108/128. The ICL gain is therefore not all specific to demonstrations.
ICL-A exceeds step-by-step by five questions and ICL-B by six; their exploratory
paired intervals include or touch zero. No claim of a robust ICL advantage
over that text cue follows from this screen.

The strict parser understates zero-shot performance: the audit recovers 15
correct completed responses, versus one for each ICL bank. The ICL benefit
remains after that correction. Only unparsed answers were reviewed; this is
not a semantic audit of all solutions. Wrong stated values were preserved,
missing answers were not solved, and generated code was never executed.
Truncated responses receive no credit. All intervals are exploratory and
unadjusted, with 10,000 paired resamples and seed 907.

## Next fixed comparison

The [next candidate](gsm8k-fixed-candidate-protocol.md) was fixed before these
scores were opened: extraction-fitted ridge, block 13, strength 0.5, last
prompt token only. It will be compared with its same-setting raw mean on
these development questions while preserving all five baseline output sets.
It must gain at least three points over zero-shot, strictly beat the mean,
match both text cues, and truncate at most 5%. There is no parameter grid or
replacement sample. Passing would authorize a separately frozen confirmation
implementation with the declared controls; it would not establish success.

The 256 questions reserved by this screen have no generated answers. The
failed conditional-v1 run remains failed, and its separate 512-question
reservation was excluded from this screen and remains untouched.

## Provenance

The [protocol](gsm8k-test-development-protocol.md) and
[declaration](../results/gsm8k-test-development-v1-declaration.json) preceded
generation. All 640 rows passed frozen-input and row-integrity checks.
The process exited with code zero and released GPU 0 before review.
The packet of 44 distinct unparsed responses was frozen in `66b6138`, and
the complete annotations were pushed in `f07f9d5` before scoring. The screen
selection returned `eligible: true`; the fixed-candidate preparation also
replayed the audit and all gate checks from the frozen records.

See [complete metrics](../results/gsm8k-test-development-v1-results.json),
[annotations](../results/gsm8k-test-development-v1-annotations.json), and the
[annotation freeze](../results/gsm8k-test-development-v1-annotation-lock.json).

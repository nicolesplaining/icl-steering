# Query-dependent GSM8K intervention

Declared September 13, 2026, after the completed v2 mean-direction test and
before any new validation or confirmation generation. This is an exploratory
follow-up, not a reinterpretation of the negative constant-mean result.

Keep Qwen2.5-Math-7B and both v2 support banks fixed. Reuse only the 128 v2
extraction questions for fitting. At each selected block, let X be the
zero-shot final-query activations and D the paired ICL-A minus zero-shot
activations. Fit an affine ridge map from X to D, with centered inputs and
targets, an unpenalized mean, and penalty trace(Xc Xc^T)/128. Use NumPy
float64 matrix operations and verify solver residuals. No mathematical
answer is a regression target, and model weights remain frozen.

For a new question, obtain its unmodified zero-shot final-query activation
x in a separate forward pass. Predict d(x) from the fitted map, then generate
from the same zero-shot prompt while adding alpha*d(x) at the selected block.
For prefill scope, inject only at the last prompt token. For all scope, also
inject the same question-specific vector at every decoded token. Do not
recompute it from already-steered or decoded states. The all-scope variant
therefore assumes a prefill-derived shift remains useful during decoding.
There is no access to a new question's ICL activation during intervention.

## Fresh questions and validation

Seed 1301 selects 96 new training questions for validation and 512 new
official test questions for confirmation. Exclude the first 128 examples of
each source split and all v2 extraction, validation, and test questions.
Also exclude normalized-text duplicates of every old question and either
support bank. The complete partition and prompts are saved before inference.
The test set is never used for fitting or selection.

The fixed grid is blocks 7 and 13, strengths 0.5 and 1, and prefill versus
all-token scope, for eight ridge candidates. These blocks had the strongest
conditional reconstruction improvements in the extraction-only diagnostic.
For every setting, evaluate the corresponding unmodified mean direction.
Also evaluate zero-shot, both actual ICL banks, First, step-by-step, and the
previously selected v2 mean intervention at block 7, strength 0.5, all scope.
This gives 22 validation conditions on identical questions. Greedy decoding,
the query template, the 1,024-token cap, and stopping rules remain unchanged.

Review all unparsed validation responses in one shuffled, deduplicated packet
with conditions and reference answers hidden. Use the existing answer-audit
rubric: transcribe stated numbers, preserve errors, do not calculate missing
totals, and leave unresolved answers unanswered. Parsed grades remain fixed
and truncated generations receive no credit. Save the complete annotation
set before scoring or selecting. The audited completed-answer score is the
selection metric for this new experiment; report the explicit-parser score
as well. Neither metric is an independent semantic audit of all solutions.

Select the ridge candidate with maximum audited completed accuracy. Break
ties by prefill scope, lower strength, then lower block. Selection does not
search among controls. Proceed to confirmation only if both ICL banks beat
zero-shot by at least 5 percentage points, the chosen candidate beats
zero-shot by at least 3 points, strictly beats its same-setting mean, and
matches or exceeds each text-prefix baseline and the old v2 mean. Zero-shot,
both ICL banks, and the chosen candidate must each truncate at most 5%.
If the selected candidate fails a gate, stop and record the failure; do not
substitute another candidate on the basis of these gates.

## Locked confirmation and controls

Freeze the selected parameters, fitted maps, query partition, source code,
validation outputs, and annotations before generating test answers. Use all
512 questions for each of these sixteen conditions:

- Zero-shot, ICL-A, ICL-B, First, and step-by-step.
- The old v2 mean direction at its original settings.
- The selected query-dependent ridge intervention.
- The same-setting constant mean, with its original magnitude.
- The same mean orientation scaled per question to the ridge vector's norm.
- A scalar affine predictor fitted to the real extraction differences.
- A ridge predictor fitted after permuting extraction target rows.
- A ridge predictor fitted to rotated-example-pair differences.
- The negative of the selected query-dependent vector.
- Three fixed random orientations with seeds 31, 59, and 83.

Scale scalar, permuted-target, rotated-pair, and random controls per question
to the real ridge vector's norm. The permutation preserves the target mean
but breaks its association with extraction inputs. It is fixed once using
seed 1301. These controls separate query association, direction, and magnitude
effects. None receives independent parameter tuning.

After all conditions complete, perform a second shared blinded answer audit
on all unparsed test responses. Freeze annotations before scoring. Report
both completed metrics and paired comparisons of the selected intervention
with every control, plus actual ICL versus zero-shot. Use 10,000 paired
bootstrap resamples, seed 907, and label intervals exploratory and unadjusted.
Do not change the selected intervention using any test result. A useful
positive interpretation needs gains over zero-shot and the mean controls,
with explicit accounting for text prompts, query-permutation controls, and
format effects. Reconstruction improvement alone is not success.

All GPU stages run sequentially on GPU 0. The process stops for the assistant
to complete validation review before selection; it does not load a model
while waiting. Raw generations, token IDs, fitted arrays, and activation
caches stay in ignored run directories. Commit compact results and provenance,
with Nicole Ma as sole author and committer.

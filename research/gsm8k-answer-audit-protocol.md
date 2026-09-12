# Supplementary answer audit

Declared September 13, 2026, during the GSM8K v2 validation sweep, before final
test inference. This audit measures how often the explicit-answer parser
misses a stated numerical answer. It supplements the frozen primary metric;
it cannot change the selected intervention or rescue truncated generations.

Use all ten final test conditions: zero, ICL-A, ICL-B, First, CoT, selected
steering, reversed steering, and random directions 31, 59, and 83. Every
condition must contain all 256 declared test questions. Do not limit the
audit to conditions or questions showing a favorable effect.

Export every unparsed response into a packet that contains only the question,
response text, and a content-derived identifier. The packet withholds the
reference answer, condition name, original grade, and dataset ID. Identical
question/response pairs receive one shared annotation. Sort by identifier,
then shuffle with seed 907. Output style may still suggest the condition, so
this procedure does not guarantee that the reviewer cannot infer it.

For each response, transcribe the numerical answer it actually settles on.
Accept a concluding sentence, prose inside a box, or a clearly stated quantity
with units. Distinguish a unit such as "per 100 km" from the requested amount.
Preserve incorrect stated numbers. A literal fraction is acceptable; do not
calculate an unstated result, repair arithmetic, execute generated code, or
use an intermediate calculation as the answer. Use null if the answer is
absent, unresolved, contradictory without a final choice, or only an
unevaluated expression with variables. Include a brief rationale and mark
every entry explicitly reviewed. Finish and save the annotations before
comparing them with the reference answers.

The scorer verifies the packet, source data, and complete annotation set.
It applies each annotation consistently to all matching responses. Parsed
answers retain their original grades. A length-truncated response gets no
completed-answer credit, even if it contains a correct number. Report both
the original and audited scores for every condition, with paired comparisons
against zero-shot, the text prefixes, reversal, and each random direction.
Use 10,000 paired bootstrap resamples with seed 907; intervals remain
exploratory and unadjusted.

This is a single-reviewer supplementary audit, performed by the assistant,
not an independently validated semantic grader. It reviews unparsed answers
only, so it cannot rule out all errors in already-parsed responses. Retain
the raw outputs, packet, annotations, and hashes for later review. The
validation annotations were made with condition names visible and serve as
an exploratory audit and software check, not as a blinded replication.

The implementation is `analysis/gsm8k_answer_audit.py`. Its packet export and
scoring commands are documented in `replication/README.md`.

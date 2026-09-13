# Fixed candidate after the ICL screen

This implementation follows the candidate and thresholds declared in the
[test-distribution screen protocol](gsm8k-test-development-protocol.md),
before that screen's scores were opened. It has no parameter search and no
confirmation-generation entry point.

Preparation requires all five screen conditions and their completed blinded
answer review, an intact annotation freeze, and all screen gates passing.
Recompute the screen audit and gates from the frozen rows and annotations.
If they fail, preparation stops before copying inputs or loading a model.
The failed conditional-v1 study stays failed; only its extraction-fitted maps
are reused. Its 512 reserved questions remain excluded.

Copy the screen's 128 development questions, five baseline output sets, and
256-question reservation exactly. Copy the existing NumPy-fitted maps without
refitting. Predict the ICL-A minus zero-shot shift from each development
question's unmodified final-query activation. Add half the predicted shift
at block 13 on the last prompt token only. Compare this one candidate with
the raw mean at the same block, strength, and position. No new question's ICL
activation or mathematical answer enters the prediction. Model weights remain
frozen. Keep the screen's prompts, model revision, greedy decoding, token cap,
batch size, and stopping rules.

Preserve the five baseline generations byte-for-byte as records; do not
regenerate them or use a different parser. Generate only the candidate and
matched mean, 128 responses each. Record each predicted vector's hash and
norm. Use the tested conditional runner, including its original-batch resume
behavior. Only physical GPU 0 is permitted.

When both conditions finish, export one shared blinded packet of all seven
conditions' unparsed responses. Carry over already-frozen screen annotations
by exact question/response hash. Review only remaining items, without
condition labels or reference answers. Do not revise old annotations. Commit
the complete merged annotation set before scoring. Wrong stated answers stay
wrong, absent or unresolved answers stay null, and truncated responses receive
no credit. Parsed grades stay fixed. This is a single-reviewer answer-format
audit, not a semantic audit of every solution.

The fixed candidate must improve audited completed accuracy over zero-shot
by at least three percentage points, strictly beat the raw matched mean,
match or exceed First and step-by-step, and truncate at most 5%. Both ICL
banks must still pass the original screen gates. If any gate fails, stop
without a replacement setting or sample. Report every count and gate.

A passing development result is not confirmation and does not establish a
useful positive effect. The reserved 256 questions may be used only by a later
frozen confirmation implementation with all sixteen controls declared in the
screen protocol. Its answers cannot be used to tune this candidate. All prior
negative results and the sequence of exploratory decisions remain visible.

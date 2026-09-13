# MATH replication screen

The rationale and evidence are in [the task review](../research/icl-task-review.md). This runner uses published benchmark problems and the pinned upstream prompt and grader. It does not run an activation-steering sweep.

Use a separate environment so the original experiment remains reproducible:

```bash
python3 -m venv .venv-benchmark
.venv-benchmark/bin/python -m pip install --upgrade pip
.venv-benchmark/bin/python -m pip install -r requirements-benchmark.txt
git clone https://github.com/TTChungC/Manyshot-CoT-ICL.git research/upstream/manyshot-cot-icl
git -C research/upstream/manyshot-cot-icl checkout c6ddffbbd8c4093a03090aa468e8048a43589340
.venv-benchmark/bin/python replication/math_screen.py prepare --output runs/math-replication-v1
VLLM_WORKER_MULTIPROC_METHOD=spawn .venv-benchmark/bin/python replication/math_screen.py run --output runs/math-replication-v1
.venv-benchmark/bin/python replication/math_screen.py report --output runs/math-replication-v1
```

`prepare` downloads the pinned data and tokenizer if needed, declares the question splits, saves all prompts, and rejects context overflow. The model weights must also be available before setting `HF_HUB_OFFLINE=1`. `run` resumes completed batches, refusing changes to configuration, script, upstream tracked files, or input data. Use only one writer per output directory.

The fixed screen has 384 generations: two subjects, three shot counts, and 64 questions. Primary metrics use the released symbolic grader. `final_only_accuracy` additionally requires a completed thinking segment and a boxed answer after it. The report includes paired bootstrap intervals and generation truncation. The first run used 8,172 new tokens to match the upstream script, but its geometry accuracy was dominated by unfinished traces. The replacement run uses 32,768 new tokens, four concurrent requests, and records the stopped run separately.

After geometry reached roughly 96% zero-shot accuracy with the larger budget, the geometry sweep was stopped as saturated. Number theory also reached 95.3% zero-shot on its 64-question screen. Its partial 16-shot comparison did not show a gain; that sweep was stopped too.

Artifacts include `manifest.json`, `prompts.jsonl`, `prompt_lengths.json`, `runtime.json`, `generations.jsonl`, `summary.json`, and `report.md`. A successful run also writes `complete.json`. The remaining official test questions and unused training indices are reserved in the manifest for later work.

The upstream repository is downloaded into an ignored directory; its source is not vendored into this repository. Record the pinned commit when comparing or reporting results.

## GSM8K audit and corrected steering experiment

The literature target is [Gadetsky et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/3e887bf77d0ba6db38802e552a0d81d2-Paper-Conference.pdf), which reports Qwen2.5-Math-7B GSM8K accuracy rising from 52.2% zero-shot to 91.4% with unsupervised ICL. Our original wrapper does not reproduce it exactly. The [audit](../research/gsm8k-audit.md) documents a parser bug, mismatched prompts, and repeated identical greedy prompts. The legacy scores are superseded and both legacy runners remain only for historical reconstruction.

Rescore the locally preserved zero-shot text on CPU:

```bash
PYTHONPATH=src python -m replication.audit_gsm8k \
  --input runs/legacy-gsm8k/heldout-preds.json \
  --output results/gsm8k-parser-audit.json
```

The corrected experiment uses matched zero/ICL query instructions and suffixes, a frozen train-only support bank, and separate extraction, validation, and test questions. Model weights stay frozen. It saves each raw generation, token IDs, termination reason, and conservative explicit-answer grade. Answer markers and balanced numeric boxes use the same grader in all conditions.

On the H100, use the saved support artifact at `runs/gsm8k-heldout-128-preds.json`. Its local backup is `runs/legacy-gsm8k/heldout-preds.json`. The artifact is a collection of model-generated training solutions, not evidence for a verified unsupervised-ICL gain.

Prepare pinned inputs, then run the validation and test gates:

```bash
PYTHONPATH=src .venv-benchmark/bin/python -m replication.gsm8k_steering prepare \
  --config configs/gsm8k_steering.json \
  --support-json runs/gsm8k-heldout-128-preds.json \
  --output runs/gsm8k-steering-v2
PYTHONPATH=src .venv-benchmark/bin/python -m replication.gsm8k_steering run \
  --config configs/gsm8k_steering.json \
  --support-json runs/gsm8k-heldout-128-preds.json \
  --output runs/gsm8k-steering-v2
```

The same invocation resumes saved batches. A kernel lock prevents simultaneous writers. Source, config, support, and data hashes must match the prepared manifest; use a new output directory after changing them. Preparation resolves a dataset revision if the config leaves it null and then pins it in the manifest.

Both ICL banks must improve validation completed-answer accuracy by at least five points, with at most 5% truncation, before fitting directions. The selected intervention must improve by three points before final testing. Test comparisons include actual ICL, text cues, sign reversal, and three random directions. Failed gates remain negative results and preserve the unseen final test. See the audit for selection rules and limitations.

### Supplementary answer audit

The [audit protocol](../research/gsm8k-answer-audit-protocol.md) supplements the
fixed explicit-answer metric. After all final conditions finish, export a
packet of unparsed responses with condition names and gold answers withheld:

```bash
PYTHONPATH=src python -m analysis.gsm8k_answer_audit export \
  --run runs/gsm8k-steering-v2 --split test \
  --conditions zero icl_a icl_b first cot steered reverse random_31 random_59 random_83 \
  --packet runs/gsm8k-steering-v2/test-review-packet.json
```

Write `test-review-annotations.json` with the packet SHA-256 printed by export
and an `answers` list. Every response needs `response_id`, `stated_answer`
as a numerical string or null, `reviewed: true`, and a `rationale`. Finish
transcribing the stated answers before comparing them with reference labels.
Then score all conditions together:

```bash
PYTHONPATH=src python -m analysis.gsm8k_answer_audit score \
  --run runs/gsm8k-steering-v2 --split test \
  --conditions zero icl_a icl_b first cot steered reverse random_31 random_59 random_83 \
  --packet runs/gsm8k-steering-v2/test-review-packet.json \
  --annotations runs/gsm8k-steering-v2/test-review-annotations.json \
  --output runs/gsm8k-steering-v2/test-answer-audit.json
```

The annotation file's top-level keys are `packet_sha256` and `answers`.
The tool rejects incomplete conditions, changed inputs, and incomplete
reviews. It leaves parsed grades and experiment selection unchanged, and
truncated generations still receive no completed-answer credit. Raw packets
and model responses stay in the ignored run directory.

### Prefix geometry controls

The extraction-only diagnostic compares real demonstrations with rotated
question/solution pairings, shuffled demonstration tokens, and repeated
single-token filler. Shuffling and filler preserve every token position of
the query and leave the instruction header unchanged. They are deliberately
unnatural controls. Their similarity to the real ICL direction can expose
a generic prefix effect; dissimilarity alone cannot prove useful ICL.

The two-H100 node is shared with another project. Use GPU 0 for this
repository and run GPU jobs sequentially. Leave GPU 1 available for the
other project. Run this extraction diagnostic when GPU 0 is free:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_prefix_controls \
  --run runs/gsm8k-steering-v2 --output runs/gsm8k-prefix-controls-v1
```

It uses only the 128 extraction questions and verifies that replaying the
original prompts reproduces the saved mean directions. Model weights stay
frozen. Inputs and activation tensors remain in the ignored output directory;
`metrics.json` records geometry and provenance. Inspect any existing control
run before starting a new one; the command refuses to overwrite its manifest.

The [fixed supplementary test](../research/gsm8k-prefix-test-protocol.md) must
be declared before the main test starts. It waits for the main selection and
test lock, then evaluates all three directions with the same selected
parameters and norms. On the shared node, resume the declared supplementary
run only after the main process exits and its test finishes. The internal
selection wait alone does not prevent GPU contention. Supply the main
Python process ID, which can have exited after completing its test:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_prefix_test \
  --run runs/gsm8k-steering-v2 --controls runs/gsm8k-prefix-controls-v1 \
  --output runs/gsm8k-prefix-test-v1 --parent-pid MAIN_PYTHON_PID
```

After both tests finish, combine their generation records for the review
packet, retaining the original prepared data and a record of both source
hashes. Include `prefix_rotated_pairs`, `prefix_token_shuffle`, and
`prefix_length_filler` alongside the ten primary conditions. Use the same
answer-audit rubric for every condition. Additional paired comparisons are
exploratory; they cannot change the selected intervention.

The combined analysis command checks completion, selection locks, question
identities, and source hashes before creating the shared review packet.
Run it on the server, where the original direction file is available:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_comparison prepare \
  --run runs/gsm8k-steering-v2 --prefix runs/gsm8k-prefix-test-v1 \
  --output runs/gsm8k-comparison-v2
```

Review only `review-packet.json` while transcribing the stated answers.
Save the complete annotations using the audit protocol, then score them:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_comparison report --run runs/gsm8k-comparison-v2 \
  --annotations runs/gsm8k-comparison-v2/review-annotations.json
```

`comparison.json` and `comparison.md` report both metrics and paired
comparisons against every control. The original grades and selection remain
unchanged. The intervals are exploratory and unadjusted.

## Query-dependent follow-up

The completed mean-direction test is summarized in
[the final v2 report](../research/gsm8k-v2-results.md). The
[conditional protocol](../research/gsm8k-conditional-protocol.md) fits an
affine map from zero-shot extraction activations to paired ICL differences,
then evaluates it on fresh questions. It selects using audited validation
scores and stops before confirmation if the declared gates fail.

Prepare on CPU using the existing extraction arrays. Fitted arrays stay in
the ignored output directory:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_conditional prepare \
  --config configs/gsm8k_conditional.json \
  --parent runs/gsm8k-steering-v2 --controls runs/gsm8k-prefix-controls-v1 \
  --output runs/gsm8k-conditional-v1
```

After verifying GPU 0 is available, run validation. It exits after exporting
the shared blinded packet, leaving the GPU free during review:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_conditional validate --output runs/gsm8k-conditional-v1
```

Review only `validation-review-packet.json`, save all annotations, then run
selection. No test generation happens in this command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_conditional select --output runs/gsm8k-conditional-v1 \
  --annotations runs/gsm8k-conditional-v1/validation-review-annotations.json
```

If eligible, the test stage locks selection and fitted maps before inference.
It also exits for a blinded review after all sixteen conditions finish:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_conditional test --output runs/gsm8k-conditional-v1
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_conditional report --output runs/gsm8k-conditional-v1 \
  --annotations runs/gsm8k-conditional-v1/test-review-annotations.json
```

Use two CPU threads for NumPy fitting and tests on the shared node. Keep
`HF_HOME=/lambda/nfs/icl/huggingface`, `USE_TF=0`, and
`TOKENIZERS_PARALLELISM=false` in the runtime environment. The original v2
source files and its selection remain unchanged.

## Fixed test-distribution screen

Conditional v1 [failed validation](../research/gsm8k-conditional-results.md),
so its test is prohibited. The separate
[development protocol](../research/gsm8k-test-development-protocol.md) checks
ICL on one fixed sample from the remaining official test distribution. These
questions are development data; the failed run's reservation is excluded.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_test_development prepare --output runs/gsm8k-test-development-v1
```

Commit the protocol, implementation, and declaration before inference. Save
`declaration.json` in the run directory with its `manifest_sha256` and source
commit. With GPU 0 available, run the five baseline conditions:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_test_development screen --output runs/gsm8k-test-development-v1
```

Review the full blinded packet and commit complete annotations before scoring.
The run's `review-freeze.json` must contain `annotations_file_sha256`,
`packet_sha256` using the canonical audit digest, and `annotations_commit`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_test_development score --output runs/gsm8k-test-development-v1 \
  --annotations runs/gsm8k-test-development-v1/validation-review-annotations.json
```

This runner has no steering or confirmation entry point. A failed screen
ends this fixed-bank setup; it does not trigger another seed or sample.

If the screen passes, the [fixed-candidate stage](../research/gsm8k-fixed-candidate-protocol.md)
replays its audit before preparing any intervention inputs. It reuses the
baseline rows and generates only the block-13, half-strength prefill ridge
candidate and its matched raw mean. It has no parameter grid or test stage.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_fixed_candidate prepare --output runs/gsm8k-fixed-candidate-v1
```

Commit the prepared manifest declaration before generating. The run-local
`declaration.json` must contain the manifest's `manifest_sha256`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_fixed_candidate validate --output runs/gsm8k-fixed-candidate-v1
```

Initialize review from `inherited-annotations.json`, keeping those annotations
unchanged. Review all remaining blinded items and commit the merged annotations.
Create `review-freeze.json` with the same fields used for the screen, then:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-benchmark/bin/python \
  -m analysis.gsm8k_fixed_candidate select --output runs/gsm8k-fixed-candidate-v1 \
  --annotations runs/gsm8k-fixed-candidate-v1/validation-review-annotations.json
```

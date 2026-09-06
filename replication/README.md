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

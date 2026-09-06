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

After geometry reached roughly 96% zero-shot accuracy with the larger budget, the geometry sweep was stopped as saturated. `configs/math_number_theory.json` runs the same screen on number theory, which leaves room for an ICL gain.

Artifacts include `manifest.json`, `prompts.jsonl`, `prompt_lengths.json`, `runtime.json`, `generations.jsonl`, `summary.json`, and `report.md`. A successful run also writes `complete.json`. The remaining official test questions and unused training indices are reserved in the manifest for later work.

The upstream repository is downloaded into an ignored directory; its source is not vendored into this repository. Record the pinned commit when comparing or reporting results.

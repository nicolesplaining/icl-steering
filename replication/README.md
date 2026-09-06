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

## GSM8K positive-control reproduction

The strongest literature-backed positive control is [Gadetsky et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/3e887bf77d0ba6db38802e552a0d81d2-Paper-Conference.pdf), which reports Qwen2.5-Math-7B GSM8K accuracy rising from 52.2% zero-shot to 91.4% with unsupervised ICL. The released code is [mlbio-epfl/joint-inference](https://github.com/mlbio-epfl/joint-inference).

Run the small paper-compatible screen on the H100 after setting `HF_HOME` to the model cache:

```bash
.venv-benchmark/bin/python replication/gsm8k_joint_inference.py \
  --mode paper --adaptation-examples 128 --evaluation-examples 128 \
  --turns 5 --num-repeats 5 --max-new-tokens 1024 \
  --output runs/gsm8k-paper-128.json
```

The released script adapts on the same test pool it scores. For a clean activation study, use `--mode heldout`, which adapts on GSM8K train and scores on test. The runner saves zero-shot and per-turn held-out evaluation results; it does not save model weights or activations.

Add `--save-predictions` when the final pseudo-labeled support bank is needed for activation extraction. The saved file contains text prompts and answers only; activation tensors remain run-local and are never committed.

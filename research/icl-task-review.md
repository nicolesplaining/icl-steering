# Which math tasks are worth replicating?

Reviewed September 5, 2026. The most relevant next experiment is MATH geometry and number theory on Qwen3-8B. GSM8K is a useful second target. A task name alone does not guarantee an ICL gain: model, demonstrations, baseline instructions, and decoding all matter.

## Evidence

| Study | Relevant result | What it establishes |
|---|---|---|
| [Wei et al., Chain-of-Thought Prompting](https://arxiv.org/html/2201.11903v6) | Worked demonstrations improve arithmetic reasoning, including GSM8K, over demonstrations containing only answers. | Evidence for the value of worked solutions. The main contrast is not zero examples versus examples, and the large models differ from ours. |
| [Xie, activation-difference steering](https://arxiv.org/html/2510.01246v1#S4.SS3) | Gemma-2-9B GSM8K raw accuracy is 22.29% zero-shot, 69.60% with eight worked examples, 53.37% with MeanActDiff, and 53.90% with the prefix "First". | The closest existing math steering comparison. It also supplies the strongest reason to include cheap textual controls. These are Gemma base-model results, not expected Qwen3 scores. |
| [Agarwal et al., Many-Shot In-Context Learning](https://arxiv.org/html/2404.11018v2#S3.SS1) | Gemini 1.5 Pro improves on MATH500 with many examples; MATH demonstrations also transfer to GSM8K. Model-generated demonstrations can outperform human solutions. | Supports MATH and GSM8K as candidate tasks, but uses a closed model and often compares many shots with a four-shot baseline. Question-only contexts can also help. |
| [Chung et al., Many-Shot CoT-ICL](https://arxiv.org/html/2605.13511v2#S4.SS2) | Studies Qwen3-8B/14B on MATH subjects and GSM8K. For Qwen3-14B geometry, 16 to 128 examples raises accuracy from 66.18% to 73.07%. | Best model-family match. It supports testing many-shot scaling, not assuming a gain over zero-shot. |

At 128 examples, Chung et al. report Qwen3-8B scores of 67.01% on geometry and 84.63% on number theory with thinking enabled. Their Table 1's enabled/disabled comparison is not an ICL-versus-zero-shot comparison. Their ordering results are also distinct from demonstrating a reusable activation direction. [Source](https://arxiv.org/html/2605.13511v2#S4.SS2)

## Replication audit

I inspected the [released code](https://github.com/TTChungC/Manyshot-CoT-ICL/tree/c6ddffbbd8c4093a03090aa468e8048a43589340), pinned at `c6ddffbbd8c4093a03090aa468e8048a43589340`. The local MATH runner uses the first N training examples in original order, gold worked solutions, a shared step-by-step query suffix, greedy decoding, an 8,172-token output limit, and repetition penalty 1.1. Its model setup uses FP16 and enables thinking for Qwen3. Our runner reads its prompt constants and imports its grader rather than substituting the synthetic pilot's numeric grader.

There are reproducibility gaps. The code does not pin the authors' dataset snapshot, model snapshot, dependency versions, or explicitly supply its stated long-context RoPE settings. Its chat helper omits `enable_thinking=False` when disabling thinking, which would leave the current Qwen3 template's default enabled. Our planned run enables thinking, so that ambiguity does not affect this comparison. We do not claim the released code reproduces every reported table exactly.

The [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-8B#best-practices) recommends sampling for thinking mode, whereas the released MATH runner uses greedy decoding. We follow the released runner for this replication and record that choice. For long contexts, we explicitly apply the model card's factor-four YaRN setting consistently across all shot counts. This can affect short prompts too; a later unscaled zero-shot reference may be useful, but must not replace the matched primary baseline.

## Our declared first run

- Qwen3-8B, pinned revision, thinking enabled for every condition.
- MATH geometry and number theory from a pinned EleutherAI mirror of the official train/test splits.
- Zero, 16, and 128 examples. The 16 examples are a prefix of the 128-example bank. Preserve original training order.
- A fixed random subset of 64 official test questions per subject, shared by every condition. Save the indices before inference. Exclude those questions from subsequent steering test sets.
- Preserve training examples after the demonstration bank for later extraction and validation. Never choose demonstrations using screening answers.
- Use the released prompt, grader, FP16, greedy decoding, repetition penalty, and answer budget. Record model/data hashes and all rendered prompts.
- Run vLLM with prefix caching and chunked prefill on the available H100. Chunked prefill and the memory settings differ from the release and are recorded in the config.

This is a reduced replication screen. It does not reproduce the paper's full test set, every demonstration count, multiple orders, or CDS algorithm. The zero-shot condition is our addition. It is essential for deciding whether there is an ICL benefit to transfer.

Report paired changes for 128 versus 16, and for each ICL condition versus zero. Include bootstrap intervals, output length, truncation, and a separate score that requires a final answer after the thinking segment. The upstream grader can count a boxed answer inside an unfinished thinking trace; preserve that primary score for comparability while exposing the distinction.

If ICL improves reliably over zero-shot, repeat with another demonstration order and untouched questions before a large steering sweep. Then extract few-shot-minus-zero-shot residual differences on separate training questions. Compare steering against zero-shot, actual demonstrations, prompt-only reasoning cues, reversed directions, and norm-matched random directions. A many-shot gain alone says nothing about whether one linear direction can reproduce it.

## What happened to the generated tasks?

The [completed synthetic screen](../results/qwen3-pilot-v2/report.md) found no positive ICL-A accuracy gain. Quadratic sums scored 50.0% zero-shot and 37.5% with examples; symmetric powers scored 87.5% in both conditions. Each comparison had only 16 questions. Quadratic zero-shot also had 12.5% truncation. These are weak, task-specific observations, not evidence against ICL in general. The runner correctly stopped before fitting or tuning any direction.

The first Qwen3 MATH run exposed a second failure mode. With the upstream 8,172-token cap, geometry scores were 51.6%, 57.8%, and 71.9% for zero, 16, and 128 examples. But every finished zero-shot and 16-shot thinking trace was correct, as were 46 of 48 finished 128-shot traces. The headline gain mostly measured whether thinking finished before the cap. We stopped after 192/384 generations and increased the budget to 32,768 for the replacement run. This is a useful negative result about evaluation design, not evidence that ICL improves geometry.

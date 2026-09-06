# ICL steering

Can a reusable activation direction recover the accuracy benefit of worked mathematical examples? This repository tests that question on Qwen3-8B with frozen weights. It also tests whether the direction transfers a particular solution method beyond what a short textual instruction achieves.

The first pilot uses non-thinking mode throughout. Qwen3 thinking mode is a separate experiment because changing it also changes the generation procedure. The model revision is pinned in `configs/pilot.json`.

## Run

Use Python 3.10 or later and a CUDA GPU. The pilot is configured for an 80 GB H100, BF16 weights, and batches of up to 32. Install an appropriate CUDA build of PyTorch first.

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[test]'
python -m pytest -q
icl-steering prepare --output runs/qwen3-pilot
icl-steering screen --output runs/qwen3-pilot
icl-steering run --output runs/qwen3-pilot
icl-steering report --output runs/qwen3-pilot
```

`run` includes screening and resumes completed generation records. It stops if neither family clears the screening threshold. Changing source, configuration, or generated data requires a new output directory. Keep only one writer per directory. An interrupted write that leaves a partial JSON line must be repaired before resuming.

Download the pinned model through Hugging Face before an offline run. Credentials belong in the environment or an interactive session, never in configuration files or committed shell scripts. Once the model is cached, `HF_HUB_OFFLINE=1` avoids network lookups.

## Experiment

The data are generated math problems with exact integer answers and independently checked solutions. They are controlled pilot tasks, not a claim about performance across a public math benchmark.

| Family | Method A | Method B |
|---|---|---|
| Sum a quadratic sequence | Finite differences and binomial coefficients | Fit polynomial coefficients and sum powers |
| Compute x^k + y^k given x+y and xy | Recurrence for power sums | Quadratic roots and binomial expansion |

All conditions ask the model to show work and box the answer. The two demonstration conditions use the same example questions and answers, with different worked solutions. Demonstrations vary deterministically by target question. No target solution is included in its prompt.

For each family, the splits contain 16 demonstration problems, 16 screening problems, 32 extraction problems, 12 validation problems, and 48 test problems. Parameter tuples are disjoint across every split. One third of test questions use a wording absent from the other splits. This does not establish out-of-distribution mathematical generalization.

Screening evaluates zero-shot and both demonstration methods. Select the family with the largest method-A ICL accuracy gain over zero-shot, requiring at least 6.25 percentage points. Ties use configuration order. This permissive pilot gate is not a significance test. If neither family passes, stop and report the failed screen.

At the last prompt token, collect the output residual of each selected decoder block. Layers are zero-indexed. Average over extraction questions:

```text
ICL direction    = mean(h_A - h_zero)
Method direction = mean(h_A - h_B)
Generic control  = mean((h_A + h_B)/2 - h_zero)
```

The first two names describe the contrasts, not proven interpretations of the resulting directions. Different demonstration lengths, formatting, and answer styles can contribute to either contrast.

Add a scaled direction to the corresponding residual at the final prompt position of a zero-shot query. The pilot edits only that position during prefill. It does not directly edit every generated token. `intervention_positions=all` is supported for a separate experiment.

For each of the ICL and method contrasts, choose the layer and strength with the best validation accuracy from layers 8, 17, 26, and 32 and raw-mean strengths 0.5, 1, and 2. Ties favor a smaller strength and then a shallower layer. Freeze both choices before test evaluation. Twelve validation questions make this selection noisy; treat it as a pilot.

Evaluate on the same 48 test questions:

- Zero-shot, "think step by step," assistant prefix "First,", and an explicit instruction describing method A.
- Four worked demonstrations using method A or method B.
- The selected ICL and method directions, with no demonstrations.
- Each selected direction with its sign reversed.
- The generic direction at each selected layer, matched to that direction's norm and strength.
- Three seeded random directions for each selected direction, also matched in layer, norm, and strength.

The generic contrast overlaps algebraically with the ICL contrast. It is a diagnostic control, not an independent null. Three random seeds are a small reference sample and do not support a strong random-direction significance claim.

## Interpretation

The primary outcome is exact numeric accuracy. Report paired accuracy differences and bootstrap intervals. The percentage of ICL gain recovered is useful only when held-out ICL actually improves over zero-shot. Comparisons and intervals are exploratory, without correction for multiple comparisons.

Greedy decoding and a shared 768-token output limit make runs reproducible under the recorded setup. The report includes truncation and output lengths because demonstrations may teach concision. A truncated answer counts as correct only if it has already produced a parseable correct final answer. A result dominated by truncation should be repeated under a larger, newly declared budget before being described as improved mathematical competence.

The grader accepts numeric boxed answers and explicit final-answer lines. It intentionally rejects unevaluated expressions. Parse failures remain visible in saved summaries. The method-signature column is an unvalidated keyword aid, not a reasoning judge. Inspect saved solutions before drawing conclusions about method use.

A positive result could show that a reusable direction elicits a known mathematical technique. It would not show that the model learned a new algorithm from the demonstrations. A convincing follow-up needs more seeds, stronger task diversity, a public benchmark, careful trace annotation, and comparison against the best cheap textual control.

`runs/` stores the configuration, source and dataset hashes, dataset, rendered prompts, generations, vector diagnostics, validation selection, runtime versions, and reports. It is excluded from Git, along with model weights and activation tensors. Copy compact reports into `results/` only after inspecting them. Raw generations should remain available for auditing.

## Related work

This is not a novel steering construction. The experimental question is whether its benefit is specific, useful, and competitive with prompting.

- [In-context learning creates task vectors](https://arxiv.org/abs/2310.15916).
- [Function vectors in large language models](https://arxiv.org/abs/2310.15213).
- [Comparing Sparse Autoencoder Representations and Mean Activation Difference for Language Model Steering](https://www.research-collection.ethz.ch/items/781252e6-528c-4061-9fb9-807f502b8a42), including a close mathematical few-shot-minus-zero-shot comparison.
- [CAST](https://arxiv.org/abs/2507.13236), using averaged contextual activation differences.

The main concern is that steering may reproduce answer style or generic reasoning cues. The explicit-method and "First," controls are included because that explanation must be tested.

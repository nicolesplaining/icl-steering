# Related work for the question-specific shift

Reviewed September 13, 2026, while the supplementary controls were running
and before opening their scores. The core linear-map idea already exists.
Our experiment should be described as a math application and controlled
evaluation, not a new method for predicting activation differences.

Kwon, Choi, and Sohn's **Linear Task Vector**, or LTV, directly fits a ridge
map from zero-shot hidden states to ICL-minus-zero-shot states. Its main
experiments use final-layer states, 256 extraction queries, 30 demonstrations,
and eight classification benchmarks, including Qwen2.5-7B and Qwen3-8B.
Appendix A.7 also compares a constant mean, linear map, and nonlinear map.
Appendix A.5 describes synthetic regression with numeric generation and
recomputing the shift from the current state at each output step. This is
direct precedent for our central construction, although it does not establish
a GSM8K result. [Paper, sections 5–6 and appendices A.5/A.7](https://arxiv.org/html/2605.20730v1).

Our current implementation instead uses a centered affine ridge map fitted
on 128 extraction questions, an internal block, and half-strength injection
at the final prompt token. Its penalty is fixed from the extraction Gram
matrix. Earlier all-token conditions reused the same question vector during
decoding. These differences make our experiment a variant, not an exact LTV
replication. See the [fitting protocol](gsm8k-conditional-protocol.md) and
[fixed candidate](gsm8k-fixed-candidate-protocol.md).

Hsu et al.'s **Contextual Linear Activation Steering**, or CLAS, learns a
state-dependent scalar multiplying a fixed direction at each block. It fits
the scalar predictor using next-token loss on prompt-completion pairs.
This credits an existing approach to conditional strength, but differs from
fitting a full activation-difference map. [Paper, section 2](https://arxiv.org/html/2604.24693v1).

Jiang et al.'s **MimIC** learns per-attention-head shift vectors and
query-dependent magnitudes. Its objective combines alignment with ICL hidden
states and supervised language-model loss. Its experiments concern visual
question answering and captioning. It supplies further precedent for
conditional ICL compression, without verifying mathematical reasoning
transfer. [Paper, sections 3–4](https://arxiv.org/html/2504.08851v2).

## Released LTV code needs a separate replication audit

The [official repository](https://github.com/Jii111/LTV) was inspected at
`e019cb49ee64c1e0914fb139d1a811321b3051e0`. No upstream code was executed.

- The regression hook explicitly skips sequence lengths of one, implementing
  prefill-only steering with cached decoding. This differs from the paper's
  every-step description.
  [Regression wrapper](https://github.com/Jii111/LTV/blob/e019cb49ee64c1e0914fb139d1a811321b3051e0/core/wrapper_regression.py#L37).
- Extraction reads `hidden_states[-1]`, while injection hooks the final
  decoder block. A replication must check whether the final normalization
  separates these tensors in its actual Transformers version. Matching the
  tensor dimension alone does not establish a matching intervention site.
  [Extraction and injection](https://github.com/Jii111/LTV/blob/e019cb49ee64c1e0914fb139d1a811321b3051e0/core/wrapper_ltv.py#L158),
  [module paths](https://github.com/Jii111/LTV/blob/e019cb49ee64c1e0914fb139d1a811321b3051e0/core/wrapper_base.py#L661).
- The released regression configuration specifies 12 demonstrations and one
  run. Its generator caps output at 15 tokens and parses the first number,
  substituting zero when none appears. Those defaults cannot be transferred
  unchanged to worked GSM8K solutions.
  [Configuration](https://github.com/Jii111/LTV/blob/e019cb49ee64c1e0914fb139d1a811321b3051e0/config/config_regression.py),
  [generation and parsing](https://github.com/Jii111/LTV/blob/e019cb49ee64c1e0914fb139d1a811321b3051e0/run/run_regression.py#L58).

An [executable boundary check](../analysis/ltv_boundary_audit.py) confirms the
normalization distinction in our Transformers 4.51.3 environment using a
tiny randomly initialized Qwen2 on CPU. `hidden_states[-1]` exactly equals
the normalized final-block output and differs from the raw block output.
Adding the same shift before and after normalization produces different
states. The hook sees sequence lengths three and one for prompt processing
and a cached decoding step, respectively. The
[saved evidence](../results/ltv-boundary-audit.json) records the versions and
script hash. This tests our installed architecture semantics; it does not
establish what runtime produced the paper's results. Our existing GSM8K
runner captures and injects at the same internal-block output, so this
finding does not invalidate the running controls.

Reproduce on CPU with an unused output path:

```bash
CUDA_VISIBLE_DEVICES='' USE_TF=0 .venv-benchmark/bin/python \
  analysis/ltv_boundary_audit.py --output runs/ltv-boundary-check.json
```

## Implication for this experiment

The pending controls address a narrower empirical question: does the fixed
math candidate need the correct association between extraction questions and
their ICL activation changes? An advantage over raw mean alone cannot answer
that. Norm-matched means, shuffled extraction targets, and random directions
are needed to interpret the gain.

Even favorable controls would remain development evidence. The candidate
failed its declared text-cue gate; this literature finding does not reopen
confirmation or justify changing that gate. A later experiment using a
different intervention site or decoding rule must declare those choices
before evaluating them. The current controls should finish first.

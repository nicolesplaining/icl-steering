# Inspecting the published complex demonstrations

September 13, 2026. This is a source inspection while the declared steering
experiment runs. No new model outputs, extraction states, or scores were
produced.

Fu et al. select eight training demonstrations with long reasoning chains.
Their GSM8K complex bank averages nine steps, compared with 3.4 for the
handcrafted bank. In Table 1, greedy accuracy rises from 48.1% to 55.4% for
text-davinci-002 and from 61.0% to 66.6% for code-davinci-002. These compare
demonstration banks, not zero-shot with ICL. Their separate majority-voting
results add a decoding intervention. Section 4.1 adds a step-by-step cue to
all prompting schemes. The paper supports testing complex demonstrations;
it does not establish their benefit on Qwen2.5-Math or demonstrate activation
transfer. [Paper](https://arxiv.org/html/2210.00720v2)

The [original release](https://github.com/FranxYao/Complexity-Based-Prompting/tree/378f7a88fb4c6a3fdb294f0dcf8a702420f76724)
contains four GSM8K prompt files. I pinned commit
`378f7a88fb4c6a3fdb294f0dcf8a702420f76724` and verified each working file
against that commit. The [inspection result](../results/gsm8k-published-prompt-inspection-v1.json)
records hashes, normalized question hashes, and overlap with our prepared
partitions. The upstream checkout remains ignored.

| Released bank | Examples | Nonempty solution lines per example | Bytes |
|---|---:|---|---:|
| Original | 8 | 3, 3, 4, 3, 3, 4, 3, 4 | 2,374 |
| Simple | 8 | 2 each | 3,168 |
| Mid | 8 | 4 each | 3,183 |
| Complex | 8 | 9 each | 8,464 |

These line counts exclude the question, step-by-step cue, and final-answer
line. They are an asset check, not a claim that each line is one semantic
operation. All banks preserve the released question framing and answer
format. This original repository contains prompt assets without a complete
generation runner, so reproducing its files alone does not reproduce the
paper's full runtime.

One complex demonstration matches `gsm8k:train:6724` in our current
256-question development set. The mid bank matches `gsm8k:train:1169` and
`gsm8k:train:7174`. None of these four banks matches a question in the new
512-question reservation or the older 512- and 256-question reservations.
Matching uses casefolding and whitespace normalization, so it does not detect
paraphrases. These findings do not affect the running experiment, which uses
its previously declared banks.

If a further ICL screen is needed, this gives a concrete replication target:
use the released original and complex banks verbatim, exclude their questions
from evaluation, and compare them with a matched zero-shot step-by-step
baseline. Keep query framing, the model revision, decoding, and answer budget
fixed across conditions. Check actual tokenizer lengths before freezing a
context budget. Report this as a Qwen adaptation of the prompting experiment,
with our added zero-shot control, rather than reproduction of the historical
model scores.

Changing demonstrations also changes the activation differences being
estimated. Any subsequent steering study would need separate extraction
questions and a newly declared map; the current maps cannot be described as
trained on these published demonstrations. Fresh evaluation identities and
all gates would need freezing before generation. The three current
reservations remain reserved for their original experiments. No follow-up
screen is launched or declared by this inspection.

Reproduce the inspection with:

```bash
python3 -m analysis.complexity_prompt_inspection \
  --output runs/published-complex-prompt-inspection.json
```

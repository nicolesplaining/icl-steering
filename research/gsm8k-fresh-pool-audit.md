# Fresh GSM8K pool inventory

September 13, 2026. The local run archive leaves 7,057 training questions
and 39 official-test questions after conservative exclusions. Of the training
questions, 2,519 have at least four calculation annotations in their reference
solutions. These counts make a separate complexity-defined training evaluation
feasible. They do not show an ICL benefit or select a new sample.

The [machine-readable inventory](../results/gsm8k-fresh-inventory-local-v1.json)
is provisional with respect to server coverage. Before SSH failed, a server
directory listing revealed `gsm8k-conditional-preflight-v1/prepared.json`,
which is absent from the local archive. The server audit did not return a
result. No questions should be declared fresh across the project until the
server observations have been compared with the local observations and their
union excluded. A connection failure is not evidence that a remote process
stopped or that an output does not exist.

## Scope and exclusions

The inventory reads all 781 JSON and JSONL artifacts in the local `runs`
directory, including prepared questions, extraction input identities, support
banks, generation records, and batch outputs. It finds 1,440 unique GSM8K
identities and 544 identities with recorded generations. It never reads a
correctness field to decide eligibility and never selects by model output.
Generated text is neither graded nor used as a difficulty measure.

Exclude every observed identity, every observed question text, both earlier
reservations, and the first 128 source rows of each split. Recover question
text for every excluded identity from the pinned dataset. Normalize with
case folding and collapsed whitespace, then deduplicate across both splits.
All observed GSM8K identities must resolve in the pinned source. The audit
fails if a reservation has recorded generations or if its inputs change
during the inventory.

The dataset is `openai/gsm8k` at revision
`740312add88f781978c0658806c59bc2815b9866`. Both downloaded Parquet hashes,
the audit source hash, the complete input-inventory hash, and hashes of the
remaining identities and calculation counts are recorded in the result.
Detailed input hashes and observations remain in the ignored run directory.

| Source split | Original rows | Excluded | Remaining | Remaining with at least four annotations |
|---|---:|---:|---:|---:|
| Train | 7,473 | 416 | 7,057 | 2,519 |
| Test | 1,319 | 1,280 | 39 | 16 |

Neither the old 512-question reservation nor the current 256-question
reservation overlaps recorded generations in this local archive. Both remain
excluded. The inventory does not authorize their use.

## What the calculation counts mean

The count is the number of complete `<<...>>` annotations in the reference
solution. It is a reproducible metadata proxy, not a measured reasoning
difficulty or a count of all necessary reasoning steps. Some solutions have
no such annotations. No reference answer value or generated answer determines
the count. A future result on a subset defined this way must be described as
applying to that subset, not to all GSM8K questions.

[Fu et al., Complexity-Based Prompting for Multi-Step Reasoning](https://arxiv.org/abs/2210.00720v2)
report gains from choosing demonstrations with more reasoning steps. Their
work motivates attention to reasoning complexity, but it does not establish
that selecting harder evaluation questions makes our existing support banks
help Qwen2.5-Math-7B. Such a screen would be a new exploratory setup, not a
replication of that paper's intervention.

The next evaluation should fix its population, sampling rule, ICL screen,
candidate, controls, and analysis before generating new answers. Keep the
existing fitted maps fixed. Preserve the earlier failed gates and both
reservations. A failed new screen must not trigger replacement questions or
a changed complexity threshold within that experiment. No new candidate,
sample, or success criterion is frozen by this inventory.

## Verification and current state

Two unit tests passed. They cover nested extraction identities, support-text
exclusions, case and whitespace duplicates, duplicates across splits, source
prefix exclusions, unknown identities, and refusal of used reservations.
An independent identity-exclusion replay against the downloaded dataset
reproduced every remaining identity and both counts without importing the
inventory implementation. Histogram totals match pool sizes.

No new model answers or activations were generated. Local CPU work continued
after proxied SSH connections closed and direct SSH timed out. Remote source
coverage remains the next unresolved check; GPU work has not been launched.

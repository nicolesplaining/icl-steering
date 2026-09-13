# Fresh inventory for the published demonstration banks

September 13, 2026. The reconciled inventory leaves 6,282 unused GSM8K
training questions and 39 unused test questions. All three earlier
reservations have zero overlap with recorded generated questions. No new
sample was selected and no model inference was performed by this inventory.

The [inventory implementation](../analysis/gsm8k_published_inventory.py)
snapshotted 1,608 JSON/JSONL files on the laptop and 1,724 on the server after
the previous experiment completed. Both contain the same 2,208 observed
question IDs and 800 generated question IDs. The laptop has 2,718 normalized
question texts including the published examples; the server has 2,470.
The union preserves every exclusion from both sources. Replaying each
snapshot and their union against the pinned dataset gives identical pools.
The [union summary](../results/gsm8k-published-inventory-union-v1.json) records
snapshot hashes, dataset hashes, counts, and replayed pool digests.

Both published prompt files are verified against their fixed hashes. Their
16 distinct demonstration questions join the normalized text exclusions.
Seven training questions that otherwise remained unused match these
demonstrations. The inventory also excludes prior extracted, evaluated, and
reserved identities, the first 128 rows of each split, and normalized
duplicates across the two splits. Matching uses casefolding and whitespace
collapse, not a semantic paraphrase check. Recorded local/server artifacts
cannot establish whether a question has been used outside this project or
appeared during model pretraining.

| Partition | Dataset size | Excluded | Unused |
|---|---:|---:|---:|
| Train | 7,473 | 1,191 | 6,282 |
| Test | 1,319 | 1,280 | 39 |

| Existing reservation | Questions | Recorded generated overlap |
|---|---:|---:|
| Conditional experiment | 512 | 0 |
| Test-development experiment | 256 | 0 |
| Complex-question screen | 512 | 0 |

The [new screen protocol](gsm8k-published-screen-protocol.md) will sample
from the entire unused training pool, without filtering on calculation count.
No earlier reservation becomes eligible. The inventory itself lists unused
IDs only; listing them does not constitute selecting or evaluating them.

Three focused tests passed. They verify that reconciliation retains distinct
exclusions from both sources plus text and reservation exclusions, rejects a
tampered eligible pool, and rejects disagreement about reservation identities.
The actual laptop/server reconciliation independently replayed both complete
pools from their observations and the pinned dataset.

Reproduce snapshots on each machine with its cached pinned dataset:

```bash
python -m analysis.gsm8k_published_inventory snapshot \
  --runs runs \
  --prompt-dir research/upstream/complexity-based-prompting/GSM8K/lib_prompt \
  --output runs/published-inventory-snapshot
```

Copy the server snapshot to the laptop, then reconcile both snapshots into a
new output directory:

```bash
python -m analysis.gsm8k_published_inventory reconcile \
  --snapshots runs/gsm8k-published-inventory-local-v1 \
              runs/gsm8k-published-inventory-server-v1 \
  --output runs/published-inventory-union
```

Snapshots refuse overwrite and require an unchanged input file set while
collecting observations. The recorded inventory precedes the new screen's
preparation; a later snapshot may contain additional deliberately selected IDs.

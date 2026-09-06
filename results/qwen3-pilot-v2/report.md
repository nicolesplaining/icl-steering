# Pilot results

Model: `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`.
Thinking enabled: `False`. Greedy decoding. Generation limit: 4096 tokens.

These are exploratory results on generated high-school math problems. Method signatures are unvalidated text heuristics; inspect the saved solutions before interpreting them.

## screen: quadratic_sum

| Condition | n | Accuracy | Truncated | Output tokens | Method A signature |
|---|---:|---:|---:|---:|---:|
| icl_a | 16 | 37.5% | 0.0% | 1124.6 | 81.2% |
| icl_b | 16 | 43.8% | 6.2% | 1328.7 | 0.0% |
| zero | 16 | 50.0% | 12.5% | 1625.8 | 0.0% |

## screen: symmetric_power

| Condition | n | Accuracy | Truncated | Output tokens | Method A signature |
|---|---:|---:|---:|---:|---:|
| icl_a | 16 | 87.5% | 0.0% | 611.9 | 100.0% |
| icl_b | 16 | 87.5% | 0.0% | 605.4 | 100.0% |
| zero | 16 | 87.5% | 0.0% | 700.9 | 68.8% |

## screen decision

```json
{
  "scores": {
    "quadratic_sum": {
      "zero": 0.5,
      "icl_a": 0.375,
      "gain": -0.125
    },
    "symmetric_power": {
      "zero": 0.875,
      "icl_a": 0.875,
      "gain": 0.0
    }
  },
  "minimum_gain": 0.0625,
  "selected_family": null,
  "rule": "Largest screen ICL-A minus zero accuracy; ties use config family order."
}
```

"""Paired estimates and compact reports; zero-shot comparisons use identical questions."""

import json
from pathlib import Path

import numpy as np


def aggregate(rows):
    if not rows:
        return {}
    return {
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows)/len(rows),
        "parse_rate": sum(r["parseable"] for r in rows)/len(rows),
        "truncation_rate": sum(r["truncated"] for r in rows)/len(rows),
        "mean_generated_tokens": sum(r["generated_tokens"] for r in rows)/len(rows),
        "method_a_signature_rate": sum(r["method_signature"] in {"a", "both"} for r in rows)/len(rows),
    }


def paired_difference(rows, reference, samples=2000, seed=0):
    left = {r["problem_id"]: r for r in rows}
    right = {r["problem_id"]: r for r in reference}
    if len(left) != len(rows) or len(right) != len(reference):
        raise ValueError("Duplicate question IDs in paired comparison")
    if left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical question IDs")
    differences = np.array([float(left[k]["correct"])-float(right[k]["correct"]) for k in sorted(left)])
    if not len(differences):
        return None
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(differences), size=(samples, len(differences)))].mean(1)
    return {"gain": float(differences.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum())}


def build_report(output):
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    config = manifest["config"]
    rows = [json.loads(line) for line in (output/"generations.jsonl").read_text().splitlines() if line.strip()]
    groups = {}
    for row in rows:
        groups.setdefault((row["split"], row["family"], row["condition"]), []).append(row)
    summary = {"model": config["model"], "revision": config["revision"], "results": []}
    lines = ["# Pilot results", "", f"Model: `{config['model']}` at `{config['revision']}`.",
             f"Thinking enabled: `{config.get('enable_thinking', False)}`. Greedy decoding. Generation limit: {config['max_new_tokens']} tokens.", "",
             "These are exploratory results on generated high-school math problems. Method signatures are unvalidated text heuristics; inspect the saved solutions before interpreting them.", ""]
    for split in ["screen", "validation", "test"]:
        for family in config["families"]:
            available = sorted(k for k in groups if k[:2] == (split, family))
            if not available:
                continue
            lines += [f"## {split}: {family}", "",
                      "| Condition | n | Accuracy | Truncated | Output tokens | Method A signature |", "|---|---:|---:|---:|---:|---:|"]
            for key in available:
                stats = aggregate(groups[key])
                entry = {"split": split, "family": family, "condition": key[2], **stats}
                lines.append(f"| {key[2]} | {stats['n']} | {stats['accuracy']:.1%} | {stats['truncation_rate']:.1%} | {stats['mean_generated_tokens']:.1f} | {stats['method_a_signature_rate']:.1%} |")
                baseline = groups.get((split, family, "zero"))
                if baseline and {r['problem_id'] for r in baseline} == {r['problem_id'] for r in groups[key]}:
                    entry["versus_zero"] = paired_difference(groups[key], baseline, config["bootstrap_samples"], config["seed"])
                if split == "test" and key[2].startswith("steer_"):
                    entry["paired_controls"] = {}
                    kind = key[2].removeprefix("steer_")
                    controls = ["zero", "cot", "first", "instruction", "icl_a", "icl_b",
                                f"negative_{kind}", f"generic_matched_{kind}"]
                    controls += [f"random_{kind}_{seed}" for seed in config["random_seeds"]]
                    for control in controls:
                        ref = groups.get((split, family, control))
                        if ref:
                            entry["paired_controls"][control] = paired_difference(groups[key], ref, config["bootstrap_samples"], config["seed"])
                summary["results"].append(entry)
            lines += [""]
    for name in ["screen_decision", "selection"]:
        path = output/f"{name}.json"
        if path.exists():
            summary[name] = json.loads(path.read_text())
            lines += [f"## {name.replace('_', ' ')}", "", "```json", json.dumps(summary[name], indent=2), "```", ""]
    tests = [r for r in summary["results"] if r["split"] == "test" and r["condition"].startswith("steer_")]
    if tests:
        lines += ["## Paired test comparisons", "", "Intervals are percentile paired-bootstrap intervals. They are exploratory and are not corrected for multiple comparisons.", ""]
        for row in tests:
            for control, stats in row.get("paired_controls", {}).items():
                lo, hi = stats["ci95"]
                lines.append(f"- {row['condition']} vs {control}: {stats['gain']:+.1%}, 95% interval [{lo:+.1%}, {hi:+.1%}]; {stats['wins']} wins, {stats['losses']} losses.")
        zero = next((r for r in summary["results"] if r["split"] == "test" and r["condition"] == "zero"), None)
        icl = next((r for r in summary["results"] if r["split"] == "test" and r["condition"] == "icl_a"), None)
        if zero and icl and icl["accuracy"] > zero["accuracy"]:
            for row in tests:
                row["icl_gain_recovered"] = (row["accuracy"]-zero["accuracy"])/(icl["accuracy"]-zero["accuracy"])
            lines += ["", "ICL gain recovery ratios are in summary.json. They are unstable when the ICL advantage is small and are omitted when it is nonpositive."]
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return summary

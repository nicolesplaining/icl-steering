"""Combine completed GSM8K runs and report the declared answer-audit contrasts."""

import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

from analysis import gsm8k_answer_audit as audit
from icl_steering.report import paired_difference


PREFIX = ["prefix_rotated_pairs", "prefix_token_shuffle", "prefix_length_filler"]
ROOT = Path(__file__).parents[1]


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def write_fixed(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Refuse to replace changed analysis input: {path}")
    path.write_bytes(content)


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def prepare(parent, prefix, output):
    if read(parent / "complete.json")["status"] != "locked_test_complete":
        raise ValueError("Primary test incomplete")
    if read(prefix / "complete.json")["status"] != "prefix_test_complete":
        raise ValueError("Prefix test incomplete")
    manifest = read(parent / "manifest.json")
    selection = read(parent / "selection.json")
    lock = read(parent / "test_lock.json")
    if not selection["eligible"] or lock != {
        "selection_sha256": file_hash(parent / "selection.json"),
        "directions_sha256": file_hash(parent / "directions.pt")
    }:
        raise ValueError("Primary selection or directions changed")
    supplement = read(prefix / "manifest.json")
    request = supplement["request"]
    if (request["inputs_sha256"]["parent_manifest"] != file_hash(parent / "manifest.json")
            or request["inputs_sha256"]["parent_directions"] != lock["directions_sha256"]
            or request["conditions"] != PREFIX
            or request["n_test"] != manifest["config"]["n_test"]
            or supplement["primary_test_lock_present"] is not False):
        raise ValueError("Supplementary run does not match the declared parent")
    if read(prefix / "test_lock.json") != {
        "parent_test_lock": lock, "supplement_manifest_sha256": file_hash(prefix / "manifest.json")
    }:
        raise ValueError("Supplementary test lock changed")
    if file_hash(parent / "prepared.json") != manifest["prepared_sha256"]:
        raise ValueError("Prepared questions changed")
    declaration = read(ROOT / "results/gsm8k-v2-audit-declaration.json")
    if file_hash(Path(audit.__file__)) != declaration["audit_code_sha256"]:
        raise ValueError("Declared audit code changed")
    prepared = read(parent / "prepared.json")
    questions = prepared["splits"]["test"]
    by_id = {r["problem_id"]: r for r in questions}
    gold = {r["problem_id"]: r["answer"] for r in questions}
    if len(gold) != len(questions) or len(gold) != manifest["config"]["n_test"]:
        raise ValueError("Unexpected test question count or duplicate IDs")
    conditions = ["zero", "icl_a", "icl_b", "first", "cot", "steered", "reverse"]
    conditions += [f"random_{seed}" for seed in manifest["config"]["random_seeds"]]
    rows = []
    for directory, expected in [(parent, conditions), (prefix, PREFIX)]:
        source = [json.loads(line) for line in (directory / "generations.jsonl").read_text().splitlines()]
        test = [r for r in source if r["split"] == "test"]
        if set(r["condition"] for r in test) != set(expected):
            raise ValueError("Missing or unexpected test conditions")
        for row in test:
            if row["problem_id"] not in gold or row["answer"] != gold[row["problem_id"]]:
                raise ValueError("Generation references a different test question or answer")
            kind = row["condition"]
            baseline = kind in {"zero", "icl_a", "icl_b", "first", "cot"}
            prompt_kind = kind if baseline else "zero"
            if row["prompt"] != by_id[row["problem_id"]]["prompts"][prompt_kind]:
                raise ValueError("Generation used a different prompt")
            parameters = {"layer": None, "alpha": 0.0, "positions": "prefill"} if baseline else {
                k: selection["chosen"][k] for k in ["layer", "alpha", "positions"]}
            if any(row[k] != value for k, value in parameters.items()):
                raise ValueError("Generation used different intervention parameters")
        rows.extend(test)
    conditions += PREFIX
    rows, _ = audit.select_rows(rows, prepared, "test", conditions)
    packet = audit.make_packet(rows, prepared, "test", conditions)
    combined = ("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode()
    sources = {label: {name: file_hash(directory / name) for name in
                      ["manifest.json", "generations.jsonl", "test_lock.json", "complete.json"]}
               for label, directory in [("primary", parent), ("prefix", prefix)]}
    provenance = {"sources_sha256": sources, "conditions": conditions,
                  "prepared_sha256": manifest["prepared_sha256"],
                  "combined_generations_sha256": sha256(combined).hexdigest(),
                  "audit_code_sha256": declaration["audit_code_sha256"],
                  "selection": selection["chosen"], "packet_sha256": audit.digest(packet)}
    output.mkdir(parents=True, exist_ok=True)
    for name, content in {"manifest.json": (parent / "manifest.json").read_bytes(),
                          "prepared.json": (parent / "prepared.json").read_bytes(),
                          "generations.jsonl": combined, "provenance.json": encoded(provenance),
                          "review-packet.json": encoded(packet)}.items():
        write_fixed(output / name, content)
    return len(packet["items"])


def compare(rows, prepared, conditions, packet, annotations, samples=10000):
    result = audit.score(rows, prepared, "test", conditions, packet, annotations, samples)
    selected, questions = audit.select_rows(rows, prepared, "test", conditions)
    answers = {r["response_id"]: Fraction(r["stated_answer"]) if r["stated_answer"] is not None else None
               for r in annotations["answers"]}
    groups = {metric: defaultdict(list) for metric in ["primary", "audited"]}
    for row in selected:
        correct = row["correct"]
        if not row["parseable"]:
            answer = answers[audit.response_item(row, questions)["response_id"]]
            correct = answer is not None and answer == Fraction(row["answer"])
        groups["primary"][row["condition"]].append({**row, "correct": row["completed_correct"]})
        groups["audited"][row["condition"]].append({**row, "correct": correct and not row["truncated"]})
    comparisons = {}
    for metric, by_condition in groups.items():
        comparisons[metric] = {
            f"{left}_vs_{right}": paired_difference(by_condition[left], by_condition[right], samples, 907)
            for left in ["steered", "icl_a", "icl_b"]
            for right in conditions if right != left and (left == "steered" or right == "zero")}
    result["comparisons"] = comparisons
    result["bootstrap_samples"] = samples
    result["bootstrap_seed"] = 907
    result["comparison_code_sha256"] = file_hash(Path(__file__))
    return result


def report(run, annotations):
    provenance = read(run / "provenance.json")
    if (file_hash(run / "generations.jsonl") != provenance["combined_generations_sha256"]
            or file_hash(run / "prepared.json") != provenance["prepared_sha256"]
            or file_hash(Path(audit.__file__)) != provenance["audit_code_sha256"]):
        raise ValueError("Combined source data or audit code changed")
    packet = read(run / "review-packet.json")
    if audit.digest(packet) != provenance["packet_sha256"]:
        raise ValueError("Review packet changed")
    rows = [json.loads(line) for line in (run / "generations.jsonl").read_text().splitlines()]
    result = compare(rows, read(run / "prepared.json"), provenance["conditions"], packet, read(annotations))
    result["provenance"] = provenance
    (run / "comparison.json").write_bytes(encoded(result))
    lines = ["# GSM8K held-out comparisons", "", result["limitations"], "",
             "| Condition | n | Primary correct | Audited correct | Unparsed | Truncated |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition in provenance["conditions"]:
        s = result["summary"][condition]
        lines.append(f"| {condition} | {s['n']} | {s['primary_completed_correct']} | "
                     f"{s['audited_completed_correct']} | {s['unparsed']} | {s['truncated']} |")
    for metric, comparisons in result["comparisons"].items():
        lines += ["", f"## {metric.capitalize()} paired differences", "",
                  "| Comparison | Gain | 95% interval | Wins | Losses |", "|---|---:|---:|---:|---:|"]
        for name, value in comparisons.items():
            low, high = value["ci95"]
            lines.append(f"| {name} | {value['gain']:+.2%} | [{low:+.2%}, {high:+.2%}] | "
                         f"{value['wins']} | {value['losses']} |")
    (run / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "report"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.prefix is None or args.output is None:
            parser.error("prepare requires --prefix and --output")
        count = prepare(args.run, args.prefix, args.output)
        print(f"Prepared a blinded packet with {count} unique responses at {args.output / 'review-packet.json'}")
    else:
        if args.annotations is None:
            parser.error("report requires --annotations")
        report(args.run, args.annotations)
        print(f"Wrote {args.run / 'comparison.md'}")


if __name__ == "__main__":
    main()

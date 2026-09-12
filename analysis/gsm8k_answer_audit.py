"""Blind review packets for unparsed GSM8K answers; never changes run selection."""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

from icl_steering.report import paired_difference


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_rows(rows, prepared, split, conditions):
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Require distinct, explicit conditions")
    questions = {r["problem_id"]: r["question"] for r in prepared["splits"][split]}
    selected = [r for r in rows if r["split"] == split and r["condition"] in conditions]
    for condition in conditions:
        ids = [r["problem_id"] for r in selected if r["condition"] == condition]
        if len(ids) != len(set(ids)) or set(ids) != set(questions):
            raise ValueError(f"Incomplete or duplicate condition: {condition}")
    return sorted(selected, key=lambda r: (r["condition"], r["problem_id"])), questions


def response_item(row, questions):
    item = {"question": questions[row["problem_id"]], "response": row["solution_text"]}
    return {"response_id": digest(item), **item}


def make_packet(rows, prepared, split, conditions, seed=907):
    selected, questions = select_rows(rows, prepared, split, conditions)
    unique = {}
    for row in selected:
        if not row["parseable"]:
            item = response_item(row, questions)
            unique[item["response_id"]] = item
    items = [unique[k] for k in sorted(unique)]
    random.Random(seed).shuffle(items)
    # No condition names, problem IDs, gold labels, or grades are exposed.
    return {"version": 1, "source_sha256": digest(selected), "items": items}


def score(rows, prepared, split, conditions, packet, annotations, samples=10000):
    expected = make_packet(rows, prepared, split, conditions)
    if packet != expected or annotations["packet_sha256"] != digest(packet):
        raise ValueError("Review packet or source data changed")
    answers = {}
    for item in annotations["answers"]:
        key = item["response_id"]
        if key in answers:
            raise ValueError("Duplicate annotation")
        rationale = item.get("rationale")
        if item.get("reviewed") is not True or not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("Every annotation requires explicit review and a rationale")
        value = item["stated_answer"]
        if value is not None and not isinstance(value, str):
            raise ValueError("Use a numeric string or null for the stated answer")
        answers[key] = Fraction(value) if value is not None else None
    if set(answers) != {item["response_id"] for item in packet["items"]}:
        raise ValueError("Missing or unexpected annotations")
    selected, questions = select_rows(rows, prepared, split, conditions)
    groups = defaultdict(list)
    for row in selected:
        correct = row["correct"]
        if not row["parseable"]:
            value = answers[response_item(row, questions)["response_id"]]
            correct = value is not None and value == Fraction(row["answer"])
        groups[row["condition"]].append({**row, "primary_correct": row["completed_correct"],
                                         "correct": correct and not row["truncated"]})
    summary = {}
    for condition, values in groups.items():
        n = len(values)
        result = {"n": n, "primary_completed_correct": sum(r["primary_correct"] for r in values),
                  "audited_completed_correct": sum(r["correct"] for r in values),
                  "unparsed": sum(not r["parseable"] for r in values),
                  "truncated": sum(r["truncated"] for r in values)}
        controls = ["zero"]
        if condition in {"steered", "icl_a", "icl_b"}:
            controls += [c for c in conditions if c in {"first", "cot", "reverse"} or c.startswith("random_")]
        result["paired"] = {c: paired_difference(values, groups[c], samples, 907)
                            for c in controls if c in groups and c != condition}
        summary[condition] = result
    return {"status": "supplementary_answer_audit", "split": split,
            "source_sha256": packet["source_sha256"], "packet_sha256": digest(packet),
            "annotations_sha256": digest(annotations),
            "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "summary": summary, "annotations": annotations["answers"],
            "limitations": "Only unparsed answers are reviewed. Parsed grades are unchanged. "
            "Truncated responses receive no completed-answer credit. This is a single-reviewer "
            "supplementary measure with exploratory, unadjusted paired intervals; it does not "
            "change the primary metric, direction, or selection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export", "score"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / "generations.jsonl").read_text().splitlines()]
    prepared_bytes = (args.run / "prepared.json").read_bytes()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if hashlib.sha256(prepared_bytes).hexdigest() != manifest["prepared_sha256"]:
        raise ValueError("Prepared questions changed from the run manifest")
    prepared = json.loads(prepared_bytes)
    if args.stage == "export":
        result = make_packet(rows, prepared, args.split, args.conditions)
        if args.packet.exists() and json.loads(args.packet.read_text()) != result:
            raise ValueError("Refuse to replace a different review packet")
        target = args.packet
    else:
        if args.annotations is None or args.output is None:
            parser.error("score requires --annotations and --output")
        result = score(rows, prepared, args.split, args.conditions,
                       json.loads(args.packet.read_text()), json.loads(args.annotations.read_text()))
        target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    if args.stage == "export":
        print(f"Wrote {target}; {len(result['items'])} responses; packet SHA-256 {digest(result)}")
    else:
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()

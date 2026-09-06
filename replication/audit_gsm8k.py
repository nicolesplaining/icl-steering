"""Rescore saved zero-shot text without rerunning the model or changing old results."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from replication.gsm8k_protocol import grade_answer


def audit(path):
    record = json.loads(path.read_text())
    rows = record["evaluation"]
    details = []
    for index, row in enumerate(rows):
        previous = row["zero_shot"]
        text = previous["raw_response"]
        old = previous["answer"] != "" and previous["answer"] == row["answer"]
        # This deliberately narrow rescue reproduces the first manual audit.
        boxes = re.findall(r"\\boxed\{\s*([-+]?\d[\d,.]*)\s*\}", text)
        box_or_old = float(boxes[-1].replace(",", "")) == row["answer"] if boxes else old
        parsed = grade_answer(text, row["answer"])
        # The legacy artifact lacks termination metadata; do not manufacture it.
        parsed.pop("completed_correct")
        details.append({"test_index": index, "text_sha256": sha256(text.encode()).hexdigest(),
                        "old_correct": old, "numeric_box_or_old_correct": box_or_old,
                        **parsed})
    base = Path(__file__).parents[1]
    parser_files = ["replication/gsm8k_protocol.py", "src/icl_steering/scoring.py"]
    return {"source_sha256": sha256(path.read_bytes()).hexdigest(),
            "parser_sha256": {name: sha256((base / name).read_bytes()).hexdigest() for name in parser_files},
            "n": len(rows),
            "legacy_correct": sum(r["old_correct"] for r in details),
            "numeric_box_or_old_correct": sum(r["numeric_box_or_old_correct"] for r in details),
            "explicit_answer_correct": sum(r["correct"] for r in details),
            "explicit_answer_parseable": sum(r["parseable"] for r in details),
            "truncation_available": False,
            "limitations": ["Legacy output has no token IDs or finish reasons.",
                            "Matched ICL per-question generations were not saved; cannot rescore the ICL contrast.",
                            "This is a parser audit on previously screened examples, not a new ICL result."],
            "rows": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis import gsm8k_answer_audit as audit
from analysis.gsm8k_comparison import PREFIX, encoded, file_hash, prepare, read, report


def fixture(tmp_path):
    parent, prefix, output = (tmp_path / name for name in ["primary", "prefix", "combined"])
    parent.mkdir()
    prefix.mkdir()
    questions = [{"problem_id": f"q{i}", "question": f"Question {i}?", "answer": "17",
                  "prompts": {kind: f"{kind}: Question {i}?" for kind in
                              ["zero", "icl_a", "icl_b", "first", "cot"]}} for i in range(2)]
    (parent / "prepared.json").write_bytes(encoded({"splits": {"test": questions}}))
    (parent / "manifest.json").write_bytes(encoded({
        "config": {"n_test": 2, "random_seeds": [31, 59, 83]},
        "prepared_sha256": file_hash(parent / "prepared.json")}))
    (parent / "directions.pt").write_bytes(b"fixed direction bytes")
    (parent / "selection.json").write_bytes(encoded({
        "eligible": True, "chosen": {"layer": 7, "alpha": 0.5, "positions": "all"}}))
    lock = {"selection_sha256": file_hash(parent / "selection.json"),
            "directions_sha256": file_hash(parent / "directions.pt")}
    (parent / "test_lock.json").write_bytes(encoded(lock))
    (parent / "complete.json").write_bytes(encoded({"status": "locked_test_complete"}))
    (prefix / "manifest.json").write_bytes(encoded({"request": {
        "inputs_sha256": {"parent_manifest": file_hash(parent / "manifest.json"),
                          "parent_directions": lock["directions_sha256"]},
        "conditions": PREFIX, "n_test": 2}, "primary_test_lock_present": False}))
    (prefix / "test_lock.json").write_bytes(encoded({
        "parent_test_lock": lock, "supplement_manifest_sha256": file_hash(prefix / "manifest.json")}))
    (prefix / "complete.json").write_bytes(encoded({"status": "prefix_test_complete"}))
    conditions = ["zero", "icl_a", "icl_b", "first", "cot", "steered", "reverse"]
    conditions += ["random_31", "random_59", "random_83"]
    for directory, kinds in [(parent, conditions), (prefix, PREFIX)]:
        rows = []
        for kind in kinds:
            for i in range(2):
                unparsed = i == 0 and kind in {"steered", "prefix_length_filler"}
                correct = (kind == "steered" and i == 1) or (i == 0 and not unparsed)
                baseline = kind in questions[i]["prompts"]
                rows.append({"split": "test", "condition": kind, "problem_id": f"q{i}",
                             "prompt": questions[i]["prompts"][kind if baseline else "zero"],
                             "layer": None if baseline else 7, "alpha": 0.0 if baseline else 0.5,
                             "positions": "prefill" if baseline else "all",
                             "answer": "17", "parseable": not unparsed,
                             "correct": correct, "completed_correct": correct,
                             "solution_text": "There are 17 apples." if unparsed else f"Answer: {17 if correct else 0}",
                             "truncated": unparsed and kind == "prefix_length_filler"})
        (directory / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return parent, prefix, output


def test_combined_report_preserves_primary_scores_and_truncation(tmp_path):
    parent, prefix, output = fixture(tmp_path)
    assert prepare(parent, prefix, output) == 1
    assert prepare(parent, prefix, output) == 1
    packet = read(output / "review-packet.json")
    annotations = {"packet_sha256": audit.digest(packet), "answers": [
        {"response_id": packet["items"][0]["response_id"], "stated_answer": "17",
         "reviewed": True, "rationale": "The final sentence states 17 apples."}]}
    path = output / "annotations.json"
    path.write_bytes(encoded(annotations))
    report(output, path)
    result = read(output / "comparison.json")
    assert result["summary"]["steered"]["primary_completed_correct"] == 1
    assert result["summary"]["steered"]["audited_completed_correct"] == 2
    assert result["summary"]["prefix_length_filler"]["audited_completed_correct"] == 0
    assert result["comparisons"]["primary"]["steered_vs_zero"]["gain"] == 0
    assert result["comparisons"]["audited"]["steered_vs_zero"]["gain"] == 0.5
    contrast = result["comparisons"]["audited"]["steered_vs_prefix_length_filler"]
    assert contrast == {"gain": 1.0, "ci95": [1.0, 1.0], "wins": 2, "losses": 0}
    assert len(result["comparisons"]["audited"]) == 14
    assert (output / "comparison.md").exists()
    with (output / "generations.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="source data"):
        report(output, path)


@pytest.mark.parametrize("mutation", ["incomplete", "selection", "lock", "missing", "duplicate", "gold",
                                     "parameters", "prompt"])
def test_merge_rejects_invalid_or_incomplete_evidence(tmp_path, mutation):
    parent, prefix, output = fixture(tmp_path)
    if mutation == "incomplete":
        (prefix / "complete.json").write_bytes(encoded({"status": "running"}))
    elif mutation == "selection":
        selection = read(parent / "selection.json")
        selection["chosen"]["alpha"] = 1.0
        (parent / "selection.json").write_bytes(encoded(selection))
    elif mutation == "lock":
        (prefix / "test_lock.json").write_bytes(encoded({"changed": True}))
    else:
        path = prefix / "generations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "missing": rows.pop()
        if mutation == "duplicate": rows.append(rows[0])
        if mutation == "gold": rows[0]["answer"] = "18"
        if mutation == "parameters": rows[0]["alpha"] = 1.0
        if mutation == "prompt": rows[0]["prompt"] += " First,"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        prepare(parent, prefix, output)
    assert not output.exists()

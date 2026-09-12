import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.gsm8k_answer_audit import digest, make_packet, score


def fixture():
    prepared = {"splits": {"test": [{"problem_id": "hidden-query-id", "question": "How many apples?"}]}}
    rows = [{"split": "test", "condition": c, "problem_id": "hidden-query-id",
             "solution_text": "Therefore, there are 17 apples.", "parseable": False,
             "answer": "17", "correct": False, "completed_correct": False,
             "truncated": c == "steered"} for c in ["zero", "steered"]]
    return rows, prepared, ["zero", "steered"]


def reviewed(packet):
    return {"packet_sha256": digest(packet), "answers": [
        {"response_id": item["response_id"], "stated_answer": "17", "reviewed": True,
         "rationale": "The concluding sentence states 17 apples."} for item in packet["items"]]}


def test_packet_hides_labels_conditions_and_deduplicates_identical_responses():
    rows, prepared, conditions = fixture()
    packet = make_packet(rows, prepared, "test", conditions)
    assert len(packet["items"]) == 1
    assert set(packet["items"][0]) == {"response_id", "question", "response"}
    assert not any(x in json.dumps(packet) for x in ["hidden-query-id", "steered", "zero", '"answer"'])
    assert packet == make_packet(list(reversed(rows)), prepared, "test", conditions)


def test_review_cannot_rescue_a_truncated_answer_or_change_primary_grade():
    rows, prepared, conditions = fixture()
    packet = make_packet(rows, prepared, "test", conditions)
    result = score(rows, prepared, "test", conditions, packet, reviewed(packet), samples=10)
    assert result["summary"]["zero"]["audited_completed_correct"] == 1
    assert result["summary"]["steered"]["audited_completed_correct"] == 0
    assert result["summary"]["zero"]["primary_completed_correct"] == 0


def test_incomplete_conditions_and_source_changes_are_rejected():
    rows, prepared, conditions = fixture()
    with pytest.raises(ValueError, match="Incomplete"):
        make_packet(rows[:1], prepared, "test", conditions)
    packet = make_packet(rows, prepared, "test", conditions)
    changed = copy.deepcopy(rows)
    changed[0]["solution_text"] = "Changed response."
    with pytest.raises(ValueError, match="changed"):
        score(changed, prepared, "test", conditions, packet, reviewed(packet))


def test_parsed_grades_stay_fixed_and_incorrect_reviews_receive_no_credit():
    rows, prepared, conditions = fixture()
    rows[0].update(parseable=True, correct=True, completed_correct=True)
    rows[1]["truncated"] = False
    packet = make_packet(rows, prepared, "test", conditions)
    annotations = reviewed(packet)
    annotations["answers"][0]["stated_answer"] = "18"
    result = score(rows, prepared, "test", conditions, packet, annotations, samples=10)
    assert result["summary"]["zero"]["audited_completed_correct"] == 1
    assert result["summary"]["steered"]["audited_completed_correct"] == 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unreviewed", "numeric_type", "invalid_rationale"])
def test_reviews_must_be_complete_explicit_and_unambiguous(mutation):
    rows, prepared, conditions = fixture()
    packet = make_packet(rows, prepared, "test", conditions)
    annotations = reviewed(packet)
    if mutation == "missing": annotations["answers"] = []
    if mutation == "duplicate": annotations["answers"] *= 2
    if mutation == "unreviewed": annotations["answers"][0]["reviewed"] = False
    if mutation == "numeric_type": annotations["answers"][0]["stated_answer"] = True
    if mutation == "invalid_rationale": annotations["answers"][0]["rationale"] = True
    with pytest.raises(ValueError):
        score(rows, prepared, "test", conditions, packet, annotations)

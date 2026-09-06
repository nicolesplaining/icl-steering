import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("math_screen", Path(__file__).parents[1] / "replication/math_screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)

joint_spec = importlib.util.spec_from_file_location(
    "gsm8k_joint_inference", Path(__file__).parents[1] / "replication/gsm8k_joint_inference.py"
)
joint = importlib.util.module_from_spec(joint_spec)
joint_spec.loader.exec_module(joint)


def test_replication_partition_is_disjoint_and_reproducible():
    train = [{"problem": f"train {i}"} for i in range(20)]
    test = [{"problem": f"test {i}"} for i in range(20)]
    plan = screen.partition(train, test, 8, 5, 71)
    assert plan == screen.partition(train, test, 8, 5, 71)
    assert not set(plan["screen_test_indices"]) & set(plan["reserved_test_indices"])
    assert len(set(plan["screen_test_indices"] + plan["reserved_test_indices"])) == 20
    assert plan["demonstration_train_indices"] == list(range(8))
    assert plan["reserved_train_indices"] == list(range(8, 20))
    test[plan["screen_test_indices"][0]] = train[0]
    with pytest.raises(ValueError, match="overlaps"):
        screen.partition(train, test, 8, 5, 71)


def test_zero_and_manyshot_have_identical_target_instruction():
    templates = {"PROBLEM_PROMPT": "Instruction\n{demo_prompt}Target: {problem}\nWork:",
                 "DEMO_TEMPLATE": "Problem: {problem}\nSolution: {solution}"}
    zero = screen.render_content(templates, "query", [])
    many = screen.render_content(templates, "query", [{"problem": "example", "solution": "answer"}])
    assert zero.endswith("Target: query\nWork:") and many.endswith("Target: query\nWork:")
    assert "answer" not in zero and "answer" in many


class GraderStub:
    @staticmethod
    def last_boxed_only_string(text):
        return "7" if "boxed7" in text else None

    remove_boxed = staticmethod(lambda value: value)
    normalize_final_answer = staticmethod(lambda value: value)
    is_equiv = staticmethod(lambda a, b: a == b)


def test_unfinished_thinking_box_is_separate_from_final_answer():
    unfinished = screen.score_response("thinking boxed7", "7", GraderStub)
    assert unfinished["correct"] and not unfinished["final_only_correct"]
    assert not unfinished["thinking_finished"]
    finished = screen.score_response("thinking</think>answer boxed7", "7", GraderStub)
    assert finished["correct"] and finished["final_only_correct"]


def test_paired_replication_checks_question_identity():
    config = {"seed": 1, "bootstrap_samples": 100}
    rows = [{"problem_id": "a", "correct": True}, {"problem_id": "b", "correct": False}]
    ref = [{"problem_id": "b", "correct": False}, {"problem_id": "a", "correct": False}]
    assert screen.paired(rows, ref, config)["gain"] == .5
    with pytest.raises(ValueError):
        screen.paired(rows, ref[:1], config)
    with pytest.raises(ValueError):
        screen.paired(rows + rows, ref + ref, config)


def test_joint_inference_prompt_and_majority_are_deterministic():
    support = [
        {"question": "one", "raw_response": "The final answer is 1."},
        {"question": "two", "raw_response": "The final answer is 2."},
    ]
    prompt = joint.icl_prompt(support, {"question": "query"}, 2, __import__("random").Random(4))
    assert prompt.endswith("Q: query\nA:")
    assert "The final answer is 1." in prompt and "The final answer is 2." in prompt
    result = joint.majority(["The final answer is 3.", "The final answer is 3.", "The final answer is 4."])
    assert result["answer"] == 3 and result["formatted"]
    supervised = joint.supervised_prompt(support, {"question": "query"}, 2)
    assert supervised.endswith("Q: query\nA:")


def test_joint_inference_can_exclude_query_from_support():
    support = [
        {"question": "query", "raw_response": "The final answer is 1."},
        {"question": "other", "raw_response": "The final answer is 2."},
    ]
    with pytest.raises(ValueError):
        joint.icl_prompt(support, {"question": "query"}, 2, __import__("random").Random(4), exclude_query=True)

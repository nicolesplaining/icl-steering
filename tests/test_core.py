import json
from pathlib import Path

import pytest

from icl_steering.data import make_dataset, demonstrations
from icl_steering.experiment import choose_candidate, choose_family
from icl_steering.prompts import messages
from icl_steering.report import paired_difference
from icl_steering.scoring import grade


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/pilot.json").read_text())


def test_disjoint_deterministic_dataset_and_exact_solutions(config):
    dataset = make_dataset(config)
    assert dataset == make_dataset(config)
    assert len(dataset) == len({p.id for p in dataset})
    for p in dataset:
        assert grade(p.method_a, p.answer)["correct"]
        assert grade(p.method_b, p.answer)["correct"]
        if p.family == "quadratic_sum":
            a, b, c, n = (p.params[k] for k in ("a", "b", "c", "n"))
            assert p.answer == a*n*(n+1)*(2*n+1)//6 + b*n*(n+1)//2 + c*n
        else:
            # Independent companion-matrix exponentiation, without floating roots.
            s, product, k = (p.params[key] for key in ("s", "p", "k"))
            matrix = ((s, -product), (1, 0))
            result = ((1, 0), (0, 1))
            for _ in range(k):
                result = tuple(tuple(sum(result[i][v]*matrix[v][j] for v in range(2))
                                     for j in range(2)) for i in range(2))
            assert result[0][0] + result[1][1] == p.answer


def test_prompts_share_examples_and_never_include_target_solution(config):
    dataset = make_dataset(config)
    p = next(p for p in dataset if p.split == "test")
    demos = demonstrations(p, dataset, config["shots"], config["seed"])
    assert all(d.split == "demonstration" and d.id != p.id for d in demos)
    assert demos == demonstrations(p, dataset, config["shots"], config["seed"])
    for condition in ("zero", "icl_a", "icl_b"):
        prompt = messages(p, demos, condition)[0]["content"]
        assert prompt.endswith(f"Problem: {p.question}\nSolution:")
        assert p.method_a not in prompt and p.method_b not in prompt
        assert prompt.count("Problem:") == (5 if condition.startswith("icl_") else 1)


@pytest.mark.parametrize("text,answer,correct", [
    (r"\boxed{12} then \boxed{13}", 13, True),
    (r"\boxed{\frac{26}{2}}", 13, True),
    (r"\boxed{1\,234}", 1234, True),
    ("Answer: -1,234.", -1234, True),
    (r"\boxed{0.5}", 1, False),
    (r"\boxed{1/0}", 1, False),
    (r"\boxed{__import__('os').system('false')}", 0, False),
    (r"\boxed{2+3}", 5, False),
    (r"\boxed{13", 13, False),
])
def test_conservative_grading(text, answer, correct):
    assert grade(text, answer)["correct"] is correct


def test_validation_ties_and_paired_ids():
    candidates = [dict(accuracy=.5, strength=a, layer=l) for a, l in [(2, 1), (1, 5), (1, 3)]]
    assert choose_candidate(candidates) == candidates[-1]
    left = [{"problem_id": "a", "correct": True}, {"problem_id": "b", "correct": False}]
    right = [{"problem_id": "b", "correct": False}, {"problem_id": "a", "correct": False}]
    assert paired_difference(left, right)["gain"] == .5
    with pytest.raises(ValueError):
        paired_difference(left, right[:1])


def test_screen_cannot_select_on_test_results():
    rows = [dict(split="screen", family="f", condition=c, correct=False,
                 parseable=True, truncated=False, generated_tokens=1, method_signature="unclear")
            for c in ("zero", "icl_a")]
    rows.append({**rows[1], "split": "test", "correct": True})
    assert choose_family(rows, ["f"], .01)["selected_family"] is None

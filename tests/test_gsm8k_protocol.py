import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from replication.gsm8k_protocol import (grade_answer, matched_prompt, parse_answer,
                                      partition, select_candidate)
from replication.gsm8k_steering import Backend, Runner, geometry


@pytest.mark.parametrize("text, answer", [
    (r"The final answer is \(\boxed{18}\) dollars.", "18"),
    (r"The total is \(\boxed{3}\).", "3"),
    (r"Thus \boxed{\frac{26}{2}}.", "13"),
    ("The final answer is [366].", "366"),
    ("The final answer is [80 sheep].", "80"),
    ("The final answer is [The profit would be $125].", "125"),
    ("The answer is -18.", "-18"),
    ("Answer: .5", "1/2"),
    ("The answer is $20,800.", "20800"),
    ("The answer is 18.\n\nQ: A new problem\nA: The answer is 99.", "18"),
    ("The answer is [80].\n\nGiven the following question, put it in \\boxed{}.", "80"),
    (r"\boxed{5} corrected to \boxed{7}", "7"),
])
def test_parser_recognizes_explicit_answers_without_gold(text, answer):
    assert parse_answer(text)["parsed_answer"] == answer
    assert grade_answer(text, answer)["correct"]
    # Changing gold never changes extraction.
    assert grade_answer(text, 999)["parsed_answer"] == answer


@pytest.mark.parametrize("text", [
    "2 plus 3 equals 5; now continue.",
    "The answer is 2 + 3.",
    "The answer is 2 or 3.",
    "The answer is sqrt(4).",
    r"\boxed{2+3}",
    r"\boxed{1/0}",
    r"\boxed{__import__('os').system('false')}",
    r"The answer is \boxed{13",
])
def test_parser_does_not_guess_an_intermediate_or_execute_code(text):
    assert not parse_answer(text)["parseable"]


def test_budget_censored_answer_is_not_completed_success():
    result = grade_answer(r"The answer is \boxed{18}", 18, truncated=True)
    assert result["correct"] and not result["completed_correct"]


def test_contrast_changes_only_demonstrations_and_keeps_last_token():
    demos = [{"question": "support", "raw_response": "The answer is 7."}]
    query = "How many?"
    zero, icl = matched_prompt(query), matched_prompt(query, demos)
    assert icl.replace("Q: support\nA:The answer is 7.\n\n", "") == zero
    assert zero.endswith(f"Q: {query}\nA:") and icl.endswith(f"Q: {query}\nA:")
    with pytest.raises(ValueError, match="Query"):
        matched_prompt("support", demos)


def test_splits_reserve_old_test_screen_and_check_text_overlap():
    train = [{"question": f"train {i}"} for i in range(80)]
    test = [{"question": f"test {i}"} for i in range(80)]
    config = dict(reserved_train=8, reserved_test=12, seed=4, n_extract=12,
                  n_validation=16, n_test=20)
    plan = partition(train, test, ["train 0"], config)
    assert not set(plan["extract"]) & set(plan["validation"])
    assert min(plan["test"]) >= 12 and min(plan["extract"]) >= 8
    assert plan == partition(train, test, ["train 0"], config)
    test[plan["test"][0]]["question"] = train[plan["extract"][0]]["question"]
    with pytest.raises(ValueError, match="Duplicate"):
        partition(train, test, ["train 0"], config)


def test_constant_shift_is_one_dimensional_even_with_zero_centered_variance():
    zero = torch.zeros(8, 4)
    shift = torch.tensor([1., 2., 3., 4.]).expand_as(zero)
    _, stat = geometry(shift, zero)
    assert stat["mean_shift_fraction"] == pytest.approx(1)
    assert stat["energy_along_mean"] == pytest.approx(1)
    assert stat["centered_rank1_energy"] == 0


def test_selection_penalizes_unfinished_answers_and_breaks_ties():
    candidates = [
        dict(layer=7, alpha=.5, positions="prefill", completed_accuracy=.6, accuracy=.9),
        dict(layer=13, alpha=.25, positions="prefill", completed_accuracy=.7, accuracy=.7),
        dict(layer=7, alpha=.25, positions="prefill", completed_accuracy=.7, accuracy=.7),
    ]
    choice = select_candidate(candidates, .65, .03)
    assert choice["eligible"] and choice["chosen"] == candidates[-1]


def test_model_records_preserve_text_and_mark_length_censoring():
    from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, prompts, **kwargs):
            width = max(map(len, prompts))
            return BatchEncoding({"input_ids": torch.tensor([[0]*(width-len(p))+p for p in prompts]),
                "attention_mask": torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts])})

        def decode(self, tokens, **kwargs):
            return " ".join(map(str, tokens))

        def batch_decode(self, tokens, **kwargs):
            return [self.decode(row.tolist(), **kwargs) for row in tokens]

    backend = Backend.__new__(Backend)
    backend.tokenizer = Tokenizer()
    torch.manual_seed(9)
    backend.model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, eos_token_id=None,
        pad_token_id=0, attention_dropout=0)).eval()
    backend.model.requires_grad_(False)
    backend.blocks = backend.model.model.layers
    backend.max_prompt_tokens = 20
    # Use tokens outside the vocabulary as stop/pad markers in the output
    # bookkeeping so this short regression is deterministically length-limited.
    backend.tokenizer.pad_token_id = 33
    config = dict(max_new_tokens=3, max_model_len=20)
    prompts = [[1, 2], [3, 4, 5]]
    original = {key: val.clone() for key, val in backend.model.state_dict().items()}
    base = backend.records(prompts, config)
    identity = backend.records(prompts, config, direction=torch.ones(32), layer=0, alpha=0)
    assert [r["token_ids"] for r in base] == [r["token_ids"] for r in identity]
    assert all(r["truncated"] and r["generated_tokens"] == 3 for r in base)
    assert all(torch.equal(original[key], val) for key, val in backend.model.state_dict().items())
    # Extraction positions must match generation prefill under left padding.
    captured = backend.activations(prompts, [0], 2)[0]
    prefill = []
    handle = backend.blocks[0].register_forward_hook(
        lambda m, i, o: prefill.append((o[0] if isinstance(o, tuple) else o)[:, -1].detach()))
    backend.records(prompts, {**config, "max_new_tokens": 1})
    handle.remove()
    torch.testing.assert_close(captured, prefill[0])


def test_test_stage_refuses_a_changed_selection_before_any_generation(tmp_path):
    runner = Runner({}, {"splits": {}}, tmp_path)
    (tmp_path / "selection.json").write_text("{}")
    (tmp_path / "directions.pt").write_bytes(b"identity")
    (tmp_path / "test_lock.json").write_text(json.dumps({"selection_sha256": "old"}))
    with pytest.raises(ValueError, match="Selection changed"):
        runner.test({"eligible": True})


def test_failed_validation_does_not_open_test_data(tmp_path):
    runner = Runner({}, {"splits": {}}, tmp_path)
    runner.test({"eligible": False})
    assert not (tmp_path / "test_lock.json").exists()


def test_staged_run_persists_selection_before_test_and_resumes_without_inference(tmp_path):
    from replication.gsm8k_steering import save

    config = dict(batch_size=2, layers=[0], strengths=[.25], positions=["prefill"],
                  random_seeds=[1, 2, 3], max_truncation_rate=.05,
                  min_icl_gain=.05, min_steering_gain=.03, bootstrap_samples=100, seed=7)
    data = {"splits": {}}
    for split in ("extract", "validation", "test"):
        data["splits"][split] = [
            {"problem_id": f"{split}:{i}", "answer": "1", "prompts": {
                kind: f"{split}/{i}/{kind}" for kind in ("zero", "icl_a", "icl_b", "first", "cot")
            }} for i in range(2)]

    class BackendFixture:
        def activations(self, prompts, layers, batch_size):
            value = 1. if "icl" in prompts[0] else 0.
            return {0: torch.full((len(prompts), 4), value)}

        def records(self, prompts, config, direction, layer, alpha, positions):
            if prompts[0].startswith("test"):
                assert (tmp_path / "selection.json").exists()
                assert (tmp_path / "test_lock.json").exists()
            return [{"text": f"The answer is {int('icl' in p or (direction is not None and direction.sum() > 0))}.",
                     "truncated": False, "generated_tokens": 5} for p in prompts]

    runner = Runner(config, data, tmp_path)
    runner.model = BackendFixture()
    selection = runner.validation()
    assert selection["eligible"]
    assert all(r["split"] != "test" for r in runner.rows)
    runner.test(selection)
    runner.report()
    assert (tmp_path / "complete.json").exists()
    count = len(runner.rows)
    rerun = Runner(config, data, tmp_path)
    # No GPU backend exists in this fixture. A resume must use saved rows.
    rerun.test(rerun.validation())
    assert len(rerun.rows) == count
    before = (tmp_path / "test_lock.json").read_text()
    save(tmp_path / "selection.json", {"changed": True})
    with pytest.raises(ValueError, match="Selection changed"):
        rerun.test(selection)
    assert (tmp_path / "test_lock.json").read_text() == before

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis import conditional_shift as shift
from analysis import gsm8k_conditional as experiment
from icl_steering.model import steering_hook


def config():
    return {"layers": [7, 13], "strengths": [0.5, 1.0], "positions": ["prefill", "all"],
            "seed": 1301, "random_seeds": [31, 59, 83], "n_validation": 64, "n_test": 4,
            "reserved_train": 2, "reserved_test": 2, "min_icl_gain": 0.05,
            "min_steering_gain": 0.03, "max_truncation_rate": 0.05, "batch_size": 4}


def training():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(128, 3))
    transform = np.array([[0., 3., 0.], [0., 0., 4.], [2., 0., 0.]])
    delta = x @ transform + np.array([5., 6., 7.])
    return x, delta, transform


def test_fit_recovers_unseen_linear_shift_and_permuting_targets_breaks_it():
    x, delta, transform = training()
    query = np.random.default_rng(43).normal(size=(64, 3))
    target = query @ transform + np.array([5., 6., 7.])
    real = shift.fit(x, delta)
    shuffled = shift.fit(x, delta[np.random.default_rng(1301).permutation(len(delta))])
    error = lambda value: np.square(value-target).sum()
    mean_error = error(shift.predict(real, query, "mean"))
    assert error(shift.predict(real, query)) < 0.002*mean_error
    assert error(shift.predict(shuffled, query)) > 0.8*mean_error
    np.testing.assert_allclose(shuffled["mean"], real["mean"], atol=1e-12)
    assert real["relative_solver_residual"] < 1e-8


def test_constant_targets_reduce_to_mean_and_degenerate_inputs_are_safe():
    x, _, _ = training()
    delta = np.broadcast_to([2., 4., 6.], x.shape)
    fitted = shift.fit(x, delta)
    np.testing.assert_allclose(shift.predict(fitted, x), delta)
    fitted = shift.fit(np.ones_like(x), delta)
    np.testing.assert_allclose(shift.predict(fitted, x), delta)


def test_predictions_do_not_mix_queries_or_depend_on_batch_composition():
    x, delta, _ = training()
    fitted = shift.fit(x, delta)
    before = shift.predict(fitted, x[:4])
    changed = x[:4].copy()
    changed[1:] += 100
    np.testing.assert_allclose(shift.predict(fitted, changed)[0], before[0])
    np.testing.assert_allclose(shift.predict(fitted, x[:4][::-1]), before[::-1])
    np.testing.assert_allclose(shift.predict(fitted, x[:1]), before[:1])


def test_scalar_predictor_recovers_known_rescaling():
    x, _, _ = training()
    fitted = shift.fit(x, -0.25*x+3)
    np.testing.assert_allclose(shift.predict(fitted, x, "scalar"), -0.25*x+3, atol=1e-12)


@pytest.mark.parametrize("x,y", [(np.zeros((1, 3)), np.zeros((1, 3))),
                                (np.zeros((3, 3)), np.zeros((3, 4))),
                                (np.full((3, 3), np.nan), np.zeros((3, 3)))])
def test_bad_fit_inputs_fail(x, y):
    with pytest.raises(ValueError):
        shift.fit(x, y)


def test_control_norms_are_matched_separately_for_every_question():
    x, delta, _ = training()
    real = shift.fit(x, delta)
    maps = {(7, "real"): real, (7, "permuted"): shift.fit(x, delta[::-1]),
            (7, "rotated"): shift.fit(x, delta+2)}
    spec = {"kind": "ridge", "layer": 7, "alpha": 0.5, "positions": "all"}
    target = experiment.directions(spec, x[:4], maps, real["mean"])
    assert np.linalg.norm(target, axis=1).ptp() > 0.1
    for kind in ["mean_norm", "scalar", "permuted", "rotated", "random"]:
        control = experiment.directions({**spec, "kind": kind, "random_seed": 31}, x[:4], maps, real["mean"])
        np.testing.assert_allclose(np.linalg.norm(control, axis=1), np.linalg.norm(target, axis=1))
    np.testing.assert_allclose(experiment.directions({**spec, "kind": "reverse"}, x[:4], maps, real["mean"]), -target)
    mean = experiment.directions({**spec, "kind": "mean"}, x[:4], maps, real["mean"])
    np.testing.assert_allclose(mean, np.broadcast_to(real["mean"], mean.shape))


def test_hook_applies_distinct_question_vectors_and_respects_scope():
    block = torch.nn.Identity()
    values = torch.zeros(2, 3, 4)
    vectors = torch.tensor([[1., 2., 3., 4.], [9., 8., 7., 6.]])
    with steering_hook(block, vectors, 0.5, "prefill"):
        output = block(values)
        torch.testing.assert_close(output[:, -1], vectors*0.5)
        torch.testing.assert_close(output[:, :-1], values[:, :-1])
        torch.testing.assert_close(block(values), values)
    with steering_hook(block, vectors, 0.5, "all"):
        block(values)
        torch.testing.assert_close(block(values[:, :1])[:, -1], vectors*0.5)
    assert not block._forward_hooks


def test_fresh_partition_excludes_old_ids_and_text_duplicates_and_ignores_gold():
    tables = {s: [{"question": f"{s} question {i}", "answer": str(i)} for i in range(40)]
              for s in ["train", "test"]}
    old = {"splits": {"extract": [{"problem_id": "gsm8k:train:4", "question": "train question 4"}],
                      "validation": [{"problem_id": "gsm8k:train:5", "question": "train question 5"}],
                      "test": [{"problem_id": "gsm8k:test:4", "question": "test question 4"}]},
           "banks": {"icl_a": [{"question": "train question 2"}], "icl_b": []}}
    tables["test"][8]["question"] = " train   question  4 "
    tables["test"][9]["question"] = "train question 0"
    tables["train"][8]["question"] = "train question 2"
    cfg = {**config(), "n_validation": 30, "n_test": 30}
    plan = shift.fresh_ids(tables, old, cfg)
    assert not set(plan["validation"]) & {0, 1, 2, 4, 5, 8}
    assert not set(plan["test"]) & {0, 1, 4, 8, 9}
    altered = copy.deepcopy(tables)
    for values in altered.values():
        for row in values:
            row["answer"] = "changed gold"
    assert plan == shift.fresh_ids(altered, old, cfg)
    with pytest.raises(ValueError, match="Insufficient"):
        shift.fresh_ids(tables, old, {**cfg, "n_test": 40})


def summary():
    values = {k: {"audited_completed_correct": 54, "truncated": 0} for k in experiment.specs(config())}
    for k, n in {"zero": 50, "icl_a": 60, "icl_b": 59, "first": 56, "cot": 56,
                 "ridge_l7_a0.5_prefill": 58}.items():
        values[k]["audited_completed_correct"] = n
    return values


def test_selection_uses_audited_accuracy_and_declared_ties():
    values = summary()
    values["ridge_l13_a1_all"]["audited_completed_correct"] = 58
    values["ridge_l13_a1_all"]["primary_completed_correct"] = 64
    selected = shift.choose(values, config())
    assert selected["eligible"]
    assert selected["chosen"]["condition"] == "ridge_l7_a0.5_prefill"
    assert len(experiment.specs(config())) == 22
    assert len(experiment.specs(config(), selected)) == 16


def test_failed_best_candidate_is_not_replaced_by_an_eligible_runner_up():
    values = summary()
    values["ridge_l13_a1_all"]["audited_completed_correct"] = 60
    values["mean_l13_a1_all"]["audited_completed_correct"] = 60
    selected = shift.choose(values, config())
    assert selected["chosen"]["condition"] == "ridge_l13_a1_all"
    assert not selected["eligible"]
    assert not selected["checks"]["beats_matched_mean"]


def test_resume_reuses_full_original_batch_and_keeps_vector_question_alignment(tmp_path, monkeypatch):
    x, delta, _ = training()
    real = shift.fit(x, delta)
    np.savez(tmp_path / "maps.npz", legacy_mean=real["mean"],
             **{f"7/real/{k}": v for k, v in real.items()})
    data = {"splits": {"validation": [{"problem_id": f"q{i}", "question": f"q{i}", "answer": "8",
                                       "prompts": {"zero": f"prompt {i}"}} for i in range(4)]}}
    seen = []

    class Backend:
        def records(self, prompts, config, direction, layer, alpha, positions):
            seen.append((prompts, direction.numpy().copy()))
            return [{"text": "The answer is 8", "solution_text": "The answer is 8", "truncated": False}
                    for _ in prompts]

    monkeypatch.setattr(experiment.ConditionalRunner, "backend", lambda self: Backend())
    monkeypatch.setattr(experiment.ConditionalRunner, "query_activations", lambda self, split: {7: x[:4]})
    spec = {"kind": "ridge", "layer": 7, "alpha": 0.5, "positions": "all"}
    runner = experiment.ConditionalRunner(config(), data, tmp_path)
    runner.evaluate_spec("validation", "ridge", spec)
    records = experiment.rows_from(tmp_path)
    (tmp_path / "generations.jsonl").write_text(json.dumps(records[0])+"\n")
    resumed = experiment.ConditionalRunner(config(), data, tmp_path)
    resumed.evaluate_spec("validation", "ridge", spec)
    assert seen[0][0] == seen[1][0]
    np.testing.assert_array_equal(seen[0][1], seen[1][1])
    assert experiment.rows_from(tmp_path) == records
    experiment.check_rows(records, data, {"ridge": spec}, "validation")
    changed = copy.deepcopy(records)
    changed[0]["prompt"] = "other prompt"
    with pytest.raises(ValueError, match="declared question"):
        experiment.check_rows(changed, data, {"ridge": spec}, "validation")


def test_frozen_array_change_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "code_hash", lambda: "source")
    for name in ["prepared.json", "maps.npz", "fit.json"]:
        (tmp_path / name).write_text("{}")
    manifest = {"request": {"code_sha256": "source"}, "config": {},
                "prepared_sha256": experiment.file_hash(tmp_path / "prepared.json"),
                "maps_sha256": experiment.file_hash(tmp_path / "maps.npz"),
                "fit_sha256": experiment.file_hash(tmp_path / "fit.json")}
    experiment.save(tmp_path / "manifest.json", manifest)
    assert experiment.verify(tmp_path) == ({}, {})
    (tmp_path / "maps.npz").write_text("changed")
    with pytest.raises(ValueError, match="Declared file changed"):
        experiment.verify(tmp_path)


def selection_fixture(tmp_path, ridge_correct=True):
    cfg = {**config(), "n_validation": 2, "layers": [7], "strengths": [0.5],
           "positions": ["prefill"], "bootstrap_samples": 100}
    data = {"splits": {"validation": [{"problem_id": f"q{i}", "question": f"Question {i}?", "answer": "8",
             "prompts": {kind: f"{kind}: Question {i}?" for kind in experiment.BASELINES}} for i in range(2)]}}
    experiment.save(tmp_path / "prepared.json", data)
    experiment.save(tmp_path / "fit.json", {})
    np.savez(tmp_path / "maps.npz", legacy_mean=np.ones(3))
    np.savez(tmp_path / "validation-queries.npz", values=np.ones((2, 3)))
    experiment.save(tmp_path / "manifest.json", {"request": {"code_sha256": experiment.code_hash()},
        "config": cfg, "prepared_sha256": experiment.file_hash(tmp_path / "prepared.json"),
        "maps_sha256": experiment.file_hash(tmp_path / "maps.npz"),
        "fit_sha256": experiment.file_hash(tmp_path / "fit.json")})
    rows = []
    for condition, spec in experiment.specs(cfg).items():
        for i, q in enumerate(data["splits"]["validation"]):
            correct = condition in {"icl_a", "icl_b"} or (spec["kind"] == "ridge" and ridge_correct)
            correct |= condition in {"first", "cot", "legacy_mean"} and i == 0
            text = f"The answer is {8 if correct else 0}"
            rows.append({"split": "validation", "condition": condition, "problem_id": q["problem_id"],
                         "prompt": q["prompts"][spec.get("prompt_kind", "zero")], "answer": q["answer"],
                         "text": text, "solution_text": text, "truncated": False,
                         **experiment.grade_answer(text, q["answer"]), "intervention": spec,
                         **{k: spec[k] for k in ["layer", "alpha", "positions"]}})
    (tmp_path / "generations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    experiment.save(tmp_path / "validation-complete.json", {"status": "awaiting_blinded_review"})
    experiment.export(tmp_path, "validation", list(experiment.specs(cfg)))
    packet = experiment.read(tmp_path / "validation-review-packet.json")
    annotation = tmp_path / "annotations.json"
    experiment.save(annotation, {"packet_sha256": experiment.audit.digest(packet), "answers": []})
    return annotation


def test_audited_selection_locks_inputs_before_allowing_confirmation(tmp_path):
    annotation = selection_fixture(tmp_path)
    experiment.select(tmp_path, annotation)
    selected = experiment.test_selection(tmp_path)
    assert selected["eligible"]
    assert selected["chosen"]["condition"] == "ridge_l7_a0.5_prefill"
    altered = experiment.read(tmp_path / "validation-annotations.json")
    altered["packet_sha256"] = "changed"
    experiment.save(tmp_path / "validation-annotations.json", altered)
    with pytest.raises(ValueError, match="Selection provenance"):
        experiment.test_selection(tmp_path)


def test_failed_validation_refuses_confirmation(tmp_path):
    annotation = selection_fixture(tmp_path, ridge_correct=False)
    experiment.select(tmp_path, annotation)
    assert not experiment.read(tmp_path / "selection.json")["eligible"]
    with pytest.raises(ValueError, match="Validation gates failed"):
        experiment.test_selection(tmp_path)
    assert not (tmp_path / "test-lock.json").exists()

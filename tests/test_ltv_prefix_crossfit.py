import numpy as np
import pytest

from analysis.ltv_prefix_crossfit import (crossfit, fold_predictions, folds,
                                         permute_targets, predict_ridge, ridge)


def fixture():
    rng = np.random.default_rng(37)
    zero = rng.normal(size=(128, 5, 6))
    target = zero @ rng.normal(size=(6, 6))*.2
    available = np.ones((128, 5), dtype=bool)
    available[::4, -1] = False
    return zero, target, available


def test_question_folds_and_excluded_targets():
    zero, target, available = fixture()
    assignment = folds()
    assert np.bincount(assignment).tolist() == [32]*4
    indices, before, meta = fold_predictions(zero, target, available, assignment, 0)
    changed = target.copy()
    changed[indices] += 1e6
    _, after, _ = fold_predictions(zero, changed, available, assignment, 0)
    assert not set(meta['train_questions']) & set(meta['evaluation_questions'])
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    assert meta['training_rows'] == available[assignment != 0].sum()


def test_ridge_matches_primal_solution():
    rng = np.random.default_rng(14)
    x, delta, query = rng.normal(size=(20, 6)), rng.normal(size=(20, 6)), rng.normal(size=(9, 6))
    fit = ridge(x, delta)
    primal = np.linalg.solve(x.T @ x + 5*np.eye(6), x.T @ delta)
    np.testing.assert_allclose(predict_ridge(fit, query), query @ primal, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        ridge(x, delta*np.nan)


def test_permutation_keeps_position_targets():
    positions = np.repeat(np.arange(5), 7)
    target = np.arange(70).reshape(35, 2)
    changed = permute_targets(target, positions, 1901)
    assert not np.array_equal(changed, target)
    for j in range(5):
        assert sorted(map(tuple, changed[positions == j])) == sorted(map(tuple, target[positions == j]))


def test_complete_crossfit_and_zero_control():
    zero, target, available = fixture()
    result, predictions = crossfit(zero, zero+target, available)
    assert all(p.shape == zero.shape and np.isfinite(p).all() for p in predictions.values())
    assert len(result['folds']) == 4 and result['available_rows'] == 608
    for point in result['summary']['zero']['positions'].values():
        assert point['normalized_squared_error'] == 1
        assert point['target_cosine']['undefined'] == point['n_questions']
    assert result['eligible_for_accuracy_design']
    for point in result['summary']['ridge']['positions'].values():
        assert point['normalized_squared_error'] < .01

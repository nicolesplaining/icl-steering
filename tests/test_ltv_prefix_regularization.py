import numpy as np

from analysis import ltv_prefix_crossfit as cv
from analysis import ltv_prefix_regularization as nested


def data():
    rng = np.random.default_rng(47)
    zero = rng.normal(size=(128, 5, 4))
    target = zero @ rng.normal(size=(4, 4))*.1
    available = np.ones((128, 5), dtype=bool)
    available[::4, -1] = False
    return zero, target, available


def test_outer_targets_cannot_select_penalty():
    zero, target, available = data()
    assignment = cv.folds()
    indices, before, record = nested.select_fold(zero, target, available, assignment, 0)
    changed = target.copy(); changed[indices] += 1e6
    _, after, second = nested.select_fold(zero, changed, available, assignment, 0)
    assert record == second
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    for inner in record['inner_folds']:
        assert not set(indices) & (set(inner['training_questions']) | set(inner['evaluation_questions']))
        assert not set(inner['training_questions']) & set(inner['evaluation_questions'])


def test_whole_grid_matches_direct_solution():
    rng = np.random.default_rng(4)
    x, y, query = rng.normal(size=(30, 6)), rng.normal(size=(30, 6)), rng.normal(size=(11, 6))
    for penalty in nested.penalties(x):
        prediction, residual = nested.fit_predict(x, y, query, penalty, check=True)
        direct = query @ np.linalg.solve(x.T @ x+penalty*np.eye(6), x.T @ y)
        np.testing.assert_allclose(prediction, direct, rtol=1e-10, atol=1e-10)
        assert residual < 1e-8


def test_nested_complete_and_inherited_controls():
    zero, target, available = data()
    _, previous = cv.crossfit(zero, zero+target, available)
    report, predictions = nested.run_arrays(zero, zero+target, available, previous)
    assert len(report['folds']) == 4 and len(predictions) == 10
    for name, value in previous.items():
        key = 'fixed_'+name if name in {'ridge', 'permuted'} else name
        np.testing.assert_array_equal(predictions[key], value)
    assert all(np.isfinite(v).all() for v in predictions.values())
    for fold in report['folds']:
        for name, selected in fold['selected'].items():
            scores = fold['inner_rule_scores'][name]
            expected = min(nested.RULES, key=lambda r:scores[r]['mean_continuation_error'])
            assert selected['rule'] == expected

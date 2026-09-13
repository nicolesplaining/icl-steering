import json

import numpy as np
import pytest

from analysis.ltv_prefix_metrics import (
    POSITIONS, PREDICTORS, paired_token_rows, prefix_token_ids, summarize,
)


def test_exact_prefix_ids_eos_and_ended_rows():
    tokens = prefix_token_ids([7, 8, 99, 4, 4], {99, 100})
    assert tokens == [7, 8]
    assert prefix_token_ids([99, 5], {99}) == []
    assert len(prefix_token_ids(list(range(200)), {999})) == 128
    z, c, available = paired_token_rows([[11], [12, 13]], [[21, 22], [23]], [tokens, []], 1)
    assert z == [[11, 7], [12, 13]] and c == [[21, 22, 7], [23]]
    assert available.tolist() == [True, False]
    assert paired_token_rows([[11]], [[21]], [[]], 0)[2].tolist() == [True]
    with pytest.raises(ValueError):
        prefix_token_ids([True], {99})
    with pytest.raises(ValueError):
        paired_token_rows([[11]], [[21]], [[]], 2)


def fixture():
    zero = np.zeros((3, len(POSITIONS), 2))
    target = np.zeros_like(zero)
    target[..., 0] = 2
    return zero, target, {k: target.copy() for k in PREDICTORS}, np.ones((3, 5), dtype=bool)


def test_known_alignment_error_and_undefined_counts_are_json_safe():
    zero, target, predictions, available = fixture()
    predictions['ltv'] = -target
    predictions['mean'] = np.zeros_like(target)
    result = summarize(zero, target, predictions, available)
    p = result['positions'][0]['predictors']
    assert p['ltv']['measurements']['target_cosine']['median'] == -1
    assert p['ltv']['normalized_squared_error'] == 4
    assert p['scalar']['normalized_squared_error'] == 0
    assert p['mean']['normalized_squared_error'] == 1
    assert p['mean']['zero_predicted_norm'] == 3
    assert p['mean']['measurements']['target_cosine'] == {
        'n': 0, 'undefined': 3, 'p10': None, 'median': None, 'p90': None}
    json.dumps(result, allow_nan=False)
    target[:] = 0
    empty_target = summarize(zero, target, predictions, available)
    assert empty_target['positions'][0]['predictors']['ltv']['normalized_squared_error'] is None
    json.dumps(empty_target, allow_nan=False)


def test_later_positions_compare_the_same_retained_questions():
    zero, target, predictions, available = fixture()
    target[0, 0, 0] = 1000
    target[1:, 0, 0] = 10
    target[1:, 1:, 0] = 12
    available[0, 1:] = False
    result = summarize(zero, target, predictions, available)
    point = result['positions'][1]
    assert point['n_questions'] == 2 and point['excluded_short_prefix'] == 1
    m = point['predictors']['ltv']
    assert m['paired_change_from_prompt']['target_norm']['median'] == 2
    assert m['prompt_normalized_squared_error_same_questions'] == .64
    available[:, 1:] = False
    result = summarize(zero, target, predictions, available)
    assert result['positions'][1]['predictors']['ltv']['normalized_squared_error'] is None
    json.dumps(result, allow_nan=False)


def test_reject_nonfinite_states_missing_predictors_and_inconsistent_availability():
    zero, target, predictions, available = fixture()
    for invalid in [np.nan, np.inf]:
        bad = zero.copy(); bad[0, 0, 0] = invalid
        with pytest.raises(ValueError): summarize(bad, target, predictions, available)
    with pytest.raises(ValueError): summarize(zero, target, {}, available)
    available[0, 1] = False
    with pytest.raises(ValueError): summarize(zero, target, predictions, available)

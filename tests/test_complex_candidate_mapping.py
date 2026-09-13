import numpy as np
import pytest

from analysis import complex_candidate_mapping as mapping
from analysis import complex_candidate_gates as gates


def test_routing_uses_the_correct_target_penalty_and_original_prompt_map():
    x = np.array([[1., 0.], [0., 1.]])
    h = np.array([[2., 3.]])
    paired = {'x': x, **{name+'/weights': (i+1)*np.eye(2)
                        for i, name in enumerate(mapping.pairing.SPECS)}}
    fitted = {'pairing': paired, 'continuation': {
        'real/x': x, 'real/weights': paired['steered/weights'], 'real/mean': np.array([1., 2.]),
        'real/x_mean': np.array([.5, 1.]), 'real/scalar': np.array(.25),
        'position_mean': np.array([[i, i+1.] for i in range(5)]),
        'position_bias': np.array([[i+2., i+3.] for i in range(5)]),
        'position_scalar': np.array([0., .1, .2, .3, .4])},
        'original': {'real/x': x, 'real/weights': 9*np.eye(2), 'real/mean': np.zeros(2)}}
    for i, name in enumerate(mapping.pairing.SPECS):
        np.testing.assert_array_equal(mapping.predict(fitted, h, name, 128), (i+1)*h)
    np.testing.assert_array_equal(mapping.predict(fitted, h, 'regularized_prefill', 0), h)
    np.testing.assert_array_equal(mapping.predict(fitted, h, 'prefill', 0), 9*h)
    np.testing.assert_array_equal(mapping.predict(fitted, h, 'mean', 7), [[1., 2.]])
    np.testing.assert_array_equal(mapping.predict(fitted, h, 'scalar', 7), [[1.375, 2.5]])
    np.testing.assert_array_equal(mapping.predict(fitted, h, 'position_mean', 20), [[2.5, 3.5]])
    np.testing.assert_array_equal(mapping.predict(fitted, h, 'position_scalar', 20), [[5., 6.25]])


def test_scope_and_archive_guards_reject_wrong_conditions_and_modified_files(tmp_path):
    assert {k for k in gates.NEW if mapping.scope(k) == 'prefill'} == {'prefill', 'regularized_prefill'}
    for name in ['prefill', 'regularized_prefill']:
        with pytest.raises(ValueError, match='after the prompt'):
            mapping.predict({}, np.zeros((1, 2)), name, 1)
    with pytest.raises(ValueError, match='Unknown'):
        mapping.scope('reverse')
    with pytest.raises(ValueError, match='three declared'):
        mapping.load_frozen({})
    path = tmp_path/'maps.npz'; path.write_bytes(b'changed map')
    with pytest.raises(ValueError, match='Frozen map changed'):
        mapping.load_frozen({k: path for k in mapping.MAP_HASHES})

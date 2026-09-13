import numpy as np
import pytest

from analysis import ltv_pairing_mapping as mapping
from analysis.ltv_mapping import fit


def test_shared_factorization_and_paired_residual_at_both_penalties():
    rng = np.random.default_rng(42)
    x, d = rng.normal(size=(20, 9)), rng.normal(size=(20, 9))
    positions = np.repeat(np.arange(5), 4)
    fitted = mapping.fit_states(x, d, positions)
    targets = mapping.targets(x, d, positions)
    for j in range(5):
        np.testing.assert_allclose(targets['permuted'][positions == j].mean(0), targets['shared'][positions == j][0])
    for real, shared, multiplier in [('steered', 'shared_low', .1), ('real_high', 'shared_high', 1.)]:
        penalty = multiplier*np.square(x).sum()/len(x)
        residual = fit(x, d-targets['shared'], penalty)['weights']
        np.testing.assert_allclose(fitted[real+'/weights'], fitted[shared+'/weights']+residual, atol=1e-12)
        assert np.linalg.matrix_rank(fitted[shared+'/weights']) <= 5
        h = rng.normal(size=(3, 9))
        primal = np.linalg.solve(x.T @ x+penalty*np.eye(9), x.T @ targets['shared'])
        np.testing.assert_allclose(mapping.predict(fitted, h, shared, 17), h @ primal, atol=1e-12)
    assert not np.allclose(fitted['steered/weights'], fitted['permuted_low/weights'])
    with pytest.raises(ValueError): mapping.predict(fitted, x, 'shared_low', 129)


def test_no_individual_residual_makes_all_three_targets_identical():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(20, 9)); positions = np.repeat(np.arange(5), 4)
    d = rng.normal(size=(5, 9))[positions]
    fitted = mapping.fit_states(x, d, positions)
    for real, shuffled, shared in [('steered', 'permuted_low', 'shared_low'), ('real_high', 'permuted', 'shared_high')]:
        np.testing.assert_array_equal(fitted[real+'/weights'], fitted[shared+'/weights'])
        np.testing.assert_array_equal(fitted[real+'/weights'], fitted[shuffled+'/weights'])
    with pytest.raises(ValueError): mapping.fit_states(x, d, positions.astype(float))

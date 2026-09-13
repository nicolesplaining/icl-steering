import numpy as np
import pytest

from analysis.ltv_state_shift import descriptors


def test_in_span_orthogonal_and_antiparallel_states():
    h = np.array([[2., 0., 0.], [0., 0., 2.], [2., 0., 2.]])
    d = -h/2
    result = descriptors(h, d, np.array([[1., 0., 0.]]))
    np.testing.assert_allclose(result['outside_extraction_span_energy'], [0., 1., .5])
    np.testing.assert_allclose(result['shift_to_state_norm'], .5)
    np.testing.assert_allclose(result['shift_state_cosine'], -1.)


def test_zero_and_nonfinite_geometry_rejected():
    for h in [np.zeros((1, 3)), np.full((1, 3), np.nan)]:
        with pytest.raises(ValueError):descriptors(h, np.ones((1, 3)), np.eye(3))

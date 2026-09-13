import numpy as np
import pytest
import torch

from analysis import ltv_continuation_mapping as mapping


def test_position_interpolation_and_affine_controls():
    rng = np.random.default_rng(93)
    x, d = rng.normal(size=(20, 8)), rng.normal(size=(20, 8))
    positions = np.repeat(np.arange(5), 4)
    fitted = mapping.fit_states(x, d, positions)
    query = rng.normal(size=(3, 8))
    for j, position in enumerate([0, 1, 8, 32, 128]):
        xx, dd = x[positions == j], d[positions == j]
        c = np.sum((xx-xx.mean(0))*(dd-dd.mean(0)))/np.square(xx-xx.mean(0)).sum()
        np.testing.assert_allclose(mapping.predict(fitted, query, 'position_scalar', position),
                                   dd.mean(0)+c*(query-xx.mean(0)), rtol=1e-12, atol=1e-12)
    for name in ['position_mean', 'position_scalar']:
        np.testing.assert_allclose(mapping.predict(fitted, query, name, 20),
            .5*mapping.predict(fitted, query, name, 8)+.5*mapping.predict(fitted, query, name, 32))
    np.testing.assert_allclose(fitted['real/penalty'], .1*np.square(x).sum()/len(x))
    np.testing.assert_allclose(fitted['permuted/penalty'], np.square(x).sum()/len(x))
    with pytest.raises(ValueError): mapping.predict(fitted, query, 'ridge', 129)


def test_cutoff_returns_original_tensor_and_checks_head():
    norm, head = torch.nn.Identity(), torch.nn.Identity()
    visited = []
    def predict(h, position):
        visited.append(position)
        return np.ones_like(h)*.5
    with mapping.normalized_hook(norm, predict, 'prefix_0_128', head) as trace:
        for step in range(133):
            state = torch.ones(2, 3 if step == 0 else 1, 4, dtype=torch.bfloat16)
            changed = norm(state); head(changed)
            if step <= 128:
                assert changed is not state
                torch.testing.assert_close(changed[:, -1], state[:, -1]+.5)
                torch.testing.assert_close(changed[:, :-1], state[:, :-1])
            else:
                assert changed is state
    assert visited == list(range(129)) and trace['head_checks'] == 129 and trace['calls'] == 133
    assert not norm._forward_hooks and not head._forward_pre_hooks


def test_prefill_and_failed_head_cleanup():
    norm, head = torch.nn.Identity(), torch.nn.Identity()
    predictor = lambda h, p: np.ones_like(h)
    with mapping.normalized_hook(norm, predictor, 'prefill', head) as trace:
        for _ in range(3): head(norm(torch.ones(2, 1, 4)))
    assert trace['prefix_positions'] == [0] and trace['head_checks'] == 1
    with pytest.raises(ValueError, match='Missing LM-head checks'):
        with mapping.normalized_hook(norm, predictor, 'prefill', head):
            norm(torch.ones(2, 1, 4))
    assert not norm._forward_hooks and not head._forward_pre_hooks

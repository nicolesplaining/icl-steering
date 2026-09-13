import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from analysis import ltv_mapping as ltv


def test_dual_ridge_matches_primal_closed_form():
    rng = np.random.default_rng(17)
    x, d, query = rng.normal(size=(12, 5)), rng.normal(size=(12, 5)), rng.normal(size=(3, 5))
    fitted = ltv.fit(x, d)
    primal = np.linalg.solve(x.T @ x + 5*np.eye(5), x.T @ d)
    np.testing.assert_allclose(ltv.predict(fitted, query), query @ primal, atol=1e-12)
    assert fitted['solver_residual'] < 1e-12


def test_controls_and_standalone_scalar():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(20, 5)); d = 2*x + np.arange(5)
    fitted = ltv.fit(x, d); query = rng.normal(size=(3, 5))
    np.testing.assert_allclose(ltv.predict(fitted, query, 'scalar'), 2*query + np.arange(5))
    shuffled = ltv.fit(x, d[rng.permutation(20)])
    for kind in ['scalar_norm', 'permuted']:
        np.testing.assert_allclose(np.linalg.norm(ltv.predict(fitted, query, kind, shuffled), axis=1),
                                   np.linalg.norm(ltv.predict(fitted, query), axis=1))
    broken = {**fitted, 'weights': np.full_like(fitted['weights'], np.nan)}
    assert np.isfinite(ltv.predict(broken, query, 'scalar')).all()
    with pytest.raises(ValueError):
        ltv.predict(broken, query, 'scalar_norm')


@pytest.mark.parametrize('scope,expected', [('all', 2), ('prefill', 1)])
@torch.inference_mode()
def test_actual_normalized_boundary_and_cached_steps(scope, expected):
    torch.manual_seed(29)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=40, hidden_size=16,
        intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=32)).eval()
    prompt, next_token = torch.tensor([[3, 7, 11]]), torch.tensor([[13]])
    zero = model(prompt, use_cache=True, output_hidden_states=True)
    next_zero = model(next_token, past_key_values=zero.past_key_values, use_cache=True,
                      output_hidden_states=True)
    predictor = lambda x: .2*x + np.arange(16)[None, :]/100
    before = len(model.model.norm._forward_hooks)
    with ltv.normalized_hook(model.model.norm, predictor, scope, model.lm_head) as trace:
        out = model(prompt, use_cache=True, output_hidden_states=True)
        step = model(next_token, past_key_values=out.past_key_values, use_cache=True,
                     output_hidden_states=True)
    assert len(model.model.norm._forward_hooks) == before
    assert trace['calls'] == 2 and len(trace['states']) == expected
    assert trace['head_checks'] == expected and not model.lm_head._forward_pre_hooks
    np.testing.assert_array_equal(trace['states'][0], zero.hidden_states[-1][:, -1].numpy())
    torch.testing.assert_close(out.hidden_states[-1][:, :-1], zero.hidden_states[-1][:, :-1])
    expected_last = zero.hidden_states[-1][:, -1] + torch.tensor(predictor(trace['states'][0]), dtype=torch.float32)
    torch.testing.assert_close(out.hidden_states[-1][:, -1], expected_last)
    torch.testing.assert_close(out.logits[:, -1], model.lm_head(expected_last))
    if scope == 'all':
        np.testing.assert_array_equal(trace['states'][1], next_zero.hidden_states[-1][:, -1].numpy())
        assert not np.array_equal(trace['vectors'][0], trace['vectors'][1])
    else:
        torch.testing.assert_close(step.logits, next_zero.logits)
    for lhs, rhs in zip(out.past_key_values.key_cache, zero.past_key_values.key_cache):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)


def test_single_token_prefill_and_hook_cleanup_on_failure():
    norm = torch.nn.Identity(); x = torch.ones(1, 1, 3)
    with ltv.normalized_hook(norm, lambda x: np.ones_like(x), 'prefill') as trace:
        assert torch.equal(norm(x), x+1)
        assert torch.equal(norm(x), x)
    assert len(trace['states']) == 1
    with pytest.raises(ValueError), ltv.normalized_hook(norm, lambda x: np.full_like(x, np.nan)):
        norm(x)
    assert not norm._forward_hooks


def test_wrong_or_missing_head_input_is_rejected():
    norm, head = torch.nn.Identity(), torch.nn.Identity()
    x = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match='did not receive'), ltv.normalized_hook(
            norm, lambda x: np.ones_like(x), head=head):
        norm(x); head(x)
    assert not norm._forward_hooks and not head._forward_pre_hooks
    with pytest.raises(ValueError, match='Missing LM-head'), ltv.normalized_hook(
            norm, lambda x: np.ones_like(x), head=head):
        norm(x)
    assert not norm._forward_hooks and not head._forward_pre_hooks

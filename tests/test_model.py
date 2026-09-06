import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, BatchEncoding

from icl_steering.model import FrozenModel, steering_hook


def test_hook_scope_positions_and_cleanup():
    block = torch.nn.Identity()
    x = torch.zeros(2, 3, 4)
    direction = torch.ones(4)
    with steering_hook(block, direction, 2, "prefill"):
        first = block(x)
        assert torch.equal(first[:, :-1], x[:, :-1])
        assert torch.equal(first[:, -1], torch.full((2, 4), 2.))
        assert torch.equal(block(x), x)
    assert torch.equal(block(x), x)
    with pytest.raises(RuntimeError):
        with steering_hook(block, direction, 1, "all"):
            assert torch.equal(block(x), block(x))
            raise RuntimeError("cleanup")
    assert not block._forward_hooks


def tiny_model():
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, eos_token_id=31, pad_token_id=0,
        attention_dropout=0.)).eval()


def test_zero_strength_generation_is_identical():
    model = tiny_model()
    inputs = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        baseline = model.generate(inputs, max_new_tokens=4, do_sample=False)
        with steering_hook(model.model.layers[0], torch.randn(32), 0):
            zero = model.generate(inputs, max_new_tokens=4, do_sample=False)
    assert torch.equal(baseline, zero)


class TokenizerStub:
    def __call__(self, prompts, **kwargs):
        width = max(map(len, prompts))
        ids = [[0]*(width-len(p))+p for p in prompts]
        mask = [[0]*(width-len(p))+[1]*len(p) for p in prompts]
        return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)})


def test_extraction_matches_generation_prefill_with_left_padding():
    backend = FrozenModel.__new__(FrozenModel)
    backend.config = {"layers": [0, 1], "max_prompt_tokens": 100}
    backend.tokenizer = TokenizerStub()
    backend.model = tiny_model()
    backend.blocks = backend.model.model.layers
    prompts = [[1, 2], [1, 4, 5, 6, 7]]
    captured = backend.activations(prompts)
    prefill = {}
    handles = []
    for layer in backend.config["layers"]:
        def hook(_m, _args, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            prefill[layer] = hidden[:, -1].detach().clone()
        handles.append(backend.blocks[layer].register_forward_hook(hook))
    inputs, _ = backend.encode(prompts)
    with torch.inference_mode():
        backend.model.generate(**inputs, max_new_tokens=1, do_sample=False)
    for handle in handles:
        handle.remove()
    for layer in captured:
        torch.testing.assert_close(captured[layer], prefill[layer], atol=1e-6, rtol=1e-5)
        for i, prompt in enumerate(prompts):
            single = backend.activations([prompt])[layer]
            torch.testing.assert_close(captured[layer][i:i+1], single, atol=1e-6, rtol=1e-5)

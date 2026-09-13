"""Check final-block versus final-state semantics using a tiny random Qwen2."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM


@torch.inference_mode()
def audit():
    torch.set_num_threads(2)
    torch.manual_seed(921)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64)
    model = Qwen2ForCausalLM(config).cpu().eval()
    captured, lengths = [], []

    def capture(module, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        captured.append(hidden.clone())
        lengths.append(hidden.shape[1])

    handle = model.model.layers[-1].register_forward_hook(capture)
    try:
        prefill = model(input_ids=torch.tensor([[3, 7, 11]]),
                        output_hidden_states=True, use_cache=True)
        final = prefill.hidden_states[-1]
        raw = captured[0]
        normalized = model.model.norm(raw)
        assert torch.equal(final, normalized)
        assert not torch.allclose(final, raw)
        model(input_ids=torch.tensor([[13]]), past_key_values=prefill.past_key_values,
              use_cache=True)
        assert lengths == [3, 1]
    finally:
        handle.remove()

    # Isolate normalization placement using the same raw activation and shift.
    delta = torch.linspace(-0.1, 0.1, config.hidden_size).reshape(1, 1, -1)
    before_norm = model.model.norm(raw + delta)
    after_norm = normalized + delta
    assert not torch.allclose(before_norm, after_norm)
    return {
        'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'device': 'cpu', 'seed': 921, 'model': 'random tiny Qwen2, not pretrained',
        'final_state_equals_normalized_block_output': True,
        'final_state_equals_raw_block_output': False,
        'raw_to_final_max_abs_difference': (final-raw).abs().max().item(),
        'prefill_and_cached_decode_sequence_lengths': lengths,
        'seq_len_greater_than_one_hook_runs_on_cached_decode': False,
        'same_delta_before_and_after_norm_max_abs_difference':
            (before_norm-after_norm).abs().max().item(),
        'limitations': 'Architecture check in our installed version only. No upstream '
            'code execution, pretrained model, GSM8K questions, or accuracy evaluation. '
            'This does not diagnose the authors recorded experimental runtime.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))

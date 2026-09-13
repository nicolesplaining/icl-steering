from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from analysis import ltv_prefix_collect as diagnostic


def test_parent_failure_and_declaration_required_before_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic.ltv, 'backend', lambda *args: pytest.fail('Unexpected model load'))
    with pytest.raises(ValueError, match='reviewed LTV failure'):
        diagnostic.prepare(tmp_path, tmp_path/'diagnostic')
    assert not (tmp_path/'diagnostic').exists()
    monkeypatch.setattr(diagnostic, 'verify', lambda path: ({}, []))
    with pytest.raises(FileNotFoundError):
        diagnostic.collect(tmp_path)


def test_preparation_preserves_frozen_inputs_and_omits_query_answers(tmp_path, monkeypatch):
    parent, output = tmp_path/'parent', tmp_path/'diagnostic'
    parent.mkdir()
    for name in ['manifest.json', 'prepared.json', 'fit.json', 'maps.npz', 'extraction.npz',
                 'selection.json', 'validation-audit.json', 'validation-annotations.json', 'review-freeze.json']:
        (parent/name).write_text('{}')
    questions = [{'problem_id': str(i), 'answer': 'unused reference answer',
        'prompts': {'zero': f'question {i}', 'icl_a': f'demos question {i}'}} for i in range(128)]
    monkeypatch.setattr(diagnostic, 'require_failed_parent', lambda path: ({}, {'splits': {'extract': questions}}))
    diagnostic.prepare(parent, output)
    before = (output/'manifest.json').read_bytes()
    diagnostic.prepare(parent, output)
    assert (output/'manifest.json').read_bytes() == before
    prepared = diagnostic.base.read(output/'prepared.json')['questions']
    assert all(set(q) == {'problem_id', 'prompts'} for q in prepared)
    (parent/'fit.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='parent input changed'):
        diagnostic.prepare(parent, output)


def small_engine(monkeypatch):
    torch.manual_seed(2911)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=8,
        intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, pad_token_id=0, bos_token_id=1, eos_token_id=2)).eval()
    model.requires_grad_(False)
    ids = {'z0': [1, 4], 'z1': [1, 5, 6], 'z2': [1, 7], 'z3': [1, 8, 9],
           'c0': [1, 10, 4], 'c1': [1, 11, 5, 6], 'c2': [1, 10, 7], 'c3': [1, 11, 8, 9]}
    def encode(prompts):
        return diagnostic.encoded_rows([ids[p] for p in prompts], 0, model.device, 256)
    def generate(**kwargs):
        assert kwargs['max_new_tokens'] == 128 and not kwargs['do_sample']
        assert 'position_ids' not in kwargs and 'stopping_criteria' not in kwargs
        continuation = torch.zeros((4, 128), dtype=torch.long)
        continuation[0, :3] = torch.tensor([12, 13, 2])
        continuation[1, 0] = 2
        continuation[2, :2] = torch.tensor([14, 2])
        continuation[3] = torch.arange(128) % 32 + 3
        return torch.cat([kwargs['input_ids'], continuation], 1)
    monkeypatch.setattr(model, 'generate', generate)
    engine = SimpleNamespace(model=model, encode=encode, tokenizer=SimpleNamespace(pad_token_id=0))
    questions = [{'problem_id': str(i), 'prompts': {'zero': f'z{i}', 'icl_a': f'c{i}'}} for i in range(4)]
    with torch.inference_mode():
        reference = {kind: model.model(**encode([q['prompts'][kind] for q in questions]),
            use_cache=False).last_hidden_state[:, -1].float().numpy() for kind in ['zero', 'icl_a']}
    return engine, questions, reference


def test_tiny_model_pairs_original_states_eos_and_saved_arrays(tmp_path, monkeypatch):
    engine, questions, reference = small_engine(monkeypatch)
    arrays, record = diagnostic.collect_batch(engine, questions, reference, 256)
    assert record['prefix_ids'][:3] == [[12, 13], [], [14]]
    assert record['prefix_ids'][3] == [i % 32 + 3 for i in range(128)]
    assert arrays['available'].tolist() == [[True, True, False, False, False],
        [True, False, False, False, False], [True, True, False, False, False],
        [True, True, True, True, True]]
    assert not engine.model.model.norm._forward_hooks
    for name in ['zero', 'icl_a']:
        np.testing.assert_array_equal(arrays[name][:, 0], reference[name])
    path = tmp_path/'000.json'
    diagnostic.ltv.arrays(path.with_suffix('.npz'), arrays)
    record['arrays_sha256'] = diagnostic.base.file_hash(path.with_suffix('.npz'))
    diagnostic.base.fixed(path, record)
    checked = diagnostic.check_batch(path, questions, reference)
    np.testing.assert_array_equal(checked['zero'], arrays['zero'])
    with pytest.raises(ValueError, match='Question order'):
        diagnostic.check_batch(path, questions[::-1], reference)
    arrays['zero'][0, 1, 0] += 1
    diagnostic.ltv.arrays(path.with_suffix('.npz'), arrays)
    with pytest.raises(ValueError, match='state file changed'):
        diagnostic.check_batch(path, questions, reference)


def test_capture_mismatch_and_context_overflow_fail(monkeypatch):
    engine, questions, reference = small_engine(monkeypatch)
    wrong = {k: v.copy() for k, v in reference.items()}
    wrong['zero'][0, 0] += 1
    with pytest.raises(AssertionError, match='Original prompt capture mismatch'):
        diagnostic.collect_batch(engine, questions, wrong, 256)
    assert not engine.model.model.norm._forward_hooks
    with pytest.raises(ValueError, match='exceed context'):
        diagnostic.collect_batch(engine, questions, reference, 128)


def test_left_padding_position_ids_and_context_limit():
    inputs = diagnostic.encoded_rows([[1, 5], [1, 6, 7]], 0, 'cpu', 3)
    assert inputs['input_ids'].tolist() == [[0, 1, 5], [1, 6, 7]]
    assert inputs['attention_mask'].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert inputs['position_ids'].tolist() == [[1, 0, 1], [0, 1, 2]]
    with pytest.raises(ValueError): diagnostic.encoded_rows([[1, 5]], 0, 'cpu', 1)

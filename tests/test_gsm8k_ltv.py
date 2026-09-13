import json

import pytest

from analysis import gsm8k_ltv as ltv


def summary():
    values = {name: {'n': 128, 'audited_completed_correct': 100, 'truncated': 0}
              for name in ltv.CONDITIONS}
    values['zero']['audited_completed_correct'] = 93
    values['steered']['audited_completed_correct'] = 111
    for name in ['icl_a', 'icl_b']:
        values[name].update(audited_completed_correct=115,
                            paired={'zero': {'gain': .17, 'ci95': [.10, .25]}})
    values['cot']['audited_completed_correct'] = 110
    return values


CONFIG = {'min_icl_gain': .05, 'max_truncation_rate': .05}


def test_fixed_registry_and_standalone_control():
    specs = ltv.specifications()
    assert len(specs) == 11
    assert specs['scalar']['kind'] == 'scalar'
    assert specs['steered'] == {'kind': 'ltv', 'site': 'final_norm',
                                'layer': None, 'alpha': 1., 'positions': 'all'}
    assert specs['prefill']['positions'] == 'prefill'


def test_gate_preserves_text_and_strict_control_requirements():
    s = summary()
    assert ltv.gates(s, CONFIG)['eligible']
    s['steered']['audited_completed_correct'] = 109
    assert not ltv.gates(s, CONFIG)['eligible']
    s['steered']['audited_completed_correct'] = 111
    for control in ['mean', 'scalar', 'scalar_norm', 'permuted']:
        copy = json.loads(json.dumps(s));copy[control]['audited_completed_correct'] = 111
        assert not ltv.gates(copy, CONFIG)['eligible']
    s['prefill']['audited_completed_correct'] = 112
    assert not ltv.gates(s, CONFIG)['eligible']


def test_gate_rejects_truncation_and_missing_condition():
    s = summary();s['steered']['truncated'] = 7
    assert not ltv.gates(s, CONFIG)['eligible']
    del s['scalar']
    with pytest.raises(ValueError):ltv.gates(s, CONFIG)


def test_reservation_rejected_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(ltv, 'verify', lambda *a, **k: ({'max_new_tokens': 1024}, {}))
    (tmp_path/'baseline-rows.json').write_text(json.dumps([{'split': 'reserved'}]))
    with pytest.raises(ValueError, match='Reserved generation'):
        ltv.collect(tmp_path)


def test_incomplete_batch_cannot_be_reviewed(tmp_path, monkeypatch):
    monkeypatch.setattr(ltv, 'verify', lambda *a, **k: ({'max_new_tokens': 1024}, {}))
    (tmp_path/'baseline-rows.json').write_text('[]')
    with pytest.raises(ValueError, match='Incomplete validation'):
        ltv.collect(tmp_path, complete=True)


def test_unexpected_batch_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ltv, 'verify', lambda *a, **k: ({'max_new_tokens': 1024}, {}))
    (tmp_path/'baseline-rows.json').write_text('[]')
    (tmp_path/'batches').mkdir(); (tmp_path/'batches'/'reserved-000.json').write_text('[]')
    with pytest.raises(ValueError, match='Unexpected batch'):
        ltv.collect(tmp_path)


def test_declaration_required_before_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(ltv, 'verify', lambda *a, **k: ({}, {}))
    monkeypatch.setattr(ltv, 'backend', lambda *a: pytest.fail('Loaded GPU before declaration'))
    with pytest.raises(FileNotFoundError):ltv.validate(tmp_path)


def test_batch_persistence_blind_packet_and_resume(tmp_path, monkeypatch):
    """Exercise the whole generation path with synthetic states and no LLM."""
    from types import SimpleNamespace
    import numpy as np
    import torch
    config = {'batch_size': 4, 'max_new_tokens': 1024}
    questions = [{'problem_id': str(i), 'question': f'Question {i}', 'answer': '1',
                  'prompts': {k: f'{k} question {i}' for k in ltv.screen.CONDITIONS}}
                 for i in range(128)]
    data = {'splits': {'validation': questions, 'reserved': [{'problem_id': 'never'}]}}
    monkeypatch.setattr(ltv, 'verify', lambda *a, **k: (config, data))
    for filename in ['manifest.json', 'fit.json']:(tmp_path/filename).write_text('{}')
    (tmp_path/'fit-declaration.json').write_text(json.dumps({
        'manifest_sha256': ltv.base.file_hash(tmp_path/'manifest.json'),
        'fit_sha256': ltv.base.file_hash(tmp_path/'fit.json')}))
    (tmp_path/'source-annotations.json').write_text(json.dumps({'answers': []}))
    def output():
        return {'text': 'The answer is 1.', 'solution_text': 'The answer is 1.',
                'token_ids': [3, 4], 'generated_tokens': 2, 'finish_reason': 'eos',
                'truncated': False, 'batch_seconds': 0.}
    baseline = []
    for name in ltv.screen.CONDITIONS:
        spec = ltv.specifications()[name]
        for p in questions:
            baseline.append({'split': 'validation', 'condition': name, 'problem_id': p['problem_id'],
                'answer': '1', 'prompt': p['prompts'][name], 'intervention': spec,
                **{k: spec[k] for k in ['layer', 'alpha', 'positions']}, **output(),
                **ltv.base.grade_answer('The answer is 1.', '1', False)})
    (tmp_path/'baseline-rows.json').write_text(json.dumps(baseline))
    rng = np.random.default_rng(13)
    x = rng.normal(size=(12, 8)); icl = x + rng.normal(size=(12, 8)); d = icl-x
    np.savez(tmp_path/'extraction.npz', zero=x, icl_a=icl)
    real = ltv.mapping.fit(x, d)
    permuted = ltv.mapping.fit(x, d[np.random.default_rng(1901).permutation(len(x))])
    monkeypatch.setattr(ltv, 'load_maps', lambda _: (real, permuted))
    norm, head = torch.nn.Identity(), torch.nn.Identity()
    calls = []
    def records(prompts, config):
        calls.append(list(prompts))
        head(norm(torch.ones(4, 3, 8))); head(norm(torch.full((4, 1, 8), 2.)))
        return [output() for _ in prompts]
    fake = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(norm=norm), lm_head=head), records=records)
    monkeypatch.setattr(ltv, 'backend', lambda *a: fake)
    ltv.validate(tmp_path)
    assert len(calls) == 192 and not norm._forward_hooks
    rows = ltv.collect(tmp_path, complete=True)
    assert len(rows) == 1408 and all(r['split'] == 'validation' for r in rows)
    for name, n in [('steered', 2), ('prefill', 1)]:
        batch = json.loads((tmp_path/f'batches/{name}-000.json').read_text())
        assert all(r['active_calls'] == n and r['hook_calls'] == 2 for r in batch)
        with np.load(tmp_path/batch[0]['trace_file']) as trace:
            assert trace['states'].shape == trace['vectors'].shape == (n, 4, 8)
            np.testing.assert_allclose(trace['vectors'][0],
                ltv.mapping.predict(real, trace['states'][0]).astype(np.float32))
    original = (tmp_path/'generations.jsonl').read_bytes()
    ltv.validate(tmp_path)
    assert len(calls) == 192 and (tmp_path/'generations.jsonl').read_bytes() == original
    from analysis import ltv_trace_audit
    checked = ltv_trace_audit.check(tmp_path, complete=True)
    assert checked['new_generation_rows'] == 768
    assert checked['saved_state_vectors_replayed'] == 1408
    with pytest.raises(ValueError, match='Trace changed'):
        (tmp_path/'traces/steered-000.npz').write_bytes(b'changed')
        ltv.collect(tmp_path)

import json

import pytest

from analysis import gsm8k_ltv_continuation as run


def values():
    summary = {name: {'n': 128, 'audited_completed_correct': 100, 'truncated': 0} for name in run.CONDITIONS}
    summary['zero']['audited_completed_correct'] = 93
    summary['steered']['audited_completed_correct'] = 114
    summary['prefill']['audited_completed_correct'] = 113
    for name in ['icl_a', 'icl_b']:
        summary[name].update(audited_completed_correct=115, paired={'zero': {'gain': .17, 'ci95': [.10, .25]}})
    return summary


CONFIG = {'min_icl_gain': .05, 'max_truncation_rate': .05}


def test_gate_requires_more_than_same_map_prefill_and_all_controls():
    summary = values()
    assert run.gates(summary, CONFIG)['eligible']
    for name in ['regularized_prefill', 'mean', 'scalar', 'position_mean', 'position_scalar', 'permuted']:
        changed = json.loads(json.dumps(summary)); changed[name]['audited_completed_correct'] = 114
        assert not run.gates(changed, CONFIG)['eligible']
    summary['steered']['truncated'] = 7
    assert not run.gates(summary, CONFIG)['eligible']


def test_no_reserved_generation_and_fit_declaration_before_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'verify', lambda *a, **k: ({'max_new_tokens': 1024}, {}))
    monkeypatch.setattr(run, 'backend', lambda *a: pytest.fail('GPU loaded before declaration'))
    with pytest.raises(FileNotFoundError): run.validate(tmp_path)
    (tmp_path/'baseline-rows.json').write_text(json.dumps([{'split': 'reserved'}]))
    with pytest.raises(ValueError, match='Reserved generation'): run.collect(tmp_path)


def test_batch_resume_and_independent_trace_replay(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import torch
    config = {'batch_size': 4, 'max_new_tokens': 1024}
    questions = [{'problem_id': str(i), 'question': f'Question {i}', 'answer': '1',
                  'prompts': {k: f'{k} question {i}' for k in run.screen.CONDITIONS}} for i in range(128)]
    data = {'splits': {'validation': questions, 'reserved': [{'problem_id': 'never'}]}}
    monkeypatch.setattr(run, 'verify', lambda *a, **k: (config, data))
    for name in ['manifest.json', 'fit.json']: (tmp_path/name).write_text('{}')
    (tmp_path/'fit-declaration.json').write_text(json.dumps({
        'manifest_sha256': run.base.file_hash(tmp_path/'manifest.json'),
        'fit_sha256': run.base.file_hash(tmp_path/'fit.json')}))
    (tmp_path/'source-annotations.json').write_text(json.dumps({'answers': []}))
    def output():
        return {'text': 'The answer is 1.', 'solution_text': 'The answer is 1.',
                'token_ids': [3, 4], 'generated_tokens': 2, 'finish_reason': 'eos',
                'truncated': False, 'batch_seconds': 0.}
    baseline = []
    for name in run.BASELINES:
        spec = run.specifications()[name]
        for p in questions:
            baseline.append({'split': 'validation', 'condition': name, 'problem_id': p['problem_id'],
                'answer': '1', 'prompt': p['prompts'][spec.get('prompt_kind', 'zero')], 'intervention': spec,
                **{k: spec[k] for k in ['layer', 'alpha', 'positions']}, **output(),
                **run.base.grade_answer('The answer is 1.', '1', False)})
    (tmp_path/'baseline-rows.json').write_text(json.dumps(baseline))
    rng = np.random.default_rng(13)
    x, d = rng.normal(size=(20, 8)), rng.normal(size=(20, 8))
    positions = np.repeat(np.arange(5), 4)
    np.savez(tmp_path/'extraction.npz', zero=x, delta=d, positions=positions)
    fitted = run.mapping.fit_states(x, d, positions)
    monkeypatch.setattr(run, 'load_maps', lambda _: fitted)
    norm, head = torch.nn.Identity(), torch.nn.Identity()
    calls = []
    def records(prompts, config):
        calls.append(prompts)
        head(norm(torch.ones(4, 3, 8))); head(norm(torch.full((4, 1, 8), 2.)))
        return [output() for _ in prompts]
    fake = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(norm=norm), lm_head=head), records=records)
    monkeypatch.setattr(run, 'backend', lambda *a: fake)
    run.validate(tmp_path)
    assert len(calls) == 224 and not norm._forward_hooks
    rows = run.collect(tmp_path, complete=True)
    assert len(rows) == 1664 and all(r['split'] == 'validation' for r in rows)
    original = (tmp_path/'generations.jsonl').read_bytes()
    run.validate(tmp_path)
    assert len(calls) == 224 and (tmp_path/'generations.jsonl').read_bytes() == original
    from analysis import ltv_continuation_audit
    checked = ltv_continuation_audit.check(tmp_path, complete=True)
    assert checked['new_generation_rows'] == 896 and checked['saved_state_vectors_replayed'] == 1664
    with pytest.raises(ValueError, match='Trace changed'):
        (tmp_path/'traces/steered-000.npz').write_bytes(b'changed')
        run.collect(tmp_path)

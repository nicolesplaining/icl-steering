import json

import pytest

from analysis import gsm8k_ltv_pairing as run


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
    assert len(calls) == 128 and not norm._forward_hooks
    rows = run.collect(tmp_path, complete=True)
    assert len(rows) == 2176 and all(r['split'] == 'validation' for r in rows)
    original = (tmp_path/'generations.jsonl').read_bytes()
    run.validate(tmp_path)
    assert len(calls) == 128 and (tmp_path/'generations.jsonl').read_bytes() == original
    from analysis import ltv_pairing_audit
    checked = ltv_pairing_audit.check(tmp_path, complete=True)
    assert checked['new_generation_rows'] == 512 and checked['saved_state_vectors_replayed'] == 1024
    packet = json.loads((tmp_path/'validation-review-packet.json').read_text())
    annotations = {'packet_sha256': run.audit.digest(packet), 'answers': [
        {'response_id': item['response_id'], 'stated_answer': '1', 'reviewed': True,
         'rationale': 'Synthetic explicit one.'} for item in packet['items']]}
    annotation_path = tmp_path/'review-annotations.json'
    annotation_path.write_text(json.dumps(annotations))
    with pytest.raises(FileNotFoundError): run.report(tmp_path, annotation_path)
    (tmp_path/'review-freeze.json').write_text(json.dumps({
        'packet_sha256': run.audit.digest(packet),
        'annotations_file_sha256': run.base.file_hash(annotation_path),
        'annotations_commit': 'synthetic'}))
    run.report(tmp_path, annotation_path)
    report = json.loads((tmp_path/'diagnostic-report.json').read_text())
    assert report['candidate_selected'] is None and report['confirmation_supported'] is False
    assert report['reserved_rows'] == 0 and len(report['summary']) == 17
    for metric in ['primary', 'audited']:
        assert len(report['contrasts'][metric]) == 7
        assert {(v['left'], v['right']) for v in report['contrasts'][metric].values()} == set(run.PAIRS)
    with pytest.raises(ValueError, match='Diagnostic already scored'): run.validate(tmp_path)
    with pytest.raises(ValueError, match='Trace changed'):
        (tmp_path/'traces/shared_low-000.npz').write_bytes(b'changed')
        run.collect(tmp_path)

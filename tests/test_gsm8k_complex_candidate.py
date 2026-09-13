import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis import gsm8k_complex_candidate as run
from analysis import complex_candidate_trace_check as trace_check


def test_failed_or_unfinished_screen_stops_before_maps_or_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(run.screen, 'verify', lambda _: ({}, {}))
    monkeypatch.setattr(run.mapping, 'load_frozen', lambda _: pytest.fail('Maps loaded before screen pass'))
    out = tmp_path/'candidate'
    with pytest.raises(FileNotFoundError):
        run.prepare(tmp_path, tmp_path/'missing-check.json', out)
    (tmp_path/'screen-selection.json').write_text(json.dumps({'eligible': False, 'checks': {'icl_a_gain': False}}))
    with pytest.raises(ValueError, match='ICL screen failed'):
        run.prepare(tmp_path, tmp_path/'missing-check.json', out)
    assert not out.exists()


def test_full_fake_run_resume_trace_first_tokens_and_score_guards(tmp_path, monkeypatch):
    config = {'batch_size': 4, 'max_new_tokens': 1024}
    questions = [{'problem_id': f'gsm8k:train:{i}', 'question': f'Question {i}', 'answer': '1',
        'prompts': {k: f'{k} question {i}' for k in run.BASELINES}} for i in range(256)]
    data = {'splits': {'validation': questions, 'reserved': [{'problem_id': 'gsm8k:train:900'}]}}
    (tmp_path/'prepared.json').write_text(json.dumps(data))
    monkeypatch.setattr(run, 'verify', lambda *a, **k: (config, data))
    for name in ['manifest.json', 'fit.json']:
        (tmp_path/name).write_text('{}')
    calls = []
    norm, head = torch.nn.Identity(), torch.nn.Identity()
    def response():
        return {'text': 'The answer is 1.', 'solution_text': 'The answer is 1.', 'token_ids': [2, 3, 4],
                'generated_tokens': 3, 'finish_reason': 'eos', 'truncated': False, 'batch_seconds': 0.}
    def records(prompts, config):
        calls.append(prompts)
        head(norm(torch.ones(4, 3, 8)))
        head(norm(torch.full((4, 1, 8), 2.)))
        head(norm(torch.full((4, 1, 8), 3.)))
        return [response() for _ in prompts]
    monkeypatch.setattr(run, 'backend', lambda *a: SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(norm=norm), lm_head=head), records=records))
    monkeypatch.setattr(run, 'load_maps', lambda _: {})
    scales = {name: i+1 for i, name in enumerate(run.NEW)}
    scales['regularized_prefill'] = scales['steered']
    expected = lambda h, name, position: h.astype(np.float64)*scales[name]
    monkeypatch.setattr(run.mapping, 'predict', lambda fitted, h, name, position: expected(h, name, position))
    monkeypatch.setattr(trace_check, 'reconstruct', lambda _: (expected, 8))
    baseline = []
    for name in run.BASELINES:
        spec = run.specifications()[name]
        for p in questions:
            baseline.append({'split': 'validation', 'condition': name, 'problem_id': p['problem_id'],
                'answer': '1', 'prompt': p['prompts'][name], 'intervention': spec,
                **{k: spec[k] for k in ['layer', 'alpha', 'positions']}, **response(),
                **run.base.grade_answer('The answer is 1.', '1', False)})
    (tmp_path/'baseline-rows.json').write_text(json.dumps(baseline))
    (tmp_path/'source-annotations.json').write_text(json.dumps({'answers': []}))
    with pytest.raises(FileNotFoundError):
        run.validate(tmp_path)
    assert calls == []
    (tmp_path/'fit-declaration.json').write_text(json.dumps({
        'manifest_sha256': run.base.file_hash(tmp_path/'manifest.json'),
        'fit_sha256': run.base.file_hash(tmp_path/'fit.json'), 'declaration_commit': 'synthetic'}))
    run.validate(tmp_path)
    assert len(calls) == 768 and not norm._forward_hooks and not head._forward_pre_hooks
    rows = run.collect(tmp_path, complete=True)
    assert len(rows) == 4352 and all(r['split'] == 'validation' for r in rows)
    original = (tmp_path/'generations.jsonl').read_bytes()
    run.validate(tmp_path)
    assert len(calls) == 768 and (tmp_path/'generations.jsonl').read_bytes() == original
    checked = trace_check.check(tmp_path, complete=True)
    assert checked['new_generation_rows'] == 3072 and checked['candidate_prefill_first_tokens_matched'] == 256
    assert checked['saved_state_vectors_replayed'] == (10*64*3+2*64)*4
    path = tmp_path/'batches/regularized_prefill-000.json'
    saved = path.read_bytes(); broken = json.loads(saved)
    broken[0]['token_ids'][0] = 999
    path.write_text(json.dumps(broken))
    with pytest.raises(AssertionError):
        trace_check.check(tmp_path, complete=True)
    path.write_bytes(saved)
    packet = run.base.read(tmp_path/'validation-review-packet.json')
    assert packet['items'] == []
    annotations = tmp_path/'annotations.json'
    annotations.write_text(json.dumps({'packet_sha256': run.audit.digest(packet), 'answers': []}))
    with pytest.raises(FileNotFoundError):
        run.select(tmp_path, annotations)
    (tmp_path/'review-freeze.json').write_text(json.dumps({'packet_sha256': run.audit.digest(packet),
        'annotations_file_sha256': run.base.file_hash(annotations), 'annotations_commit': 'synthetic'}))
    run.select(tmp_path, annotations)
    decision = run.base.read(tmp_path/'selection.json')
    assert not decision['eligible'] and not decision['reserved_generation_authorized']
    report = run.base.read(tmp_path/'validation-audit.json')
    assert all(len(report['contrasts'][metric]) == 16 for metric in ['primary', 'audited'])
    with pytest.raises(ValueError, match='Selection already frozen'):
        run.validate(tmp_path)


def test_source_input_tamper_and_screen_prerequisite_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'code_hash', lambda: 'fixed')
    (tmp_path/'prepared.json').write_text('{}')
    source = tmp_path/'source.json'; source.write_text('{}')
    manifest = {'config': run.base.read(run.screen.CONFIG), 'code_sha256': 'fixed',
        'conditions': run.specifications(), 'icl_screen_passed': True, 'confirmation_supported': False,
        'files_sha256': {'prepared.json': run.base.file_hash(tmp_path/'prepared.json')},
        'inputs_sha256': {str(source): run.base.file_hash(source)}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    run.verify(tmp_path)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Frozen source input'):
        run.verify(tmp_path)
    source.write_text('{}'); manifest['icl_screen_passed'] = False
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='prerequisite'):
        run.verify(tmp_path)

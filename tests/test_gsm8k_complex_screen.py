import json
from types import SimpleNamespace

import pytest

from analysis import gsm8k_complex_screen as screen


def test_fixed_complexity_partition_and_failure_without_replacement():
    pool = [{'problem_id': f'gsm8k:train:{i}', 'calculation_annotations': i % 6}
            for i in range(100)]
    config = {'seed': 3401, 'n_screen': 8, 'n_reserved': 12, 'min_calculation_annotations': 4}
    plan = screen.partition(pool, config)
    ids = plan['validation']+plan['reserved']
    assert len(set(ids)) == 20
    assert all(int(i.rsplit(':', 1)[1]) % 6 >= 4 for i in ids)
    assert not set(plan['validation']) & set(plan['reserved'])
    for row in pool:
        row['answer'] = 'changed numerical reference, not a selection input'
    assert screen.partition(pool, config) == plan
    with pytest.raises(ValueError, match='Insufficient'):
        screen.partition(pool, {**config, 'n_screen': 100})
    with pytest.raises(ValueError, match='distinct, ordered'):
        screen.partition(list(reversed(pool)), config)


def test_fake_full_screen_resume_freeze_and_reserved_guard(tmp_path, monkeypatch):
    config = {'batch_size': 4, 'bootstrap_samples': 64, 'min_icl_gain': .05,
              'max_truncation_rate': .05}
    questions = [{'problem_id': f'gsm8k:train:{i}', 'question': f'Question {i}', 'answer': '1',
        'prompts': {k: f'{k} question {i}' for k in screen.CONDITIONS}} for i in range(4)]
    data = {'splits': {'validation': questions, 'reserved': [{'problem_id': 'gsm8k:train:9'}]}}
    monkeypatch.setattr(screen, 'verify', lambda _: (config, data))
    (tmp_path/'manifest.json').write_text('{}')
    calls = []
    def records(prompts, config, *args):
        calls.append(prompts)
        answer = '0' if prompts[0].startswith('zero ') else '1'
        text = 'The answer is '+answer+'.'
        return [{'text': text, 'solution_text': text, 'token_ids': [1], 'generated_tokens': 1,
                 'finish_reason': 'eos', 'truncated': False, 'batch_seconds': 0.} for p in prompts]
    monkeypatch.setattr(screen.previous.ScreenRunner, 'backend', lambda _: SimpleNamespace(records=records))
    with pytest.raises(FileNotFoundError):
        screen.run(tmp_path)
    assert not calls
    (tmp_path/'declaration.json').write_text(json.dumps({
        'manifest_sha256': screen.base.file_hash(tmp_path/'manifest.json'), 'declaration_commit': 'synthetic'}))
    screen.run(tmp_path)
    assert len(calls) == 5
    original = (tmp_path/'generations.jsonl').read_bytes()
    screen.run(tmp_path)
    assert len(calls) == 5 and (tmp_path/'generations.jsonl').read_bytes() == original
    rows = screen.previous.check_rows(tmp_path, data, complete=True)
    assert len(rows) == 20 and {r['split'] for r in rows} == {'validation'}
    packet = screen.base.read(tmp_path/'validation-review-packet.json')
    assert packet['items'] == []
    annotations = tmp_path/'annotations.json'
    annotations.write_text(json.dumps({'packet_sha256': screen.audit.digest(packet), 'answers': []}))
    with pytest.raises(FileNotFoundError):
        screen.score(tmp_path, annotations)
    (tmp_path/'review-freeze.json').write_text(json.dumps({'packet_sha256': screen.audit.digest(packet),
        'annotations_file_sha256': screen.base.file_hash(annotations), 'annotations_commit': 'synthetic'}))
    screen.score(tmp_path, annotations)
    assert screen.base.read(tmp_path/'screen-selection.json')['eligible']
    with pytest.raises(ValueError, match='frozen after scoring'):
        screen.run(tmp_path)
    with (tmp_path/'generations.jsonl').open('a') as out:
        out.write(json.dumps({'split': 'reserved', 'condition': 'zero', 'problem_id': 'gsm8k:train:9'})+'\n')
    with pytest.raises(ValueError, match='non-development'):
        screen.score(tmp_path, annotations)


def test_manifest_input_and_preparation_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, 'code_hash', lambda: 'fixed')
    config_path = tmp_path/'config.json'
    config_path.write_text('{}')
    monkeypatch.setattr(screen, 'CONFIG', config_path)
    (tmp_path/'prepared.json').write_text('{}')
    dependency = tmp_path/'input.json'; dependency.write_text('{}')
    manifest = {'code_sha256': 'fixed', 'config': {},
        'prepared_sha256': screen.base.file_hash(tmp_path/'prepared.json'),
        'inputs_sha256': {str(dependency): screen.base.file_hash(dependency)}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    screen.verify(tmp_path)
    dependency.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='input changed'):
        screen.verify(tmp_path)
    dependency.write_text('{}')
    (tmp_path/'prepared.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Prepared screen changed'):
        screen.verify(tmp_path)

import copy
import json

import pytest

from analysis import gsm8k_test_development as screen


def fixture():
    tables = {s: [{'question': f'{s} question {i}', 'answer': str(i)} for i in range(180)]
              for s in ['train', 'test']}
    old = {'banks': {'icl_a': [{'question': 'test question 130'}]},
           'splits': {'test': [{'question': 'test question 129'}]}}
    prior = {'splits': {'validation': [{'question': 'test question 131'}],
                        'test': [{'question': 'test question 132'}]}}
    tables['test'][133]['question'] = '  test   question 132  '
    tables['test'][134]['question'] = 'train question 4'
    return tables, old, prior


def test_partition_excludes_reservations_and_text_duplicates():
    tables, old, prior = fixture()
    p = screen.partition(tables, old, prior, 1701, 16, 24)
    ids = p['validation'] + p['reserved']
    assert len(ids) == len(set(ids)) == 40
    assert not set(ids) & set(range(128))
    assert not set(ids) & {129, 130, 131, 132, 133, 134}
    assert p['eligible_count'] == 46


def test_partition_does_not_select_by_reference_answer():
    tables, old, prior = fixture()
    first = screen.partition(tables, old, prior, 1701, 16, 24)
    for row in tables['test']:
        row['answer'] = 'changed reference answer'
    assert screen.partition(tables, old, prior, 1701, 16, 24) == first
    with pytest.raises(ValueError, match='Insufficient'):
        screen.partition(tables, old, prior, 1701, 32, 32)


def test_both_banks_intervals_and_every_truncation_gate_required():
    summary = {k: {'n': 128, 'truncated': 0, 'paired': {
        'zero': {'gain': .1, 'ci95': [.01, .2]}}} for k in screen.CONDITIONS}
    config = {'min_icl_gain': .05, 'max_truncation_rate': .05}
    assert screen.gates(summary, config)['eligible']
    for bank in ['icl_a', 'icl_b']:
        for field, bad in [('gain', .04), ('ci95', [0., .2])]:
            changed = copy.deepcopy(summary)
            changed[bank]['paired']['zero'][field] = bad
            assert not screen.gates(changed, config)['eligible']
    summary['first']['truncated'] = 7
    assert not screen.gates(summary, config)['eligible']


def test_reserved_generation_rejected(tmp_path):
    (tmp_path / 'generations.jsonl').write_text(json.dumps({
        'split': 'reserved', 'condition': 'zero', 'problem_id': 'gsm8k:test:150'})+'\n')
    with pytest.raises(ValueError, match='non-development'):
        screen.check_rows(tmp_path, {'splits': {}})


def test_gpu_one_rejected_before_backend_load(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    runner = screen.ScreenRunner({}, {'splits': {}}, tmp_path)
    with pytest.raises(ValueError, match='GPU 0'):
        runner.backend()


def test_prepared_tamper_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, 'code_hash', lambda: 'fixed')
    (tmp_path / 'prepared.json').write_text('{}')
    original = screen.parent.file_hash(tmp_path / 'prepared.json')
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'code_sha256': 'fixed', 'prepared_sha256': original, 'config': {}}))
    screen.verify(tmp_path)
    (tmp_path / 'prepared.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Prepared screen changed'):
        screen.verify(tmp_path)

import copy
import json

import pytest

from analysis import gsm8k_fixed_candidate as fixed


def test_candidate_cannot_enter_a_grid():
    specs = fixed.specifications()
    assert list(specs) == ['zero', 'icl_a', 'icl_b', 'first', 'cot', 'steered', 'mean']
    for name in ['steered', 'mean']:
        assert {k: specs[name][k] for k in fixed.SETTING} == {
            'layer': 13, 'alpha': .5, 'positions': 'prefill'}


def test_each_fixed_candidate_gate():
    scores = {'zero': 95, 'icl_a': 115, 'icl_b': 113, 'first': 99, 'cot': 100,
              'steered': 101, 'mean': 100}
    summary = {k: {'n': 128, 'audited_completed_correct': n, 'truncated': 0}
               for k, n in scores.items()}
    assert fixed.gates(summary)['eligible']
    for name, field, bad in [('zero', 'audited_completed_correct', 98),
                             ('mean', 'audited_completed_correct', 101),
                             ('cot', 'audited_completed_correct', 102),
                             ('steered', 'truncated', 7)]:
        changed = copy.deepcopy(summary)
        changed[name][field] = bad
        assert not fixed.gates(changed)['eligible']


def test_no_preparation_after_failed_screen(tmp_path, monkeypatch):
    def failed(_):
        raise ValueError('ICL screen failed; fixed candidate is prohibited')
    monkeypatch.setattr(fixed, 'verified_screen', failed)
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='ICL screen failed'):
        fixed.prepare(tmp_path / 'screen', tmp_path / 'maps', out)
    assert not out.exists()


def test_cannot_change_baseline_generations(tmp_path):
    row = {'split': 'validation', 'problem_id': 'p', 'condition': 'zero', 'text': 'frozen'}
    (tmp_path / 'baseline-rows.json').write_text(json.dumps([row]))
    changed = {**row, 'text': 'replacement'}
    (tmp_path / 'generations.jsonl').write_text(json.dumps(changed)+'\n')
    with pytest.raises(ValueError, match='baseline rows changed'):
        fixed.check_rows(tmp_path, {})


def test_reserved_questions_never_allowed(tmp_path):
    (tmp_path / 'generations.jsonl').write_text(json.dumps({
        'split': 'reserved', 'problem_id': 'p', 'condition': 'steered'})+'\n')
    with pytest.raises(ValueError, match='Unexpected'):
        fixed.check_rows(tmp_path, {})


def test_selection_rejects_changed_inherited_annotation(tmp_path, monkeypatch):
    original = {'response_id': 'r', 'stated_answer': '7', 'reviewed': True, 'rationale': 'literal'}
    packet = {'items': [{'response_id': 'r'}]}
    annotations = {'packet_sha256': fixed.audit.digest(packet),
                   'answers': [{**original, 'stated_answer': '8'}]}
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps(annotations))
    (tmp_path / 'validation-review-packet.json').write_text(json.dumps(packet))
    (tmp_path / 'inherited-annotations.json').write_text(json.dumps({'answers': [original]}))
    (tmp_path / 'review-freeze.json').write_text(json.dumps({
        'annotations_file_sha256': fixed.base.file_hash(path),
        'packet_sha256': fixed.audit.digest(packet), 'annotations_commit': 'frozen'}))
    monkeypatch.setattr(fixed, 'verify', lambda _: ({}, {}))
    monkeypatch.setattr(fixed, 'check_rows', lambda *a, **k: [])
    with pytest.raises(ValueError, match='Frozen screen annotation changed'):
        fixed.select(tmp_path, path)


def test_verified_screen_replays_and_rejects_failed_or_changed_gates(tmp_path, monkeypatch):
    config = {'bootstrap_samples': 10, 'min_icl_gain': .05, 'max_truncation_rate': .05}
    data = {'splits': {'validation': [{'problem_id': str(i), 'question': f'q{i}'} for i in range(2)]}}
    rows = [{'split': 'validation', 'condition': kind, 'problem_id': str(i),
             'answer': '1', 'solution_text': 'The answer is 1.', 'parseable': True,
             'correct': True, 'completed_correct': True, 'truncated': False}
            for kind in fixed.screen.CONDITIONS for i in range(2)]
    packet = fixed.audit.make_packet(rows, data, 'validation', fixed.screen.CONDITIONS)
    annotations = {'packet_sha256': fixed.audit.digest(packet), 'answers': []}
    result = fixed.audit.score(rows, data, 'validation', fixed.screen.CONDITIONS, packet, annotations, 10)
    for name, value in [('manifest.json', {}), ('validation-review-packet.json', packet),
                        ('screen-annotations.json', annotations), ('screen-audit.json', result)]:
        fixed.base.fixed(tmp_path / name, value)
    fixed.base.fixed(tmp_path / 'review-freeze.json', {
        'annotations_file_sha256': fixed.base.file_hash(tmp_path / 'screen-annotations.json'),
        'packet_sha256': fixed.audit.digest(packet), 'annotations_commit': 'frozen'})
    selection = {**fixed.screen.gates(result['summary'], config),
        'manifest_sha256': fixed.base.file_hash(tmp_path / 'manifest.json'),
        'rows_sha256': fixed.audit.digest(rows), 'annotations_sha256': fixed.audit.digest(annotations),
        'audit_sha256': fixed.base.file_hash(tmp_path / 'screen-audit.json')}
    fixed.base.fixed(tmp_path / 'screen-selection.json', selection)
    monkeypatch.setattr(fixed.screen, 'verify', lambda _: (config, data))
    monkeypatch.setattr(fixed.screen, 'check_rows', lambda *a, **k: rows)
    with pytest.raises(ValueError, match='ICL screen failed'):
        fixed.verified_screen(tmp_path)
    selection['eligible'] = True
    (tmp_path / 'screen-selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='selection rule mismatch'):
        fixed.verified_screen(tmp_path)

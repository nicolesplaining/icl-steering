import json

import pytest

from analysis import gsm8k_fixed_controls as controls


def test_control_registry_has_no_new_settings():
    specs = controls.specifications({'random_seeds': [31, 59, 83]})
    assert len(specs) == 16
    new = set(specs) - set(controls.fixed.CONDITIONS)
    assert new == {'legacy_mean', 'mean_norm', 'scalar', 'permuted', 'rotated_pairs',
                   'reverse', 'random_31', 'random_59', 'random_83'}
    for name in new - {'legacy_mean'}:
        assert {k: specs[name][k] for k in controls.fixed.SETTING} == controls.fixed.SETTING
    assert specs['legacy_mean']['layer'] == 7 and specs['legacy_mean']['positions'] == 'all'


@pytest.mark.parametrize('wins,losses,expected', [(0, 0, 1), (4, 4, 1),
    (8, 0, .0078125), (12, 1, .00341796875), (1, 12, .00341796875)])
def test_exact_mcnemar_known_values(wins, losses, expected):
    assert controls.exact_mcnemar(wins, losses) == expected


def test_holm_adjustment_and_monotonicity():
    assert controls.holm({'a': .01, 'b': .02, 'c': .9}) == {'a': .03, 'b': .04, 'c': .9}
    assert controls.holm({'a': .03, 'b': .031, 'c': .032}) == {'a': .09, 'b': .09, 'c': .09}


def test_reservation_rejected(tmp_path):
    (tmp_path / 'generations.jsonl').write_text(json.dumps({
        'split': 'reserved', 'condition': 'permuted', 'problem_id': 'q'})+'\n')
    with pytest.raises(ValueError, match='Unexpected'):
        controls.check_rows(tmp_path, {}, {})


def test_existing_candidate_records_immutable(tmp_path):
    row = {'split': 'validation', 'condition': 'steered', 'problem_id': 'q', 'text': 'original'}
    (tmp_path / 'original-rows.json').write_text(json.dumps([row]))
    (tmp_path / 'generations.jsonl').write_text(json.dumps({**row, 'text': 'changed'})+'\n')
    with pytest.raises(ValueError, match='records changed'):
        controls.check_rows(tmp_path, {}, {})


def test_no_inherited_answer_revision_in_report(tmp_path, monkeypatch):
    packet = {'items': [{'response_id': 'r'}]}
    original = {'response_id': 'r', 'stated_answer': '5', 'reviewed': True, 'rationale': 'literal'}
    annotations = {'packet_sha256': controls.audit.digest(packet), 'answers': [{**original, 'stated_answer': '6'}]}
    path = tmp_path / 'annotations.json';path.write_text(json.dumps(annotations))
    (tmp_path / 'validation-review-packet.json').write_text(json.dumps(packet))
    (tmp_path / 'inherited-annotations.json').write_text(json.dumps({'answers': [original]}))
    (tmp_path / 'review-freeze.json').write_text(json.dumps({
        'annotations_file_sha256': controls.base.file_hash(path),
        'packet_sha256': controls.audit.digest(packet), 'annotations_commit': 'frozen'}))
    monkeypatch.setattr(controls, 'verify', lambda _: ({}, {}))
    monkeypatch.setattr(controls, 'check_rows', lambda *a, **k: [])
    with pytest.raises(ValueError, match='inherited annotation changed'):
        controls.report(tmp_path, path)

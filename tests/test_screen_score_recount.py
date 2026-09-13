from fractions import Fraction
import json

import pytest

from analysis import gsm8k_answer_audit as production
from analysis import gsm8k_test_development as screen
from analysis import screen_score_recount as independent


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def fixture(root):
    config = {'n_screen': 4, 'n_reserved': 2, 'bootstrap_seed': 907, 'bootstrap_samples': 101,
              'min_icl_gain': .05, 'max_truncation_rate': .05}
    questions = [{'problem_id': f'gsm8k:train:{i}', 'question': f'Question {i}', 'answer': str(i+1),
        'prompts': {k: f'{k}: question {i}' for k in screen.CONDITIONS}} for i in range(4)]
    data = {'splits': {'validation': questions, 'reserved': [
        {'problem_id': 'gsm8k:train:10'}, {'problem_id': 'gsm8k:train:11'}]}}
    save(root/'prepared.json', data)
    manifest = {'config': config, 'conditions': screen.CONDITIONS,
                'prepared_sha256': independent.file_hash(root/'prepared.json')}
    save(root/'manifest.json', manifest)
    save(root/'declaration.json', {'manifest_sha256': independent.file_hash(root/'manifest.json'),
                                  'declaration_commit': 'synthetic'})
    # Tuple: explicit parse available, stated value, truncated. Gold is i+1.
    outcomes = {
        'zero': [(True, '1', False), (True, '9', False), (False, '3', False), (False, None, False)],
        'icl_a': [(True, '1', False), (True, '2', False), (False, '3', False), (True, '4', False)],
        'icl_b': [(True, '1', False), (True, '2', False), (False, '3', False), (True, '4', True)],
        'first': [(False, '10', False), (True, '2', False), (False, None, False), (True, '4', True)],
        'cot': [(True, '1', False), (False, '4/2', False), (False, '5', False), (True, '4', False)]}
    rows, transcriptions = [], {}
    for condition in screen.CONDITIONS:
        for question, (parseable, value, truncated) in zip(questions, outcomes[condition]):
            text = f'Synthetic response: {value}'
            correct = bool(parseable and Fraction(value) == Fraction(question['answer']))
            row = {'split': 'validation', 'condition': condition, 'problem_id': question['problem_id'],
                'answer': question['answer'], 'prompt': question['prompts'][condition],
                'text': text, 'solution_text': text, 'parseable': parseable,
                'parsed_answer': value if parseable else None, 'correct': correct,
                'completed_correct': correct and not truncated, 'truncated': truncated}
            rows.append(row)
            if not parseable:
                key = production.digest({'question': question['question'], 'response': text})
                transcriptions[key] = value
    (root/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    packet = production.make_packet(rows, data, 'validation', screen.CONDITIONS)
    save(root/'validation-review-packet.json', packet)
    save(root/'screen-complete.json', {'status': 'awaiting_blinded_review', 'rows': 20,
                                     'packet_sha256': production.digest(packet)})
    annotations = {'packet_sha256': production.digest(packet), 'answers': [
        {'response_id': item['response_id'], 'stated_answer': transcriptions[item['response_id']],
         'reviewed': True, 'rationale': 'Hand-specified synthetic outcome.'} for item in packet['items']]}
    for name in ['validation-review-annotations.json', 'screen-annotations.json']:
        save(root/name, annotations)
    save(root/'review-freeze.json', {'packet_sha256': production.digest(packet),
        'annotations_file_sha256': independent.file_hash(root/'validation-review-annotations.json'),
        'annotations_commit': 'synthetic'})
    report = production.score(rows, data, 'validation', screen.CONDITIONS, packet, annotations, 101)
    save(root/'screen-audit.json', report)
    decision = screen.gates(report['summary'], config)
    decision.update(manifest_sha256=independent.file_hash(root/'manifest.json'),
        rows_sha256=production.digest(rows), annotations_sha256=production.digest(annotations),
        audit_sha256=independent.file_hash(root/'screen-audit.json'))
    save(root/'screen-selection.json', decision)
    return report, decision


def test_recounts_hand_specified_parsed_unparsed_wrong_fractional_and_truncated_answers(tmp_path):
    fixture(tmp_path)
    result = independent.recount(tmp_path)
    assert result['status'] == 'passed' and result['paired_intervals_checked'] == 8
    assert {k: v['audited'] for k, v in result['counts'].items()} == {
        'zero': 2, 'icl_a': 4, 'icl_b': 3, 'first': 1, 'cot': 3}
    assert {k: v['primary'] for k, v in result['counts'].items()} == {
        'zero': 1, 'icl_a': 3, 'icl_b': 2, 'first': 1, 'cot': 2}
    assert result['checks']['truncation'] is False and result['eligible'] is False
    assert result['reserved_rows'] == 0


@pytest.mark.parametrize('corruption', ['count', 'interval', 'gate'])
def test_rejects_corrupted_scores_even_when_report_hash_is_updated(tmp_path, corruption):
    report, decision = fixture(tmp_path)
    if corruption == 'count':
        report['summary']['icl_a']['audited_completed_correct'] -= 1
    elif corruption == 'interval':
        report['summary']['icl_a']['paired']['zero']['ci95'][0] += .125
    else:
        decision['eligible'] = True
    save(tmp_path/'screen-audit.json', report)
    decision['audit_sha256'] = independent.file_hash(tmp_path/'screen-audit.json')
    save(tmp_path/'screen-selection.json', decision)
    with pytest.raises(AssertionError):
        independent.recount(tmp_path)


def test_requires_frozen_annotations(tmp_path):
    fixture(tmp_path)
    freeze = independent.read(tmp_path/'review-freeze.json')
    freeze['annotations_commit'] = ''
    save(tmp_path/'review-freeze.json', freeze)
    with pytest.raises(AssertionError):
        independent.recount(tmp_path)

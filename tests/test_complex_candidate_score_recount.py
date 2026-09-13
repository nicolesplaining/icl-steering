from fractions import Fraction
import json
import shutil

import pytest

from analysis import complex_candidate_score_recount as check
from analysis import complex_candidate_gates as gates
from analysis import gsm8k_answer_audit as audit
from analysis.gsm8k_fixed_controls import exact_mcnemar, holm
from icl_steering.report import paired_difference


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def build(root, candidate_limit=254):
    questions = [{'problem_id': f'gsm8k:train:{i}', 'question': f'Synthetic question {i}', 'answer': '1',
                  'prompts': {k: f'{k}: {i}' for k in gates.BASELINES}} for i in range(256)]
    data = {'splits': {'validation': questions,
                      'reserved': [{'problem_id': f'gsm8k:train:{i}'} for i in range(256, 768)]}}
    rows, stated, grouped = [], {}, {k: {} for k in ['primary', 'audited']}
    for condition in gates.CONDITIONS:
        # Four missing credits at indices 0,2,3,4; all remaining values below limit are correct.
        limit = candidate_limit if condition == 'steered' else 244
        for metric in grouped:
            grouped[metric][condition] = []
        for i, question in enumerate(questions):
            parseable = i not in {1, 2, 3}
            value = '1' if 0 < i < limit else '9'
            if i == 1:
                value = '2/2'
            elif i == 2:
                value = '2'
            elif i == 3:
                value = None
            truncated = i == 4
            correct = parseable and value == '1'
            text = f'Synthetic stated value {value}'
            prompt_kind = condition if condition in gates.BASELINES else 'zero'
            row = {'split': 'validation', 'condition': condition, 'problem_id': question['problem_id'],
                   'prompt': question['prompts'][prompt_kind], 'answer': '1', 'text': text, 'solution_text': text,
                   'parseable': parseable, 'parsed_answer': value if parseable else None,
                   'correct': correct, 'completed_correct': correct and not truncated, 'truncated': truncated}
            rows.append(row)
            if not parseable:
                stated[audit.digest({'question': question['question'], 'response': text})] = value
            for metric in grouped:
                ok = (correct if metric == 'primary' else value is not None and Fraction(value) == 1) and not truncated
                grouped[metric][condition].append({'problem_id': question['problem_id'], 'correct': ok})
    packet = audit.make_packet(rows, data, 'validation', gates.CONDITIONS)
    annotations = {'packet_sha256': audit.digest(packet), 'answers': [
        {'response_id': x['response_id'], 'stated_answer': stated[x['response_id']],
         'reviewed': True, 'rationale': 'Hand-specified synthetic response.'} for x in packet['items']]}
    inherited = {'packet_sha256': 'old-screen-packet', 'answers': annotations['answers']}
    for name, value in [('prepared.json', data), ('baseline-rows.json', [r for r in rows if r['condition'] in gates.BASELINES]),
                        ('source-annotations.json', inherited)]:
        save(root/name, value)
    manifest = {'conditions': gates.CONDITIONS, 'icl_screen_passed': True, 'confirmation_supported': False,
                'config': {'n_screen': 256, 'n_reserved': 512, 'bootstrap_samples': 10000, 'bootstrap_seed': 907},
                'files_sha256': {name: check.file_hash(root/name) for name in
                                ['prepared.json', 'baseline-rows.json', 'source-annotations.json']}}
    save(root/'manifest.json', manifest)
    mh = check.file_hash(root/'manifest.json')
    save(root/'fit.json', {'manifest_sha256': mh, 'new_fits': 0})
    fh = check.file_hash(root/'fit.json')
    save(root/'fit-declaration.json', {'manifest_sha256': mh, 'fit_sha256': fh, 'declaration_commit': 'synthetic'})
    (root/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    save(root/'validation-review-packet.json', packet)
    save(root/'validation-annotations.json', annotations)
    save(root/'review-freeze.json', {'packet_sha256': audit.digest(packet),
                                   'annotations_file_sha256': check.file_hash(root/'validation-annotations.json'),
                                   'annotations_commit': 'synthetic'})
    save(root/'validation-complete.json', {'status': 'awaiting_blinded_review',
                                         'packet_sha256': audit.digest(packet), 'rows': len(rows)})
    save(root/'trajectory-check.json', {'status': 'passed', 'complete': True,
        'manifest_sha256': mh, 'fit_sha256': fh, 'rows_sha256': audit.digest(rows),
        'candidate_prefill_first_tokens_matched': 256, 'completed_batches': 768,
        'new_generation_rows': 3072, 'reserved_rows': 0})
    report = audit.score(rows, data, 'validation', gates.CONDITIONS, packet, annotations, 10000)
    report['contrasts'] = {}
    for metric, groups in grouped.items():
        pairs = {name: paired_difference(groups['steered'], groups[name], 10000, 907)
                 for name in gates.CONDITIONS if name != 'steered'}
        p = {name: exact_mcnemar(v['wins'], v['losses']) for name, v in pairs.items()}
        adjusted = holm(p)
        for name in pairs:
            pairs[name].update(mcnemar_two_sided_p=p[name], holm_p=adjusted[name])
        report['contrasts'][metric] = pairs
    decision = gates.development_gate(report['summary'], report['contrasts']['audited']['zero'], True)
    decision.update(manifest_sha256=mh, fit_sha256=fh, rows_sha256=audit.digest(rows),
                    annotations_sha256=audit.digest(annotations))
    save(root/'validation-audit.json', report)
    save(root/'selection.json', decision)


@pytest.fixture(scope='module')
def golden(tmp_path_factory):
    root = tmp_path_factory.mktemp('complex-score-golden')
    build(root)
    return root


@pytest.fixture
def example(golden, tmp_path):
    shutil.copytree(golden, tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_independent_recount_matches_hand_counts_and_exact_tests(example):
    result = check.recount(example)
    assert result['eligible'] is True and result['paired_contrasts_checked'] == 32
    assert result['counts']['steered'] == {'primary': 249, 'audited': 250, 'unparsed': 3, 'truncated': 1}
    assert result['counts']['zero'] == {'primary': 239, 'audited': 240, 'unparsed': 3, 'truncated': 1}
    report = check.read(example/'validation-audit.json')
    for metric in ['primary', 'audited']:
        value = report['contrasts'][metric]['zero']
        assert value['wins'] == 10 and value['losses'] == 0
        assert value['mcnemar_two_sided_p'] == 1/512 and value['holm_p'] == 1/32
    assert result['inherited_annotations'] == 3 and result['reserved_rows'] == 0


@pytest.mark.parametrize('field', ['count', 'interval', 'exact_p', 'holm', 'gate'])
def test_rejects_corrupted_results(example, field):
    report = check.read(example/'validation-audit.json')
    decision = check.read(example/'selection.json')
    if field == 'count':
        report['summary']['steered']['audited_completed_correct'] -= 1
    elif field == 'interval':
        report['contrasts']['audited']['zero']['ci95'][0] += .01
    elif field == 'exact_p':
        report['contrasts']['primary']['zero']['mcnemar_two_sided_p'] *= 2
    elif field == 'holm':
        report['contrasts']['audited']['zero']['holm_p'] /= 2
    else:
        decision['eligible'] = False
    save(example/'validation-audit.json', report)
    save(example/'selection.json', decision)
    with pytest.raises(AssertionError):
        check.recount(example)


@pytest.mark.parametrize('field', ['freeze', 'runtime', 'inherited'])
def test_rejects_missing_review_incomplete_runtime_or_changed_inheritance(example, field):
    if field == 'freeze':
        path = example/'review-freeze.json'; value = check.read(path)
        value['annotations_commit'] = ''
    elif field == 'runtime':
        path = example/'trajectory-check.json'; value = check.read(path)
        value['complete'] = False
    else:
        path = example/'validation-annotations.json'; value = check.read(path)
        value['answers'][0]['rationale'] = 'Changed after the inherited review.'
        save(path, value)
        freeze = check.read(example/'review-freeze.json')
        freeze['annotations_file_sha256'] = check.file_hash(path)
        save(example/'review-freeze.json', freeze)
        for name in ['validation-audit.json', 'selection.json']:
            parent = check.read(example/name)
            parent['annotations_sha256'] = check.digest(value)
            if name == 'validation-audit.json':
                parent['annotations'] = value['answers']
            save(example/name, parent)
    save(path, value)
    with pytest.raises(AssertionError):
        check.recount(example)


def test_tied_candidate_is_a_verified_failure(tmp_path):
    build(tmp_path, candidate_limit=244)
    result = check.recount(tmp_path)
    assert result['eligible'] is False
    assert result['checks']['beats_controls'] is False
    assert result['checks']['beats_same_map_prefill'] is False
    assert result['checks']['zero_gain'] is False and result['checks']['zero_interval'] is False

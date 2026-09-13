"""Independently verify a completed, reviewed five-condition GSM8K ICL screen."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random

from analysis.paired_bootstrap_check import interval


CONDITIONS = {'zero', 'icl_a', 'icl_b', 'first', 'cot'}


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recount(run, annotation_path=None):
    annotation_path = annotation_path or run/'validation-review-annotations.json'
    manifest, data, declaration = [read(run/name) for name in
        ['manifest.json', 'prepared.json', 'declaration.json']]
    config = manifest['config']
    assert manifest['prepared_sha256'] == file_hash(run/'prepared.json')
    assert declaration['manifest_sha256'] == file_hash(run/'manifest.json') and declaration['declaration_commit']
    assert set(manifest['conditions']) == CONDITIONS
    assert config['bootstrap_seed'] == 907
    questions = {p['problem_id']: p for p in data['splits']['validation']}
    reserved = {p['problem_id'] for p in data['splits']['reserved']}
    n = config['n_screen']
    assert len(questions) == len(data['splits']['validation']) == n
    assert len(reserved) == len(data['splits']['reserved']) == config['n_reserved']
    assert not set(questions) & reserved
    report, decision, packet, freeze = [read(run/name) for name in
        ['screen-audit.json', 'screen-selection.json', 'validation-review-packet.json', 'review-freeze.json']]
    complete = read(run/'screen-complete.json')
    annotations = read(annotation_path)
    assert freeze['annotations_commit'] and freeze['annotations_file_sha256'] == file_hash(annotation_path)
    assert freeze['packet_sha256'] == annotations['packet_sha256'] == digest(packet)
    assert complete['status'] == 'awaiting_blinded_review' and complete['packet_sha256'] == digest(packet)
    assert decision['annotations_sha256'] == report['annotations_sha256'] == digest(annotations)
    assert read(run/'screen-annotations.json') == annotations
    assert decision['audit_sha256'] == file_hash(run/'screen-audit.json')
    assert decision['manifest_sha256'] == file_hash(run/'manifest.json')
    rows = [json.loads(line) for line in (run/'generations.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == complete['rows'] == n*5 and digest(rows) == decision['rows_sha256']
    selected = sorted(rows, key=lambda r: (r['condition'], r['problem_id']))
    assert digest(selected) == report['source_sha256'] == packet['source_sha256']
    assert report['packet_sha256'] == digest(packet)
    unique = {}
    for row in rows:
        assert row['split'] == 'validation' and row['problem_id'] in questions
        assert row['condition'] in CONDITIONS and row['problem_id'] not in reserved
        question = questions[row['problem_id']]
        assert row['answer'] == question['answer'] and row['prompt'] == question['prompts'][row['condition']]
        if not row['parseable']:
            body = {'question': question['question'], 'response': row['solution_text']}
            key = digest(body)
            unique[key] = {'response_id': key, **body}
    items = [unique[key] for key in sorted(unique)]
    random.Random(907).shuffle(items)
    assert packet == {'version': 1, 'source_sha256': digest(selected), 'items': items}
    lookup = {}
    for annotation in annotations['answers']:
        key = annotation['response_id']
        assert key not in lookup and annotation['reviewed'] is True
        assert isinstance(annotation['rationale'], str) and annotation['rationale'].strip()
        value = annotation['stated_answer']
        assert value is None or isinstance(value, str)
        lookup[key] = Fraction(value) if value is not None else None
    assert set(lookup) == set(unique)
    groups = {name: {} for name in CONDITIONS}
    for row in rows:
        pid = row['problem_id']; group = groups[row['condition']]
        assert pid not in group
        if row['parseable']:
            correct = Fraction(str(row['parsed_answer'])) == Fraction(row['answer'])
        else:
            assert row['parsed_answer'] is None
            correct = False
        assert correct == row['correct']
        assert row['completed_correct'] == (correct and not row['truncated'])
        audited = correct
        if not row['parseable']:
            key = digest({'question': questions[pid]['question'], 'response': row['solution_text']})
            audited = lookup[key] is not None and lookup[key] == Fraction(row['answer'])
        group[pid] = {'primary': int(correct and not row['truncated']),
                     'audited': int(audited and not row['truncated']),
                     'unparsed': int(not row['parseable']), 'truncated': int(row['truncated'])}
    assert set(report['summary']) == CONDITIONS
    intervals, counts = {}, {}
    for name, group in groups.items():
        assert set(group) == set(questions)
        summary = report['summary'][name]
        assert summary['n'] == n
        counts[name] = {k: sum(r[k] for r in group.values()) for k in ['primary', 'audited', 'unparsed', 'truncated']}
        for local, published in [('primary', 'primary_completed_correct'), ('audited', 'audited_completed_correct'),
                                 ('unparsed', 'unparsed'), ('truncated', 'truncated')]:
            assert counts[name][local] == summary[published], (name, local)
        controls = set() if name == 'zero' else {'zero'}
        if name in {'icl_a', 'icl_b'}:
            controls.update(['first', 'cot'])
        assert set(summary['paired']) == controls
        for control in sorted(controls):
            result = summary['paired'][control]
            differences = [group[k]['audited']-groups[control][k]['audited'] for k in sorted(questions)]
            assert differences.count(1) == result['wins'] and differences.count(-1) == result['losses']
            assert sum(differences)/n == result['gain']
            actual = interval(differences, samples=config['bootstrap_samples'], seed=config['bootstrap_seed'])
            assert all(math.isclose(a, b, rel_tol=0, abs_tol=2e-15) for a, b in zip(actual, result['ci95']))
            assert len(result['ci95']) == 2
            intervals[name+'__minus__'+control] = actual
    checks = {}
    for name in ['icl_a', 'icl_b']:
        checks[name+'_gain'] = (counts[name]['audited']-counts['zero']['audited'])/n >= config['min_icl_gain']
        checks[name+'_interval'] = intervals[name+'__minus__zero'][0] > 0
    checks['truncation'] = all(v['truncated']/n <= config['max_truncation_rate'] for v in counts.values())
    assert checks == decision['checks'] and decision['eligible'] == all(checks.values())
    return {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': file_hash(Path(__file__)),
        'interval_checker_sha256': file_hash(Path(__file__).with_name('paired_bootstrap_check.py')),
        'manifest_sha256': file_hash(run/'manifest.json'), 'report_file_sha256': file_hash(run/'screen-audit.json'),
        'generation_file_sha256': file_hash(run/'generations.jsonl'),
        'annotations_file_sha256': file_hash(annotation_path), 'annotations_commit': freeze['annotations_commit'],
        'rows': len(rows), 'conditions': 5, 'reserved_rows': 0, 'counts': counts,
        'paired_intervals_checked': len(intervals), 'ci95': intervals,
        'checks': checks, 'eligible': decision['eligible'],
        'method': 'No production scorer imports. Rebuild blind packet, compare exact rational answers, '
                  'recount paired outcomes, replay bootstrap with count-weighted draws and explicit interpolation, and recompute every screen gate.',
        'limits': 'Uses saved explicit parses without reimplementing the parsing policy; shares the declared PCG64 random draws with production.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite a recount')
    result = recount(args.run)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

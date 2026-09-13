"""Recount the matched-penalty pairing diagnostic without importing the production scorer."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def recount(run):
    rows = [json.loads(line) for line in (run/'generations.jsonl').read_text().splitlines()]
    report = read(run/'diagnostic-report.json')
    data = read(run/'prepared.json')
    annotations = read(run/'validation-review-annotations.json')
    lookup = {a['response_id']: a for a in annotations['answers']}
    questions = {p['problem_id']: p['question'] for p in data['splits']['validation']}
    reserved = {p['problem_id'] for p in data['splits']['reserved']}
    assert len(questions) == 128 and len(rows) == 2176
    assert all(r['split'] == 'validation' and r['problem_id'] not in reserved for r in rows)
    assert digest(rows) == report['rows_sha256']
    assert digest(annotations) == report['annotations_sha256']
    groups = {}
    for row in rows:
        group = groups.setdefault(row['condition'], {})
        assert row['problem_id'] not in group
        if row['parseable']:
            correct = row['correct']
        else:
            key = digest({'question': questions[row['problem_id']], 'response': row['solution_text']})
            stated = lookup[key]['stated_answer']
            correct = stated is not None and Fraction(stated) == Fraction(row['answer'])
        assert row['completed_correct'] == (row['correct'] and not row['truncated'])
        group[row['problem_id']] = {
            'primary': bool(row['completed_correct']),
            'audited': bool(correct and not row['truncated']),
            'unparsed': not row['parseable'], 'truncated': row['truncated']}
    assert set(groups) == set(report['summary']) == {
        'zero', 'icl_a', 'icl_b', 'first', 'cot', 'prefill', 'steered',
        'regularized_prefill', 'mean', 'scalar', 'position_mean',
        'position_scalar', 'permuted', 'shared_low', 'permuted_low', 'real_high', 'shared_high'}
    for name, group in groups.items():
        assert set(group) == set(questions)
        summary = report['summary'][name]
        for local, published in [('primary', 'primary_completed_correct'),
                                 ('audited', 'audited_completed_correct'),
                                 ('unparsed', 'unparsed'), ('truncated', 'truncated')]:
            assert sum(r[local] for r in group.values()) == summary[published], (name, local)
    pairs = [('steered', 'shared_low'), ('steered', 'permuted_low'),
             ('real_high', 'shared_high'), ('real_high', 'permuted'),
             ('steered', 'real_high'), ('permuted_low', 'permuted'), ('shared_low', 'shared_high')]
    for metric in ['primary', 'audited']:
        p_values = {}
        assert set(report['contrasts'][metric]) == {left+'__minus__'+right for left, right in pairs}
        for name, result in report['contrasts'][metric].items():
            left, right = name.split('__minus__')
            assert (result['left'], result['right']) == (left, right)
            differences = [int(groups[left][k][metric])-int(groups[right][k][metric])
                           for k in sorted(questions)]
            wins, losses = differences.count(1), differences.count(-1)
            assert (wins, losses) == (result['wins'], result['losses'])
            assert sum(differences)/128 == result['gain']
            discordant = wins+losses
            probability = min(Fraction(1), 2*sum(
                (Fraction(math.comb(discordant, k), 2**discordant)
                 for k in range(min(wins, losses)+1)), Fraction(0)))
            p_values[name] = float(probability)
            assert math.isclose(float(probability), result['mcnemar_two_sided_p'],
                                rel_tol=1e-12, abs_tol=0)
        running = 0.
        for rank, name in enumerate(sorted(p_values, key=p_values.get)):
            running = max(running, min(1., (7-rank)*p_values[name]))
            assert math.isclose(running, report['contrasts'][metric][name]['holm_p'],
                                rel_tol=1e-12, abs_tol=0)
    assert report['candidate_selected'] is None and report['confirmation_supported'] is False
    assert report['reserved_rows'] == 0
    return {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'report_file_sha256': hashlib.sha256((run/'diagnostic-report.json').read_bytes()).hexdigest(),
            'recount_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows': len(rows), 'conditions': len(groups), 'paired_contrasts': 14,
            'counts_wins_losses_exact_p_and_holm_recounted': True,
            'bootstrap_intervals': 'Produced by the frozen reporter; not independently recomputed here.',
            'reserved_rows': 0, 'candidate_selected': None, 'confirmation_supported': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = recount(args.run)
    if args.output.exists():
        raise ValueError('Refuse to replace a prior recount')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

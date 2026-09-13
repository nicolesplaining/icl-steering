"""Replay declared intervals with count-weighted draws and explicit interpolation.

Uses NumPy's declared PCG64 generator, but does not import the production
reporter or use its indexed-array bootstrap and quantile implementation.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(differences, samples=10000, seed=907):
    n = len(differences)
    if not n or not set(differences) <= {-1, 0, 1}:
        raise ValueError('Require paired binary-score differences')
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = []
    for _ in range(samples):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        boot.append(sum(int(count)*difference for count, difference in zip(counts, differences))/n)
    boot.sort()
    values = []
    for probability in [Fraction(1, 40), Fraction(39, 40)]:
        position = probability*(samples-1)
        low = position.numerator//position.denominator
        fraction = float(position-low)
        values.append(boot[low]+fraction*(boot[min(low+1, samples-1)]-boot[low]))
    return values


def check(run, report_name):
    report_path = run/report_name
    report = json.loads(report_path.read_text())
    rows = [json.loads(line) for line in (run/'generations.jsonl').read_text().splitlines()]
    data = json.loads((run/'prepared.json').read_text())
    annotations = json.loads((run/'validation-review-annotations.json').read_text())
    freeze = json.loads((run/'review-freeze.json').read_text())
    assert freeze['annotations_commit'] and freeze['annotations_file_sha256'] == file_hash(run/'validation-review-annotations.json')
    assert digest(annotations) == report['annotations_sha256']
    questions = {p['problem_id']: p['question'] for p in data['splits']['validation']}
    reserved = {p['problem_id'] for p in data['splits']['reserved']}
    lookup = {a['response_id']: a['stated_answer'] for a in annotations['answers']}
    groups = {}
    for row in rows:
        assert row['split'] == 'validation' and row['problem_id'] not in reserved
        group = groups.setdefault(row['condition'], {})
        assert row['problem_id'] not in group
        correct = row['correct']
        if not row['parseable']:
            stated = lookup[digest({'question': questions[row['problem_id']], 'response': row['solution_text']})]
            correct = stated is not None and Fraction(stated) == Fraction(row['answer'])
        group[row['problem_id']] = {'primary': int(row['correct'] and not row['truncated']),
                                    'audited': int(correct and not row['truncated'])}
    assert all(set(g) == set(questions) for g in groups.values())
    checked = {}
    for metric in ['primary', 'audited']:
        for name, result in report['contrasts'][metric].items():
            left, right = result.get('left', 'steered'), result.get('right', name)
            differences = [groups[left][k][metric]-groups[right][k][metric] for k in sorted(questions)]
            actual = interval(differences)
            np.testing.assert_allclose(actual, result['ci95'], rtol=0, atol=2e-15,
                                       err_msg=metric+':'+name)
            assert sum(differences)/len(differences) == result['gain']
            checked[metric+':'+name] = actual
    return {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'script_sha256': file_hash(Path(__file__)), 'run': run.name,
            'report_file_sha256': file_hash(report_path),
            'generation_file_sha256': file_hash(run/'generations.jsonl'),
            'annotations_file_sha256': file_hash(run/'validation-review-annotations.json'),
            'annotations_commit': freeze['annotations_commit'], 'intervals_checked': len(checked),
            'samples': 10000, 'seed': 907, 'reserved_rows': 0,
            'method': 'Same declared PCG64 draws; independent count-weighted resampling, Python sums and sorting, explicit linear percentile interpolation. No production reporter imports.',
            'ci95': checked}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', choices=['validation-audit.json', 'diagnostic-report.json'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Refuse to overwrite verification')
    result = check(args.run, args.report)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'ci95'}, indent=2))

"""Recount LTV development scores without importing the production scorer."""

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
    report = read(run/'validation-audit.json')
    decision = read(run/'selection.json')
    data = read(run/'prepared.json')
    annotations = read(run/'validation-review-annotations.json')
    lookup = {a['response_id']: a for a in annotations['answers']}
    questions = {p['problem_id']: p['question'] for p in data['splits']['validation']}
    reserved = {p['problem_id'] for p in data['splits']['reserved']}
    assert len(questions) == 128 and len(rows) == 1408
    assert all(r['split'] == 'validation' and r['problem_id'] not in reserved for r in rows)
    assert digest(rows) == decision['rows_sha256']
    assert digest(annotations) == decision['annotations_sha256']
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
    assert set(groups) == set(report['summary']) and len(groups) == 11
    for name, group in groups.items():
        assert set(group) == set(questions)
        summary = report['summary'][name]
        for local, published in [('primary', 'primary_completed_correct'),
                                 ('audited', 'audited_completed_correct'),
                                 ('unparsed', 'unparsed'), ('truncated', 'truncated')]:
            assert sum(r[local] for r in group.values()) == summary[published], (name, local)
    for metric in ['primary', 'audited']:
        p_values = {}
        assert set(report['contrasts'][metric]) == set(groups)-{'steered'}
        for name, result in report['contrasts'][metric].items():
            differences = [int(groups['steered'][k][metric])-int(groups[name][k][metric])
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
            running = max(running, min(1., (10-rank)*p_values[name]))
            assert math.isclose(running, report['contrasts'][metric][name]['holm_p'],
                                rel_tol=1e-12, abs_tol=0)
    counts = {k: sum(r['audited'] for r in v.values()) for k, v in groups.items()}
    # The inherited ICL screen was frozen in the parent manifest before this run.
    assert decision['checks']['icl_screen'] is True
    checks = {'icl_screen': True, 'zero_gain': (counts['steered']-counts['zero'])/128 >= .03,
              'beats_controls': all(counts['steered'] > counts[k] for k in
                                    ['mean', 'scalar', 'scalar_norm', 'permuted']),
              'matches_text_and_prefill': all(counts['steered'] >= counts[k] for k in
                                             ['first', 'cot', 'prefill']),
              'truncation': sum(r['truncated'] for r in groups['steered'].values())/128 <= .05}
    assert checks == decision['checks'] and decision['eligible'] == all(checks.values())
    return {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'report_file_sha256': hashlib.sha256((run/'validation-audit.json').read_bytes()).hexdigest(),
            'recount_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows': len(rows), 'conditions': len(groups), 'paired_contrasts': 20,
            'counts_wins_losses_exact_p_and_holm_recounted': True,
            'bootstrap_intervals': 'Produced by the frozen reporter; not independently recomputed here.',
            'inherited_icl_gate': 'Asserted preserved; not independently re-estimated here.',
            'new_candidate_gates_recounted': True, 'reserved_rows': 0,
            'eligible': decision['eligible']}


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

"""Recount the frozen complex-question candidate without production scorer imports."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

from analysis.paired_bootstrap_check import interval


BASELINES = {'zero', 'icl_a', 'icl_b', 'first', 'cot'}
NULLS = {'mean', 'scalar', 'position_mean', 'position_scalar', 'permuted',
         'shared_low', 'permuted_low', 'shared_high'}
CONDITIONS = BASELINES | NULLS | {'steered', 'regularized_prefill', 'prefill', 'real_high'}


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_p(wins, losses):
    n = wins+losses
    return float(min(Fraction(1), 2*sum(
        (Fraction(math.comb(n, k), 2**n) for k in range(min(wins, losses)+1)), Fraction(0))))


def recount(run):
    manifest, fit, declaration, data = [read(run/name) for name in
        ['manifest.json', 'fit.json', 'fit-declaration.json', 'prepared.json']]
    mh, fh = file_hash(run/'manifest.json'), file_hash(run/'fit.json')
    assert declaration['manifest_sha256'] == fit['manifest_sha256'] == mh
    assert declaration['fit_sha256'] == fh and declaration['declaration_commit']
    assert manifest['icl_screen_passed'] is True and manifest['confirmation_supported'] is False
    assert set(manifest['conditions']) == CONDITIONS and fit['new_fits'] == 0
    for name, expected in manifest['files_sha256'].items():
        assert file_hash(run/name) == expected, name
    config = manifest['config']
    assert (config['n_screen'], config['n_reserved'], config['bootstrap_samples'],
            config['bootstrap_seed']) == (256, 512, 10000, 907)
    questions = {p['problem_id']: p for p in data['splits']['validation']}
    reserved = {p['problem_id'] for p in data['splits']['reserved']}
    assert len(questions) == len(data['splits']['validation']) == 256
    assert len(reserved) == len(data['splits']['reserved']) == 512
    assert not set(questions) & reserved
    report, decision, packet, annotations, freeze, complete, runtime = [read(run/name) for name in
        ['validation-audit.json', 'selection.json', 'validation-review-packet.json',
         'validation-annotations.json', 'review-freeze.json', 'validation-complete.json',
         'trajectory-check.json']]
    assert freeze['annotations_commit'] and freeze['annotations_file_sha256'] == file_hash(run/'validation-annotations.json')
    assert freeze['packet_sha256'] == annotations['packet_sha256'] == digest(packet)
    assert report['annotations_sha256'] == decision['annotations_sha256'] == digest(annotations)
    assert report['annotations'] == annotations['answers']
    assert complete['status'] == 'awaiting_blinded_review' and complete['packet_sha256'] == digest(packet)
    assert decision['manifest_sha256'] == runtime['manifest_sha256'] == mh
    assert decision['fit_sha256'] == runtime['fit_sha256'] == fh
    assert runtime['status'] == 'passed' and runtime['complete'] is True
    assert runtime['candidate_prefill_first_tokens_matched'] == 256
    assert runtime['completed_batches'] == 768 and runtime['new_generation_rows'] == 3072
    assert runtime['reserved_rows'] == 0
    rows = [json.loads(line) for line in (run/'generations.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == complete['rows'] == 4352
    assert digest(rows) == decision['rows_sha256'] == runtime['rows_sha256']
    selected = sorted(rows, key=lambda r: (r['condition'], r['problem_id']))
    assert digest(selected) == report['source_sha256'] == packet['source_sha256']
    assert report['packet_sha256'] == digest(packet)
    assert sorted(read(run/'baseline-rows.json'), key=lambda r: (r['condition'], r['problem_id'])) == [
        r for r in selected if r['condition'] in BASELINES]
    unique = {}
    for row in rows:
        assert row['split'] == 'validation' and row['problem_id'] in questions
        assert row['condition'] in CONDITIONS and row['problem_id'] not in reserved
        question = questions[row['problem_id']]
        prompt_kind = row['condition'] if row['condition'] in BASELINES else 'zero'
        assert row['answer'] == question['answer'] and row['prompt'] == question['prompts'][prompt_kind]
        if not row['parseable']:
            body = {'question': question['question'], 'response': row['solution_text']}
            key = digest(body)
            unique[key] = {'response_id': key, **body}
    items = [unique[key] for key in sorted(unique)]
    random.Random(907).shuffle(items)
    assert packet == {'version': 1, 'source_sha256': digest(selected), 'items': items}
    lookup, records = {}, {}
    for annotation in annotations['answers']:
        key = annotation['response_id']
        assert key not in lookup and annotation['reviewed'] is True
        assert isinstance(annotation['rationale'], str) and annotation['rationale'].strip()
        value = annotation['stated_answer']
        assert value is None or isinstance(value, str)
        lookup[key] = Fraction(value) if value is not None else None
        records[key] = annotation
    assert set(lookup) == set(unique)
    inherited = read(run/'source-annotations.json')['answers']
    assert len({a['response_id'] for a in inherited}) == len(inherited)
    assert all(records.get(a['response_id']) == a for a in inherited)
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
    counts = {}
    assert set(report['summary']) == CONDITIONS
    for name, group in groups.items():
        assert set(group) == set(questions) and report['summary'][name]['n'] == 256
        counts[name] = {k: sum(r[k] for r in group.values()) for k in ['primary', 'audited', 'unparsed', 'truncated']}
        for local, published in [('primary', 'primary_completed_correct'), ('audited', 'audited_completed_correct'),
                                 ('unparsed', 'unparsed'), ('truncated', 'truncated')]:
            assert counts[name][local] == report['summary'][name][published], (name, local)

    @lru_cache(maxsize=None)
    def replay(differences):
        return interval(differences, samples=10000, seed=907)

    intervals = {}
    assert set(report['contrasts']) == {'primary', 'audited'}
    for metric in ['primary', 'audited']:
        contrasts = report['contrasts'][metric]
        assert set(contrasts) == CONDITIONS-{'steered'}
        probabilities = {}
        for name, result in contrasts.items():
            differences = tuple(groups['steered'][k][metric]-groups[name][k][metric] for k in sorted(questions))
            wins, losses = differences.count(1), differences.count(-1)
            assert (wins, losses) == (result['wins'], result['losses'])
            assert sum(differences)/256 == result['gain']
            actual = replay(differences)
            assert len(result['ci95']) == 2
            assert all(math.isclose(a, b, rel_tol=0, abs_tol=2e-15) for a, b in zip(actual, result['ci95']))
            intervals[metric+':'+name] = actual
            probabilities[name] = exact_p(wins, losses)
            assert math.isclose(probabilities[name], result['mcnemar_two_sided_p'], rel_tol=1e-12, abs_tol=0)
        running = 0.
        for rank, name in enumerate(sorted(probabilities, key=probabilities.get)):
            running = max(running, min(1., (16-rank)*probabilities[name]))
            assert math.isclose(running, contrasts[name]['holm_p'], rel_tol=1e-12, abs_tol=0)
    # Recompute the frozen gates, without importing their production implementation.
    candidate = counts['steered']
    checks = {
        'icl_screen': manifest['icl_screen_passed'] is True,
        'zero_gain': (candidate['audited']-counts['zero']['audited'])/256 >= .03,
        'zero_interval': intervals['audited:zero'][0] > 0,
        'beats_controls': all(candidate['audited'] > counts[k]['audited'] for k in NULLS),
        'beats_same_map_prefill': candidate['audited'] > counts['regularized_prefill']['audited'],
        'matches_text_and_prefill': all(candidate['audited'] >= counts[k]['audited'] for k in ['first', 'cot', 'prefill']),
        'parser_not_below_zero': candidate['primary'] >= counts['zero']['primary'],
        'truncation': candidate['truncated']/256 <= .05,
    }
    assert checks == decision['checks'] and decision['eligible'] == all(checks.values())
    assert decision['candidate'] == 'steered' and decision['reserved_generation_authorized'] is False
    return {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': file_hash(Path(__file__)),
        'interval_checker_sha256': file_hash(Path(__file__).with_name('paired_bootstrap_check.py')),
        'manifest_sha256': mh, 'fit_sha256': fh, 'report_file_sha256': file_hash(run/'validation-audit.json'),
        'generation_file_sha256': file_hash(run/'generations.jsonl'),
        'annotations_file_sha256': file_hash(run/'validation-annotations.json'),
        'annotations_commit': freeze['annotations_commit'], 'rows': len(rows), 'conditions': 17,
        'inherited_baseline_rows': 1280, 'inherited_annotations': len(inherited),
        'paired_contrasts_checked': 32, 'counts': counts, 'ci95': intervals,
        'checks': checks, 'eligible': decision['eligible'], 'reserved_rows': 0,
        'method': 'Rebuild the blind packet, compare rational answers, recount paired outcomes, '
                  'replay bootstrap intervals, independently calculate exact McNemar and Holm, '
                  'and recompute every candidate gate. No production scorer imports.',
        'limits': 'Uses saved parses without reimplementing parsing. Shares declared PCG64 draws. '
                  'Checks the prior screen flag and runtime audit provenance; does not repeat their underlying audits.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite a recount')
    result = recount(args.run)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'ci95'}, indent=2))

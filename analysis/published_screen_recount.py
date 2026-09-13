"""Recount all three published-screen conditions without production scorer imports."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re

from analysis.paired_bootstrap_check import interval


CONDITIONS = ['zero', 'icl_original', 'icl_complex']
PAIRS = [('icl_complex', 'zero'), ('icl_original', 'zero'), ('icl_complex', 'icl_original')]


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability(wins, losses):
    n = wins+losses
    return float(min(Fraction(1), Fraction(2*sum(math.comb(n, k)
                 for k in range(min(wins, losses)+1)), 2**n)))


def recount(run):
    manifest, data, declaration = [read(run/name) for name in
                                  ['manifest.json', 'prepared.json', 'declaration.json']]
    mh = sha(run/'manifest.json')
    assert declaration['manifest_sha256'] == mh and declaration['declaration_commit']
    assert manifest['prepared_sha256'] == sha(run/'prepared.json')
    assert manifest['conditions'] == CONDITIONS and manifest['fixed_target'] == 'icl_complex'
    assert manifest['reserved_generation_supported'] is False
    assert manifest['extraction_generation_supported'] is False
    config = manifest['config']
    assert [config[k] for k in ['seed', 'n_extract', 'n_screen', 'n_reserved', 'batch_size',
                                'max_new_tokens', 'bootstrap_samples', 'bootstrap_seed']] == [3402, 256, 512, 512, 4, 1024, 10000, 907]
    assert (config['min_icl_gain'], config['max_truncation_rate'], config['max_adjusted_p']) == (.05, .05, .05)
    assert set(data['splits']) == set(data['plan']) == {'extract', 'validation', 'reserved'}
    all_ids = []
    for name, count in [('extract', 256), ('validation', 512), ('reserved', 512)]:
        ids = [r['problem_id'] for r in data['splits'][name]]
        assert ids == data['plan'][name] and len(ids) == count
        assert ids == sorted(ids, key=lambda pid: int(pid.rsplit(':', 1)[1]))
        assert all(re.fullmatch(r'gsm8k:train:\d+', pid) for pid in ids)
        all_ids.extend(ids)
    assert len(set(all_ids)) == 1280
    questions = {r['problem_id']: r for r in data['splits']['validation']}
    report, decision, packet, annotations, freeze, complete, runtime = [read(run/name) for name in
        ['screen-audit.json', 'screen-selection.json', 'validation-review-packet.json',
         'screen-annotations.json', 'review-freeze.json', 'screen-complete.json', 'runtime.json']]
    assert runtime['physical_gpu'] == 0 and runtime['frozen'] is True and runtime['dtype'] == 'torch.bfloat16'
    assert complete['status'] == 'awaiting_blinded_review'
    assert complete['extraction_rows'] == complete['reserved_rows'] == 0
    assert complete['rows'] == 1536 and complete['batches_verified'] == 384
    assert complete['generation_file_sha256'] == sha(run/'generations.jsonl')
    assert freeze['annotations_commit']
    assert freeze['annotations_file_sha256'] == sha(run/'screen-annotations.json')
    assert freeze['packet_sha256'] == annotations['packet_sha256'] == complete['packet_sha256'] == digest(packet)
    assert report['annotations'] == annotations['answers']
    assert report['annotations_sha256'] == decision['annotations_sha256'] == digest(annotations)
    assert decision['audit_sha256'] == sha(run/'screen-audit.json')
    assert decision['manifest_sha256'] == mh
    assert report['packet_sha256'] == digest(packet)
    rows = [json.loads(line) for line in (run/'generations.jsonl').read_text().splitlines()]
    assert len(rows) == 1536 and digest(rows) == decision['rows_sha256']
    filenames = [f'{kind}-{start:04d}.json' for kind in CONDITIONS for start in range(0, 512, 4)]
    assert {p.name for p in (run/'batches').glob('*.json')} == set(filenames)
    from_batches = []
    for name in filenames:
        batch = read(run/'batches'/name)
        assert len(batch) == 4
        kind, start = name[:-5].rsplit('-', 1)
        assert [r['problem_id'] for r in batch] == data['plan']['validation'][int(start):int(start)+4]
        assert all(r['condition'] == kind for r in batch)
        from_batches.extend(batch)
    assert rows == from_batches
    ordered = sorted(rows, key=lambda row: (row['condition'], row['problem_id']))
    assert digest(ordered) == report['source_sha256'] == packet['source_sha256']
    unique = {}
    marker = re.compile(r'(?:^|\n)[ \t]*(?:question:|q:|problem:|human:|user:|'
                        r'given the following (?:question|problem)[, ])', re.I)
    for row in rows:
        assert row['split'] == 'validation' and row['problem_id'] in questions
        assert row['condition'] in CONDITIONS
        problem = questions[row['problem_id']]
        assert row['answer'] == problem['answer'] and row['prompt'] == problem['prompts'][row['condition']]
        ids = row['token_ids']
        assert len(ids) == row['generated_tokens'] and 0 < len(ids) <= 1024
        assert all(type(t) is int and t >= 0 for t in ids)
        assert not any(t in runtime['eos_token_ids'] for t in ids[:-1])
        match = marker.search(row['text'])
        cut = row['text'][:match.start()] if match else row['text']
        assert row['solution_text'] == cut.rstrip()
        reason = 'next_question' if match else ('eos' if ids[-1] in runtime['eos_token_ids'] else 'length')
        assert row['finish_reason'] == reason and row['truncated'] == (reason == 'length')
        assert reason != 'length' or len(ids) == 1024
        if not row['parseable']:
            body = {'question': problem['question'], 'response': row['solution_text']}
            key = digest(body)
            unique[key] = {'response_id': key, **body}
    items = [unique[key] for key in sorted(unique)]
    random.Random(907).shuffle(items)
    assert packet == {'version': 1, 'source_sha256': digest(ordered), 'items': items}
    lookup = {}
    for row in annotations['answers']:
        key = row['response_id']
        assert key not in lookup and row['reviewed'] is True
        assert isinstance(row['rationale'], str) and row['rationale'].strip()
        value = row['stated_answer']
        assert value is None or isinstance(value, str)
        lookup[key] = None if value is None else Fraction(value)
    assert set(lookup) == set(unique)
    groups = {name: {} for name in CONDITIONS}
    for row in rows:
        group, pid = groups[row['condition']], row['problem_id']
        assert pid not in group
        if row['parseable']:
            correct = Fraction(str(row['parsed_answer'])) == Fraction(row['answer'])
        else:
            assert row['parsed_answer'] is None
            correct = False
        assert row['correct'] == correct and row['completed_correct'] == (correct and not row['truncated'])
        audited = correct
        if not row['parseable']:
            key = digest({'question': questions[pid]['question'], 'response': row['solution_text']})
            audited = lookup[key] is not None and lookup[key] == Fraction(row['answer'])
        group[pid] = {'primary': int(correct and not row['truncated']),
                     'audited': int(audited and not row['truncated']),
                     'unparsed': int(not row['parseable']), 'truncated': int(row['truncated'])}
    counts = {}
    assert set(report['summary']) == set(CONDITIONS)
    for name, group in groups.items():
        assert set(group) == set(questions) and report['summary'][name]['n'] == 512
        counts[name] = {key: sum(row[key] for row in group.values())
                        for key in ['primary', 'audited', 'unparsed', 'truncated']}
        for local, published in [('primary', 'primary_completed_correct'), ('audited', 'audited_completed_correct'),
                                 ('unparsed', 'unparsed'), ('truncated', 'truncated')]:
            assert counts[name][local] == report['summary'][name][published]
    assert set(report['contrasts']) == {'primary', 'audited'}
    intervals, probabilities = {}, {}
    for metric in ['primary', 'audited']:
        contrasts = report['contrasts'][metric]
        assert set(contrasts) == {a+'-'+b for a, b in PAIRS}
        pvalues = {}
        for left, right in PAIRS:
            key = left+'-'+right
            result = contrasts[key]
            assert (result['left'], result['right']) == (left, right)
            diffs = [groups[left][pid][metric]-groups[right][pid][metric] for pid in sorted(questions)]
            wins, losses = diffs.count(1), diffs.count(-1)
            assert (result['wins'], result['losses']) == (wins, losses)
            assert result['gain'] == sum(diffs)/512
            actual = interval(diffs, samples=10000, seed=907)
            assert len(result['ci95']) == 2 and all(abs(a-b) < 2e-15 for a, b in zip(actual, result['ci95']))
            intervals[metric+':'+key] = actual
            pvalues[key] = probability(wins, losses)
            assert abs(result['mcnemar_two_sided_p']-pvalues[key]) < 2e-15
        sorted_keys = sorted(pvalues, key=pvalues.get)
        for index, key in enumerate(sorted_keys):
            adjusted = min(1., max((3-j)*pvalues[k] for j, k in enumerate(sorted_keys[:index+1])))
            assert abs(contrasts[key]['holm_p']-adjusted) < 2e-15
            probabilities[metric+':'+key] = adjusted
    gain = (counts['icl_complex']['audited']-counts['zero']['audited'])/512
    checks = {'complete': True, 'complex_gain': gain >= .05,
        'complex_interval': intervals['audited:icl_complex-zero'][0] > 0,
        'complex_adjusted_p': probabilities['audited:icl_complex-zero'] < .05,
        'primary_not_below_zero': counts['icl_complex']['primary'] >= counts['zero']['primary'],
        'truncation': all(values['truncated']/512 <= .05 for values in counts.values())}
    assert decision['checks'] == checks and decision['eligible'] == all(checks.values())
    assert decision['fixed_target'] == 'icl_complex'
    assert decision['reserved_generation_authorized'] is decision['steering_generation_authorized'] is False
    return {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(Path(__file__)), 'interval_script_sha256': sha(Path(__file__).with_name('paired_bootstrap_check.py')),
        'manifest_sha256': mh, 'generation_file_sha256': sha(run/'generations.jsonl'),
        'audit_sha256': sha(run/'screen-audit.json'), 'annotations_sha256': digest(annotations),
        'rows': 1536, 'batches': 384, 'contrasts_checked': 6, 'counts': counts,
        'checks': checks, 'screen_eligible': all(checks.values()), 'extraction_rows': 0, 'reserved_rows': 0,
        'method': 'No production scorer imports. Independent identities, batches/export, review packet, '
                  'annotation joins, stored-parse grading, counts, paired wins/losses, exact binomial tails, '
                  'Holm families, gates, and count-weighted bootstrap interpolation. Same declared PCG64 draws. '
                  'Does not independently parse answers, decode token IDs, or rerun model inference.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite recount')
    result = recount(args.run)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

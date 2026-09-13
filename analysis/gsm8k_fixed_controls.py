"""Supplementary controls on completed development data; never changes eligibility."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from math import comb
from pathlib import Path

from analysis import gsm8k_fixed_candidate as fixed
from analysis.gsm8k_comparison import write_fixed
from icl_steering.report import paired_difference


base, audit = fixed.base, fixed.audit
ROOT = Path(__file__).parents[1]


def code_hash():
    return audit.digest({'fixed_candidate': fixed.code_hash(), 'files': {
        str(p.relative_to(ROOT)): base.file_hash(p) for p in [Path(__file__),
        ROOT / 'research/gsm8k-fixed-controls-protocol.md']}})


def specifications(config):
    return base.specs(config, {'chosen': fixed.SETTING})


def exact_mcnemar(wins, losses):
    if min(wins, losses) < 0:
        raise ValueError('Negative discordant count')
    n = wins + losses
    return min(1., 2 * sum(comb(n, k) for k in range(min(wins, losses)+1)) / 2**n)


def holm(values):
    ordered = sorted(values, key=lambda k: (values[k], k))
    adjusted, previous = {}, 0.
    for i, k in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered)-i)*values[k]))
        adjusted[k] = previous
    return adjusted


def verify(output):
    manifest = base.read(output / 'manifest.json')
    if manifest['code_sha256'] != code_hash():
        raise ValueError('Diagnostic code or protocol changed')
    for name, expected in manifest['files_sha256'].items():
        if base.file_hash(output / name) != expected:
            raise ValueError('Frozen diagnostic input changed: ' + name)
    for path, expected in manifest['inputs_sha256'].items():
        if base.file_hash(Path(path)) != expected:
            raise ValueError('Diagnostic parent changed: ' + path)
    return manifest['config'], base.read(output / 'prepared.json')


def prepare(parent, output):
    config, data = fixed.verify(parent)
    rows = fixed.check_rows(parent, data, complete=True)
    selection = base.read(parent / 'selection.json')
    annotations = base.read(parent / 'validation-annotations.json')
    packet = base.read(parent / 'validation-review-packet.json')
    freeze = base.read(parent / 'review-freeze.json')
    if (freeze['annotations_file_sha256'] != base.file_hash(parent / 'validation-annotations.json')
            or freeze['packet_sha256'] != audit.digest(packet) or not freeze['annotations_commit']):
        raise ValueError('Candidate annotation freeze changed')
    replay = audit.score(rows, data, 'validation', fixed.CONDITIONS, packet, annotations,
                         config['bootstrap_samples'])
    if replay != base.read(parent / 'validation-audit.json'):
        raise ValueError('Candidate audit replay mismatch')
    expected = {**fixed.gates(replay['summary']), 'manifest_sha256': base.file_hash(parent / 'manifest.json'),
        'validation_rows_sha256': audit.digest(rows), 'annotations_sha256': audit.digest(annotations),
        'validation_audit_sha256': base.file_hash(parent / 'validation-audit.json')}
    if selection != expected or selection['eligible'] or selection['checks']['matches_text_cues']:
        raise ValueError('Require the unchanged, documented text-gate failure')
    names = ['prepared.json', 'maps.npz', 'validation-queries.npz', 'validation-queries.json',
             'generations.jsonl', 'validation-annotations.json', 'manifest.json', 'selection.json',
             'validation-audit.json', 'review-freeze.json']
    provenance = {str(parent / name): base.file_hash(parent / name) for name in names}
    if (output / 'manifest.json').exists():
        if base.read(output / 'manifest.json')['inputs_sha256'] != provenance:
            raise ValueError('Diagnostic request changed')
        return verify(output)
    if (output / 'generations.jsonl').exists():
        raise ValueError('Unmanifested diagnostic outputs')
    output.mkdir(parents=True, exist_ok=True)
    copies = ['prepared.json', 'maps.npz', 'validation-queries.npz', 'validation-queries.json', 'generations.jsonl']
    for name in copies:
        write_fixed(output / name, (parent / name).read_bytes())
    base.fixed(output / 'original-rows.json', rows)
    write_fixed(output / 'original-annotations.json', (parent / 'validation-annotations.json').read_bytes())
    files = [n for n in copies if n != 'generations.jsonl'] + ['original-rows.json', 'original-annotations.json']
    base.fixed(output / 'manifest.json', {'config': config, 'code_sha256': code_hash(),
        'inputs_sha256': provenance, 'files_sha256': {name: base.file_hash(output / name) for name in files},
        'parent_selection': selection, 'conditions': list(specifications(config)),
        'created_at': datetime.now(timezone.utc).isoformat(), 'confirmation_supported': False})
    return verify(output)


def check_rows(output, data, config, complete=False):
    rows = base.rows_from(output)
    keys = [(r['split'], r['condition'], r['problem_id']) for r in rows]
    if len(keys) != len(set(keys)) or any(r['split'] != 'validation' for r in rows):
        raise ValueError('Unexpected or duplicate diagnostic generation')
    old = [r for r in rows if r['condition'] in fixed.CONDITIONS]
    if old != base.read(output / 'original-rows.json'):
        raise ValueError('Completed candidate or baseline records changed')
    base.check_rows(rows, data, specifications(config), 'validation')
    if complete:
        audit.select_rows(rows, data, 'validation', list(specifications(config)))
    return rows


def run(output):
    config, data = verify(output)
    if base.read(output / 'declaration.json')['manifest_sha256'] != base.file_hash(output / 'manifest.json'):
        raise ValueError('Diagnostic declaration mismatch')
    if (output / 'diagnostic-report.json').exists():
        raise ValueError('Diagnostic frozen after reporting')
    check_rows(output, data, config)
    runner = base.ConditionalRunner(config, data, output)
    for name, spec in specifications(config).items():
        if name not in fixed.CONDITIONS:
            runner.evaluate_spec('validation', name, spec)
    rows = check_rows(output, data, config, complete=True)
    packet = audit.make_packet(rows, data, 'validation', list(specifications(config)))
    base.fixed(output / 'validation-review-packet.json', packet)
    ids = {r['response_id'] for r in packet['items']}
    inherited = [r for r in base.read(output / 'original-annotations.json')['answers'] if r['response_id'] in ids]
    base.fixed(output / 'inherited-annotations.json', {'packet_sha256': audit.digest(packet), 'answers': inherited})
    base.fixed(output / 'diagnostic-complete.json', {'status': 'awaiting_blinded_review', 'rows': len(rows),
        'packet_sha256': audit.digest(packet), 'inherited_annotations': len(inherited)})
    print(f'Complete: {len(rows)} rows; {len(packet["items"])-len(inherited)} new blinded items.', flush=True)


def report(output, annotation_path):
    config, data = verify(output)
    rows = check_rows(output, data, config, complete=True)
    packet, annotations = base.read(output / 'validation-review-packet.json'), base.read(annotation_path)
    freeze = base.read(output / 'review-freeze.json')
    if (freeze['annotations_file_sha256'] != base.file_hash(annotation_path)
            or freeze['packet_sha256'] != audit.digest(packet) or not freeze['annotations_commit']):
        raise ValueError('Diagnostic annotation freeze changed')
    lookup = {r['response_id']: r for r in annotations['answers']}
    if any(lookup.get(r['response_id']) != r for r in base.read(output / 'inherited-annotations.json')['answers']):
        raise ValueError('An inherited annotation changed')
    names = list(specifications(config))
    result = audit.score(rows, data, 'validation', names, packet, annotations, config['bootstrap_samples'])
    selected, questions = audit.select_rows(rows, data, 'validation', names)
    groups = {metric: {name: [] for name in names} for metric in ['primary', 'audited']}
    for row in selected:
        correct = row['correct']
        if not row['parseable']:
            value = lookup[audit.response_item(row, questions)['response_id']]['stated_answer']
            correct = value is not None and Fraction(value) == Fraction(row['answer'])
        groups['primary'][row['condition']].append({'problem_id': row['problem_id'], 'correct': row['completed_correct']})
        groups['audited'][row['condition']].append({'problem_id': row['problem_id'], 'correct': correct and not row['truncated']})
    contrasts = {}
    for metric, by_condition in groups.items():
        pairs = {name: paired_difference(by_condition['steered'], by_condition[name], 10000, 907)
                 for name in names if name != 'steered'}
        pvalues = {name: exact_mcnemar(v['wins'], v['losses']) for name, v in pairs.items()}
        adjusted = holm(pvalues)
        for name in pairs:
            pairs[name].update(mcnemar_two_sided_p=pvalues[name], holm_p=adjusted[name])
        contrasts[metric] = pairs
    result.update(status='supplementary_development_controls_complete', contrasts=contrasts,
        parent_selection=base.read(output / 'manifest.json')['parent_selection'],
        manifest_sha256=base.file_hash(output / 'manifest.json'), generation_rows_sha256=audit.digest(rows),
        limitations='Supplementary controls chosen after observing the fixed candidate development signal. '
        'No eligibility change and no independent confirmation. Inherited annotations remain fixed. '
        'Only unparsed responses were reviewed; truncations receive no credit. Bootstrap intervals are '
        'exploratory and unadjusted. Holm p-values adjust fifteen contrasts separately for each metric.')
    base.fixed(output / 'diagnostic-annotations.json', annotations)
    base.fixed(output / 'diagnostic-report.json', result)
    print('Supplementary report complete; original selection remains ineligible.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=ROOT / 'runs/gsm8k-fixed-candidate-v1')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / '.writer.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            prepare(args.parent, args.output)
            print('Prepared supplementary development controls; no new generations.', flush=True)
        elif args.stage == 'run':
            run(args.output)
        else:
            if args.annotations is None:
                parser.error('report requires --annotations')
            report(args.output, args.annotations)


if __name__ == '__main__':
    main()

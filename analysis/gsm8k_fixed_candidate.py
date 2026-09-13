"""Validate one predeclared ridge candidate only after a passing frozen ICL screen."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from analysis import gsm8k_answer_audit as audit
from analysis import gsm8k_conditional as base
from analysis import gsm8k_test_development as screen
from analysis.gsm8k_comparison import write_fixed


ROOT = Path(__file__).parents[1]
SETTING = {'layer': 13, 'alpha': 0.5, 'positions': 'prefill'}
CONDITIONS = [*screen.CONDITIONS, 'steered', 'mean']


def code_hash():
    return audit.digest({'screen': screen.code_hash(), 'files': {
        str(p.relative_to(ROOT)): base.file_hash(p) for p in [Path(__file__),
        ROOT / 'research/gsm8k-fixed-candidate-protocol.md']}})


def specifications():
    return {**screen.specifications(), 'steered': {'kind': 'ridge', **SETTING},
            'mean': {'kind': 'mean', **SETTING}}


def verified_screen(path):
    config, data = screen.verify(path)
    rows = screen.check_rows(path, data, complete=True)
    selection = base.read(path / 'screen-selection.json')
    annotations = base.read(path / 'screen-annotations.json')
    packet = base.read(path / 'validation-review-packet.json')
    freeze = base.read(path / 'review-freeze.json')
    if (freeze['annotations_file_sha256'] != base.file_hash(path / 'screen-annotations.json')
            or freeze['packet_sha256'] != audit.digest(packet) or not freeze['annotations_commit']):
        raise ValueError('Screen annotation freeze changed')
    expected = {'manifest_sha256': base.file_hash(path / 'manifest.json'),
        'rows_sha256': audit.digest(rows), 'annotations_sha256': audit.digest(annotations),
        'audit_sha256': base.file_hash(path / 'screen-audit.json')}
    if any(selection[k] != value for k, value in expected.items()):
        raise ValueError('Screen selection provenance changed')
    replay = audit.score(rows, data, 'validation', screen.CONDITIONS, packet, annotations,
                         config['bootstrap_samples'])
    if replay != base.read(path / 'screen-audit.json'):
        raise ValueError('Screen audit replay mismatch')
    decision = screen.gates(replay['summary'], config)
    if any(selection[k] != v for k, v in decision.items()):
        raise ValueError('Screen selection rule mismatch')
    if not decision['eligible']:
        raise ValueError('ICL screen failed; fixed candidate is prohibited')
    return config, data, rows


def gates(summary):
    selected, zero = summary['steered'], summary['zero']
    if len({r['n'] for r in summary.values()}) != 1:
        raise ValueError('Unequal development sample sizes')
    checks = {
        'zero_gain': (selected['audited_completed_correct'] - zero['audited_completed_correct']) / selected['n'] >= .03,
        'beats_mean': selected['audited_completed_correct'] > summary['mean']['audited_completed_correct'],
        'matches_text_cues': all(selected['audited_completed_correct'] >= summary[k]['audited_completed_correct']
                                 for k in ['first', 'cot']),
        'truncation': selected['truncated']/selected['n'] <= .05}
    return {'eligible': all(checks.values()), 'chosen': SETTING, 'checks': checks}


def verify(output):
    manifest = base.read(output / 'manifest.json')
    if manifest['code_sha256'] != code_hash() or manifest['chosen'] != SETTING:
        raise ValueError('Fixed candidate code or setting changed')
    for filename in ['prepared.json', 'maps.npz', 'baseline-rows.json', 'screen-annotations.json']:
        if base.file_hash(output / filename) != manifest['files_sha256'][filename]:
            raise ValueError('Frozen candidate input changed: ' + filename)
    for path, expected in manifest['inputs_sha256'].items():
        if base.file_hash(Path(path)) != expected:
            raise ValueError('Candidate parent input changed: ' + path)
    return manifest['config'], base.read(output / 'prepared.json')


def prepare(screen_path, map_path, output):
    config, data, rows = verified_screen(screen_path)
    base.verify(map_path)
    # These are extraction-only fitted maps, not a rescued parent selection.
    if base.read(map_path / 'selection.json')['eligible']:
        raise ValueError('Unexpected parent conditional result')
    source_files = [screen_path / name for name in ['manifest.json', 'prepared.json', 'generations.jsonl',
        'screen-selection.json', 'screen-audit.json', 'screen-annotations.json', 'review-freeze.json']]
    source_files += [map_path / name for name in ['manifest.json', 'maps.npz', 'fit.json', 'selection.json']]
    provenance = {str(p): base.file_hash(p) for p in source_files}
    if (output / 'manifest.json').exists():
        if base.read(output / 'manifest.json')['inputs_sha256'] != provenance:
            raise ValueError('Candidate preparation request changed')
        return verify(output)
    if (output / 'generations.jsonl').exists():
        raise ValueError('Unmanifested candidate rows')
    output.mkdir(parents=True, exist_ok=True)
    config = {**config, 'layers': [7, 13], 'random_seeds': [31, 59, 83]}
    base.fixed(output / 'prepared.json', data)
    write_fixed(output / 'maps.npz', (map_path / 'maps.npz').read_bytes())
    base.fixed(output / 'baseline-rows.json', rows)
    write_fixed(output / 'generations.jsonl', (screen_path / 'generations.jsonl').read_bytes())
    write_fixed(output / 'screen-annotations.json', (screen_path / 'screen-annotations.json').read_bytes())
    base.fixed(output / 'manifest.json', {'config': config, 'chosen': SETTING, 'code_sha256': code_hash(),
        'inputs_sha256': provenance, 'files_sha256': {name: base.file_hash(output / name) for name in
            ['prepared.json', 'maps.npz', 'baseline-rows.json', 'screen-annotations.json']},
        'created_at': datetime.now(timezone.utc).isoformat(), 'test_generation_supported': False})
    return verify(output)


def check_rows(output, data, complete=False):
    rows = base.rows_from(output)
    keys = [(r['split'], r['problem_id'], r['condition']) for r in rows]
    if len(keys) != len(set(keys)) or any(r['split'] != 'validation' for r in rows):
        raise ValueError('Unexpected or duplicate candidate rows')
    baseline = [r for r in rows if r['condition'] in screen.CONDITIONS]
    if baseline != base.read(output / 'baseline-rows.json'):
        raise ValueError('Screen baseline rows changed')
    base.check_rows(rows, data, specifications(), 'validation')
    if complete:
        audit.select_rows(rows, data, 'validation', CONDITIONS)
    return rows


def run(output):
    config, data = verify(output)
    if base.read(output / 'declaration.json')['manifest_sha256'] != base.file_hash(output / 'manifest.json'):
        raise ValueError('Candidate declaration mismatch')
    if (output / 'selection.json').exists():
        raise ValueError('Candidate is frozen after selection')
    check_rows(output, data)
    runner = base.ConditionalRunner(config, data, output)
    for kind in ['steered', 'mean']:
        runner.evaluate_spec('validation', kind, specifications()[kind])
    rows = check_rows(output, data, complete=True)
    packet = audit.make_packet(rows, data, 'validation', CONDITIONS)
    base.fixed(output / 'validation-review-packet.json', packet)
    original = base.read(output / 'screen-annotations.json')
    ids = {r['response_id'] for r in packet['items']}
    inherited = [r for r in original['answers'] if r['response_id'] in ids]
    base.fixed(output / 'inherited-annotations.json', {'packet_sha256': audit.digest(packet), 'answers': inherited})
    base.fixed(output / 'validation-complete.json', {'status': 'awaiting_blinded_review', 'rows': len(rows),
        'packet_sha256': audit.digest(packet), 'inherited_annotations': len(inherited)})
    print(f'Complete: {len(rows)} rows; {len(packet["items"])-len(inherited)} new blinded items.', flush=True)


def select(output, annotation_path):
    config, data = verify(output)
    rows = check_rows(output, data, complete=True)
    packet = base.read(output / 'validation-review-packet.json')
    annotations = base.read(annotation_path)
    freeze = base.read(output / 'review-freeze.json')
    if (freeze['annotations_file_sha256'] != base.file_hash(annotation_path)
            or freeze['packet_sha256'] != audit.digest(packet) or not freeze['annotations_commit']):
        raise ValueError('Candidate annotation freeze changed')
    lookup = {r['response_id']: r for r in annotations['answers']}
    if any(lookup.get(r['response_id']) != r for r in base.read(output / 'inherited-annotations.json')['answers']):
        raise ValueError('Frozen screen annotation changed')
    result = audit.score(rows, data, 'validation', CONDITIONS, packet, annotations, config['bootstrap_samples'])
    screen_decision = screen.gates(result['summary'], config)
    if not screen_decision['eligible']:
        raise ValueError('Previously eligible ICL baselines changed')
    decision = gates(result['summary'])
    base.fixed(output / 'validation-annotations.json', annotations)
    base.fixed(output / 'validation-audit.json', result)
    decision.update(manifest_sha256=base.file_hash(output / 'manifest.json'),
        validation_rows_sha256=audit.digest(rows), annotations_sha256=audit.digest(annotations),
        validation_audit_sha256=base.file_hash(output / 'validation-audit.json'))
    base.fixed(output / 'selection.json', decision)
    print(json.dumps(decision, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validate', 'select'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--screen', type=Path, default=ROOT / 'runs/gsm8k-test-development-v1')
    parser.add_argument('--maps', type=Path, default=ROOT / 'runs/gsm8k-conditional-v1')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / '.writer.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            prepare(args.screen, args.maps, args.output)
            print('Fixed candidate prepared after verified passing screen.', flush=True)
        elif args.stage == 'validate':
            run(args.output)
        else:
            if args.annotations is None:
                parser.error('select requires --annotations')
            select(args.output, args.annotations)


if __name__ == '__main__':
    main()

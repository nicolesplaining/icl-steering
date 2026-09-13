"""Matched-penalty development diagnostic of shared and paired ICL targets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from analysis import gsm8k_ltv_continuation as parent
from analysis import ltv_pairing_mapping as mapping

base, audit, screen, previous = parent.base, parent.audit, parent.screen, parent.previous
ROOT = Path(__file__).parents[1]
BASELINES = list(parent.CONDITIONS)
NEW = ['shared_low', 'permuted_low', 'real_high', 'shared_high']
CONDITIONS = [*BASELINES, *NEW]
PAIRS = [('steered', 'shared_low'), ('steered', 'permuted_low'),
         ('real_high', 'shared_high'), ('real_high', 'permuted'),
         ('steered', 'real_high'), ('permuted_low', 'permuted'), ('shared_low', 'shared_high')]
backend, arrays = parent.backend, parent.arrays


def code_hash():
    files = [Path(__file__), ROOT/'analysis/ltv_pairing_mapping.py',
             ROOT/'analysis/ltv_pairing_audit.py', ROOT/'research/ltv-pairing-protocol.md']
    return audit.digest({'parent': parent.code_hash(),
        'files': {str(p.relative_to(ROOT)): base.file_hash(p) for p in files}})


def specifications():
    return {**parent.specifications(), **{name: {'kind': name, 'site': 'final_norm',
        'layer': None, 'alpha': 1., 'positions': 'prefix_0_128'} for name in NEW}}


def prepare(source, output):
    config, data = parent.verify(source, fitted=True)
    rows = parent.collect(source, complete=True)
    decision, report = base.read(source/'selection.json'), base.read(source/'validation-audit.json')
    annotations = base.read(source/'validation-annotations.json')
    if (decision['eligible'] or decision['rows_sha256'] != audit.digest(rows)
            or report['annotations_sha256'] != audit.digest(annotations)
            or decision['annotations_sha256'] != audit.digest(annotations)):
        raise ValueError('Require the reviewed continuation failure')
    if (output/'manifest.json').exists():
        return verify(output)
    inputs = dict(base.read(source/'manifest.json')['inputs_sha256'])
    inputs.update({str((source/n).resolve()): base.file_hash(source/n) for n in
        ['manifest.json', 'fit.json', 'maps.npz', 'extraction.npz', 'generations.jsonl',
         'selection.json', 'validation-audit.json', 'validation-annotations.json',
         'review-freeze.json', 'trajectory-check.json']})
    if not base.read(source/'trajectory-check.json')['complete']:
        raise ValueError('Require completed parent trajectory audit')
    for name, value in [('prepared.json', data), ('baseline-rows.json', rows),
                         ('source-annotations.json', annotations)]:
        base.fixed(output/name, value)
    base.fixed(output/'manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'config': config, 'parent': str(source.resolve()), 'code_sha256': code_hash(),
        'conditions': specifications(), 'inputs_sha256': inputs,
        'files_sha256': {n: base.file_hash(output/n) for n in
                         ['prepared.json', 'baseline-rows.json', 'source-annotations.json']},
        'scales': [.1, 1.], 'permutation_seed': 2390, 'pairs': PAIRS,
        'max_intervention_prefix': 128, 'confirmation_supported': False,
        'purpose': 'Diagnostic only; no candidate selection or reserved generation'})
    return verify(output)


def verify(output, fitted=False):
    m = base.read(output/'manifest.json')
    if m['code_sha256'] != code_hash() or m['conditions'] != specifications():
        raise ValueError('Declared source or conditions changed')
    for name, expected in m['files_sha256'].items():
        if base.file_hash(output/name) != expected:
            raise ValueError('Prepared input changed: '+name)
    for path, expected in m['inputs_sha256'].items():
        if base.file_hash(Path(path)) != expected:
            raise ValueError('Parent input changed: '+path)
    if fitted:
        f = base.read(output/'fit.json')
        if f['manifest_sha256'] != base.file_hash(output/'manifest.json'):
            raise ValueError('Fit provenance mismatch')
        for name, expected in f['files_sha256'].items():
            if base.file_hash(output/name) != expected:
                raise ValueError('Fitted input changed: '+name)
    return m['config'], base.read(output/'prepared.json')


def fit(output):
    verify(output)
    if (output/'fit.json').exists():
        return verify(output, fitted=True)
    if base.read(output/'declaration.json')['manifest_sha256'] != base.file_hash(output/'manifest.json'):
        raise ValueError('Matching declaration required before fitting')
    source = Path(base.read(output/'manifest.json')['parent'])
    with np.load(source/'extraction.npz', allow_pickle=False) as packed:
        extraction = {k: packed[k] for k in ['zero', 'delta', 'positions']}
    x, d, positions = [extraction[k] for k in ['zero', 'delta', 'positions']]
    if x.shape != (620, 3584) or np.bincount(positions).tolist() != [128, 128, 128, 128, 108]:
        raise ValueError('Unexpected extraction coverage')
    fitted = mapping.fit_states(x, d, positions)
    prior = parent.load_maps(source)
    np.testing.assert_array_equal(fitted['x'], prior['real/x'])
    for name, previous_name in [('steered', 'real'), ('permuted', 'permuted')]:
        for key in ['weights', 'penalty']:
            np.testing.assert_array_equal(fitted[name+'/'+key], prior[previous_name+'/'+key])
    arrays(output/'extraction.npz', extraction)
    arrays(output/'maps.npz', fitted)
    base.fixed(output/'fit.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': base.file_hash(output/'manifest.json'),
        'files_sha256': {n: base.file_hash(output/n) for n in ['extraction.npz', 'maps.npz']},
        'n_extract': 128, 'n_states': 620, 'width': x.shape[1],
        'position_counts': np.bincount(positions).tolist(),
        'parent_maps_reconstructed_exactly': ['steered', 'permuted'],
        'maps': {name: {'target': target, 'scale': scale,
            'penalty': float(fitted[name+'/penalty']),
            'solver_residual': float(fitted[name+'/solver_residual'])}
            for name, (target, scale) in mapping.SPECS.items()}})
    print('Full diagnostic fit saved; declare it before generation.', flush=True)


def load_maps(output):
    with np.load(output/'maps.npz', allow_pickle=False) as packed:
        return {k: packed[k] for k in packed.files}


def collect(output, complete=False):
    config, data = verify(output, fitted=True)
    rows = list(base.read(output/'baseline-rows.json'))
    expected = {f'{name}-{start:03d}.json': (name, start) for name in NEW for start in range(0, 128, 4)}
    found = set()
    for path in sorted((output/'batches').glob('*.json')):
        if path.name not in expected:
            raise ValueError('Unexpected batch file')
        name, start = expected[path.name]; found.add(path.name)
        batch = base.read(path)
        if [r['problem_id'] for r in batch] != [p['problem_id'] for p in data['splits']['validation'][start:start+4]]:
            raise ValueError('Batch identities changed')
        if any(r['condition'] != name or r['split'] != 'validation' for r in batch):
            raise ValueError('Unexpected condition or reserved row')
        trace_path = f'traces/{name}-{start:03d}.npz'
        trace_hash = base.file_hash(output/trace_path)
        if any(r['trace_file'] != trace_path or r['trace_file_sha256'] != trace_hash for r in batch):
            raise ValueError('Trace changed')
        for i, row in enumerate(batch):
            if (row['trace_index'] != i or row['head_checks'] != row['active_calls']
                    or row['active_calls'] != min(row['hook_calls'], 129)
                    or row['prefix_positions'] != list(range(row['active_calls']))):
                raise ValueError('Intervention trace accounting changed')
        rows.extend(batch)
    if complete and found != set(expected):
        raise ValueError('Incomplete validation')
    if any(r['split'] != 'validation' for r in rows):
        raise ValueError('Reserved generation prohibited')
    base.check_rows(rows, data, specifications(), 'validation')
    if len({(r['condition'], r['problem_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate generation')
    for row in rows:
        if row['generated_tokens'] != len(row['token_ids']) or len(row['token_ids']) > config['max_new_tokens']:
            raise ValueError('Invalid token count')
    temporary = output/'generations.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    temporary.replace(output/'generations.jsonl')
    return rows


def validate(output):
    config, data = verify(output, fitted=True)
    declaration = base.read(output/'fit-declaration.json')
    if (declaration['fit_sha256'] != base.file_hash(output/'fit.json')
            or declaration['manifest_sha256'] != base.file_hash(output/'manifest.json')):
        raise ValueError('Fit declaration mismatch')
    if (output/'diagnostic-report.json').exists():
        raise ValueError('Diagnostic already scored')
    collect(output)
    fitted = load_maps(output)
    model = backend(config, data, output)
    (output/'batches').mkdir(exist_ok=True); (output/'traces').mkdir(exist_ok=True)
    for name in NEW:
        spec = specifications()[name]
        for start in range(0, 128, 4):
            record_path = output/f'batches/{name}-{start:03d}.json'
            if record_path.exists():
                continue
            problems = data['splits']['validation'][start:start+4]
            predictor = lambda h, position: mapping.predict(fitted, h, spec['kind'], position)
            with mapping.normalized_hook(model.model.model.norm, predictor, spec['positions'],
                                         model.model.lm_head) as trace:
                outputs = model.records([p['prompts']['zero'] for p in problems], config)
            states, vectors = np.stack(trace['states']), np.stack(trace['vectors'])
            path = output/f'traces/{name}-{start:03d}.npz'
            arrays(path, {'states': states, 'vectors': vectors,
                          'applied_states': np.stack(trace['applied_states']),
                          'prefix_positions': np.asarray(trace['prefix_positions'])})
            trace_hash = base.file_hash(path)
            batch = []
            for i, (p, result) in enumerate(zip(problems, outputs)):
                batch.append({'split': 'validation', 'condition': name, 'problem_id': p['problem_id'],
                    'prompt': p['prompts']['zero'], 'answer': p['answer'], 'intervention': spec,
                    **{k: spec[k] for k in ['layer', 'alpha', 'positions']}, **result,
                    **base.grade_answer(result['text'], p['answer'], result['truncated']),
                    'trace_file': str(path.relative_to(output)), 'trace_file_sha256': trace_hash,
                    'trace_index': i, 'hook_calls': trace['calls'], 'active_calls': len(states),
                    'head_checks': trace['head_checks'], 'activation_dtype': trace['dtype'],
                    'sequence_lengths': trace['sequence_lengths'], 'prefix_positions': trace['prefix_positions'],
                    'vector_chain_sha256': hashlib.sha256(vectors[:, i].copy().tobytes()).hexdigest()})
            base.fixed(record_path, batch)
            print(f'Validation {name}: {start+len(problems)}/128', flush=True)
        collect(output)
    rows = collect(output, complete=True)
    packet = audit.make_packet(rows, data, 'validation', CONDITIONS)
    ids = {x['response_id'] for x in packet['items']}
    inherited = [x for x in base.read(output/'source-annotations.json')['answers'] if x['response_id'] in ids]
    base.fixed(output/'validation-review-packet.json', packet)
    base.fixed(output/'inherited-annotations.json', {'packet_sha256': audit.digest(packet), 'answers': inherited})
    base.fixed(output/'validation-complete.json', {'status': 'awaiting_blinded_review', 'rows': len(rows),
        'packet_sha256': audit.digest(packet), 'inherited': len(inherited), 'new': len(ids)-len(inherited)})
    print('Validation complete; freeze and review the packet before scoring.', flush=True)


def report(output, annotation_path):
    config, data = verify(output, fitted=True)
    rows = collect(output, complete=True)
    packet, annotations = base.read(output/'validation-review-packet.json'), base.read(annotation_path)
    freeze = base.read(output/'review-freeze.json')
    if (freeze['packet_sha256'] != audit.digest(packet)
            or freeze['annotations_file_sha256'] != base.file_hash(annotation_path)
            or not freeze['annotations_commit']):
        raise ValueError('Annotation freeze mismatch')
    lookup = {x['response_id']: x for x in annotations['answers']}
    if any(lookup.get(x['response_id']) != x for x in base.read(output/'inherited-annotations.json')['answers']):
        raise ValueError('Inherited annotation changed')
    runtime = base.read(output/'trajectory-check.json')
    if (not runtime['complete'] or runtime['status'] != 'passed'
            or runtime['manifest_sha256'] != base.file_hash(output/'manifest.json')
            or runtime['fit_sha256'] != base.file_hash(output/'fit.json')):
        raise ValueError('Complete runtime audit required')
    result = audit.score(rows, data, 'validation', CONDITIONS, packet, annotations, 10000)
    from fractions import Fraction
    groups = {metric: {name: [] for name in CONDITIONS} for metric in ['primary', 'audited']}
    selected, questions = audit.select_rows(rows, data, 'validation', CONDITIONS)
    for row in selected:
        correct = row['correct']
        if not row['parseable']:
            value = lookup[audit.response_item(row, questions)['response_id']]['stated_answer']
            correct = value is not None and Fraction(value) == Fraction(row['answer'])
        for metric, ok in [('primary', row['completed_correct']), ('audited', correct and not row['truncated'])]:
            groups[metric][row['condition']].append({'problem_id': row['problem_id'], 'correct': ok})
    result['contrasts'] = {}
    for metric, by_name in groups.items():
        pairs = {left+'__minus__'+right: {'left': left, 'right': right,
                    **previous.paired_difference(by_name[left], by_name[right], 10000, 907)}
                 for left, right in PAIRS}
        p = {name: previous.exact_mcnemar(v['wins'], v['losses']) for name, v in pairs.items()}
        adjusted = previous.holm(p)
        for name in pairs:
            pairs[name].update(mcnemar_two_sided_p=p[name], holm_p=adjusted[name])
        result['contrasts'][metric] = pairs
    result.update(status='completed_development_diagnostic', confirmation_supported=False,
        candidate_selected=None, reserved_rows=0,
        manifest_sha256=base.file_hash(output/'manifest.json'), fit_sha256=base.file_hash(output/'fit.json'),
        rows_sha256=audit.digest(rows), annotations_commit=freeze['annotations_commit'])
    result['limitations'] = ('Reused development questions informed this design. Seven contrasts adjusted '
        'separately per metric. No candidate selection or confirmation authorization. '
        'Inherited continuation gate remains failed. Truncated responses receive no credit.')
    base.fixed(output/'diagnostic-report.json', result)
    print('Diagnostic report saved; no candidate selected and no reserved generation authorized.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'validate', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=ROOT/'runs/gsm8k-continuation-v1')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare': prepare(args.parent, args.output)
    elif args.stage == 'fit': fit(args.output)
    elif args.stage == 'validate': validate(args.output)
    else:
        if args.annotations is None: parser.error('--annotations required for report')
        report(args.output, args.annotations)


if __name__ == '__main__':
    main()

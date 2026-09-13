"""Reuse frozen continuation maps on the independently screened complex population."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import gsm8k_complex_screen as screen
from analysis import gsm8k_ltv_pairing as prior
from analysis import complex_candidate_mapping as mapping
from analysis import complex_candidate_gates as gate

base, audit, previous = prior.base, prior.audit, prior.previous
ROOT = Path(__file__).parents[1]
BASELINES, NEW, CONDITIONS = gate.BASELINES, gate.NEW, gate.CONDITIONS
backend, arrays = prior.backend, prior.arrays


def code_hash():
    files = [Path(__file__), ROOT/'analysis/complex_candidate_mapping.py',
        ROOT/'analysis/complex_candidate_map_check.py', ROOT/'analysis/complex_candidate_gates.py',
        ROOT/'analysis/complex_candidate_trace_check.py', ROOT/'analysis/screen_score_recount.py',
        ROOT/'research/gsm8k-complex-candidate-protocol.md']
    return audit.digest({'parent': prior.code_hash(), 'screen': screen.code_hash(),
        'files': {str(p.relative_to(ROOT)): base.file_hash(p) for p in files}})


def specifications():
    return {**screen.previous.specifications(), **{name: {'kind': name, 'site': 'final_norm',
        'layer': None, 'alpha': 1., 'positions': mapping.scope(name)} for name in NEW}}


def prepare(source, map_check, output):
    config, data = screen.verify(source)
    selection = base.read(source/'screen-selection.json')
    if not selection['eligible'] or not all(selection['checks'].values()):
        raise ValueError('ICL screen failed; steering generation prohibited')
    report = base.read(source/'screen-audit.json')
    annotations = base.read(source/'screen-annotations.json')
    rows = screen.previous.check_rows(source, data, complete=True)
    recount = base.read(source/'screen-recount.json')
    if (selection['rows_sha256'] != audit.digest(rows)
            or selection['annotations_sha256'] != audit.digest(annotations)
            or selection['audit_sha256'] != base.file_hash(source/'screen-audit.json')
            or recount['status'] != 'passed' or recount['eligible'] is not True
            or recount['manifest_sha256'] != base.file_hash(source/'manifest.json')
            or recount['report_file_sha256'] != base.file_hash(source/'screen-audit.json')
            or recount['generation_file_sha256'] != base.file_hash(source/'generations.jsonl')
            or recount['script_sha256'] != base.file_hash(ROOT/'analysis/screen_score_recount.py')
            or report['annotations_sha256'] != audit.digest(annotations)):
        raise ValueError('Require the independently verified screen pass')
    if (len(data['splits']['validation']) != 256 or len(data['splits']['reserved']) != 512
            or config['batch_size'] != 4 or config['max_new_tokens'] != 1024):
        raise ValueError('Screen sizes or inference settings changed')
    checked = base.read(map_check)
    if (checked['status'] != 'passed' or checked['conditions'] != 12
            or checked['source_maps_sha256'] != mapping.MAP_HASHES
            or checked['script_sha256'] != base.file_hash(ROOT/'analysis/complex_candidate_map_check.py')
            or checked['mapping_code_sha256'] != base.file_hash(ROOT/'analysis/complex_candidate_mapping.py')):
        raise ValueError('Require the frozen-map routing check')
    roots = {k: ROOT/'runs'/name for k, name in mapping.RUN_NAMES.items()}
    paths = {k: p/'maps.npz' for k, p in roots.items()}
    mapping.load_frozen(paths)
    inputs = {str((source/name).resolve()): base.file_hash(source/name) for name in
        ['manifest.json', 'prepared.json', 'declaration.json', 'generations.jsonl', 'screen-selection.json',
         'screen-audit.json', 'screen-annotations.json', 'screen-recount.json', 'review-freeze.json']}
    inputs[str(map_check.resolve())] = base.file_hash(map_check)
    for root in roots.values():
        inputs.update({str((root/name).resolve()): base.file_hash(root/name)
                       for name in ['manifest.json', 'fit.json', 'maps.npz', 'extraction.npz']})
    if (output/'manifest.json').exists():
        if base.read(output/'manifest.json')['inputs_sha256'] != inputs:
            raise ValueError('Prepared request changed')
        return verify(output, fitted=True)
    if (output/'generations.jsonl').exists():
        raise ValueError('Unmanifested generations')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('prepared.json', data), ('baseline-rows.json', rows),
                        ('source-annotations.json', annotations)]:
        base.fixed(output/name, value)
    base.fixed(output/'manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'config': config, 'code_sha256': code_hash(), 'inputs_sha256': inputs,
        'map_paths': {k: str(p.resolve()) for k, p in paths.items()},
        'conditions': specifications(), 'icl_screen_passed': True, 'confirmation_supported': False,
        'files_sha256': {name: base.file_hash(output/name) for name in
                         ['prepared.json', 'baseline-rows.json', 'source-annotations.json']}})
    base.fixed(output/'fit.json', {'manifest_sha256': base.file_hash(output/'manifest.json'),
        'strategy': 'Reuse existing arrays byte for byte; no new fit',
        'source_maps_sha256': mapping.MAP_HASHES, 'new_fits': 0})
    return verify(output, fitted=True)


def verify(output, fitted=False):
    manifest = base.read(output/'manifest.json')
    if (manifest['code_sha256'] != code_hash() or manifest['conditions'] != specifications()
            or manifest['config'] != base.read(screen.CONFIG)
            or manifest['icl_screen_passed'] is not True or manifest['confirmation_supported'] is not False):
        raise ValueError('Source, conditions, or screen prerequisite changed')
    for name, expected in manifest['files_sha256'].items():
        if base.file_hash(output/name) != expected:
            raise ValueError('Prepared input changed: '+name)
    for path, expected in manifest['inputs_sha256'].items():
        if base.file_hash(Path(path)) != expected:
            raise ValueError('Frozen source input changed: '+path)
    if fitted:
        fit = base.read(output/'fit.json')
        if (fit['manifest_sha256'] != base.file_hash(output/'manifest.json')
                or fit['source_maps_sha256'] != mapping.MAP_HASHES or fit['new_fits'] != 0):
            raise ValueError('Map reuse provenance changed')
    return manifest['config'], base.read(output/'prepared.json')


def load_maps(output):
    return mapping.load_frozen({k: Path(p) for k, p in base.read(output/'manifest.json')['map_paths'].items()})

def collect(output, complete=False):
    config, data = verify(output, fitted=True)
    rows = list(base.read(output/'baseline-rows.json'))
    expected = {f'{name}-{start:03d}.json': (name, start) for name in NEW for start in range(0, 256, 4)}
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
                    or row['active_calls'] != (1 if name in {'regularized_prefill', 'prefill'} else min(row['hook_calls'], 129))
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
            or declaration['manifest_sha256'] != base.file_hash(output/'manifest.json')
            or not declaration['declaration_commit']):
        raise ValueError('Fit declaration mismatch')
    if (output/'selection.json').exists():
        raise ValueError('Selection already frozen')
    collect(output)
    fitted = load_maps(output)
    model = backend(config, data, output)
    (output/'batches').mkdir(exist_ok=True); (output/'traces').mkdir(exist_ok=True)
    for name in NEW:
        spec = specifications()[name]
        for start in range(0, 256, 4):
            record_path = output/f'batches/{name}-{start:03d}.json'
            if record_path.exists():
                continue
            problems = data['splits']['validation'][start:start+4]
            predictor = lambda h, position: mapping.predict(fitted, h, spec['kind'], position)
            with mapping.normalized_hook(model.model.model.norm, predictor, spec['positions'],
                                         model.model.lm_head) as trace:
                outputs = model.records([p['prompts']['zero'] for p in problems], config)
            if len(outputs) != len(problems):
                raise ValueError('Generation row count mismatch')
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
            print(f'Validation {name}: {start+len(problems)}/256', flush=True)
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



def select(output, annotation_path):
    config, data = verify(output, fitted=True)
    rows = collect(output, complete=True)
    packet, annotations = base.read(output/'validation-review-packet.json'), base.read(annotation_path)
    freeze = base.read(output/'review-freeze.json')
    if (freeze['packet_sha256'] != audit.digest(packet)
            or freeze['annotations_file_sha256'] != base.file_hash(annotation_path) or not freeze['annotations_commit']):
        raise ValueError('Annotation freeze mismatch')
    lookup = {x['response_id']: x for x in annotations['answers']}
    if any(lookup.get(x['response_id']) != x for x in base.read(output/'inherited-annotations.json')['answers']):
        raise ValueError('Inherited annotation changed')
    runtime = base.read(output/'trajectory-check.json')
    if (runtime['status'] != 'passed' or not runtime['complete']
            or runtime['manifest_sha256'] != base.file_hash(output/'manifest.json')
            or runtime['fit_sha256'] != base.file_hash(output/'fit.json')
            or runtime['rows_sha256'] != audit.digest(rows)
            or runtime['script_sha256'] != base.file_hash(ROOT/'analysis/complex_candidate_trace_check.py')
            or runtime['candidate_prefill_first_tokens_matched'] != 256):
        raise ValueError('Complete trace and first-token verification required')
    result = audit.score(rows, data, 'validation', CONDITIONS, packet, annotations, 10000)
    # Shared reporter computes the declared exact tests and Holm family.
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
        pairs = {name: previous.paired_difference(by_name['steered'], by_name[name], 10000, 907)
                 for name in CONDITIONS if name != 'steered'}
        p = {name: previous.exact_mcnemar(v['wins'], v['losses']) for name, v in pairs.items()}
        adjusted = previous.holm(p)
        for name in pairs:
            pairs[name].update(mcnemar_two_sided_p=p[name], holm_p=adjusted[name])
        result['contrasts'][metric] = pairs
    result['limitations'] = 'Fresh complex-training development questions; ICL-screened before steering. No independent confirmation. Only unparsed responses reviewed; truncations receive no credit. Sixteen contrasts adjusted separately for each metric.'
    decision = {**gate.development_gate(result['summary'], result['contrasts']['audited']['zero'], True), 'manifest_sha256': base.file_hash(output/'manifest.json'),
        'fit_sha256': base.file_hash(output/'fit.json'), 'rows_sha256': audit.digest(rows),
        'annotations_sha256': audit.digest(annotations)}
    base.fixed(output/'validation-audit.json', result)
    base.fixed(output/'validation-annotations.json', annotations)
    base.fixed(output/'selection.json', decision)
    print(json.dumps(decision, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validate', 'select'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--screen', type=Path, default=ROOT/'runs/gsm8k-complex-screen-v1')
    parser.add_argument('--map-check', type=Path, default=ROOT/'runs/gsm8k-complex-map-check-v1.json')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            prepare(args.screen, args.map_check, args.output)
        elif args.stage == 'validate':
            validate(args.output)
        else:
            if args.annotations is None:
                parser.error('select requires --annotations')
            select(args.output, args.annotations)


if __name__ == '__main__':
    main()

"""Fixed development experiment for regularized continuation steering."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import gsm8k_ltv as parent
from analysis import ltv_prefix_collect as prefix
from analysis import ltv_prefix_regularization as nested
from analysis import ltv_continuation_mapping as mapping

base, audit, screen, previous = parent.base, parent.audit, parent.screen, parent.previous
ROOT = Path(__file__).parents[1]
BASELINES = [*screen.CONDITIONS, 'prefill']
NEW = ['steered', 'regularized_prefill', 'mean', 'scalar', 'position_mean', 'position_scalar', 'permuted']
CONDITIONS = [*BASELINES, *NEW]
backend, arrays = parent.backend, parent.arrays


def code_hash():
    files = [Path(__file__), ROOT/'analysis/ltv_continuation_mapping.py',
             ROOT/'analysis/ltv_continuation_audit.py', ROOT/'research/gsm8k-continuation-protocol.md']
    return audit.digest({'parent': parent.code_hash(), 'nested': nested.source_hash(),
                         'files': {str(p.relative_to(ROOT)): base.file_hash(p) for p in files}})


def specifications():
    old = parent.specifications()
    return {**{k: old[k] for k in BASELINES}, **{name: {
        'kind': 'ridge' if name in {'steered', 'regularized_prefill'} else name,
        'site': 'final_norm', 'layer': None, 'alpha': 1.,
        'positions': 'prefill' if name == 'regularized_prefill' else 'prefix_0_128'} for name in NEW}}


def prepare(source, diagnostic, output):
    config, data = parent.verify(source, fitted=True)
    report, decision = base.read(source/'validation-audit.json'), base.read(source/'selection.json')
    rows = parent.collect(source, complete=True)
    if (decision['eligible'] or decision['rows_sha256'] != audit.digest(rows)
            or report['annotations_sha256'] != audit.digest(base.read(source/'validation-annotations.json'))):
        raise ValueError('Require the reviewed original LTV failure')
    manifest, result = base.read(diagnostic/'manifest.json'), base.read(diagnostic/'nested-report.json')
    if (manifest['source_sha256'] != nested.source_hash() or not result['eligible_for_accuracy_design']
            or result['manifest_sha256'] != base.file_hash(diagnostic/'manifest.json')
            or result['source_sha256'] != nested.source_hash()
            or result['predictions_file_sha256'] != base.file_hash(diagnostic/'predictions.npz')):
        raise ValueError('Require the verified nested activation pass')
    if any(f['selected']['ridge']['rule'] != 'scaled_0.1' or
           f['selected']['permuted']['rule'] != 'scaled_1' for f in result['folds']):
        raise ValueError('Declared regularization differs from nested selections')
    crossfit = base.read(Path(manifest['parent'])/'manifest.json')
    prefix_run = Path(crossfit['parent'])
    _, questions = prefix.verify(prefix_run)
    if [q['problem_id'] for q in questions] != [q['problem_id'] for q in data['splits']['extract']]:
        raise ValueError('Extraction question identities differ')
    if (output/'manifest.json').exists():
        return verify(output)
    inputs = dict(manifest['inputs_sha256'])
    inputs.update({str((diagnostic/n).resolve()): base.file_hash(diagnostic/n) for n in
                   ['manifest.json', 'nested-report.json', 'declaration.json', 'predictions.npz']})
    inputs.update({str((source/n).resolve()): base.file_hash(source/n) for n in
                   ['manifest.json', 'fit.json', 'generations.jsonl', 'selection.json',
                    'validation-audit.json', 'validation-annotations.json', 'review-freeze.json']})
    for path, expected in inputs.items():
        if base.file_hash(Path(path)) != expected:
            raise ValueError('Changed parent input')
    for name, value in [('prepared.json', data),
                        ('baseline-rows.json', [r for r in rows if r['condition'] in BASELINES]),
                        ('source-annotations.json', base.read(source/'validation-annotations.json'))]:
        base.fixed(output/name, value)
    base.fixed(output/'manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'config': config, 'prefix_run': str(prefix_run.resolve()), 'code_sha256': code_hash(),
        'conditions': specifications(), 'inputs_sha256': inputs,
        'files_sha256': {n: base.file_hash(output/n) for n in
                         ['prepared.json', 'baseline-rows.json', 'source-annotations.json']},
        'primary_scale': .1, 'permuted_scale': 1., 'permutation_seed': 2390,
        'max_intervention_prefix': 128, 'confirmation_supported': False})
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
    _, data = verify(output)
    if (output/'fit.json').exists():
        return verify(output, fitted=True)
    if base.read(output/'declaration.json')['manifest_sha256'] != base.file_hash(output/'manifest.json'):
        raise ValueError('Matching declaration required before fitting')
    prefix_run = Path(base.read(output/'manifest.json')['prefix_run'])
    prior, questions = prefix.verify(prefix_run)
    with np.load(Path(prior['parent'])/'extraction.npz', allow_pickle=False) as packed:
        reference = {k: packed[k] for k in ['zero', 'icl_a']}
    chunks = [prefix.check_batch(prefix_run/'batches'/f'{start:03d}.json', questions[start:start+4],
                                 {k: v[start:start+4] for k, v in reference.items()})
              for start in range(0, 128, 4)]
    values = {k: np.concatenate([c[k] for c in chunks]) for k in ['zero', 'icl_a', 'available']}
    keep = values['available']; x = values['zero'][keep].astype(np.float64)
    delta = values['icl_a'][keep].astype(np.float64)-x
    positions = np.broadcast_to(np.arange(5), keep.shape)[keep]
    if x.shape[0] != 620 or len(questions) != 128:
        raise ValueError('Unexpected extraction coverage')
    fitted = mapping.fit_states(x, delta, positions)
    arrays(output/'extraction.npz', {'zero': x, 'delta': delta, 'positions': positions})
    arrays(output/'maps.npz', fitted)
    base.fixed(output/'fit.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': base.file_hash(output/'manifest.json'),
        'files_sha256': {n: base.file_hash(output/n) for n in ['extraction.npz', 'maps.npz']},
        'n_extract': 128, 'n_states': 620, 'width': x.shape[1],
        'position_counts': fitted['position_counts'].tolist(),
        'real_penalty': float(fitted['real/penalty']), 'permuted_penalty': float(fitted['permuted/penalty']),
        'real_solver_residual': float(fitted['real/solver_residual']),
        'permuted_solver_residual': float(fitted['permuted/solver_residual'])})
    print('Full fit saved; declare it before generation.', flush=True)


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
                    or row['active_calls'] != (1 if name == 'regularized_prefill' else min(row['hook_calls'], 129))
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
    if (output/'selection.json').exists():
        raise ValueError('Selection already frozen')
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


def gates(summary, config):
    candidate = summary['steered']; n = candidate['n']
    if set(summary) != set(CONDITIONS) or any(s['n'] != n for s in summary.values()):
        raise ValueError('Incomplete or unequal condition sizes')
    checks = {'icl_screen': screen.gates(summary, config)['eligible'],
        'zero_gain': (candidate['audited_completed_correct']-summary['zero']['audited_completed_correct'])/n >= .03,
        'beats_controls': all(candidate['audited_completed_correct'] > summary[k]['audited_completed_correct']
                              for k in ['mean', 'scalar', 'position_mean', 'position_scalar', 'permuted']),
        'matches_text_and_prefill': all(candidate['audited_completed_correct'] >= summary[k]['audited_completed_correct']
                                        for k in ['first', 'cot', 'prefill']),
        'beats_same_map_prefill': candidate['audited_completed_correct'] > summary['regularized_prefill']['audited_completed_correct'],
        'truncation': candidate['truncated']/n <= .05}
    return {'eligible': all(checks.values()), 'checks': checks, 'chosen': specifications()['steered']}


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
    result['limitations'] = 'Reused development questions. No independent confirmation. Only unparsed responses reviewed; truncations receive no credit. Twelve contrasts adjusted separately for each metric.'
    decision = {**gates(result['summary'], config), 'manifest_sha256': base.file_hash(output/'manifest.json'),
        'fit_sha256': base.file_hash(output/'fit.json'), 'rows_sha256': audit.digest(rows),
        'annotations_sha256': audit.digest(annotations)}
    base.fixed(output/'validation-audit.json', result)
    base.fixed(output/'validation-annotations.json', annotations)
    base.fixed(output/'selection.json', decision)
    print(json.dumps(decision, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'validate', 'select'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=ROOT/'runs/gsm8k-ltv-v1')
    parser.add_argument('--diagnostic', type=Path, default=ROOT/'runs/gsm8k-ltv-regularization-v1')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'.writer.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            prepare(args.parent, args.diagnostic, args.output)
        elif args.stage == 'fit':
            fit(args.output)
        elif args.stage == 'validate':
            validate(args.output)
        else:
            if args.annotations is None:
                parser.error('select requires --annotations')
            select(args.output, args.annotations)


if __name__ == '__main__':
    main()

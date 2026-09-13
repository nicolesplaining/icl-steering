"""Fixed final-normalized-state LTV development experiment, with no test stage."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from analysis import gsm8k_fixed_controls as previous
from analysis import ltv_mapping as mapping

base, audit, screen = previous.base, previous.audit, previous.fixed.screen
ROOT = Path(__file__).parents[1]
NEW = ['steered', 'mean', 'scalar', 'scalar_norm', 'permuted', 'prefill']
CONDITIONS = [*screen.CONDITIONS, *NEW]


def code_hash():
    return audit.digest({'parent': previous.code_hash(), 'files': {
        str(p.relative_to(ROOT)): base.file_hash(p) for p in [Path(__file__),
        ROOT/'analysis/ltv_mapping.py', ROOT/'research/gsm8k-ltv-protocol.md']}})


def specifications():
    return {**screen.specifications(), **{name: {'kind': 'ltv' if name in {'steered', 'prefill'} else name,
        'site': 'final_norm', 'layer': None, 'alpha': 1.,
        'positions': 'prefill' if name == 'prefill' else 'all'} for name in NEW}}


def prepare(parent, extraction, output):
    config, data = previous.verify(parent)
    rows = previous.check_rows(parent, data, config, complete=True)
    report = base.read(parent/'diagnostic-report.json')
    if (report['manifest_sha256'] != base.file_hash(parent/'manifest.json')
            or report['generation_rows_sha256'] != audit.digest(rows)
            or report['parent_selection']['eligible']):
        raise ValueError('Require the completed failed-candidate diagnostic')
    source = base.read(extraction/'prepared.json')
    old_manifest = base.read(extraction/'manifest.json')
    if (old_manifest['prepared_sha256'] != base.file_hash(extraction/'prepared.json')
            or old_manifest['source_sha256'] != base.source_hash()
            or any(old_manifest['config'][key] != config[key] for key in
                   ['model', 'revision', 'dataset', 'dataset_revision', 'max_new_tokens', 'max_model_len'])):
        raise ValueError('Extraction source provenance mismatch')
    if source['banks'] != data['banks']:
        raise ValueError('Demonstration banks differ')
    data = {**data, 'splits': {**data['splits'], 'extract': source['splits']['extract']}}
    if {k: len(v) for k, v in data['splits'].items()} != {'extract': 128, 'validation': 128, 'reserved': 256}:
        raise ValueError('Unexpected split sizes')
    seen_ids, seen_text = set(), set()
    for values in data['splits'].values():
        for p in values:
            text = ' '.join(p['question'].split())
            if p['problem_id'] in seen_ids or text in seen_text:
                raise ValueError('Extraction, development, or reservation overlap')
            seen_ids.add(p['problem_id']); seen_text.add(text)
    if any(' '.join(p['question'].split()) in seen_text for b in data['banks'].values() for p in b):
        raise ValueError('Demonstration overlap')
    paths = [parent/n for n in ['manifest.json', 'prepared.json', 'generations.jsonl',
        'diagnostic-report.json', 'diagnostic-annotations.json', 'review-freeze.json']]
    paths += [extraction/'prepared.json', extraction/'manifest.json']
    provenance = {str(p.resolve()): base.file_hash(p) for p in paths}
    if (output/'manifest.json').exists():
        if base.read(output/'manifest.json')['inputs_sha256'] != provenance:
            raise ValueError('Preparation inputs changed')
        return verify(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = [r for r in rows if r['condition'] in screen.CONDITIONS]
    annotations = base.read(parent/'diagnostic-annotations.json')
    if audit.digest(annotations) != report['annotations_sha256']:
        raise ValueError('Parent annotation mismatch')
    for name, value in [('prepared.json', data), ('baseline-rows.json', baseline),
                        ('source-annotations.json', annotations)]:
        base.fixed(output/name, value)
    config = {**config, 'ltv_penalty': 5., 'permutation_seed': 1901}
    base.fixed(output/'manifest.json', {'config': config, 'code_sha256': code_hash(),
        'inputs_sha256': provenance, 'files_sha256': {n: base.file_hash(output/n) for n in
        ['prepared.json', 'baseline-rows.json', 'source-annotations.json']},
        'conditions': specifications(), 'created_at': datetime.now(timezone.utc).isoformat(),
        'confirmation_supported': False})
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


def backend(config, data, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Require physical GPU 0')
    return base.Runner(config, data, output).backend()


def arrays(path, values):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


@torch.inference_mode()
def extract(output):
    config, data = verify(output)
    if (output/'fit.json').exists():
        verify(output, fitted=True)
        return
    declaration = base.read(output/'declaration.json')
    if declaration['manifest_sha256'] != base.file_hash(output/'manifest.json'):
        raise ValueError('Missing matching declaration')
    model = backend(config, data, output)
    values = {}
    for kind in ['zero', 'icl_a']:
        chunks = []
        for start in range(0, 128, 4):
            batch = data['splits']['extract'][start:start+4]
            captured = []
            def capture(module, inputs, states):
                captured.append(states[:, -1, :].detach().float().cpu().numpy().copy())
            handle = model.model.model.norm.register_forward_hook(capture)
            try:
                model.model.model(**model.encode([p['prompts'][kind] for p in batch]), use_cache=False)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise ValueError('Unexpected normalization calls')
            chunks.append(captured[0])
            print(f'Extraction {kind}: {start+len(batch)}/128', flush=True)
        values[kind] = np.concatenate(chunks)
    x = values['zero'].astype(np.float64)
    delta = values['icl_a'].astype(np.float64)-x
    permutation = np.random.default_rng(config['permutation_seed']).permutation(128)
    real = mapping.fit(x, delta, config['ltv_penalty'])
    permuted = mapping.fit(x, delta[permutation], config['ltv_penalty'])
    arrays(output/'extraction.npz', values)
    arrays(output/'maps.npz', {f'{kind}/{k}': v for kind, fit in [('real', real), ('permuted', permuted)]
                              for k, v in fit.items()})
    base.fixed(output/'fit.json', {'manifest_sha256': base.file_hash(output/'manifest.json'),
        'files_sha256': {n: base.file_hash(output/n) for n in ['extraction.npz', 'maps.npz']},
        'n_extract': 128, 'site': 'final_norm', 'width': x.shape[1], 'permutation': permutation.tolist(),
        'penalty': config['ltv_penalty'], 'real_solver_residual': float(real['solver_residual']),
        'permuted_solver_residual': float(permuted['solver_residual']),
        'mean_norm': float(np.linalg.norm(real['mean'])), 'scalar': float(real['scalar']),
        'created_at': datetime.now(timezone.utc).isoformat()})
    print('Extraction complete; fit must be declared before validation.', flush=True)


def load_maps(output):
    with np.load(output/'maps.npz', allow_pickle=False) as packed:
        return tuple({key.split('/')[1]: packed[key] for key in packed.files if key.startswith(kind+'/')}
                     for kind in ['real', 'permuted'])


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
                    or row['active_calls'] != (1 if name == 'prefill' else row['hook_calls'])):
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
    real, permuted = load_maps(output)
    model = backend(config, data, output)
    (output/'batches').mkdir(exist_ok=True); (output/'traces').mkdir(exist_ok=True)
    for name in NEW:
        spec = specifications()[name]
        for start in range(0, 128, 4):
            record_path = output/f'batches/{name}-{start:03d}.json'
            if record_path.exists():
                continue
            problems = data['splits']['validation'][start:start+4]
            predictor = lambda h: mapping.predict(real, h, spec['kind'], permuted)
            with mapping.normalized_hook(model.model.model.norm, predictor, spec['positions'],
                                         model.model.lm_head) as trace:
                outputs = model.records([p['prompts']['zero'] for p in problems], config)
            states, vectors = np.stack(trace['states']), np.stack(trace['vectors'])
            path = output/f'traces/{name}-{start:03d}.npz'
            arrays(path, {'states': states, 'vectors': vectors,
                          'applied_states': np.stack(trace['applied_states'])})
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
                    'sequence_lengths': trace['sequence_lengths'],
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
                              for k in ['mean', 'scalar', 'scalar_norm', 'permuted']),
        'matches_text_and_prefill': all(candidate['audited_completed_correct'] >= summary[k]['audited_completed_correct']
                                        for k in ['first', 'cot', 'prefill']),
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
    result['limitations'] = 'Reused development questions. No independent confirmation. Only unparsed responses reviewed; truncations receive no credit. Ten contrasts adjusted separately for each metric.'
    decision = {**gates(result['summary'], config), 'manifest_sha256': base.file_hash(output/'manifest.json'),
        'fit_sha256': base.file_hash(output/'fit.json'), 'rows_sha256': audit.digest(rows),
        'annotations_sha256': audit.digest(annotations)}
    base.fixed(output/'validation-audit.json', result)
    base.fixed(output/'validation-annotations.json', annotations)
    base.fixed(output/'selection.json', decision)
    print(json.dumps(decision, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'extract', 'validate', 'select'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=ROOT/'runs/gsm8k-fixed-controls-v1')
    parser.add_argument('--extraction', type=Path, default=ROOT/'runs/gsm8k-steering-v2')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'.writer.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            prepare(args.parent, args.extraction, args.output)
        elif args.stage == 'extract':
            extract(args.output)
        elif args.stage == 'validate':
            validate(args.output)
        else:
            if args.annotations is None:
                parser.error('select requires --annotations')
            select(args.output, args.annotations)


if __name__ == '__main__':
    main()

"""One predeclared test-distribution ICL screen, without a steering/test stage."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import random

from analysis import gsm8k_answer_audit as audit
from analysis import gsm8k_conditional as parent
from replication.gsm8k_protocol import matched_prompt
from replication.gsm8k_steering import Runner


ROOT = Path(__file__).parents[1]
CONDITIONS = ['zero', 'icl_a', 'icl_b', 'first', 'cot']


def code_hash():
    return audit.digest({'parent': parent.code_hash(), 'files': {
        str(p.relative_to(ROOT)): parent.file_hash(p) for p in [Path(__file__),
        ROOT / 'research/gsm8k-test-development-protocol.md']}})


def partition(tables, old, conditional, seed, n_screen, n_reserved):
    norm = lambda s: ' '.join(s.split())
    excluded = {norm(x['question']) for data in [old, conditional]
                for rows in data['splits'].values() for x in rows}
    excluded.update(norm(x['question']) for bank in old['banks'].values() for x in bank)
    excluded.update(norm(x['question']) for rows in tables.values() for x in rows[:128])
    seen, eligible = set(excluded), []
    for i, row in enumerate(tables['test']):
        question = norm(row['question'])
        if question not in seen:
            eligible.append(i)
            seen.add(question)
    if min(n_screen, n_reserved) < 2 or len(eligible) < n_screen + n_reserved:
        raise ValueError('Insufficient fresh questions; never relax exclusions')
    random.Random(seed).shuffle(eligible)
    return {'validation': sorted(eligible[:n_screen]),
            'reserved': sorted(eligible[n_screen:n_screen+n_reserved]),
            'eligible_count': len(eligible)}


def gates(summary, config):
    checks = {}
    for kind in ['icl_a', 'icl_b']:
        pair = summary[kind]['paired']['zero']
        checks[kind + '_gain'] = pair['gain'] >= config['min_icl_gain']
        checks[kind + '_interval'] = pair['ci95'][0] > 0
    checks['truncation'] = all(summary[k]['truncated']/summary[k]['n'] <=
                               config['max_truncation_rate'] for k in CONDITIONS)
    return {'eligible': all(checks.values()), 'checks': checks}


def verify(output):
    manifest = parent.read(output / 'manifest.json')
    if manifest['code_sha256'] != code_hash():
        raise ValueError('Screen source or protocol changed')
    if manifest['prepared_sha256'] != parent.file_hash(output / 'prepared.json'):
        raise ValueError('Prepared screen changed')
    return manifest['config'], parent.read(output / 'prepared.json')


def prepare(config, old_path, conditional_path, output):
    output.mkdir(parents=True, exist_ok=True)
    old = parent.read(old_path / 'prepared.json')
    parent_config, conditional = parent.verify(conditional_path)
    original_hash = parent.read(conditional_path / 'manifest.json')['request']['inputs_sha256']['parent_prepared']
    if parent.file_hash(old_path / 'prepared.json') != original_hash:
        raise ValueError('Original support/extraction partition changed')
    if old['banks'] != conditional['banks']:
        raise ValueError('Support banks changed')
    for k in ['model', 'revision', 'dataset', 'dataset_revision', 'max_model_len',
              'max_new_tokens', 'batch_size']:
        if config[k] != parent_config[k]:
            raise ValueError('Screen changed matched inference settings: ' + k)
    provenance = {str(p): parent.file_hash(p) for p in [old_path / 'prepared.json',
                  conditional_path / 'prepared.json', conditional_path / 'manifest.json',
                  conditional_path / 'selection.json']}
    if parent.read(conditional_path / 'selection.json')['eligible']:
        raise ValueError('This follow-up requires the documented failed parent')
    if (output / 'manifest.json').exists():
        prior = parent.read(output / 'manifest.json')
        if prior['inputs_sha256'] != provenance or prior['config'] != config:
            raise ValueError('Screen request changed')
        return verify(output)
    if (output / 'generations.jsonl').exists():
        raise ValueError('Unmanifested generations')
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tables, hashes = {}, {}
    for split in ['train', 'test']:
        path = Path(hf_hub_download(config['dataset'], f'main/{split}-00000-of-00001.parquet',
            repo_type='dataset', revision=config['dataset_revision'], local_files_only=True))
        tables[split] = pq.read_table(path).to_pylist()
        hashes[split] = parent.file_hash(path)
    plan = partition(tables, old, conditional, config['seed'], config['n_screen'], config['n_reserved'])
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'],
                                               local_files_only=True)
    data = {'banks': old['banks'], 'plan': plan, 'splits': {}}
    for split in ['validation', 'reserved']:
        data['splits'][split] = []
        for i in plan[split]:
            row = tables['test'][i]
            q = row['question']
            prompts = {'zero': matched_prompt(q), 'first': matched_prompt(q, suffix=' First,'),
                'cot': matched_prompt(q, suffix=" Let's think step by step."),
                **{k: matched_prompt(q, bank) for k, bank in old['banks'].items()}}
            if any(len(tokenizer.encode(p, add_special_tokens=False)) + config['max_new_tokens'] >
                   config['max_model_len'] for p in prompts.values()):
                raise ValueError('Prompt overflow; no silent truncation')
            data['splits'][split].append({'problem_id': f'gsm8k:test:{i}', 'question': q,
                'answer': row['answer'].split('####')[-1].strip().replace(',', ''), 'prompts': prompts})
    parent.fixed(output / 'prepared.json', data)
    parent.fixed(output / 'manifest.json', {'config': config, 'code_sha256': code_hash(),
        'inputs_sha256': provenance, 'data_sha256': hashes,
        'prepared_sha256': parent.file_hash(output / 'prepared.json'),
        'created_at': datetime.now(timezone.utc).isoformat(), 'conditions': CONDITIONS,
        'eligible_pool_count': plan['eligible_count'], 'test_generation_supported': False})
    return verify(output)


def specifications():
    return {k: {'kind': 'baseline', 'prompt_kind': k, 'layer': None,
                'alpha': 0., 'positions': 'prefill'} for k in CONDITIONS}


def check_rows(output, data, complete=False):
    rows = parent.rows_from(output)
    keys = [(r['split'], r['condition'], r['problem_id']) for r in rows]
    if len(keys) != len(set(keys)) or any(r['split'] != 'validation' for r in rows):
        raise ValueError('Duplicate or non-development generation')
    parent.check_rows(rows, data, specifications(), 'validation')
    if complete:
        audit.select_rows(rows, data, 'validation', CONDITIONS)
    return rows


class ScreenRunner(Runner):
    # The tested parent implementation preserves original batches on resume.
    evaluate_spec = parent.ConditionalRunner.evaluate_spec

    def backend(self):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('Use physical GPU 0 only')
        return super().backend()


def run(output):
    config, data = verify(output)
    declaration = parent.read(output / 'declaration.json')
    if declaration['manifest_sha256'] != parent.file_hash(output / 'manifest.json'):
        raise ValueError('Declaration mismatch')
    if (output / 'screen-selection.json').exists():
        raise ValueError('Screen is frozen after scoring')
    check_rows(output, data)
    runner = ScreenRunner(config, data, output)
    for kind, spec in specifications().items():
        runner.evaluate_spec('validation', kind, spec)
    rows = check_rows(output, data, complete=True)
    packet = audit.make_packet(rows, data, 'validation', CONDITIONS)
    parent.fixed(output / 'validation-review-packet.json', packet)
    parent.fixed(output / 'screen-complete.json', {'status': 'awaiting_blinded_review',
        'rows': len(rows), 'packet_sha256': audit.digest(packet)})
    print(f'Complete: {len(rows)} rows, {len(packet["items"])} blinded items. Scores unopened.', flush=True)


def score(output, annotation_path):
    config, data = verify(output)
    complete = parent.read(output / 'screen-complete.json')
    rows = check_rows(output, data, complete=True)
    packet = parent.read(output / 'validation-review-packet.json')
    if complete['status'] != 'awaiting_blinded_review' or complete['packet_sha256'] != audit.digest(packet):
        raise ValueError('Completion packet mismatch')
    annotations = parent.read(annotation_path)
    freeze = parent.read(output / 'review-freeze.json')
    if (freeze['annotations_file_sha256'] != parent.file_hash(annotation_path)
            or freeze['packet_sha256'] != audit.digest(packet)
            or not freeze['annotations_commit']):
        raise ValueError('Complete annotation freeze required before scoring')
    result = audit.score(rows, data, 'validation', CONDITIONS, packet, annotations,
                         config['bootstrap_samples'])
    decision = gates(result['summary'], config)
    parent.fixed(output / 'screen-annotations.json', annotations)
    parent.fixed(output / 'screen-audit.json', result)
    decision.update(manifest_sha256=parent.file_hash(output / 'manifest.json'),
        rows_sha256=audit.digest(rows), annotations_sha256=audit.digest(annotations),
        audit_sha256=parent.file_hash(output / 'screen-audit.json'))
    parent.fixed(output / 'screen-selection.json', decision)
    print(decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'screen', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/gsm8k_test_development.json')
    parser.add_argument('--old', type=Path, default=ROOT / 'runs/gsm8k-steering-v2')
    parser.add_argument('--conditional', type=Path, default=ROOT / 'runs/gsm8k-conditional-v1')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / '.writer.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            prepare(parent.read(args.config), args.old, args.conditional, args.output)
            print('Prepared development and reservation without inference.', flush=True)
        elif args.stage == 'screen':
            run(args.output)
        else:
            if args.annotations is None:
                parser.error('score requires --annotations')
            score(args.output, args.annotations)


if __name__ == '__main__':
    main()

"""One fixed ICL screen on fresh GSM8K training questions with longer calculations."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random

from analysis import gsm8k_fresh_pool as fresh
from analysis import gsm8k_test_development as previous

base, audit = previous.parent, previous.audit
ROOT = Path(__file__).parents[1]
CONDITIONS = previous.CONDITIONS
CONFIG = ROOT/'configs/gsm8k_complex_screen.json'


def code_hash():
    files = [Path(__file__), ROOT/'analysis/gsm8k_fresh_pool.py', CONFIG,
             ROOT/'research/gsm8k-complex-screen-protocol.md',
             ROOT/'results/gsm8k-fresh-inventory-union-v1.json']
    return audit.digest({'parent': previous.code_hash(), 'files': {
        str(p.relative_to(ROOT)): base.file_hash(p) for p in files}})


def partition(pool, config):
    ids = [row['problem_id'] for row in pool if row['calculation_annotations'] >=
           config['min_calculation_annotations']]
    if (len(ids) != len(set(ids)) or any(not i.startswith('gsm8k:train:') for i in ids)
            or ids != sorted(ids, key=lambda i: int(i.rsplit(':', 1)[1]))):
        raise ValueError('Require distinct, ordered training identities')
    n, reserved = config['n_screen'], config['n_reserved']
    if min(n, reserved) < 2 or len(ids) < n+reserved:
        raise ValueError('Insufficient eligible pool; never relax exclusions')
    count = len(ids)
    random.Random(config['seed']).shuffle(ids)
    order = lambda rows: sorted(rows, key=lambda i: int(i.rsplit(':', 1)[1]))
    return {'validation': order(ids[:n]), 'reserved': order(ids[n:n+reserved]),
            'eligible_count': count}


def verify(output):
    manifest = base.read(output/'manifest.json')
    if manifest['code_sha256'] != code_hash() or manifest['config'] != base.read(CONFIG):
        raise ValueError('Declared source, protocol, or configuration changed')
    if manifest['prepared_sha256'] != base.file_hash(output/'prepared.json'):
        raise ValueError('Prepared screen changed')
    for path, expected in manifest['inputs_sha256'].items():
        if base.file_hash(Path(path)) != expected:
            raise ValueError('Inventory or support input changed: '+path)
    return manifest['config'], base.read(output/'prepared.json')


def prepare(inventory, support, output):
    if (output/'manifest.json').exists():
        return verify(output)
    if (output/'generations.jsonl').exists():
        raise ValueError('Unmanifested generations')
    config = base.read(CONFIG)
    summary, observations, pool = [base.read(inventory/name) for name in
                                  ['summary.json', 'observations.json', 'fresh-pool.json']]
    if summary != base.read(ROOT/'results/gsm8k-fresh-inventory-union-v1.json'):
        raise ValueError('Require the fixed, reconciled inventory')
    if fresh.digest(observations) != summary['observations_sha256']:
        raise ValueError('Reconciled observations changed')
    for split, rows in pool.items():
        if fresh.digest(rows) != summary['splits'][split]['identities_and_counts_sha256']:
            raise ValueError('Fresh pool changed')
    inputs = {str((inventory/name).resolve()): base.file_hash(inventory/name)
              for name in ['summary.json', 'observations.json', 'fresh-pool.json']}
    old, old_manifest = base.read(support/'prepared.json'), base.read(support/'manifest.json')
    if base.file_hash(support/'prepared.json') != old_manifest['prepared_sha256']:
        raise ValueError('Original support preparation changed')
    for key in ['model', 'revision', 'dataset', 'dataset_revision', 'max_model_len', 'max_new_tokens']:
        if old_manifest['config'][key] != config[key]:
            raise ValueError('Matched inference setting changed: '+key)
    if set(old['banks']) != {'icl_a', 'icl_b'} or any(len(b) != 8 for b in old['banks'].values()):
        raise ValueError('Require the two original eight-example banks')
    inputs.update({str((support/name).resolve()): base.file_hash(support/name)
                   for name in ['prepared.json', 'manifest.json']})
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tables, hashes = {}, {}
    for split in ['train', 'test']:
        path = Path(hf_hub_download(config['dataset'], f'main/{split}-00000-of-00001.parquet',
            repo_type='dataset', revision=config['dataset_revision'], local_files_only=True))
        tables[split], hashes[split] = pq.read_table(path).to_pylist(), base.file_hash(path)
        if hashes[split] != old_manifest['data_sha256'][f'main/{split}-00000-of-00001.parquet']:
            raise ValueError('Pinned dataset bytes changed')
    # Replay exclusions using the union, rather than trusting a list of eligible IDs.
    reservations = {}
    for name, split in [('gsm8k-conditional-v1', 'test'), ('gsm8k-test-development-v1', 'reserved')]:
        path = ROOT/'runs'/name/'prepared.json'
        reservations[name] = {r['problem_id'] for r in base.read(path)['splits'][split]}
        inputs[str(path.resolve())] = base.file_hash(path)
    replay, replay_pool = fresh.partition_inventory(tables, set(observations['ids']),
        set(observations['question_texts']), set(observations['generated_ids']), reservations)
    if replay_pool != pool or audit.digest(replay['reservations']) != audit.digest(summary['reservations']):
        raise ValueError('Inventory replay or old reservations differ')
    plan = partition(pool['train'], config)
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)
    data = {'banks': old['banks'], 'plan': plan, 'splits': {}}
    for split in ['validation', 'reserved']:
        data['splits'][split] = []
        for pid in plan[split]:
            row = tables['train'][int(pid.rsplit(':', 1)[1])]
            q = row['question']
            prompts = {'zero': previous.matched_prompt(q),
                'first': previous.matched_prompt(q, suffix=' First,'),
                'cot': previous.matched_prompt(q, suffix=" Let's think step by step."),
                **{k: previous.matched_prompt(q, bank) for k, bank in old['banks'].items()}}
            if any(len(tokenizer.encode(p, add_special_tokens=False))+config['max_new_tokens'] >
                   config['max_model_len'] for p in prompts.values()):
                raise ValueError('Prompt overflow; no replacement or truncation')
            data['splits'][split].append({'problem_id': pid, 'question': q,
                'answer': row['answer'].split('####')[-1].strip().replace(',', ''), 'prompts': prompts})
    output.mkdir(parents=True, exist_ok=True)
    base.fixed(output/'prepared.json', data)
    base.fixed(output/'manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'config': config, 'code_sha256': code_hash(), 'inputs_sha256': inputs,
        'data_sha256': hashes, 'prepared_sha256': base.file_hash(output/'prepared.json'),
        'conditions': CONDITIONS, 'eligible_pool_count': plan['eligible_count'],
        'steering_generation_supported': False, 'test_generation_supported': False})
    return verify(output)


def run(output):
    config, data = verify(output)
    declaration = base.read(output/'declaration.json')
    if (declaration['manifest_sha256'] != base.file_hash(output/'manifest.json')
            or not declaration['declaration_commit']):
        raise ValueError('Committed declaration required before generation')
    if (output/'screen-selection.json').exists():
        raise ValueError('Screen is frozen after scoring')
    previous.check_rows(output, data)
    runner = previous.ScreenRunner(config, data, output)
    for kind, spec in previous.specifications().items():
        runner.evaluate_spec('validation', kind, spec)
    rows = previous.check_rows(output, data, complete=True)
    packet = audit.make_packet(rows, data, 'validation', CONDITIONS)
    base.fixed(output/'validation-review-packet.json', packet)
    base.fixed(output/'screen-complete.json', {'status': 'awaiting_blinded_review',
        'rows': len(rows), 'packet_sha256': audit.digest(packet)})
    print(f'Complete: {len(rows)} rows, {len(packet["items"])} blinded items. Scores unopened.', flush=True)


def score(output, annotation_path):
    config, data = verify(output)
    rows = previous.check_rows(output, data, complete=True)
    complete, packet, freeze = [base.read(output/name) for name in
        ['screen-complete.json', 'validation-review-packet.json', 'review-freeze.json']]
    if (complete['status'] != 'awaiting_blinded_review' or complete['packet_sha256'] != audit.digest(packet)
            or freeze['annotations_file_sha256'] != base.file_hash(annotation_path)
            or freeze['packet_sha256'] != audit.digest(packet) or not freeze['annotations_commit']):
        raise ValueError('Complete frozen annotations required before scoring')
    annotations = base.read(annotation_path)
    result = audit.score(rows, data, 'validation', CONDITIONS, packet, annotations, config['bootstrap_samples'])
    decision = previous.gates(result['summary'], config)
    base.fixed(output/'screen-annotations.json', annotations)
    base.fixed(output/'screen-audit.json', result)
    decision.update(manifest_sha256=base.file_hash(output/'manifest.json'), rows_sha256=audit.digest(rows),
        annotations_sha256=audit.digest(annotations), audit_sha256=base.file_hash(output/'screen-audit.json'))
    base.fixed(output/'screen-selection.json', decision)
    print(decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'screen', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, default=ROOT/'runs/gsm8k-fresh-inventory-union-v1')
    parser.add_argument('--support', type=Path, default=ROOT/'runs/gsm8k-steering-v2')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'.writer.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            prepare(args.inventory, args.support, args.output)
            print('Prepared screen and reservation without inference.', flush=True)
        elif args.stage == 'screen':
            run(args.output)
        else:
            if args.annotations is None:
                parser.error('score requires --annotations')
            score(args.output, args.annotations)


if __name__ == '__main__':
    main()

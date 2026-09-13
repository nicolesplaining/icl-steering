"""Fixed three-condition screen of released original and complex demonstrations."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random

from analysis import gsm8k_answer_audit as audit
from analysis import gsm8k_published_inventory as inventory
from analysis.published_screen_backend import BOUNDARY, PublishedBackend, solution
from icl_steering.report import paired_difference
from replication.gsm8k_protocol import grade_answer


ROOT = Path(__file__).parents[1]
CONFIG = ROOT/'configs/gsm8k_published_screen.json'
INVENTORY = ROOT/'results/gsm8k-published-inventory-union-v1.json'
CONDITIONS = ['zero', 'icl_original', 'icl_complex']
PAIRS = [('icl_complex', 'zero'), ('icl_original', 'zero'), ('icl_complex', 'icl_original')]
read, sha = inventory.read, inventory.sha


def fixed(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Refuse changed artifact: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def code_hash():
    names = ['analysis/gsm8k_published_screen.py', 'analysis/published_screen_backend.py',
             'analysis/gsm8k_published_inventory.py', 'analysis/gsm8k_fresh_pool.py',
             'analysis/complexity_prompt_budget.py', 'analysis/complexity_prompt_inspection.py',
             'analysis/gsm8k_answer_audit.py', 'replication/gsm8k_protocol.py',
             'replication/gsm8k_activation.py', 'src/icl_steering/model.py',
             'src/icl_steering/scoring.py', 'src/icl_steering/report.py',
             'configs/gsm8k_published_screen.json', 'research/gsm8k-published-screen-protocol.md',
             'results/gsm8k-published-inventory-union-v1.json',
             'results/gsm8k-published-prompt-budget-v1.json']
    return audit.digest({name: sha(ROOT/name) for name in names})


def partition(pool, config):
    ids = [r['problem_id'] for r in pool]
    order = lambda values: sorted(values, key=lambda x: int(x.rsplit(':', 1)[1]))
    if (len(set(ids)) != len(ids) or ids != order(ids)
            or any(not i.startswith('gsm8k:train:') for i in ids)):
        raise ValueError('Require distinct dataset-ordered training identities')
    sizes = [config[k] for k in ['n_extract', 'n_screen', 'n_reserved']]
    if min(sizes) < 2 or sum(sizes) > len(ids):
        raise ValueError('Insufficient unused questions; never relax exclusions')
    random.Random(config['seed']).shuffle(ids)
    plan, start = {}, 0
    for name, n in zip(['extract', 'validation', 'reserved'], sizes):
        plan[name] = order(ids[start:start+n]); start += n
    return plan


def tokenizer(config):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    budget = read(ROOT/'results/gsm8k-published-prompt-budget-v1.json')
    if budget['model'] != config['model'] or budget['model_revision'] != config['revision']:
        raise ValueError('Tokenizer revision differs from the inspected budget')
    for name, expected in budget['tokenizer_files_sha256'].items():
        path = Path(hf_hub_download(config['model'], name, revision=config['revision'], local_files_only=True))
        if sha(path) != expected:
            raise ValueError('Tokenizer/config bytes changed: '+name)
    return AutoTokenizer.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)


def prompts(question, banks):
    query = f"Question: {question}\nLet's think step by step\n"
    return {'zero': query, **{'icl_'+name: text+'\n'+query for name, text in banks.items()}}


def verify(output):
    manifest = read(output/'manifest.json')
    if (manifest['config'] != read(CONFIG) or manifest['code_sha256'] != code_hash()
            or manifest['conditions'] != CONDITIONS or manifest['fixed_target'] != 'icl_complex'
            or manifest['reserved_generation_supported'] is not False):
        raise ValueError('Frozen configuration, source, or scope changed')
    if manifest['prepared_sha256'] != sha(output/'prepared.json'):
        raise ValueError('Prepared questions changed')
    for path, expected in manifest['inputs_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Frozen input changed: '+path)
    return manifest['config'], read(output/'prepared.json')


def prepare(source, prompt_dir, output):
    if (output/'manifest.json').exists():
        return verify(output)
    if (output/'batches').exists() or (output/'generations.jsonl').exists():
        raise ValueError('Unmanifested generation artifacts')
    config = read(CONFIG)
    summary, seen, reserved, pool = [read(source/name) for name in
        ['summary.json', 'observations.json', 'reservations.json', 'fresh-pool.json']]
    if (summary != read(INVENTORY) or summary['kind'] != 'laptop_server_union'
            or summary['observations_sha256'] != inventory.fresh.digest(seen)
            or summary['reservations_sha256'] != inventory.fresh.digest(reserved)):
        raise ValueError('Require the frozen reconciled inventory')
    tables = inventory.tables()
    replay, actual_pool = inventory.fresh.partition_inventory(tables, set(seen['ids']),
        set(seen['question_texts']), set(seen['generated_ids']), {k: set(v) for k, v in reserved.items()})
    if actual_pool != pool or replay['reservations'] != summary['reservations']:
        raise ValueError('Unused pool or reservations do not replay')
    demos = inventory.demonstrations(prompt_dir)
    if not demos <= set(seen['question_texts']):
        raise ValueError('Published demonstrations missing from exclusions')
    banks = {name: (prompt_dir/f'prompt_{name}.txt').read_text() for name in inventory.PROMPT_HASHES}
    plan = partition(pool['train'], config)
    tok = tokenizer(config)
    data = {'banks': banks, 'plan': plan, 'splits': {}}
    lengths = {kind: [] for kind in CONDITIONS}
    for split, ids in plan.items():
        data['splits'][split] = []
        for pid in ids:
            row = tables['train'][int(pid.rsplit(':', 1)[1])]
            values = prompts(row['question'], banks)
            for kind, prompt in values.items():
                length = len(tok.encode(prompt, add_special_tokens=False)); lengths[kind].append(length)
                if length+config['max_new_tokens'] > config['max_model_len']:
                    raise ValueError('Context overflow; no question replacement or truncation')
            data['splits'][split].append({'problem_id': pid, 'question': row['question'],
                'answer': row['answer'].split('####')[-1].strip().replace(',', ''), 'prompts': values})
    inputs = {str((source/name).resolve()): sha(source/name) for name in
              ['summary.json', 'observations.json', 'reservations.json', 'fresh-pool.json']}
    inputs.update({str((prompt_dir/f'prompt_{name}.txt').resolve()): expected
                   for name, expected in inventory.PROMPT_HASHES.items()})
    fixed(output/'prepared.json', data)
    fixed(output/'manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'config': config, 'code_sha256': code_hash(), 'inputs_sha256': inputs,
        'prepared_sha256': sha(output/'prepared.json'), 'data_sha256': inventory.DATA_HASHES,
        'conditions': CONDITIONS, 'fixed_target': 'icl_complex', 'eligible_questions': len(pool['train']),
        'prompt_tokens': {k: {'min': min(v), 'max': max(v)} for k, v in lengths.items()},
        'reserved_generation_supported': False, 'extraction_generation_supported': False})
    return verify(output)


def check_batch(batch, condition, problems, config, runtime, tok):
    if len(batch) != len(problems):
        raise ValueError('Batch length mismatch')
    for row, problem in zip(batch, problems):
        if (row['problem_id'] != problem['problem_id'] or row['condition'] != condition
                or row['split'] != 'validation' or row['prompt'] != problem['prompts'][condition]
                or row['answer'] != problem['answer']):
            raise ValueError('Unexpected question, condition, split, or prompt')
        ids = row['token_ids']
        if (not ids or len(ids) != row['generated_tokens'] or len(ids) > config['max_new_tokens']
                or any(not isinstance(t, int) or t < 0 for t in ids)):
            raise ValueError('Invalid token accounting')
        raw = tok.decode(ids, skip_special_tokens=True)
        if raw != row['text'] or solution(raw) != row['solution_text']:
            raise ValueError('Token decode or query boundary changed')
        eos = runtime['eos_token_ids']
        reason = 'next_question' if BOUNDARY.search(raw) else ('eos' if ids[-1] in eos else 'length')
        if row['finish_reason'] != reason or row['truncated'] != (reason == 'length'):
            raise ValueError('Stop reason mismatch')
        if reason == 'length' and len(ids) != config['max_new_tokens']:
            raise ValueError('Premature length stop')
        if any(t in eos for t in ids[:-1]):
            raise ValueError('Tokens after EOS')
        if reason == 'next_question' and BOUNDARY.search(tok.decode(ids[:-1], skip_special_tokens=True)):
            raise ValueError('Tokens after first question boundary')
        grades = grade_answer(row['solution_text'], problem['answer'], row['truncated'])
        if any(row[k] != v for k, v in grades.items()):
            raise ValueError('Stored query grade changed')


def collect(output, config, data, tok, complete=False):
    paths = {(f'{kind}-{start:04d}.json'): (kind, start) for kind in CONDITIONS
             for start in range(0, config['n_screen'], config['batch_size'])}
    files = sorted((output/'batches').glob('*.json'))
    if any(f.name not in paths for f in files) or (complete and len(files) != len(paths)):
        raise ValueError('Unexpected or incomplete batches')
    if not files:
        return []
    runtime = read(output/'runtime.json')
    if runtime['physical_gpu'] != 0 or runtime['frozen'] is not True or runtime['dtype'] != 'torch.bfloat16':
        raise ValueError('Unexpected model runtime')
    rows = []
    for kind in CONDITIONS:
        for start in range(0, config['n_screen'], config['batch_size']):
            path = output/'batches'/f'{kind}-{start:04d}.json'
            if path.exists():
                batch = read(path)
                check_batch(batch, kind, data['splits']['validation'][start:start+config['batch_size']],
                            config, runtime, tok)
                rows.extend(batch)
    if complete:
        audit.select_rows(rows, data, 'validation', CONDITIONS)
    return rows


def run(output, backend_factory=PublishedBackend):
    config, data = verify(output)
    declaration = read(output/'declaration.json')
    if declaration['manifest_sha256'] != sha(output/'manifest.json') or not declaration['declaration_commit']:
        raise ValueError('Committed declaration required')
    if (output/'screen-selection.json').exists():
        raise ValueError('Screen already scored')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Use physical GPU 0 only')
    tok = tokenizer(config)
    collect(output, config, data, tok)
    backend = None
    for kind in CONDITIONS:
        for start in range(0, config['n_screen'], config['batch_size']):
            path = output/'batches'/f'{kind}-{start:04d}.json'
            if path.exists():
                continue
            if backend is None:
                backend = backend_factory(config)
                fixed(output/'runtime.json', backend.runtime)
            problems = data['splits']['validation'][start:start+config['batch_size']]
            outputs = backend.records([p['prompts'][kind] for p in problems], config)
            if len(outputs) != len(problems):
                raise ValueError('Backend batch size mismatch')
            batch = [{'split': 'validation', 'condition': kind, 'problem_id': p['problem_id'],
                      'prompt': p['prompts'][kind], 'answer': p['answer'], **result,
                      **grade_answer(result['solution_text'], p['answer'], result['truncated'])}
                     for p, result in zip(problems, outputs)]
            check_batch(batch, kind, problems, config, backend.runtime, tok)
            fixed(path, batch)
            print(f'Screen {kind}: {start+len(problems)}/{config["n_screen"]}', flush=True)
    rows = collect(output, config, data, tok, complete=True)
    content = ''.join(json.dumps(r)+'\n' for r in rows)
    export = output/'generations.jsonl'
    if export.exists() and export.read_text() != content:
        raise ValueError('Generation export changed')
    if not export.exists():
        temp = export.with_suffix('.jsonl.tmp'); temp.write_text(content); temp.replace(export)
    packet = audit.make_packet(rows, data, 'validation', CONDITIONS)
    fixed(output/'validation-review-packet.json', packet)
    fixed(output/'screen-complete.json', {'status': 'awaiting_blinded_review', 'rows': len(rows),
        'packet_sha256': audit.digest(packet), 'generation_file_sha256': sha(export),
        'batches_verified': len(CONDITIONS)*math.ceil(config['n_screen']/config['batch_size']),
        'extraction_rows': 0, 'reserved_rows': 0})
    print('Screen complete; freeze blind review before scoring.', flush=True)


def exact_p(wins, losses):
    n = wins+losses
    return min(1., 2*sum(math.comb(n, k) for k in range(min(wins, losses)+1))/2**n)


def holm(values):
    adjusted, previous = {}, 0.
    for rank, name in enumerate(sorted(values, key=values.get)):
        previous = max(previous, min(1., (len(values)-rank)*values[name]))
        adjusted[name] = previous
    return adjusted


def gates(summary, contrast, config):
    complex_, zero = summary['icl_complex'], summary['zero']
    checks = {'complete': set(summary) == set(CONDITIONS) and all(v['n'] == 512 for v in summary.values()),
        'complex_gain': contrast['gain'] >= config['min_icl_gain'],
        'complex_interval': contrast['ci95'][0] > 0,
        'complex_adjusted_p': contrast['holm_p'] < config['max_adjusted_p'],
        'primary_not_below_zero': complex_['primary_completed_correct'] >= zero['primary_completed_correct'],
        'truncation': all(v['truncated']/v['n'] <= config['max_truncation_rate'] for v in summary.values())}
    return {'eligible': all(checks.values()), 'checks': checks, 'fixed_target': 'icl_complex',
            'reserved_generation_authorized': False, 'steering_generation_authorized': False}


def score(output, annotation_path):
    config, data = verify(output)
    rows = collect(output, config, data, tokenizer(config), complete=True)
    complete, packet, freeze = [read(output/name) for name in
        ['screen-complete.json', 'validation-review-packet.json', 'review-freeze.json']]
    if (complete['generation_file_sha256'] != sha(output/'generations.jsonl')
            or complete['packet_sha256'] != audit.digest(packet)
            or freeze['packet_sha256'] != audit.digest(packet)
            or freeze['annotations_file_sha256'] != sha(annotation_path) or not freeze['annotations_commit']):
        raise ValueError('Complete frozen annotations required')
    annotations = read(annotation_path)
    result = audit.score(rows, data, 'validation', CONDITIONS, packet, annotations, config['bootstrap_samples'])
    lookup = {r['response_id']: r['stated_answer'] for r in annotations['answers']}
    questions = {r['problem_id']: r['question'] for r in data['splits']['validation']}
    groups = {metric: {k: [] for k in CONDITIONS} for metric in ['primary', 'audited']}
    for row in rows:
        correct = row['correct']
        if not row['parseable']:
            value = lookup[audit.response_item(row, questions)['response_id']]
            correct = value is not None and Fraction(value) == Fraction(row['answer'])
        for metric, ok in [('primary', row['completed_correct']), ('audited', correct and not row['truncated'])]:
            groups[metric][row['condition']].append({'problem_id': row['problem_id'], 'correct': ok})
    result['contrasts'] = {}
    for metric, values in groups.items():
        contrasts = {left+'-'+right: {'left': left, 'right': right,
            **paired_difference(values[left], values[right], config['bootstrap_samples'], config['bootstrap_seed'])}
            for left, right in PAIRS}
        probabilities = {k: exact_p(v['wins'], v['losses']) for k, v in contrasts.items()}
        adjusted = holm(probabilities)
        for k in contrasts:
            contrasts[k].update(mcnemar_two_sided_p=probabilities[k], holm_p=adjusted[k])
        result['contrasts'][metric] = contrasts
    result['limitations'] = 'Qwen adaptation of published prompt assets on unused training questions. Three fixed contrasts per metric; intervals unadjusted. Single blinded reviewer for unparsed outputs. No independent steering confirmation.'
    decision = gates(result['summary'], result['contrasts']['audited']['icl_complex-zero'], config)
    fixed(output/'screen-annotations.json', annotations)
    fixed(output/'screen-audit.json', result)
    decision.update(manifest_sha256=sha(output/'manifest.json'), rows_sha256=audit.digest(rows),
                    annotations_sha256=audit.digest(annotations), audit_sha256=sha(output/'screen-audit.json'))
    fixed(output/'screen-selection.json', decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'screen', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--prompt-dir', type=Path)
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            if args.inventory is None or args.prompt_dir is None:
                parser.error('prepare requires --inventory and --prompt-dir')
            prepare(args.inventory, args.prompt_dir, args.output)
            print('Prepared partitions without inference.', flush=True)
        elif args.stage == 'screen':
            run(args.output)
        else:
            if args.annotations is None:
                parser.error('score requires --annotations')
            score(args.output, args.annotations)

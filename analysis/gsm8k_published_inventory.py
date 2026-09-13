"""Reconcile unused questions and published demonstration exclusions, without sampling."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from analysis import gsm8k_fresh_pool as fresh
from analysis.complexity_prompt_budget import PROMPT_HASHES, TRAIN_HASH
from analysis.complexity_prompt_inspection import UPSTREAM


DATA_HASHES = {'train': TRAIN_HASH,
               'test': 'ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59'}
RESERVATIONS = [('gsm8k-conditional-v1', 'test', 512),
                ('gsm8k-test-development-v1', 'reserved', 256),
                ('gsm8k-complex-screen-v1', 'reserved', 512)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def tables():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    result = {}
    for split, expected in DATA_HASHES.items():
        path = Path(hf_hub_download('openai/gsm8k', f'main/{split}-00000-of-00001.parquet',
                    repo_type='dataset', revision=fresh.REVISION, local_files_only=True))
        if sha(path) != expected:
            raise ValueError('Pinned dataset changed: '+split)
        result[split] = pq.read_table(path).to_pylist()
    return result


def demonstrations(prompt_dir):
    questions = set()
    for name, expected in PROMPT_HASHES.items():
        path = prompt_dir/f'prompt_{name}.txt'
        if sha(path) != expected:
            raise ValueError('Released prompt changed: '+name)
        blocks = re.split(r'(?m)^Question: ', path.read_text())
        if blocks[0] or len(blocks) != 9:
            raise ValueError('Require eight demonstrations per bank')
        questions.update(fresh.normalize(block.split('\n', 1)[0]) for block in blocks[1:])
    return questions


def write_snapshot(output, observations, reservations, metadata):
    if output.exists():
        raise ValueError('Refuse to overwrite an inventory')
    summary, pools = fresh.partition_inventory(tables(), set(observations['ids']),
        set(observations['question_texts']), set(observations['generated_ids']),
        {name: set(ids) for name, ids in reservations.items()})
    summary.update(created_at=datetime.now(timezone.utc).isoformat(),
        dataset='openai/gsm8k', dataset_revision=fresh.REVISION, dataset_sha256=DATA_HASHES,
        no_sampling_or_generation=True, observations_sha256=fresh.digest(observations),
        reservations_sha256=fresh.digest(reservations), published_prompt_sha256=PROMPT_HASHES,
        upstream_commit=UPSTREAM, source_sha256=sha(Path(__file__)),
        exclusion_source_sha256=sha(Path(fresh.__file__)), **metadata)
    output.mkdir(parents=True)
    for name, value in [('summary.json', summary), ('observations.json', observations),
                        ('reservations.json', reservations), ('fresh-pool.json', pools)]:
        (output/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    return summary


def snapshot(runs, prompt_dir, output):
    if output.exists():
        raise ValueError('Refuse to overwrite an inventory')
    ids, questions, generated, files = fresh.inventory(runs)
    demos = demonstrations(prompt_dir)
    questions.update(demos)
    reservations = {}
    for name, split, n in RESERVATIONS:
        rows = read(runs/name/'prepared.json')['splits'][split]
        values = sorted(r['problem_id'] for r in rows)
        if len(values) != len(set(values)) or len(values) != n:
            raise ValueError('Reservation identities changed: '+name)
        reservations[name] = values
    # Require quiescent inputs, including the file set, before snapshot publication.
    for name, expected in files.items():
        if sha(runs/name) != expected:
            raise ValueError('Source changed during inventory: '+name)
    current = {str(p.relative_to(runs)) for p in runs.rglob('*')
               if p.is_file() and p.suffix in {'.json', '.jsonl'}}
    if current != set(files):
        raise ValueError('Source file set changed during inventory')
    observations = {'ids': sorted(ids), 'question_texts': sorted(questions),
                    'generated_ids': sorted(generated)}
    summary = write_snapshot(output, observations, reservations,
        {'kind': 'single_checkout', 'input_files_sha256': files,
         'published_unique_questions': len(demos),
         'scope': 'All recorded JSON/JSONL under one runs directory, published original/complex '
                  'demonstrations, first 128 rows of each split, and all three existing reservations. '
                  'Exact normalized text exclusion, not a paraphrase audit.'})
    return summary


def reconcile(snapshots, output):
    if len(snapshots) != 2:
        raise ValueError('Require laptop and server snapshots')
    observations = {'ids': set(), 'question_texts': set(), 'generated_ids': set()}
    reservations = None
    inputs = {}
    counts = []
    for source in snapshots:
        summary, seen, reserved, pools = [read(source/name) for name in
            ['summary.json', 'observations.json', 'reservations.json', 'fresh-pool.json']]
        if (summary['kind'] != 'single_checkout' or summary['dataset_sha256'] != DATA_HASHES
                or summary['published_prompt_sha256'] != PROMPT_HASHES
                or summary['observations_sha256'] != fresh.digest(seen)
                or summary['reservations_sha256'] != fresh.digest(reserved)
                or summary['source_sha256'] != sha(Path(__file__))
                or summary['exclusion_source_sha256'] != sha(Path(fresh.__file__))):
            raise ValueError('Snapshot provenance mismatch')
        if reservations is not None and reservations != reserved:
            raise ValueError('Laptop and server reservations disagree')
        reservations = reserved
        replay, replay_pool = fresh.partition_inventory(tables(), set(seen['ids']),
            set(seen['question_texts']), set(seen['generated_ids']),
            {k: set(v) for k, v in reserved.items()})
        if replay_pool != pools or replay['reservations'] != summary['reservations']:
            raise ValueError('Snapshot pool does not replay')
        for key in observations:
            observations[key].update(seen[key])
        counts.append({key: len(values) for key, values in seen.items()})
        inputs.update({str((source/name).resolve()): sha(source/name) for name in
                       ['summary.json', 'observations.json', 'reservations.json', 'fresh-pool.json']})
    return write_snapshot(output, {k: sorted(v) for k, v in observations.items()}, reservations,
        {'kind': 'laptop_server_union', 'inputs_sha256': inputs, 'source_observation_counts': counts,
         'scope': 'Union of independently snapshotted laptop/server records with published-bank '
                  'exclusions and three unchanged reservations. No sample selected or model called.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['snapshot', 'reconcile'])
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--prompt-dir', type=Path)
    parser.add_argument('--snapshots', type=Path, nargs=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'snapshot':
        if args.runs is None or args.prompt_dir is None:
            parser.error('snapshot requires --runs and --prompt-dir')
        result = snapshot(args.runs, args.prompt_dir, args.output)
    else:
        if args.snapshots is None:
            parser.error('reconcile requires --snapshots')
        result = reconcile(args.snapshots, args.output)
    print(json.dumps({k: result[k] for k in ['kind', 'splits', 'reservations', 'observed_unique_ids',
                                          'generated_unique_ids', 'observed_question_texts']}))

"""Independently check the published screen's partitions before inference."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(run, source, cache):
    import pyarrow.parquet as pq
    manifest, data = read(run/'manifest.json'), read(run/'prepared.json')
    assert manifest['prepared_sha256'] == sha(run/'prepared.json')
    assert manifest['extraction_generation_supported'] is False
    assert manifest['reserved_generation_supported'] is False
    assert manifest['conditions'] == ['zero', 'icl_original', 'icl_complex']
    assert manifest['fixed_target'] == 'icl_complex'
    assert not (run/'batches').exists() and not (run/'generations.jsonl').exists()
    config = manifest['config']
    assert [config[k] for k in ['seed', 'n_extract', 'n_screen', 'n_reserved']] == [3402, 256, 512, 512]
    tables = {}
    for split, expected in manifest['data_sha256'].items():
        path = cache/'hub/datasets--openai--gsm8k/blobs'/expected
        assert sha(path) == expected
        tables[split] = pq.read_table(path).to_pylist()
    seen, reserved, pool = [read(source/name) for name in
                            ['observations.json', 'reservations.json', 'fresh-pool.json']]
    for name in ['summary.json', 'observations.json', 'reservations.json', 'fresh-pool.json']:
        hashes = [value for path, value in manifest['inputs_sha256'].items()
                  if Path(path).name == name]
        assert hashes == [sha(source/name)]
    norm = lambda text: ' '.join(text.casefold().split())
    lookup = {f'gsm8k:{split}:{i}': row for split, rows in tables.items() for i, row in enumerate(rows)}
    used = set(seen['ids']) | set(seen['generated_ids'])
    for name, values in reserved.items():
        assert not set(values) & set(seen['generated_ids']), name
        used.update(values)
    used.update(f'gsm8k:{split}:{i}' for split in tables for i in range(128))
    excluded = set(seen['question_texts']) | {norm(lookup[pid]['question']) for pid in used}
    eligible, visited = [], set(excluded)
    for index, row in enumerate(tables['train']):
        pid, text = f'gsm8k:train:{index}', norm(row['question'])
        if pid not in used and text not in visited:
            eligible.append(pid)
            visited.add(text)
    assert eligible == [r['problem_id'] for r in pool['train']]
    assert len(eligible) == manifest['eligible_questions'] == 6282
    random.Random(3402).shuffle(eligible)
    plan = {name: sorted(eligible[start:stop], key=lambda pid: int(pid.rsplit(':', 1)[1]))
            for name, start, stop in [('extract', 0, 256), ('validation', 256, 768), ('reserved', 768, 1280)]}
    assert data['plan'] == plan
    assert len(set().union(*map(set, plan.values()))) == 1280
    assert set(data['splits']) == set(plan)
    assert set(data['banks']) == {'original', 'complex'}
    for name, bank in data['banks'].items():
        expected = [value for path, value in manifest['inputs_sha256'].items()
                    if Path(path).name == f'prompt_{name}.txt']
        assert expected == [hashlib.sha256(bank.encode()).hexdigest()]
        demos = re.split(r'(?m)^Question: ', bank)
        assert demos[0] == '' and len(demos) == 9
        assert all(norm(block.split('\n', 1)[0]) in excluded for block in demos[1:])
    for split, ids in plan.items():
        rows = data['splits'][split]
        assert [row['problem_id'] for row in rows] == ids
        for row in rows:
            original = lookup[row['problem_id']]
            assert row['question'] == original['question']
            assert row['answer'] == original['answer'].split('####')[-1].strip().replace(',', '')
            query = f"Question: {original['question']}\nLet's think step by step\n"
            assert row['prompts'] == {'zero': query,
                **{'icl_'+k: v+'\n'+query for k, v in data['banks'].items()}}
    return {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'script_sha256': sha(Path(__file__)), 'manifest_sha256': sha(run/'manifest.json'),
            'prepared_sha256': sha(run/'prepared.json'), 'eligible_questions': 6282,
            'partition_sizes': {k: len(v) for k, v in plan.items()}, 'selected_ids': plan,
            'generation_rows': 0, 'prompts_checked': 3840,
            'scope': 'Independent pinned-parquet exclusion, shuffle, identity, gold, and exact prompt replay. '
                     'No production preparation imports. Token lengths are checked separately with the pinned tokenizer.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite preflight')
    result = check(args.run, args.inventory, args.cache)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'selected_ids'}, indent=2))

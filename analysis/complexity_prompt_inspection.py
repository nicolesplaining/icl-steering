"""Inspect released prompt assets and question overlap without model inference."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


UPSTREAM = '378f7a88fb4c6a3fdb294f0dcf8a702420f76724'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(text.casefold().split())


def inspect(upstream, runs):
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == UPSTREAM
    partitions, inputs = {}, {}
    for name, split in [('gsm8k-complex-screen-v1', 'validation'),
                        ('gsm8k-complex-screen-v1', 'reserved'),
                        ('gsm8k-conditional-v1', 'test'),
                        ('gsm8k-test-development-v1', 'reserved')]:
        path = runs/name/'prepared.json'
        inputs[str(path)] = file_hash(path)
        problems = json.loads(path.read_text())['splits'][split]
        mapping = {normalize(p['question']): p['problem_id'] for p in problems}
        assert len(mapping) == len(problems)
        partitions[name+':'+split] = mapping
    prompts = {}
    for name in ['original', 'simple', 'mid', 'complex']:
        path = upstream/'GSM8K/lib_prompt'/f'prompt_{name}.txt'
        tracked = subprocess.check_output(['git', '-C', str(upstream), 'show',
                                          f'{UPSTREAM}:GSM8K/lib_prompt/{path.name}'])
        assert tracked == path.read_bytes()
        text = path.read_text()
        blocks = re.split(r'(?m)^Question: ', text)
        assert blocks[0] == '' and len(blocks[1:]) == 8
        questions = [block.split('\n', 1)[0] for block in blocks[1:]]
        assert len({normalize(q) for q in questions}) == 8
        lines = [block.splitlines()[1:] for block in blocks[1:]]
        assert all(sum(line.startswith("Let's think step by step") for line in group) == 1 for group in lines)
        assert all(sum(line.startswith('The answer is') for line in group) == 1 for group in lines)
        counts = [sum(bool(line.strip()) and not line.startswith("Let's think step by step")
                      and not line.startswith('The answer is') for line in group) for group in lines]
        prompts[name] = {'upstream_path': str(path.relative_to(upstream)), 'sha256': file_hash(path),
                        'bytes': path.stat().st_size, 'demonstrations': 8, 'solution_line_counts': counts,
                        'question_sha256': [hashlib.sha256(normalize(q).encode()).hexdigest() for q in questions],
                        'overlap': {key: [mapping[normalize(q)] for q in questions if normalize(q) in mapping]
                                    for key, mapping in partitions.items()}}
    return {'status': 'inspected_without_inference', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'upstream_repository': 'https://github.com/FranxYao/Complexity-Based-Prompting',
            'upstream_commit': commit, 'script_sha256': file_hash(Path(__file__)),
            'input_files_sha256': inputs, 'partition_sizes': {k: len(v) for k, v in partitions.items()},
            'prompts': prompts, 'generation_rows': 0,
            'limits': 'Counts nonempty solution lines, not semantic reasoning steps. Overlap uses casefolded '
                      'whitespace-normalized exact question text, not paraphrase detection. This is an asset '
                      'inspection, not a model replication or a new experiment declaration.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=Path('research/upstream/complexity-based-prompting'))
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite an inspection')
    result = inspect(args.upstream, args.runs)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'upstream_commit': result['upstream_commit'],
                      'demonstrations_per_bank': 8, 'generation_rows': 0}))

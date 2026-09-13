"""Inventory unused GSM8K questions without selecting a sample or scoring outputs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


REVISION = '740312add88f781978c0658806c59bc2815b9866'
IDENTITY = re.compile(r'gsm8k:(train|test):(\d+)\Z')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def normalize(text):
    return ' '.join(text.casefold().split())


def identities(value, ids, questions, generated):
    """Conservatively include IDs anywhere, and question fields in any nested record."""
    if isinstance(value, dict):
        question = value.get('question')
        if isinstance(question, str):
            questions.add(normalize(question))
        pid = value.get('problem_id')
        if isinstance(pid, str) and IDENTITY.fullmatch(pid) and 'condition' in value and 'text' in value:
            generated.add(pid)
        for key, item in value.items():
            identities(key, ids, questions, generated)
            identities(item, ids, questions, generated)
    elif isinstance(value, list):
        for item in value:
            identities(item, ids, questions, generated)
    elif isinstance(value, str) and IDENTITY.fullmatch(value):
        ids.add(value)


def inventory(runs):
    ids, questions, generated, files = set(), set(), set(), {}
    paths = sorted(p for p in runs.rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl'})
    for path in paths:
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Run file changed during inventory: '+str(path))
        files[str(path.relative_to(runs))] = hashlib.sha256(content).hexdigest()
        if path.suffix == '.jsonl':
            values = (json.loads(line) for line in content.splitlines() if line.strip())
        else:
            values = [json.loads(content)]
        for value in values:
            identities(value, ids, questions, generated)
    return ids, questions, generated, files


def partition_inventory(tables, ids, questions, generated, reservations, prefix=128):
    """Return an inventory only. No sampling seed, target count, or accuracy is read."""
    lookup = {f'gsm8k:{split}:{i}': row for split, rows in tables.items() for i, row in enumerate(rows)}
    unknown = (ids | generated | set().union(*reservations.values())) - lookup.keys()
    if unknown:
        raise ValueError('Observed IDs absent from pinned dataset: '+str(sorted(unknown)))
    used = set(ids) | generated | set().union(*reservations.values())
    excluded_text = set(questions) | {normalize(lookup[i]['question']) for i in used}
    for split, rows in tables.items():
        for i, row in enumerate(rows[:prefix]):
            used.add(f'gsm8k:{split}:{i}')
            excluded_text.add(normalize(row['question']))
    reservation_checks = {}
    for name, reserved in reservations.items():
        overlap = reserved & generated
        if overlap:
            raise ValueError('Reserved questions already generated: '+name+' '+str(sorted(overlap)))
        reservation_checks[name] = {'n': len(reserved), 'generated_overlap': 0}
    pools, summaries = {}, {}
    seen = set(excluded_text)
    for split in ['train', 'test']:
        fresh, counts = [], Counter()
        for i, row in enumerate(tables[split]):
            pid, question = f'gsm8k:{split}:{i}', normalize(row['question'])
            if pid in used or question in excluded_text:
                counts['previous_or_reserved_or_prefix'] += 1
            elif question in seen:
                counts['fresh_text_duplicate'] += 1
            else:
                fresh.append({'problem_id': pid, 'calculation_annotations': len(re.findall(r'<<[^<>]*>>', row['answer']))})
                seen.add(question)
        hist = Counter(r['calculation_annotations'] for r in fresh)
        pools[split] = fresh
        summaries[split] = {'total': len(tables[split]), 'excluded': dict(counts),
            'fresh': len(fresh), 'calculation_annotation_histogram': dict(sorted(hist.items())),
            'fresh_at_least_four_annotations': sum(n for steps, n in hist.items() if steps >= 4),
            'identities_and_counts_sha256': digest(fresh)}
    return {'splits': summaries, 'reservations': reservation_checks,
            'observed_unique_ids': len(ids), 'generated_unique_ids': len(generated),
            'observed_question_texts': len(questions)}, pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite inventory')
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    tables, dataset_hashes = {}, {}
    for split in ['train', 'test']:
        path = Path(hf_hub_download('openai/gsm8k', f'main/{split}-00000-of-00001.parquet',
            repo_type='dataset', revision=REVISION, local_files_only=True))
        tables[split] = pq.read_table(path).to_pylist()
        dataset_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    ids, questions, generated, files = inventory(args.runs)
    reservations = {}
    for name, split, expected in [('gsm8k-conditional-v1', 'test', 512),
                                   ('gsm8k-test-development-v1', 'reserved', 256)]:
        rows = json.loads((args.runs/name/'prepared.json').read_text())['splits'][split]
        reservations[name] = {r['problem_id'] for r in rows}
        if len(rows) != expected or len(reservations[name]) != expected:
            raise ValueError('Unexpected reservation size: '+name)
    result, pools = partition_inventory(tables, ids, questions, generated, reservations)
    # Recheck source bytes after the complete audit, so a concurrent writer invalidates it.
    for name, expected in files.items():
        if hashlib.sha256((args.runs/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed after inventory: '+name)
    current = {str(p.relative_to(args.runs)) for p in args.runs.rglob('*')
               if p.is_file() and p.suffix in {'.json', '.jsonl'}}
    if current != set(files):
        raise ValueError('Run file set changed during audit')
    result.update(created_at=datetime.now(timezone.utc).isoformat(), dataset='openai/gsm8k',
        dataset_revision=REVISION, dataset_sha256=dataset_hashes,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_file_count=len(files), input_inventory_sha256=digest(files),
        run_directories=sorted({p.split('/')[0] for p in files}),
        no_sampling_or_generation=True, first_rows_excluded_per_split=128,
        normalization='casefold and whitespace collapse; deduplicate across both splits',
        scope='All JSON and JSONL artifacts under the audited runs directory; no claim about unrecorded external use')
    args.output.mkdir(parents=True)
    observations = {'ids': sorted(ids), 'question_texts': sorted(questions), 'generated_ids': sorted(generated)}
    for name, value in [('summary.json', result), ('input-files.json', files),
                        ('fresh-pool.json', pools), ('observations.json', observations)]:
        (args.output/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

"""Describe the first-token effect of final-norm prompt-only steering."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def inspect(run):
    source = run/'generations.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    groups = {name: {r['problem_id']: r for r in rows if r['condition'] == name}
              for name in ['zero', 'prefill']}
    assert all(len(g) == 128 for g in groups.values())
    assert groups['zero'].keys() == groups['prefill'].keys()
    assert all(r['split'] == 'validation' and r['token_ids'] for g in groups.values() for r in g.values())
    assert all(groups['zero'][k]['prompt'] == groups['prefill'][k]['prompt'] for k in groups['zero'])
    same = [k for k in groups['zero'] if
            groups['zero'][k]['token_ids'][0] == groups['prefill'][k]['token_ids'][0]]
    return {
        'status': 'post_score_descriptive_check',
        'source_file_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'questions': 128,
        'first_token_ids': {name: dict(Counter(r['token_ids'][0] for r in group.values()))
                            for name, group in groups.items()},
        'first_whitespace_delimited_words': {
            name: dict(Counter(r['solution_text'].lstrip().split()[0] for r in group.values()))
            for name, group in groups.items()},
        'same_first_token': len(same),
        'identical_complete_token_sequences_given_same_first_token': sum(
            groups['zero'][k]['token_ids'] == groups['prefill'][k]['token_ids'] for k in same),
        'interpretation': 'Final-norm prompt-only intervention changes next-token logits after KV construction. '
                          'The changed first token is its only route into subsequent greedy decoding. '
                          'Matching sequences support that mechanism on seven observed questions; '
                          'no forced-token GPU replay was performed. This is not a new accuracy test.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to replace a prior result')
    args.output.write_text(json.dumps(inspect(args.run), indent=2)+'\n')

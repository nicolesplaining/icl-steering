"""Replay new candidate traces using independent reconstructions of the unchanged maps."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis import gsm8k_complex_candidate as run
from analysis import ltv_pairing_audit, ltv_continuation_audit


def reconstruct(output):
    roots = {k: Path(path).parent for k, path in run.base.read(output/'manifest.json')['map_paths'].items()}
    paired, width = ltv_pairing_audit.reconstruct(roots['pairing'])
    continued, other_width = ltv_continuation_audit.reconstruct(roots['continuation'])
    assert width == other_width
    with np.load(roots['original']/'extraction.npz', allow_pickle=False) as packed:
        x = packed['zero'].astype(np.float64)
        delta = packed['icl_a'].astype(np.float64)-x
    matrix = x @ x.T+5*np.eye(len(x))
    weights = np.linalg.solve(matrix, delta)
    residual = np.linalg.norm(matrix @ weights-delta)/max(np.linalg.norm(delta), 1.)
    assert residual <= 1e-8
    old = run.mapping.load_frozen({k: p/'maps.npz' for k, p in roots.items()})['original']
    np.testing.assert_array_equal(old['real/x'], x)
    np.testing.assert_allclose(old['real/weights'], weights, rtol=1e-12, atol=1e-12)
    def predict(h, name, position):
        if name in {'steered', 'permuted_low', 'shared_low', 'real_high', 'permuted', 'shared_high'}:
            return paired(h, name, position)
        if name == 'prefill':
            return (h.astype(np.float64) @ x.T) @ weights
        return continued(h, name, position)
    return predict, width

def check(output, complete=False):
    _, data = run.verify(output, fitted=True)
    predict, width = reconstruct(output)
    valid = {f'{name}-{start:03d}.json': (name, start) for name in run.NEW for start in range(0, 256, 4)}
    files = sorted((output/'batches').glob('*.json'))
    if complete and {p.name for p in files} != set(valid):
        raise ValueError('Incomplete trajectory set')
    vectors_checked, heads = 0, 0
    for path in files:
        if path.name not in valid:
            raise ValueError('Unexpected batch')
        name, start = valid[path.name]; rows = run.base.read(path)
        assert [r['problem_id'] for r in rows] == [p['problem_id'] for p in data['splits']['validation'][start:start+4]]
        assert all(r['condition'] == name and r['split'] == 'validation' for r in rows)
        trace_path = output/'traces'/(path.stem+'.npz')
        trace_hash = run.base.file_hash(trace_path)
        assert all(r['trace_file'] == str(trace_path.relative_to(output)) and r['trace_file_sha256'] == trace_hash for r in rows)
        with np.load(trace_path, allow_pickle=False) as packed:
            states, vectors, applied, positions = [packed[k] for k in
                ['states', 'vectors', 'applied_states', 'prefix_positions']]
        assert states.shape == vectors.shape == applied.shape and states.shape[1:] == (4, width)
        assert np.isfinite(states).all() and np.isfinite(vectors).all() and np.isfinite(applied).all()
        assert positions.tolist() == list(range(len(states))) and len(states) <= 129
        for j, h in enumerate(states):
            delta = predict(h, name, int(positions[j]))
            np.testing.assert_array_equal(vectors[j], delta.astype(np.float32))
            dtype = {'torch.bfloat16': torch.bfloat16, 'torch.float32': torch.float32}[rows[0]['activation_dtype']]
            expected = (torch.from_numpy(h.copy()).to(dtype)+torch.from_numpy(delta.copy()).to(dtype)).float().numpy()
            np.testing.assert_array_equal(applied[j], expected)
        for i, row in enumerate(rows):
            assert row['trace_index'] == i and row['active_calls'] == len(states) == row['head_checks']
            assert len(states) == (1 if name in {'regularized_prefill', 'prefill'} else min(row['hook_calls'], 129))
            assert row['prefix_positions'] == positions.tolist()
            assert row['sequence_lengths'][1:] == [1]*(len(states)-1)
            assert row['vector_chain_sha256'] == hashlib.sha256(vectors[:, i].copy().tobytes()).hexdigest()
        vectors_checked += len(states)*4; heads += rows[0]['head_checks']
    matched = 0
    for start in range(0, 256, 4):
        left = output/f'batches/steered-{start:03d}.json'
        right = output/f'batches/regularized_prefill-{start:03d}.json'
        if not left.exists() or not right.exists():
            continue
        left_rows, right_rows = run.base.read(left), run.base.read(right)
        with np.load(output/f'traces/steered-{start:03d}.npz', allow_pickle=False) as a, np.load(
                output/f'traces/regularized_prefill-{start:03d}.npz', allow_pickle=False) as b:
            for field in ['states', 'vectors', 'applied_states']:
                np.testing.assert_array_equal(a[field][0], b[field][0])
        for a, b in zip(left_rows, right_rows):
            assert a['problem_id'] == b['problem_id'] and a['prompt'] == b['prompt']
            assert a['sequence_lengths'][0] == b['sequence_lengths'][0]
            assert a['token_ids'] and b['token_ids'] and a['token_ids'][0] == b['token_ids'][0]
            matched += 1
    if complete and matched != 256:
        raise ValueError('Incomplete candidate/prompt-only first-token match')
    audited_rows = run.collect(output, complete=complete)
    result = {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'rows_sha256': run.audit.digest(audited_rows),
        'script_sha256': run.base.file_hash(Path(__file__)), 'manifest_sha256': run.base.file_hash(output/'manifest.json'),
        'fit_sha256': run.base.file_hash(output/'fit.json'), 'completed_batches': len(files),
        'new_generation_rows': len(files)*4, 'saved_state_vectors_replayed': vectors_checked,
        'candidate_prefill_first_tokens_matched': matched, 'head_calls_verified': heads, 'complete': complete, 'reserved_rows': 0, 'scores_not_displayed': True}
    run.base.save(output/'trajectory-check.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.output, args.complete), indent=2))

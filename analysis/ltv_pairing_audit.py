"""Replay continuation fits and actual head inputs before answer scoring."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis import gsm8k_ltv_pairing as run


def reconstruct(output):
    with np.load(output/'extraction.npz', allow_pickle=False) as packed:
        x, d, positions = packed['zero'], packed['delta'], packed['positions']
    saved = run.load_maps(output)
    np.testing.assert_array_equal(saved['x'], x)
    means = np.stack([d[positions == j].mean(0) for j in range(5)])
    shared = means[positions]
    shuffled = d.copy(); rng = np.random.default_rng(2390)
    for j in range(5):
        idx = np.flatnonzero(positions == j)
        shuffled[idx] = d[rng.permutation(idx)]
    np.testing.assert_array_equal(saved['shared_targets'], shared)
    np.testing.assert_array_equal(saved['permuted_targets'], shuffled)
    scale = np.square(x).sum()/len(x)
    targets = {'real': d, 'shared': shared, 'permuted': shuffled}
    specs = {'steered': ('real', .1), 'permuted_low': ('permuted', .1), 'shared_low': ('shared', .1),
             'real_high': ('real', 1.), 'permuted': ('permuted', 1.), 'shared_high': ('shared', 1.)}
    weights = {}
    for name, (target, multiplier) in specs.items():
        penalty = multiplier*scale
        system = x @ x.T+penalty*np.eye(len(x))
        weights[name] = np.linalg.solve(system, targets[target])
        residual = np.linalg.norm(system @ weights[name]-targets[target])/max(np.linalg.norm(targets[target]), 1.)
        assert residual <= 1e-8
        np.testing.assert_array_equal(saved[name+'/weights'], weights[name])
        np.testing.assert_array_equal(saved[name+'/penalty'], penalty)
    for real, average, multiplier in [('steered', 'shared_low', .1), ('real_high', 'shared_high', 1.)]:
        system = x @ x.T+multiplier*scale*np.eye(len(x))
        residual_map = np.linalg.solve(system, d-shared)
        np.testing.assert_allclose(weights[real], weights[average]+residual_map, rtol=1e-10, atol=1e-11)
        # A distinct five-column solve verifies the shared map's factorization.
        factors = np.linalg.solve(system, np.eye(5)[positions])
        np.testing.assert_allclose(weights[average], factors @ means, rtol=1e-10, atol=1e-11)
    def predict(h, name, position):
        return (h.astype(np.float64) @ x.T) @ weights[name]
    return predict, x.shape[1]


def check(output, complete=False):
    _, data = run.verify(output, fitted=True)
    predict, width = reconstruct(output)
    valid = {f'{name}-{start:03d}.json': (name, start) for name in run.NEW for start in range(0, 128, 4)}
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
            assert len(states) == min(row['hook_calls'], 129)
            assert row['prefix_positions'] == positions.tolist()
            assert row['sequence_lengths'][1:] == [1]*(len(states)-1)
            assert row['vector_chain_sha256'] == hashlib.sha256(vectors[:, i].copy().tobytes()).hexdigest()
        vectors_checked += len(states)*4; heads += rows[0]['head_checks']
    result = {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': run.base.file_hash(Path(__file__)), 'manifest_sha256': run.base.file_hash(output/'manifest.json'),
        'fit_sha256': run.base.file_hash(output/'fit.json'), 'completed_batches': len(files),
        'new_generation_rows': len(files)*4, 'saved_state_vectors_replayed': vectors_checked,
        'head_calls_verified': heads, 'complete': complete, 'reserved_rows': 0, 'scores_not_displayed': True}
    run.base.save(output/'trajectory-check.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.output, args.complete), indent=2))

"""Replay saved LTV fits and trajectories without displaying answers or scores."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import torch

from analysis import gsm8k_ltv as run


def check(output, complete=False):
    config, data = run.verify(output, fitted=True)
    real, permuted = run.load_maps(output)
    with np.load(output/'extraction.npz', allow_pickle=False) as packed:
        x = packed['zero'].astype(np.float64)
        d = packed['icl_a'].astype(np.float64)-x
    permutation = np.random.default_rng(1901).permutation(len(x))
    system = x @ x.T + 5*np.eye(len(x))
    weights = np.linalg.solve(system, d)
    shuffled = np.linalg.solve(system, d[permutation])
    for fitted, expected in [(real, weights), (permuted, shuffled)]:
        np.testing.assert_array_equal(fitted['x'], x)
        np.testing.assert_allclose(fitted['weights'], expected, rtol=1e-12, atol=1e-12)
    xm, mean = x.mean(0), d.mean(0)
    scalar = ((x-xm)*(d-mean)).sum()/np.square(x-xm).sum()
    np.testing.assert_array_equal(real['x_mean'], xm)
    np.testing.assert_array_equal(real['mean'], mean)
    np.testing.assert_array_equal(real['scalar'], scalar)
    valid = {f'{name}-{start:03d}.json': (name, start) for name in run.NEW for start in range(0, 128, 4)}
    files = sorted((output/'batches').glob('*.json'))
    if complete and {p.name for p in files} != set(valid):
        raise ValueError('Incomplete trajectory set')
    steps, head_checks = 0, 0
    for path in files:
        if path.name not in valid:
            raise ValueError('Unexpected batch')
        name, start = valid[path.name]; rows = run.base.read(path)
        assert [r['problem_id'] for r in rows] == [p['problem_id'] for p in data['splits']['validation'][start:start+4]]
        assert all(r['condition'] == name and r['split'] == 'validation' for r in rows)
        trace_path = output/'traces'/(path.stem+'.npz')
        trace_hash = run.base.file_hash(trace_path)
        assert all(r['trace_file'] == str(trace_path.relative_to(output)) and
                   r['trace_file_sha256'] == trace_hash for r in rows)
        with np.load(trace_path, allow_pickle=False) as packed:
            states, vectors, applied = [packed[k] for k in ['states', 'vectors', 'applied_states']]
        assert states.shape == vectors.shape == applied.shape
        assert states.shape[1:] == (4, x.shape[1])
        assert np.isfinite(states).all() and np.isfinite(vectors).all() and np.isfinite(applied).all()
        for t, h in enumerate(states):
            h64 = h.astype(np.float64); ridge = (h64 @ x.T) @ weights
            if name in ['steered', 'prefill']: delta = ridge
            elif name == 'mean': delta = np.broadcast_to(mean, h64.shape).copy()
            elif name in ['scalar', 'scalar_norm']: delta = mean + scalar*(h64-xm)
            elif name == 'permuted': delta = (h64 @ x.T) @ shuffled
            if name in ['scalar_norm', 'permuted']:
                delta = delta/np.linalg.norm(delta, axis=1, keepdims=True)*np.linalg.norm(ridge, axis=1, keepdims=True)
            np.testing.assert_array_equal(vectors[t], delta.astype(np.float32))
            dtype = {'torch.bfloat16': torch.bfloat16, 'torch.float32': torch.float32}[rows[0]['activation_dtype']]
            expected = (torch.from_numpy(h.copy()).to(dtype) + torch.from_numpy(delta.copy()).to(dtype)).float().numpy()
            np.testing.assert_array_equal(applied[t], expected)
        for i, row in enumerate(rows):
            assert row['trace_index'] == i and row['active_calls'] == len(states) == row['head_checks']
            assert len(states) == (1 if name == 'prefill' else row['hook_calls'])
            assert row['sequence_lengths'][1:] == [1]*(len(states)-1)
            assert row['vector_chain_sha256'] == hashlib.sha256(vectors[:, i].copy().tobytes()).hexdigest()
        steps += len(states)*4; head_checks += rows[0]['head_checks']
    result = {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': run.base.file_hash(Path(__file__)), 'manifest_sha256': run.base.file_hash(output/'manifest.json'),
        'fit_sha256': run.base.file_hash(output/'fit.json'), 'completed_batches': len(files),
        'new_generation_rows': len(files)*4, 'saved_state_vectors_replayed': steps,
        'head_calls_verified': head_checks, 'complete': complete, 'reserved_rows': 0,
        'scores_not_displayed': True, 'note': 'Vector count includes inactive rows padded within a running batch.'}
    run.base.save(output/'trajectory-check.json', result)
    return result


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--complete', action='store_true')
    a = p.parse_args()
    print(json.dumps(check(a.output, a.complete), indent=2))

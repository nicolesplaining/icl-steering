"""Replay continuation fits and actual head inputs before answer scoring."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analysis import gsm8k_ltv_continuation as run


def reconstruct(output):
    with np.load(output/'extraction.npz', allow_pickle=False) as packed:
        x, d, position = packed['zero'], packed['delta'], packed['positions']
    saved = run.load_maps(output)
    scale = np.square(x).sum()/len(x)
    shuffled = d.copy(); rng = np.random.default_rng(2390)
    for j in range(5):
        idx = np.flatnonzero(position == j)
        shuffled[idx] = d[rng.permutation(idx)]
    weights = {}
    for name, target, penalty in [('real', d, .1*scale), ('permuted', shuffled, scale)]:
        matrix = x @ x.T+penalty*np.eye(len(x))
        weights[name] = np.linalg.solve(matrix, target)
        residual = np.linalg.norm(matrix @ weights[name]-target)/max(np.linalg.norm(target), 1)
        if residual > 1e-8:
            raise ValueError('Independent solve failed')
        np.testing.assert_array_equal(saved[name+'/x'], x)
        np.testing.assert_allclose(saved[name+'/weights'], weights[name], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(saved[name+'/penalty'], penalty, rtol=1e-12, atol=1e-12)
    def scalar(xx, dd):
        xm, dm = xx.mean(0), dd.mean(0)
        energy = np.square(xx-xm).sum()
        c = float(np.sum((xx-xm)*(dd-dm))/energy) if energy > 1e-12 else 0.
        return xm, dm, c
    xm, dm, c = scalar(x, d)
    np.testing.assert_array_equal(saved['real/x_mean'], xm)
    np.testing.assert_array_equal(saved['real/mean'], dm)
    np.testing.assert_array_equal(saved['real/scalar'], c)
    per_position = [scalar(x[position == j], d[position == j]) for j in range(5)]
    means = np.stack([v[1] for v in per_position])
    coefficients = np.array([v[2] for v in per_position])
    biases = np.stack([m-cc*xx for xx, m, cc in per_position])
    for name, expected in [('position_mean', means), ('position_scalar', coefficients), ('position_bias', biases)]:
        np.testing.assert_array_equal(saved[name], expected)
    np.testing.assert_array_equal(saved['position_counts'], np.bincount(position, minlength=5))
    anchors = [0, 1, 8, 32, 128]
    def interpolate(v, t):
        if t in anchors:
            return v[anchors.index(t)]
        lo = max(i for i, p in enumerate(anchors) if p < t); hi = lo+1
        weight = (t-anchors[lo])/(anchors[hi]-anchors[lo])
        return (1-weight)*v[lo]+weight*v[hi]
    def predict(h, name, t):
        h = h.astype(np.float64)
        if name in ['steered', 'regularized_prefill']:
            return (h @ x.T) @ weights['real']
        if name == 'permuted':
            return (h @ x.T) @ weights['permuted']
        if name == 'mean':
            return np.broadcast_to(dm, h.shape).copy()
        if name == 'scalar':
            return dm+c*(h-xm)
        if name == 'position_mean':
            return np.broadcast_to(interpolate(means, t), h.shape).copy()
        if name == 'position_scalar':
            return interpolate(biases, t)+interpolate(coefficients, t)*h
        raise ValueError('Unexpected predictor')
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
            assert len(states) == (1 if name == 'regularized_prefill' else min(row['hook_calls'], 129))
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

"""Separate position-average and individually paired targets at fixed penalties."""

import numpy as np

from analysis.ltv_mapping import fit
from analysis.ltv_prefix_crossfit import permute_targets
from analysis.ltv_continuation_mapping import normalized_hook

SPECS = {
    'steered': ('real', .1), 'permuted_low': ('permuted', .1), 'shared_low': ('shared', .1),
    'real_high': ('real', 1.), 'permuted': ('permuted', 1.), 'shared_high': ('shared', 1.)}


def targets(x, delta, positions):
    x, delta, positions = np.asarray(x, dtype=np.float64), np.asarray(delta, dtype=np.float64), np.asarray(positions)
    if (x.ndim != 2 or x.shape != delta.shape or positions.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(delta).all()
            or set(positions) != set(range(5)) or not np.issubdtype(positions.dtype, np.integer)):
        raise ValueError('Invalid paired extraction states')
    if any(np.sum(positions == j) < 2 for j in range(5)):
        raise ValueError('Insufficient position coverage')
    means = np.stack([delta[positions == j].mean(0) for j in range(5)])
    return {'real': delta, 'shared': means[positions],
            'permuted': permute_targets(delta, positions, 2390)}


def fit_states(x, delta, positions):
    x = np.asarray(x, dtype=np.float64)
    y = targets(x, delta, positions)
    scale = np.square(x).sum()/len(x)
    fitted = {'x': x, 'shared_targets': y['shared'], 'permuted_targets': y['permuted']}
    for name, (target, multiplier) in SPECS.items():
        result = fit(x, y[target], multiplier*scale)
        for key in ['weights', 'penalty', 'solver_residual']:
            fitted[name+'/'+key] = result[key]
    return fitted


def predict(fitted, h, kind, position):
    h = np.asarray(h, dtype=np.float64)
    if (kind not in SPECS or h.ndim != 2 or h.shape[1] != fitted['x'].shape[1]
            or not np.isfinite(h).all() or type(position) is not int or not 0 <= position <= 128):
        raise ValueError('Invalid predictor inputs')
    result = (h @ fitted['x'].T) @ fitted[kind+'/weights']
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite predicted shift')
    return result

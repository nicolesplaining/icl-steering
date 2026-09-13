"""Regularized maps and position controls within the measured prefix range."""

from contextlib import contextmanager

import numpy as np
import torch

from analysis import ltv_mapping as original
from analysis.ltv_prefix_crossfit import permute_targets


POSITIONS = np.array([0, 1, 8, 32, 128])


def fit_states(zero, delta, positions):
    zero, delta = np.asarray(zero, dtype=np.float64), np.asarray(delta, dtype=np.float64)
    if zero.ndim != 2 or zero.shape != delta.shape or positions.shape != (len(zero),):
        raise ValueError('Invalid continuation extraction')
    if set(positions) != set(range(5)):
        raise ValueError('Missing or unexpected extraction position')
    scale = float(np.square(zero).sum()/len(zero))
    real = original.fit(zero, delta, .1*scale)
    permuted = original.fit(zero, permute_targets(delta, positions, 2390), scale)
    fitted = {f'{name}/{k}': v for name, fit in [('real', real), ('permuted', permuted)] for k, v in fit.items()}
    means, biases, coefficients, counts = [], [], [], []
    for j in range(5):
        x, d = zero[positions == j], delta[positions == j]
        if len(x) < 2:
            raise ValueError('Insufficient position-specific training pairs')
        xm, dm = x.mean(0), d.mean(0)
        energy = np.square(x-xm).sum()
        c = float(np.sum((x-xm)*(d-dm))/energy) if energy > 1e-12 else 0.
        means.append(dm); biases.append(dm-c*xm); coefficients.append(c); counts.append(len(x))
    fitted.update(position_mean=np.stack(means), position_bias=np.stack(biases),
                  position_scalar=np.asarray(coefficients), position_counts=np.asarray(counts))
    return fitted


def interpolate(values, position):
    if type(position) is not int or not 0 <= position <= 128:
        raise ValueError('Position outside declared intervention range')
    upper = int(np.searchsorted(POSITIONS, position, side='left'))
    if POSITIONS[upper] == position:
        return values[upper]
    lower = upper-1
    weight = (position-POSITIONS[lower])/(POSITIONS[upper]-POSITIONS[lower])
    return (1-weight)*values[lower]+weight*values[upper]


def predict(fitted, h, kind, position):
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 2 or not np.isfinite(h).all() or type(position) is not int or not 0 <= position <= 128:
        raise ValueError('Invalid current activation or prefix position')
    if kind in {'ridge', 'permuted'}:
        prefix = 'real' if kind == 'ridge' else 'permuted'
        delta = (h @ fitted[prefix+'/x'].T) @ fitted[prefix+'/weights']
    elif kind == 'mean':
        delta = np.broadcast_to(fitted['real/mean'], h.shape).copy()
    elif kind == 'scalar':
        delta = fitted['real/mean']+fitted['real/scalar']*(h-fitted['real/x_mean'])
    elif kind == 'position_mean':
        delta = np.broadcast_to(interpolate(fitted['position_mean'], position), h.shape).copy()
    elif kind == 'position_scalar':
        delta = interpolate(fitted['position_bias'], position)+interpolate(fitted['position_scalar'], position)*h
    else:
        raise ValueError('Unknown continuation predictor')
    if delta.shape != h.shape or not np.isfinite(delta).all():
        raise ValueError('Invalid continuation prediction')
    return delta


@contextmanager
def normalized_hook(norm, predictor, scope, head):
    if scope not in {'prefix_0_128', 'prefill'}:
        raise ValueError('Unknown intervention scope')
    limit = 0 if scope == 'prefill' else 128
    trace = {'states': [], 'vectors': [], 'applied_states': [], 'sequence_lengths': [],
             'prefix_positions': [], 'calls': 0, 'head_checks': 0}
    pending = None

    def hook(module, args, output):
        nonlocal pending
        position = trace['calls']; trace['calls'] += 1
        pending = None
        if position > limit:
            return output
        state = output[:, -1, :].detach().float().cpu().numpy().copy()
        delta = np.asarray(predictor(state, position), dtype=np.float64)
        if delta.shape != state.shape or not np.isfinite(delta).all():
            raise ValueError('Invalid proposed shift')
        changed = output.clone()
        changed[:, -1, :] += torch.as_tensor(delta, dtype=output.dtype, device=output.device)
        pending = changed[:, -1, :].detach().clone()
        trace['states'].append(state); trace['vectors'].append(delta.astype(np.float32))
        trace['applied_states'].append(pending.float().cpu().numpy().copy())
        trace['prefix_positions'].append(position); trace['sequence_lengths'].append(output.shape[1])
        trace['dtype'] = str(output.dtype)
        return changed

    def check_head(module, args):
        nonlocal pending
        if pending is not None:
            if not torch.equal(args[0][:, -1, :], pending):
                raise ValueError('LM-head input differs from applied state')
            trace['head_checks'] += 1; pending = None

    handle, head_handle = norm.register_forward_hook(hook), head.register_forward_pre_hook(check_head)
    try:
        yield trace
        if trace['head_checks'] != len(trace['states']):
            raise ValueError('Missing LM-head checks')
    finally:
        handle.remove(); head_handle.remove()

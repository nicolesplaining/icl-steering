"""Uncentered ridge for ICL shifts at the final normalized state."""

from contextlib import contextmanager

import numpy as np
import torch


def fit(x, delta, penalty=5.0):
    x, delta = np.asarray(x, dtype=np.float64), np.asarray(delta, dtype=np.float64)
    if x.ndim != 2 or x.shape != delta.shape or len(x) < 2:
        raise ValueError('Require paired extraction states')
    if not np.isfinite(x).all() or not np.isfinite(delta).all() or penalty <= 0:
        raise ValueError('Invalid fit inputs')
    system = x @ x.T + penalty * np.eye(len(x))
    weights = np.linalg.solve(system, delta)
    residual = np.linalg.norm(system @ weights - delta) / max(np.linalg.norm(delta), 1.)
    if not np.isfinite(residual) or residual > 1e-8:
        raise ValueError('Ridge solve failed')
    xm, mean = x.mean(0), delta.mean(0)
    xc, dc = x-xm, delta-mean
    energy = np.square(xc).sum()
    scalar = (xc*dc).sum()/energy if energy > 1e-12 else 0.
    return {'x': x, 'weights': weights, 'mean': mean, 'x_mean': xm,
            'scalar': np.asarray(scalar), 'penalty': np.asarray(penalty),
            'solver_residual': np.asarray(residual)}


def predict(fitted, x, kind='ltv', permuted=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1:] != fitted['mean'].shape or not np.isfinite(x).all():
        raise ValueError('Invalid query states')
    if kind == 'mean':
        delta = np.broadcast_to(fitted['mean'], x.shape).copy()
    elif kind in {'scalar', 'scalar_norm'}:
        delta = fitted['mean'] + fitted['scalar']*(x-fitted['x_mean'])
    elif kind == 'permuted':
        if permuted is None:
            raise ValueError('Missing permuted map')
        delta = (x @ permuted['x'].T) @ permuted['weights']
    elif kind == 'ltv':
        delta = (x @ fitted['x'].T) @ fitted['weights']
    else:
        raise ValueError('Unknown intervention')
    if kind in {'scalar_norm', 'permuted'}:
        reference = predict(fitted, x)
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        if (norm <= 1e-12).any():
            raise ValueError('Cannot norm-match zero control')
        delta = delta/norm*np.linalg.norm(reference, axis=1, keepdims=True)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite intervention')
    return delta


@contextmanager
def normalized_hook(norm, predictor, scope='all', head=None):
    """Recompute from each unmodified norm output; retain states for replay."""
    if scope not in {'all', 'prefill'}:
        raise ValueError('Unknown intervention scope')
    trace = {'states': [], 'vectors': [], 'applied_states': [], 'sequence_lengths': [],
             'calls': 0, 'head_checks': 0}
    pending = None

    def hook(module, inputs, output):
        nonlocal pending
        pending = None
        first = trace['calls'] == 0
        trace['calls'] += 1
        if scope == 'prefill' and not first:
            return output
        x = output[:, -1, :].detach().float().cpu().numpy().copy()
        delta = np.asarray(predictor(x), dtype=np.float64)
        if delta.shape != x.shape or not np.isfinite(delta).all():
            raise ValueError('Invalid predicted shift')
        trace['states'].append(x)
        trace['vectors'].append(delta.astype(np.float32))
        trace['sequence_lengths'].append(output.shape[1])
        changed = output.clone()
        changed[:, -1, :] += torch.as_tensor(delta, device=output.device, dtype=output.dtype)
        pending = changed[:, -1, :].detach().clone()
        trace['applied_states'].append(pending.float().cpu().numpy().copy())
        trace['dtype'] = str(output.dtype)
        return changed

    def check_head(module, inputs):
        nonlocal pending
        if pending is not None:
            if not torch.equal(inputs[0][:, -1, :], pending):
                raise ValueError('LM head did not receive the declared intervention')
            trace['head_checks'] += 1
            pending = None

    handle = norm.register_forward_hook(hook)
    head_handle = head.register_forward_pre_hook(check_head) if head is not None else None
    try:
        yield trace
        if head is not None and trace['head_checks'] != len(trace['states']):
            raise ValueError('Missing LM-head intervention check')
    finally:
        handle.remove()
        if head_handle is not None:
            head_handle.remove()

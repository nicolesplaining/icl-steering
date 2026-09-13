"""Describe a fixed early trajectory cohort without reading answer scores."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


STEPS = [0, 1, 8, 32, 128]
N_QUESTIONS = 32


def descriptors(states, shifts, basis):
    h, d = np.asarray(states, dtype=np.float64), np.asarray(shifts, dtype=np.float64)
    if h.ndim != 2 or h.shape != d.shape or basis.ndim != 2 or basis.shape[1] != h.shape[1]:
        raise ValueError('Incompatible states, shifts, and basis')
    if not all(np.isfinite(a).all() for a in [h, d, basis]):
        raise ValueError('Nonfinite geometry input')
    hn, dn = np.linalg.norm(h, axis=1), np.linalg.norm(d, axis=1)
    if (hn <= 1e-12).any() or (dn <= 1e-12).any():
        raise ValueError('Undefined geometry for a zero state or shift')
    residual = h-(h @ basis.T) @ basis
    return {'state_norm': hn, 'shift_norm': dn, 'shift_to_state_norm': dn/hn,
        'shift_state_cosine': np.sum(h*d, axis=1)/(hn*dn),
        'outside_extraction_span_energy': np.square(residual).sum(1)/np.square(hn)}


def summarize(values):
    return {k: {'median': float(np.median(v)), 'p10': float(np.quantile(v, .1)),
                'p90': float(np.quantile(v, .9))} for k, v in values.items()}


def analyze(directory):
    from analysis import gsm8k_ltv as run
    _, data = run.verify(directory, fitted=True)
    with np.load(directory/'extraction.npz', allow_pickle=False) as packed:
        training = packed['zero'].astype(np.float64)
    _, singular, vt = np.linalg.svd(training, full_matrices=False)
    rank = int((singular > singular[0]*max(training.shape)*np.finfo(np.float64).eps).sum())
    basis = vt[:rank]
    buckets = {t: {'h': [], 'd': [], 'h0': [], 'd0': []} for t in STEPS}
    sources = {}
    for start in range(0, N_QUESTIONS, 4):
        record_path = directory/f'batches/steered-{start:03d}.json'
        trace_path = directory/f'traces/steered-{start:03d}.npz'
        rows = run.base.read(record_path)
        assert [r['problem_id'] for r in rows] == [p['problem_id'] for p in data['splits']['validation'][start:start+4]]
        trace_hash = run.base.file_hash(trace_path)
        assert len(rows) == 4 and all(r['trace_file_sha256'] == trace_hash for r in rows)
        assert all(r['condition'] == 'steered' and r['split'] == 'validation' for r in rows)
        sources[record_path.name] = {'records_sha256': run.base.file_hash(record_path),
                                    'trace_sha256': trace_hash}
        with np.load(trace_path, allow_pickle=False) as packed:
            states, shifts = packed['states'], packed['vectors']
        for i, row in enumerate(rows):
            for t in STEPS:
                # Exclude the final recorded token decision, which can be EOS
                # or the first padding token after a custom stop. No grades,
                # finish reasons, output text, or reference answers are used.
                if t >= len(states) or (t > 0 and t >= row['generated_tokens']-1):
                    continue
                for key, value in [('h', states[t, i]), ('d', shifts[t, i]),
                                   ('h0', states[0, i]), ('d0', shifts[0, i])]:
                    buckets[t][key].append(value)
    result = {}
    for step, values in buckets.items():
        if not values['h']:
            result[str(step)] = {'n': 0}
            continue
        current = descriptors(values['h'], values['d'], basis)
        reference = descriptors(values['h0'], values['d0'], basis)
        result[str(step)] = {'n': len(values['h']), 'states': summarize(current),
            'paired_change_from_own_prefill': summarize({k: v-reference[k] for k, v in current.items()})}
    return {'status': 'exploratory_state_shift_snapshot', 'created_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': run.base.file_hash(Path(__file__)),
        'manifest_sha256': run.base.file_hash(directory/'manifest.json'),
        'fit_sha256': run.base.file_hash(directory/'fit.json'), 'cohort': 'First 32 development questions in their fixed original order, steered condition only.',
        'steps': STEPS, 'training_span_rank': rank, 'hidden_width': training.shape[1],
        'source_files': sources, 'geometry': result, 'scores_not_read': True,
        'limitations': 'Descriptive snapshot, not an accuracy result or a selection rule. '
            'States follow prefixes generated under steering, so this cannot separate ordinary '
            'decoding distribution shift from effects of the steered prefix. Later steps omit '
            'questions that have ended; changes compare each retained state with its own prefill. '
            'A large span residual or shift norm does not itself establish harmful behavior.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.run)
    with a.report.open('x') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_files'}, indent=2))

"""Independently replay frozen predictions and identical-prefix alignment metrics."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from analysis import ltv_prefix_collect as collector
from analysis import ltv_prefix_metrics as metrics

ltv, base = collector.ltv, collector.base


def reconstructed_predictions(parent, states):
    real, permuted = ltv.load_maps(parent)
    with np.load(parent/'extraction.npz', allow_pickle=False) as packed:
        x = packed['zero'].astype(np.float64)
        delta = packed['icl_a'].astype(np.float64)-x
    order = np.random.default_rng(1901).permutation(len(x))
    system = x @ x.T + 5*np.eye(len(x))
    weights = np.linalg.solve(system, delta)
    shuffled = np.linalg.solve(system, delta[order])
    for fitted, target, solution in [(real, delta, weights), (permuted, delta[order], shuffled)]:
        np.testing.assert_array_equal(fitted['x'], x)
        np.testing.assert_allclose(fitted['weights'], solution, rtol=1e-12, atol=1e-12)
        if np.linalg.norm(system @ solution-target)/max(np.linalg.norm(target), 1) > 1e-8:
            raise ValueError('Independent frozen-fit reconstruction failed')
    mean, xm = delta.mean(0), x.mean(0)
    scalar = np.sum((x-xm)*(delta-mean))/np.square(x-xm).sum()
    np.testing.assert_array_equal(real['mean'], mean)
    np.testing.assert_array_equal(real['x_mean'], xm)
    np.testing.assert_array_equal(real['scalar'], scalar)
    h = states.reshape(-1, states.shape[-1]).astype(np.float64)
    ridge = (h @ x.T) @ weights
    affine = mean + scalar*(h-xm)
    random_pairs = (h @ x.T) @ shuffled
    def match(v):
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        if (norm <= 1e-12).any():
            raise ValueError('Undefined norm-matched control')
        return v/norm*np.linalg.norm(ridge, axis=1, keepdims=True)
    replay = {'ltv': ridge, 'mean': np.broadcast_to(mean, h.shape), 'scalar': affine,
              'scalar_norm': match(affine), 'permuted': match(random_pairs)}
    predictions = {}
    for name in metrics.PREDICTORS:
        value = ltv.mapping.predict(real, h, name, permuted)
        np.testing.assert_allclose(value, replay[name], rtol=1e-12, atol=1e-12)
        predictions[name] = value.reshape(states.shape)
    return predictions


def replay_summary(summary, zero, icl, predictions, available):
    """Recount per question using dot products and explicit quantile interpolation."""
    def measurement(i, j, prediction):
        target = icl[i, j].astype(np.float64)-zero[i, j].astype(np.float64)
        guess = prediction[i, j].astype(np.float64)
        tn, pn = math.sqrt(float(target @ target)), math.sqrt(float(guess @ guess))
        return {'target_norm': tn, 'predicted_norm': pn,
            'predicted_to_target_norm': pn/tn if tn else None,
            'target_cosine': float(guess @ target)/(pn*tn) if pn and tn else None}
    def stats(values):
        finite = sorted(v for v in values if v is not None)
        def at(q):
            if not finite: return None
            offset = (len(finite)-1)*q
            low, high = math.floor(offset), math.ceil(offset)
            return finite[low]+(finite[high]-finite[low])*(offset-low)
        return {'n': len(finite), 'undefined': len(values)-len(finite),
                'p10': at(.1), 'median': at(.5), 'p90': at(.9)}
    def error(indices, j, prediction):
        errors, energies = [], []
        for i in indices:
            target = icl[i, j].astype(np.float64)-zero[i, j].astype(np.float64)
            residual = prediction[i, j]-target
            errors.append(float(residual @ residual)); energies.append(float(target @ target))
        denominator = math.fsum(energies)
        return math.fsum(errors)/denominator if denominator else None
    expected = {'n_questions': len(zero), 'positions': []}
    for j, position in enumerate(metrics.POSITIONS):
        indices = [i for i in range(len(zero)) if available[i, j]]
        point = {'prefix_tokens': position, 'n_questions': len(indices),
            'excluded_short_prefix': len(zero)-len(indices), 'predictors': {}}
        for name, prediction in predictions.items():
            current = [measurement(i, j, prediction) for i in indices]
            prompt = [measurement(i, 0, prediction) for i in indices]
            keys = ['target_norm', 'predicted_norm', 'predicted_to_target_norm', 'target_cosine']
            point['predictors'][name] = {
                'zero_target_norm': sum(x['target_norm'] == 0 for x in current),
                'zero_predicted_norm': sum(x['predicted_norm'] == 0 for x in current),
                'measurements': {k: stats([x[k] for x in current]) for k in keys},
                'paired_change_from_prompt': {k: stats([
                    a[k]-b[k] if a[k] is not None and b[k] is not None else None
                    for a, b in zip(current, prompt)]) for k in keys},
                'normalized_squared_error': error(indices, j, prediction),
                'prompt_normalized_squared_error_same_questions': error(indices, 0, prediction)}
        expected['positions'].append(point)
    compare(summary, expected)


def compare(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Replayed report fields differ')
        for k, v in expected.items(): compare(actual[k], v)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Replayed report lengths differ')
        for a, e in zip(actual, expected): compare(a, e)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    elif actual != expected:
        raise ValueError('Replayed report value differs')


def build(output):
    manifest, questions = collector.verify(output)
    completion = base.read(output/'collection-complete.json')
    expected = {f'{i:03d}.json' for i in range(0, 128, 4)}
    paths = sorted((output/'batches').glob('*.json'))
    if (len(questions) != 128 or completion['n_extract'] != 128
            or completion['manifest_sha256'] != base.file_hash(output/'manifest.json')
            or {p.name for p in paths} != expected
            or completion['batch_files_sha256'] != {p.name: base.file_hash(p) for p in paths}):
        raise ValueError('Incomplete or changed diagnostic collection')
    parent = Path(manifest['parent'])
    with np.load(parent/'extraction.npz', allow_pickle=False) as packed:
        reference = {k: packed[k] for k in ['zero', 'icl_a']}
    chunks = []
    for start, path in zip(range(0, 128, 4), paths):
        chunks.append(collector.check_batch(path, questions[start:start+4],
            {k: v[start:start+4] for k, v in reference.items()}))
    arrays = {k: np.concatenate([v[k] for v in chunks]) for k in ['zero', 'icl_a', 'available']}
    predictions = reconstructed_predictions(parent, arrays['zero'])
    summary = metrics.summarize(arrays['zero'], arrays['icl_a'], predictions, arrays['available'])
    replay_summary(summary, arrays['zero'], arrays['icl_a'], predictions, arrays['available'])
    result = {'status': 'verified_activation_diagnostic', 'summary': summary,
        'manifest_sha256': base.file_hash(output/'manifest.json'),
        'collection_sha256': base.file_hash(output/'collection-complete.json'),
        'parent_fit_sha256': base.file_hash(parent/'fit.json'),
        'code_sha256': collector.code_hash(), 'n_extraction_questions': 128,
        'n_development_questions': 0, 'n_reserved_questions': 0,
        'independent_prediction_replay': True, 'independent_metric_replay': True,
        'limitations': 'Prompt states are in-sample for the frozen fit. Continuation positions use the same extraction questions and zero-shot prefixes, which need not be correct or typical of ICL. This measures activation alignment, not accuracy or causal steering benefit.'}
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['report', 'replay'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output)
    path = args.output/'alignment-report.json'
    if args.stage == 'replay':
        saved = base.read(path)
        saved.pop('created_at')
        compare(saved, result)
        print('Saved alignment report independently replayed.')
    else:
        base.fixed(path, {**result, 'created_at': datetime.now(timezone.utc).isoformat()})
        print('Alignment report saved after independent prediction and metric replay.')


if __name__ == '__main__':
    main()

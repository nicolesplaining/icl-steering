"""Question-disjoint evaluation of a continuation-trained ICL activation map."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


POSITIONS = (0, 1, 8, 32, 128)
NAMES = ('ridge', 'zero', 'prompt_ridge', 'mean', 'scalar',
         'position_mean', 'position_scalar', 'permuted')
ROOT = Path(__file__).parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed(path, value):
    if path.exists():
        raise ValueError('Refuse to overwrite '+str(path))
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def folds(n=128):
    if n != 128:
        raise ValueError('Require 128 question groups')
    order = np.random.default_rng(2713).permutation(n)
    result = np.empty(n, dtype=int)
    for fold, indices in enumerate(order.reshape(4, 32)):
        result[indices] = fold
    return result


def ridge(x, delta):
    x, delta = np.asarray(x, dtype=np.float64), np.asarray(delta, dtype=np.float64)
    if x.ndim != 2 or x.shape != delta.shape or not len(x):
        raise ValueError('Invalid paired training arrays')
    if not np.isfinite(x).all() or not np.isfinite(delta).all():
        raise ValueError('Nonfinite training data')
    system = x @ x.T + 5*np.eye(len(x))
    weights = np.linalg.solve(system, delta)
    residual = float(np.linalg.norm(system @ weights-delta)/max(np.linalg.norm(delta), 1))
    if not math.isfinite(residual) or residual > 1e-8:
        raise ValueError('Ridge solve failed')
    eigenvalues, vectors = np.linalg.eigh(system)
    if eigenvalues.min() <= 0:
        raise ValueError('Nonpositive ridge system')
    independent = vectors @ ((vectors.T @ delta)/eigenvalues[:, None])
    np.testing.assert_allclose(weights, independent, rtol=1e-7, atol=1e-8)
    return {'x': x, 'weights': weights, 'independent': independent, 'residual': residual}


def predict_ridge(fit, query):
    result = (query @ fit['x'].T) @ fit['weights']
    alternate = np.einsum('ij,kj->ik', query, fit['x']) @ fit['independent']
    np.testing.assert_allclose(result, alternate, rtol=1e-7, atol=1e-7)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite prediction')
    return result


def affine(x, delta, query):
    xm, dm = x.mean(0), delta.mean(0)
    xc, dc = x-xm, delta-dm
    energy = np.square(xc).sum()
    c = float(np.sum(xc*dc)/energy) if energy > 1e-12 else 0.
    return dm+c*(query-xm)


def permute_targets(target, positions, seed):
    shuffled = target.copy()
    rng = np.random.default_rng(seed)
    for position in range(5):
        indices = np.flatnonzero(positions == position)
        shuffled[indices] = target[rng.permutation(indices)]
    return shuffled


def fold_predictions(zero, target, available, assignment, fold):
    train, test = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
    if len(train) != 96 or len(test) != 32 or set(train) & set(test):
        raise ValueError('Invalid question separation')
    keep = available[train]
    x, delta = zero[train][keep], target[train][keep]
    position_rows = np.broadcast_to(np.arange(5), keep.shape)[keep]
    query = zero[test].reshape(-1, zero.shape[-1])
    primary = ridge(x, delta)
    prompt = ridge(zero[train, 0], target[train, 0])
    permuted = ridge(x, permute_targets(delta, position_rows, 1901+fold))
    predictions = {'ridge': predict_ridge(primary, query), 'zero': np.zeros_like(query),
        'prompt_ridge': predict_ridge(prompt, query), 'mean': np.broadcast_to(delta.mean(0), query.shape).copy(),
        'scalar': affine(x, delta, query), 'permuted': predict_ridge(permuted, query)}
    predictions = {k: v.reshape(zero[test].shape) for k, v in predictions.items()}
    predictions['position_mean'] = np.empty_like(zero[test], dtype=np.float64)
    predictions['position_scalar'] = np.empty_like(zero[test], dtype=np.float64)
    for j in range(5):
        at = train[available[train, j]]
        if len(at) < 2:
            raise ValueError('Insufficient position-specific training data')
        predictions['position_mean'][:, j] = target[at, j].mean(0)
        predictions['position_scalar'][:, j] = affine(zero[at, j], target[at, j], zero[test, j])
    return test, predictions, {'fold': fold, 'train_questions': train.tolist(),
        'evaluation_questions': test.tolist(), 'training_rows': int(keep.sum()),
        'training_rows_per_position': keep.sum(0).tolist(),
        'solve_residuals': {k: v['residual'] for k, v in
                            [('ridge', primary), ('prompt_ridge', prompt), ('permuted', permuted)]}}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    answer = {'n': len(finite), 'undefined': len(values)-len(finite)}
    answer.update(zip(['p10', 'median', 'p90'],
                      np.quantile(finite, [.1, .5, .9]).tolist() if len(finite) else [None]*3))
    return answer


def measurement(prediction, target):
    tn, pn = np.linalg.norm(target, axis=1), np.linalg.norm(prediction, axis=1)
    ratio, cosine = np.full(len(tn), np.nan), np.full(len(tn), np.nan)
    np.divide(pn, tn, out=ratio, where=tn > 0)
    np.divide(np.sum(prediction*target, axis=1), pn*tn, out=cosine, where=pn*tn > 0)
    energy = np.square(target).sum()
    error = float(np.square(prediction-target).sum()/energy) if energy else None
    # A second calculation uses individual dot products and stable scalar summation.
    independent_energy = math.fsum(float(t @ t) for t in target)
    independent_error = math.fsum(float((p-t) @ (p-t)) for p, t in zip(prediction, target))
    if energy:
        np.testing.assert_allclose(error, independent_error/independent_energy, rtol=1e-12, atol=1e-12)
    for i, (p, t) in enumerate(zip(prediction, target)):
        a, b = math.sqrt(float(p @ p)), math.sqrt(float(t @ t))
        np.testing.assert_allclose([pn[i], tn[i]], [a, b], rtol=1e-12, atol=1e-12)
        if a*b:
            np.testing.assert_allclose(cosine[i], float(p @ t)/(a*b), rtol=1e-12, atol=1e-12)
    return {'n_questions': len(target), 'normalized_squared_error': error,
            'target_norm': stats(tn), 'predicted_norm': stats(pn),
            'norm_ratio': stats(ratio), 'target_cosine': stats(cosine),
            'zero_target_norm': int((tn == 0).sum()), 'zero_predicted_norm': int((pn == 0).sum())}


def summary(predictions, target, available):
    output = {}
    for name in NAMES:
        points = {str(position): measurement(predictions[name][available[:, j], j],
                                              target[available[:, j], j])
                  for j, position in enumerate(POSITIONS)}
        errors = [points[str(p)]['normalized_squared_error'] for p in POSITIONS[1:]]
        output[name] = {'positions': points,
                       'mean_continuation_normalized_error': float(np.mean(errors))
                       if all(v is not None for v in errors) else None}
    return output


def crossfit(zero, icl, available):
    zero, icl = np.asarray(zero, dtype=np.float64), np.asarray(icl, dtype=np.float64)
    if (zero.shape != icl.shape or zero.ndim != 3 or zero.shape[:2] != (128, 5)
            or available.shape != (128, 5) or available.dtype != np.bool_
            or not available[:, 0].all() or np.any(available[:, 1:] & ~available[:, :-1])
            or not np.isfinite(zero).all() or not np.isfinite(icl).all()):
        raise ValueError('Invalid paired prefix states')
    target, assignment = icl-zero, folds()
    predictions = {k: np.full(zero.shape, np.nan) for k in NAMES}
    fold_results, seen = [], set()
    for fold in range(4):
        indices, values, provenance = fold_predictions(zero, target, available, assignment, fold)
        assert not (seen & set(indices)); seen.update(indices)
        for name in NAMES:
            predictions[name][indices] = values[name]
        fold_results.append({**provenance, 'summary': summary(values, target[indices], available[indices])})
    assert len(seen) == 128 and all(np.isfinite(v).all() for v in predictions.values())
    pooled = summary(predictions, target, available)
    primary = pooled['ridge']
    points = [primary['positions'][str(p)] for p in POSITIONS[1:]]
    controls = {k: primary['mean_continuation_normalized_error'] <=
                .9*pooled[k]['mean_continuation_normalized_error'] for k in NAMES if k not in {'ridge', 'zero'}}
    checks = {'error_below_zero_at_every_continuation': all(p['normalized_squared_error'] < 1 for p in points),
              'positive_median_cosine_at_every_continuation': all(p['target_cosine']['median'] > 0 for p in points),
              'ten_percent_lower_macro_error_than_each_control': controls}
    passed = checks['error_below_zero_at_every_continuation'] and checks['positive_median_cosine_at_every_continuation'] and all(controls.values())
    return {'status': 'completed_question_disjoint_activation_crossfit', 'summary': pooled,
            'folds': fold_results, 'checks': checks, 'eligible_for_accuracy_design': passed,
            'question_count': 128, 'available_rows': int(available.sum()),
            'independent_eigensystem_prediction_check': True, 'independent_metric_check': True,
            'limitations': 'Post-diagnostic method development on original extraction questions. '
                          'All states of each evaluation question excluded from its fit. '
                          'Folds overlap in training data and are not independent replications. '
                          'No answer accuracy or causal steering benefit is measured.'}, predictions


def source_hash():
    paths = [Path(__file__), ROOT/'research/ltv-prefix-crossfit-plan.md', ROOT/'tests/test_ltv_prefix_crossfit.py']
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in paths}, sort_keys=True).encode()).hexdigest()


def prepare(parent, output):
    from analysis import ltv_prefix_collect as collector
    manifest, questions = collector.verify(parent)
    report, completed = read(parent/'alignment-report.json'), read(parent/'collection-complete.json')
    if (report['status'] != 'verified_activation_diagnostic' or completed['n_extract'] != 128
            or report['manifest_sha256'] != sha(parent/'manifest.json')
            or report['collection_sha256'] != sha(parent/'collection-complete.json')):
        raise ValueError('Require the completed verified prefix diagnostic')
    inputs = [parent/name for name in ['manifest.json', 'prepared.json', 'alignment-report.json', 'collection-complete.json']]
    inputs += sorted((parent/'batches').glob('*.json')) + sorted((parent/'batches').glob('*.npz'))
    if len(inputs) != 68:
        raise ValueError('Require 32 complete prefix batches')
    output.mkdir(parents=True, exist_ok=True)
    fixed(output/'manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'parent': str(parent.resolve()), 'source_sha256': source_hash(),
        'inputs_sha256': {str(p.resolve()): sha(p) for p in inputs},
        'question_ids': [q['problem_id'] for q in questions], 'fold_assignment': folds().tolist(),
        'fold_seed': 2713, 'penalty': 5, 'positions': list(POSITIONS), 'predictors': list(NAMES),
        'n_development': 0, 'n_reserved': 0, 'answer_supervision': False})


def evaluate(output, replay=False):
    from analysis import ltv_prefix_collect as collector
    manifest, declaration = read(output/'manifest.json'), read(output/'declaration.json')
    if (manifest['source_sha256'] != source_hash() or declaration['manifest_sha256'] != sha(output/'manifest.json')
            or not declaration.get('source_commit') or manifest['fold_assignment'] != folds().tolist()):
        raise ValueError('Crossfit declaration or source changed')
    for path, expected in manifest['inputs_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Frozen prefix input changed')
    parent = Path(manifest['parent'])
    prior, questions = collector.verify(parent)
    if manifest['question_ids'] != [q['problem_id'] for q in questions]:
        raise ValueError('Question order changed')
    with np.load(Path(prior['parent'])/'extraction.npz', allow_pickle=False) as packed:
        reference = {k: packed[k] for k in ['zero', 'icl_a']}
    chunks = [collector.check_batch(parent/'batches'/f'{start:03d}.json', questions[start:start+4],
                                   {k: v[start:start+4] for k, v in reference.items()})
              for start in range(0, 128, 4)]
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in ['zero', 'icl_a', 'available']}
    result, predictions = crossfit(arrays['zero'], arrays['icl_a'], arrays['available'])
    result.update(manifest_sha256=sha(output/'manifest.json'), source_sha256=source_hash())
    if replay:
        saved = read(output/'crossfit-report.json')
        saved.pop('created_at'); expected_hash = saved.pop('predictions_file_sha256')
        if sha(output/'predictions.npz') != expected_hash:
            raise ValueError('Prediction arrays changed')
        from analysis.ltv_prefix_report import compare
        compare(saved, result)
        with np.load(output/'predictions.npz', allow_pickle=False) as packed:
            if set(packed.files) != set(predictions):
                raise ValueError('Prediction fields differ')
            for name in predictions:
                np.testing.assert_array_equal(packed[name], predictions[name])
        print('Crossfit report and saved predictions replayed.', flush=True)
    else:
        path = output/'predictions.npz'
        if path.exists() or (output/'crossfit-report.json').exists():
            raise ValueError('Refuse to overwrite crossfit results')
        np.savez_compressed(path, **predictions)
        fixed(output/'crossfit-report.json', {**result, 'predictions_file_sha256': sha(path),
                                             'created_at': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'eligible_for_accuracy_design': result['eligible_for_accuracy_design'],
                          'checks': result['checks']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'evaluate', 'replay'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('runs/gsm8k-ltv-prefix-v1'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'.writer.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare': prepare(args.parent, args.output)
        else: evaluate(args.output, args.stage == 'replay')

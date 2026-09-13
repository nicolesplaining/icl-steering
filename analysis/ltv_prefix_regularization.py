"""Nested question folds for selecting continuation ridge regularization."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import ltv_prefix_crossfit as cv


RULES = ('fixed_5', 'scaled_0.001', 'scaled_0.01', 'scaled_0.1', 'scaled_1', 'scaled_10')
SCALES = (.001, .01, .1, 1., 10.)


def penalties(x):
    values = [5., *(float(np.square(x).sum()/len(x))*scale for scale in SCALES)]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Invalid training-only penalty scale')
    return values


def spectral(x):
    gram = x @ x.T
    values, vectors = np.linalg.eigh(gram)
    return gram, values, vectors


def fit_predict(x, target, query, penalty, basis=None, check=False):
    gram, eigenvalues, vectors = spectral(x) if basis is None else basis
    if penalty <= 0 or (eigenvalues+penalty).min() <= 0:
        raise ValueError('Invalid regularized eigensystem')
    weights = vectors @ ((vectors.T @ target)/(eigenvalues[:, None]+penalty))
    prediction = (query @ x.T) @ weights
    residual = None
    if check:
        system = gram+penalty*np.eye(len(x))
        direct = np.linalg.solve(system, target)
        residual = float(np.linalg.norm(system @ direct-target)/max(np.linalg.norm(target), 1))
        if not np.isfinite(residual) or residual > 1e-8:
            raise ValueError('Direct selected-fit solve failed')
        alternate = np.einsum('ij,kj->ik', query, x) @ direct
        np.testing.assert_allclose(prediction, alternate, rtol=1e-7, atol=1e-7)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite nested-fit prediction')
    return prediction, residual


def inner_folds(train, outer_fold):
    if len(train) != 96 or len(set(train)) != 96:
        raise ValueError('Require 96 distinct outer training questions')
    return np.random.default_rng(2714+outer_fold).permutation(np.sort(train)).reshape(3, 32)


def training_arrays(zero, target, available, indices):
    keep = available[indices]
    positions = np.broadcast_to(np.arange(5), keep.shape)[keep]
    return zero[indices][keep], target[indices][keep], positions


def select_fold(zero, target, available, assignment, fold):
    train, test = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
    if len(test) != 32 or set(train) & set(test):
        raise ValueError('Invalid outer question fold')
    errors = {name: np.zeros((6, 5)) for name in ['ridge', 'permuted']}
    energy = np.zeros(5)
    inner_records = []
    for inner, evaluate in enumerate(inner_folds(train, fold)):
        fit_indices = np.array(sorted(set(train)-set(evaluate)))
        assert len(fit_indices) == 64 and not set(test) & (set(fit_indices) | set(evaluate))
        x, delta, positions = training_arrays(zero, target, available, fit_indices)
        targets = {'ridge': delta, 'permuted': cv.permute_targets(delta, positions, 1901+100*fold+inner)}
        values, basis = penalties(x), spectral(x)
        query = zero[evaluate].reshape(-1, zero.shape[-1])
        for j in range(5):
            energy[j] += np.square(target[evaluate, j][available[evaluate, j]]).sum()
        for rule, penalty in enumerate(values):
            for name, labels in targets.items():
                predicted, _ = fit_predict(x, labels, query, penalty, basis)
                predicted = predicted.reshape(zero[evaluate].shape)
                for j in range(5):
                    keep = available[evaluate, j]
                    errors[name][rule, j] += np.square(predicted[keep, j]-target[evaluate, j][keep]).sum()
        inner_records.append({'inner_fold': inner, 'training_questions': fit_indices.tolist(),
            'evaluation_questions': evaluate.tolist(), 'training_rows': len(x),
            'penalties': dict(zip(RULES, values))})
    if (energy <= 0).any():
        raise ValueError('Missing inner evaluation target energy')
    normalized = {name: value/energy for name, value in errors.items()}
    scores = {name: value[:, 1:].mean(1) for name, value in normalized.items()}
    choices = {name: int(np.argmin(value)) for name, value in scores.items()}
    x, delta, positions = training_arrays(zero, target, available, train)
    targets = {'ridge': delta, 'permuted': cv.permute_targets(delta, positions, 1901+100*fold+99)}
    values, basis = penalties(x), spectral(x)
    predictions, selected = {}, {}
    for name, labels in targets.items():
        choice = choices[name]
        predicted, residual = fit_predict(x, labels, zero[test].reshape(-1, zero.shape[-1]),
                                          values[choice], basis, check=True)
        predictions[name] = predicted.reshape(zero[test].shape)
        selected[name] = {'rule': RULES[choice], 'outer_penalty': values[choice],
                          'inner_mean_continuation_error': float(scores[name][choice]),
                          'direct_solve_residual': residual}
    return test, predictions, {'fold': fold, 'training_questions': train.tolist(),
        'evaluation_questions': test.tolist(), 'training_rows': len(x),
        'inner_folds': inner_records, 'selected': selected,
        'inner_rule_scores': {name: {rule: {'mean_continuation_error': float(scores[name][i]),
                                            'position_errors': normalized[name][i].tolist()}
                                    for i, rule in enumerate(RULES)} for name in scores}}


def summarize(predictions, target, available):
    output = {}
    for name, prediction in predictions.items():
        points = {str(p): cv.measurement(prediction[available[:, j], j], target[available[:, j], j])
                  for j, p in enumerate(cv.POSITIONS)}
        output[name] = {'positions': points, 'mean_continuation_normalized_error': float(np.mean(
            [points[str(p)]['normalized_squared_error'] for p in cv.POSITIONS[1:]]))}
    return output


def run_arrays(zero, icl, available, previous):
    zero, icl = np.asarray(zero, dtype=np.float64), np.asarray(icl, dtype=np.float64)
    if (zero.shape != icl.shape or zero.shape[:2] != (128, 5) or available.shape != (128, 5)
            or available.dtype != np.bool_ or not available[:, 0].all()
            or np.any(available[:, 1:] & ~available[:, :-1])
            or not np.isfinite(zero).all() or not np.isfinite(icl).all() or set(previous) != set(cv.NAMES)):
        raise ValueError('Invalid paired data or inherited controls')
    target, assignment = icl-zero, cv.folds()
    predictions = {('fixed_'+k if k in {'ridge', 'permuted'} else k): v.copy() for k, v in previous.items()}
    for name in ['ridge', 'permuted']:
        predictions[name] = np.full_like(zero, np.nan)
    records, covered = [], set()
    for fold in range(4):
        indices, values, record = select_fold(zero, target, available, assignment, fold)
        assert not covered & set(indices); covered.update(indices)
        for name in values:
            predictions[name][indices] = values[name]
        records.append({**record, 'summary': summarize(values, target[indices], available[indices])})
    assert len(covered) == 128
    if any(v.shape != zero.shape or not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError('Invalid final prediction arrays')
    pooled = summarize(predictions, target, available)
    primary = pooled['ridge']
    points = [primary['positions'][str(p)] for p in cv.POSITIONS[1:]]
    checks = {'error_below_zero_at_every_continuation': all(p['normalized_squared_error'] < 1 for p in points),
        'positive_median_cosine_at_every_continuation': all(p['target_cosine']['median'] > 0 for p in points),
        'ten_percent_lower_macro_error_than_each_control': {
            k: primary['mean_continuation_normalized_error'] <= .9*v['mean_continuation_normalized_error']
            for k, v in pooled.items() if k not in {'ridge', 'zero'}}}
    passed = checks['error_below_zero_at_every_continuation'] and checks['positive_median_cosine_at_every_continuation'] and all(checks['ten_percent_lower_macro_error_than_each_control'].values())
    return {'status': 'completed_nested_activation_crossfit', 'summary': pooled, 'folds': records,
        'checks': checks, 'eligible_for_accuracy_design': passed, 'n_questions': 128,
        'available_rows': int(available.sum()), 'selected_predictions_checked_against_direct_solve': True,
        'limitations': 'Further development after inspecting prior results on the same extraction pool. '
                      'Outer targets are excluded from inner penalty selection and outer fitting. '
                      'Training folds overlap. This is activation prediction, not math accuracy or causal steering evidence.'}, predictions


def source_hash():
    files = [Path(__file__), cv.ROOT/'research/ltv-prefix-regularization-plan.md',
             cv.ROOT/'tests/test_ltv_prefix_regularization.py']
    return hashlib.sha256(json.dumps({'parent_source': cv.source_hash(), 'files': {
        str(p.relative_to(cv.ROOT)): cv.sha(p) for p in files}}, sort_keys=True).encode()).hexdigest()


def prepare(parent, output):
    manifest, report = cv.read(parent/'manifest.json'), cv.read(parent/'crossfit-report.json')
    if (manifest['source_sha256'] != cv.source_hash() or report['eligible_for_accuracy_design']
            or report['manifest_sha256'] != cv.sha(parent/'manifest.json')
            or report['predictions_file_sha256'] != cv.sha(parent/'predictions.npz')):
        raise ValueError('Require the completed fixed-penalty failure')
    inputs = dict(manifest['inputs_sha256'])
    inputs.update({str((parent/n).resolve()): cv.sha(parent/n) for n in
                   ['manifest.json', 'declaration.json', 'crossfit-report.json', 'predictions.npz']})
    cv.fixed(output/'manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'parent': str(parent.resolve()), 'source_sha256': source_hash(), 'inputs_sha256': inputs,
        'fold_assignment': cv.folds().tolist(), 'rules': list(RULES),
        'inner_folds': {str(f): inner_folds(np.flatnonzero(cv.folds() != f), f).tolist() for f in range(4)},
        'n_development': 0, 'n_reserved': 0, 'cpu_only': True})


def evaluate(output, replay=False):
    manifest, declaration = cv.read(output/'manifest.json'), cv.read(output/'declaration.json')
    if (manifest['source_sha256'] != source_hash() or declaration['manifest_sha256'] != cv.sha(output/'manifest.json')
            or not declaration.get('source_commit')):
        raise ValueError('Declaration or source changed')
    for path, expected in manifest['inputs_sha256'].items():
        if cv.sha(Path(path)) != expected:
            raise ValueError('Frozen input changed: '+path)
    parent = Path(manifest['parent'])
    earlier = cv.read(parent/'manifest.json')
    from analysis import ltv_prefix_collect as collector
    prefix = Path(earlier['parent'])
    prior, questions = collector.verify(prefix)
    if earlier['question_ids'] != [p['problem_id'] for p in questions]:
        raise ValueError('Question order changed')
    with np.load(Path(prior['parent'])/'extraction.npz', allow_pickle=False) as packed:
        reference = {k: packed[k] for k in ['zero', 'icl_a']}
    chunks = [collector.check_batch(prefix/'batches'/f'{start:03d}.json', questions[start:start+4],
                                    {k: v[start:start+4] for k, v in reference.items()})
              for start in range(0, 128, 4)]
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in ['zero', 'icl_a', 'available']}
    with np.load(parent/'predictions.npz', allow_pickle=False) as packed:
        previous = {k: packed[k] for k in packed.files}
    result, predictions = run_arrays(arrays['zero'], arrays['icl_a'], arrays['available'], previous)
    result.update(manifest_sha256=cv.sha(output/'manifest.json'), source_sha256=source_hash())
    if replay:
        saved = cv.read(output/'nested-report.json')
        saved.pop('created_at'); expected_hash = saved.pop('predictions_file_sha256')
        if cv.sha(output/'predictions.npz') != expected_hash:
            raise ValueError('Saved prediction arrays changed')
        from analysis.ltv_prefix_report import compare
        compare(saved, result)
        with np.load(output/'predictions.npz', allow_pickle=False) as packed:
            if set(packed.files) != set(predictions):
                raise ValueError('Saved prediction fields differ')
            for name in predictions:
                np.testing.assert_array_equal(packed[name], predictions[name])
        print('Nested report and prediction arrays replayed.', flush=True)
    else:
        if (output/'predictions.npz').exists() or (output/'nested-report.json').exists():
            raise ValueError('Refuse to replace results')
        np.savez_compressed(output/'predictions.npz', **predictions)
        cv.fixed(output/'nested-report.json', {**result, 'predictions_file_sha256': cv.sha(output/'predictions.npz'),
                                              'created_at': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'eligible_for_accuracy_design': result['eligible_for_accuracy_design'],
                          'checks': result['checks']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'evaluate', 'replay'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('runs/gsm8k-ltv-crossfit-v1'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'.writer.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare': prepare(args.parent, args.output)
        else: evaluate(args.output, args.stage == 'replay')

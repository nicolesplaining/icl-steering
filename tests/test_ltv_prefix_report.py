import copy
import json

import numpy as np
import pytest

from analysis import ltv_prefix_report as report


def test_independent_recount_detects_changed_error_and_cohort():
    rng = np.random.default_rng(2141)
    zero = rng.normal(size=(4, 5, 3))
    icl = zero+rng.normal(size=zero.shape)
    predictions = {k: rng.normal(size=zero.shape) for k in report.metrics.PREDICTORS}
    predictions['mean'][:] = 0
    icl[1] = zero[1]
    available = np.ones((4, 5), dtype=bool); available[0, 1:] = False
    summary = report.metrics.summarize(zero, icl, predictions, available)
    report.replay_summary(summary, zero, icl, predictions, available)
    changed = copy.deepcopy(summary)
    changed['positions'][1]['predictors']['ltv']['normalized_squared_error'] += .1
    with pytest.raises(AssertionError):
        report.replay_summary(changed, zero, icl, predictions, available)
    changed = copy.deepcopy(summary); changed['positions'][1]['n_questions'] += 1
    with pytest.raises(ValueError):
        report.replay_summary(changed, zero, icl, predictions, available)


def test_saved_capture_report_reconstruction_and_incomplete_rejection(tmp_path, monkeypatch):
    rng = np.random.default_rng(2143)
    parent, output = tmp_path/'parent', tmp_path/'diagnostic'
    parent.mkdir(); (output/'batches').mkdir(parents=True)
    x, delta = rng.normal(size=(128, 8)), rng.normal(size=(128, 8))
    np.savez(parent/'extraction.npz', zero=x, icl_a=x+delta)
    # Fit from the exact saved arithmetic, including its floating-point rounding.
    d = (x+delta)-x
    real = report.ltv.mapping.fit(x, d)
    permuted = report.ltv.mapping.fit(x, d[np.random.default_rng(1901).permutation(128)])
    np.savez(parent/'maps.npz', **{f'{kind}/{k}': v for kind, fit in [('real', real), ('permuted', permuted)]
                                 for k, v in fit.items()})
    (parent/'fit.json').write_text('{}')
    (output/'manifest.json').write_text('{}')
    questions = [{'problem_id': str(i), 'prompts': {'zero': 'unused', 'icl_a': 'unused'}} for i in range(128)]
    monkeypatch.setattr(report.collector, 'verify', lambda path: ({'parent': str(parent)}, questions))
    for start in range(0, 128, 4):
        z, c = rng.normal(size=(4, 5, 8)), rng.normal(size=(4, 5, 8))
        z[:, 0], c[:, 0] = x[start:start+4], (x+delta)[start:start+4]
        prefixes = [[3]*n for n in [0, 1, 8, 128]]
        prompts = {'zero': [[1, 11]]*4, 'icl_a': [[1, 21, 11]]*4}
        masks, hashes = [], {}
        for position in report.metrics.POSITIONS:
            zs, cs, mask = report.metrics.paired_token_rows(prompts['zero'], prompts['icl_a'], prefixes, position)
            masks.append(mask)
            hashes[str(position)] = {'zero': report.ltv.audit.digest(zs), 'icl_a': report.ltv.audit.digest(cs)}
        path = output/'batches'/f'{start:03d}.json'
        report.ltv.arrays(path.with_suffix('.npz'), {'zero': z, 'icl_a': c, 'available': np.stack(masks, 1)})
        report.base.fixed(path, {'problem_ids': [q['problem_id'] for q in questions[start:start+4]],
            'prompt_ids': prompts, 'prefix_ids': prefixes, 'eos_ids': [2],
            'generated_token_ids': [p+[2] if len(p) < 128 else p for p in prefixes],
            'paired_input_sha256': hashes, 'arrays_sha256': report.base.file_hash(path.with_suffix('.npz'))})
    report.base.fixed(output/'collection-complete.json', {'n_extract': 128,
        'manifest_sha256': report.base.file_hash(output/'manifest.json'),
        'batch_files_sha256': {p.name: report.base.file_hash(p) for p in sorted((output/'batches').glob('*.json'))}})
    result = report.build(output)
    assert result['independent_prediction_replay'] and result['independent_metric_replay']
    assert result['n_development_questions'] == result['n_reserved_questions'] == 0
    assert [p['n_questions'] for p in result['summary']['positions']] == [128, 96, 64, 32, 32]
    json.dumps(result, allow_nan=False)
    changed = copy.deepcopy(real); changed['weights'][0, 0] += .01
    np.savez(parent/'maps.npz', **{f'{kind}/{k}': v for kind, fit in [('real', changed), ('permuted', permuted)]
                                 for k, v in fit.items()})
    with pytest.raises(AssertionError): report.build(output)
    (output/'batches'/'124.json').unlink()
    with pytest.raises(ValueError, match='Incomplete'): report.build(output)

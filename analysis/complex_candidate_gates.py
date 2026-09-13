"""Frozen development criteria for the single fresh-complexity steering candidate."""

import math


BASELINES = ['zero', 'icl_a', 'icl_b', 'first', 'cot']
NEW = ['steered', 'regularized_prefill', 'shared_low', 'permuted_low', 'permuted',
       'shared_high', 'real_high', 'prefill', 'mean', 'scalar', 'position_mean', 'position_scalar']
CONDITIONS = [*BASELINES, *NEW]
NULL_CONTROLS = ['mean', 'scalar', 'position_mean', 'position_scalar', 'permuted',
                 'shared_low', 'permuted_low', 'shared_high']


def development_gate(summary, audited_zero_contrast, icl_screen_passed):
    if set(summary) != set(CONDITIONS):
        raise ValueError('Require all 17 declared conditions')
    for name, row in summary.items():
        if row['n'] != 256:
            raise ValueError('Require all 256 development questions: '+name)
        fields = ['primary_completed_correct', 'audited_completed_correct', 'unparsed', 'truncated']
        if any(type(row[k]) is not int or not 0 <= row[k] <= 256 for k in fields):
            raise ValueError('Invalid counts: '+name)
        if not (row['primary_completed_correct'] <= row['audited_completed_correct'] <= 256-row['truncated']):
            raise ValueError('Completed-answer counts contradict truncation: '+name)
    counts = {k: v['audited_completed_correct'] for k, v in summary.items()}
    gain = (counts['steered']-counts['zero'])/256
    ci = audited_zero_contrast['ci95']
    if (len(ci) != 2 or not all(math.isfinite(x) for x in ci)
            or not -1 <= ci[0] <= ci[1] <= 1 or audited_zero_contrast['gain'] != gain):
        raise ValueError('Invalid or mismatched audited zero-shot contrast')
    checks = {
        'icl_screen': icl_screen_passed is True,
        'zero_gain': gain >= .03,
        'zero_interval': ci[0] > 0,
        'beats_controls': all(counts['steered'] > counts[k] for k in NULL_CONTROLS),
        'beats_same_map_prefill': counts['steered'] > counts['regularized_prefill'],
        'matches_text_and_prefill': all(counts['steered'] >= counts[k] for k in ['first', 'cot', 'prefill']),
        'parser_not_below_zero': summary['steered']['primary_completed_correct'] >= summary['zero']['primary_completed_correct'],
        'truncation': summary['steered']['truncated']/256 <= .05,
    }
    return {'eligible': all(checks.values()), 'checks': checks,
            'candidate': 'steered', 'reserved_generation_authorized': False}

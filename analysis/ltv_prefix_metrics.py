"""Numerical checks for the declared identical-prefix activation diagnostic."""

import numpy as np


POSITIONS = (0, 1, 8, 32, 128)
PREDICTORS = ('ltv', 'mean', 'scalar', 'scalar_norm', 'permuted')


def prefix_token_ids(generated, eos_ids):
    """Keep at most 128 generated IDs before EOS; never decode or retokenize."""
    if not eos_ids or any(type(t) is not int or t < 0 for t in eos_ids):
        raise ValueError('Require nonnegative integer EOS IDs')
    if any(type(t) is not int or t < 0 for t in generated):
        raise ValueError('Invalid generated token IDs')
    stop = next((i for i, t in enumerate(generated) if t in eos_ids), len(generated))
    return list(generated[:min(stop, POSITIONS[-1])])


def paired_token_rows(zero_prompts, icl_prompts, continuations, position):
    """Append the same available token IDs to both contexts, retaining all rows."""
    if position not in POSITIONS or not len(zero_prompts) == len(icl_prompts) == len(continuations):
        raise ValueError('Invalid position or mismatched question rows')
    if not zero_prompts or any(not p for p in [*zero_prompts, *icl_prompts]):
        raise ValueError('Empty prompt batch')
    rows = [[], []]
    available = []
    for z, c, tokens in zip(zero_prompts, icl_prompts, continuations):
        if any(type(t) is not int or t < 0 for t in [*z, *c, *tokens]):
            raise ValueError('Invalid input token ID')
        prefix = list(tokens[:position])
        rows[0].append([*z, *prefix])
        rows[1].append([*c, *prefix])
        available.append(len(tokens) >= position)
    return rows[0], rows[1], np.asarray(available, dtype=bool)


def values(prediction, target):
    """Undefined ratios/cosines are NaN internally and counted in summaries."""
    target_norm = np.linalg.norm(target, axis=-1)
    predicted_norm = np.linalg.norm(prediction, axis=-1)
    ratio = np.full(target_norm.shape, np.nan)
    cosine = np.full(target_norm.shape, np.nan)
    np.divide(predicted_norm, target_norm, out=ratio, where=target_norm > 0)
    denominator = predicted_norm * target_norm
    np.divide(np.sum(prediction * target, axis=-1), denominator,
              out=cosine, where=denominator > 0)
    return {'target_norm': target_norm, 'predicted_norm': predicted_norm,
            'predicted_to_target_norm': ratio, 'target_cosine': cosine}


def quantiles(x):
    finite = np.isfinite(x)
    result = {'n': int(finite.sum()), 'undefined': int((~finite).sum())}
    result.update(zip(['p10', 'median', 'p90'],
                      np.quantile(x[finite], [.1, .5, .9]).tolist() if finite.any()
                      else [None, None, None]))
    return result


def normalized_error(prediction, target):
    energy = np.square(target).sum()
    return float(np.square(prediction-target).sum()/energy) if energy > 0 else None


def summarize(zero, icl, predictions, available):
    """Question x position x feature arrays; no answers or grades are accepted."""
    zero, icl = np.asarray(zero, dtype=np.float64), np.asarray(icl, dtype=np.float64)
    available = np.asarray(available)
    if (zero.ndim != 3 or zero.shape != icl.shape or zero.shape[1] != len(POSITIONS)
            or not zero.shape[0] or not zero.shape[2]
            or available.shape != zero.shape[:2] or available.dtype != np.bool_
            or not available[:, 0].all()
            or np.any(available[:, 1:] & ~available[:, :-1])):
        raise ValueError('Invalid paired state shapes or prefix availability')
    if set(predictions) != set(PREDICTORS):
        raise ValueError('Require all five declared predictors')
    predicted = {k: np.asarray(v, dtype=np.float64) for k, v in predictions.items()}
    if (not np.isfinite(zero).all() or not np.isfinite(icl).all()
            or any(v.shape != zero.shape or not np.isfinite(v).all() for v in predicted.values())):
        raise ValueError('Invalid or nonfinite states or predictions')
    target = icl-zero
    measurements = {k: values(v, target) for k, v in predicted.items()}
    result = {'n_questions': len(zero), 'positions': []}
    for j, position in enumerate(POSITIONS):
        keep = available[:, j]
        point = {'prefix_tokens': position, 'n_questions': int(keep.sum()),
                 'excluded_short_prefix': int((~keep).sum()), 'predictors': {}}
        for name, prediction in predicted.items():
            measured = measurements[name]
            current = {k: v[keep, j] for k, v in measured.items()}
            baseline = {k: v[keep, 0] for k, v in measured.items()}
            point['predictors'][name] = {
                'zero_target_norm': int((current['target_norm'] == 0).sum()),
                'zero_predicted_norm': int((current['predicted_norm'] == 0).sum()),
                'measurements': {k: quantiles(v) for k, v in current.items()},
                'paired_change_from_prompt': {k: quantiles(v-baseline[k]) for k, v in current.items()},
                'normalized_squared_error': normalized_error(prediction[keep, j], target[keep, j]),
                'prompt_normalized_squared_error_same_questions': normalized_error(
                    prediction[keep, 0], target[keep, 0])}
        result['positions'].append(point)
    return result

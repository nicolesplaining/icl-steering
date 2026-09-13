"""Replay fixed candidate routing against independent reconstructions of old fits."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import complex_candidate_mapping as mapping
from analysis import complex_candidate_gates as gates
from analysis import ltv_continuation_audit, ltv_pairing_audit


def check(runs):
    roots = {k: runs/name for k, name in mapping.RUN_NAMES.items()}
    paths = {k: p/'maps.npz' for k, p in roots.items()}
    fitted = mapping.load_frozen(paths)
    paired, width = ltv_pairing_audit.reconstruct(roots['pairing'])
    continued, other_width = ltv_continuation_audit.reconstruct(roots['continuation'])
    assert width == other_width == 3584
    with np.load(roots['original']/'extraction.npz', allow_pickle=False) as packed:
        old_x = packed['zero'].astype(np.float64)
        old_delta = packed['icl_a'].astype(np.float64)-old_x
    matrix = old_x @ old_x.T+5*np.eye(len(old_x))
    old_weights = np.linalg.solve(matrix, old_delta)
    residual = np.linalg.norm(matrix @ old_weights-old_delta)/max(np.linalg.norm(old_delta), 1.)
    assert residual <= 1e-8
    np.testing.assert_array_equal(fitted['original']['real/x'], old_x)
    np.testing.assert_allclose(fitted['original']['real/weights'], old_weights, rtol=1e-12, atol=1e-12)
    states = fitted['pairing']['x'][:4].copy()
    checked = {}
    for name in gates.NEW:
        positions = [0] if mapping.scope(name) == 'prefill' else [0, 1, 4, 8, 16, 32, 64, 128]
        for position in positions:
            actual = mapping.predict(fitted, states, name, position)
            if name in mapping.pairing.SPECS:
                expected = paired(states, name, position)
            elif name == 'prefill':
                expected = (states @ old_x.T) @ old_weights
            else:
                expected = continued(states, name, position)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
            np.testing.assert_array_equal(actual.astype(np.float32), expected.astype(np.float32))
        checked[name] = positions
    np.testing.assert_array_equal(mapping.predict(fitted, states, 'steered', 0),
                                  mapping.predict(fitted, states, 'regularized_prefill', 0))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(Path(__file__)), 'mapping_code_sha256': sha(Path(mapping.__file__)),
        'source_maps_sha256': {k: sha(p) for k, p in paths.items()},
        'source_extraction_sha256': {k: sha(p/'extraction.npz') for k, p in roots.items()},
        'reconstruction_code_sha256': {p.name: sha(p) for p in
            [Path(ltv_pairing_audit.__file__), Path(ltv_continuation_audit.__file__)]},
        'conditions': len(checked), 'condition_positions_checked': sum(map(len, checked.values())),
        'states_per_check': len(states), 'width': width, 'old_prompt_solver_residual': float(residual),
        'candidate_and_same_map_prefill_predictions_equal': True, 'positions': checked,
        'new_fits_saved': False, 'new_model_generations': 0,
        'scope': 'Independent algebraic reconstruction and condition routing only. Does not validate new generation traces or establish accuracy.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite a map check')
    result = check(args.runs)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

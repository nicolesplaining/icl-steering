"""Route the fixed candidate conditions to their original, immutable map archives."""

import hashlib
from pathlib import Path

import numpy as np

from analysis import complex_candidate_gates as gates
from analysis import ltv_continuation_mapping as continuation
from analysis import ltv_mapping as original
from analysis import ltv_pairing_mapping as pairing


MAP_HASHES = {
    'pairing': 'ee9c4fa68737abe5671c54fa740f41fc6a2f8a0aec1e5c7668b4b641593a9137',
    'continuation': '512875b4246793687eb5227d7c774bb26220ab3ac7cfff33f7971b46bd3633d7',
    'original': '85a10858a81a53b94bb814e8a4705f38d34f4a036b4b1fa0303784e0a41342b3',
}
RUN_NAMES = {'pairing': 'gsm8k-pairing-v1', 'continuation': 'gsm8k-continuation-v1',
             'original': 'gsm8k-ltv-v1'}
normalized_hook = continuation.normalized_hook


def load_frozen(paths):
    if set(paths) != set(MAP_HASHES):
        raise ValueError('Require the three declared map archives')
    fitted = {}
    for name, path in paths.items():
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != MAP_HASHES[name]:
            raise ValueError('Frozen map changed: '+name)
        with np.load(path, allow_pickle=False) as packed:
            fitted[name] = {k: packed[k] for k in packed.files}
        if any(not np.isfinite(v).all() for v in fitted[name].values()):
            raise ValueError('Nonfinite map archive: '+name)
        for value in fitted[name].values():
            value.flags.writeable = False
    paired, continued = fitted['pairing'], fitted['continuation']
    np.testing.assert_array_equal(paired['x'], continued['real/x'])
    np.testing.assert_array_equal(paired['x'], continued['permuted/x'])
    np.testing.assert_array_equal(paired['steered/weights'], continued['real/weights'])
    np.testing.assert_array_equal(paired['permuted/weights'], continued['permuted/weights'])
    np.testing.assert_array_equal(paired['steered/penalty'], continued['real/penalty'])
    np.testing.assert_array_equal(paired['permuted/penalty'], continued['permuted/penalty'])
    return fitted


def scope(name):
    if name not in gates.NEW:
        raise ValueError('Unknown candidate condition')
    return 'prefill' if name in {'prefill', 'regularized_prefill'} else 'prefix_0_128'


def predict(fitted, h, name, position):
    condition_scope = scope(name)
    if condition_scope == 'prefill' and position != 0:
        raise ValueError('Prompt-only predictor called after the prompt')
    if name in pairing.SPECS:
        return pairing.predict(fitted['pairing'], h, name, position)
    if name == 'prefill':
        real = {k.split('/', 1)[1]: v for k, v in fitted['original'].items() if k.startswith('real/')}
        return original.predict(real, h, 'ltv')
    kind = 'ridge' if name == 'regularized_prefill' else name
    return continuation.predict(fitted['continuation'], h, kind, position)

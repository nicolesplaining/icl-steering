"""Collect identical-prefix state pairs after a reviewed LTV development failure."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import torch

from analysis import gsm8k_ltv as ltv
from analysis import ltv_prefix_metrics as metrics

base = ltv.base
ROOT = Path(__file__).parents[1]


def code_hash():
    return ltv.audit.digest({'parent': ltv.code_hash(), 'files': {
        str(p.relative_to(ROOT)): base.file_hash(p) for p in [Path(__file__),
        ROOT/'analysis/ltv_prefix_metrics.py', ROOT/'research/ltv-prefix-alignment-plan.md']}})


def require_failed_parent(parent):
    if not (parent/'selection.json').exists():
        raise ValueError('Require a completed, reviewed LTV failure before preparation')
    state = base.read(parent/'process.json')
    if (state['status'] != 'awaiting_blinded_review' or state['returncode'] != 0
            or any(Path(f'/proc/{state[k]}').exists() for k in ['pid', 'supervisor_pid'])):
        raise ValueError('Parent GPU processes must finish first')
    config, data = ltv.verify(parent, fitted=True)
    decision = base.read(parent/'selection.json')
    report = base.read(parent/'validation-audit.json')
    annotations = base.read(parent/'validation-annotations.json')
    freeze = base.read(parent/'review-freeze.json')
    rows = ltv.collect(parent, complete=True)
    if (decision['eligible'] or ltv.gates(report['summary'], config)['eligible']
            or decision['checks'] != ltv.gates(report['summary'], config)['checks']
            or decision['manifest_sha256'] != base.file_hash(parent/'manifest.json')
            or decision['fit_sha256'] != base.file_hash(parent/'fit.json')
            or decision['rows_sha256'] != ltv.audit.digest(rows)
            or decision['annotations_sha256'] != ltv.audit.digest(annotations)
            or not freeze['annotations_commit']
            or freeze['annotations_file_sha256'] != base.file_hash(parent/'validation-annotations.json')):
        raise ValueError('Parent failure or annotation provenance mismatch')
    return config, data


def prepare(parent, output):
    config, data = require_failed_parent(parent)
    questions = [{'problem_id': p['problem_id'],
                  'prompts': {k: p['prompts'][k] for k in ['zero', 'icl_a']}}
                 for p in data['splits']['extract']]
    if len(questions) != 128 or len({p['problem_id'] for p in questions}) != 128:
        raise ValueError('Require the original 128 extraction questions')
    output.mkdir(parents=True, exist_ok=True)
    base.fixed(output/'prepared.json', {'questions': questions})
    inputs = [parent/n for n in ['manifest.json', 'prepared.json', 'fit.json', 'maps.npz',
        'extraction.npz', 'selection.json', 'validation-audit.json', 'validation-annotations.json',
        'review-freeze.json']]
    base.fixed(output/'manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'parent': str(parent.resolve()), 'config': config, 'code_sha256': code_hash(),
        'prepared_sha256': base.file_hash(output/'prepared.json'),
        'inputs_sha256': {str(p.resolve()): base.file_hash(p) for p in inputs},
        'positions': list(metrics.POSITIONS), 'n_extract': 128, 'batch_size': 4,
        'prefix_source': 'unsteered_zero_shot_greedy_EOS_only_128_tokens',
        'answer_supervision': False, 'new_fit': False, 'accuracy_evaluation': False})


def verify(output):
    manifest = base.read(output/'manifest.json')
    if (manifest['code_sha256'] != code_hash()
            or manifest['prepared_sha256'] != base.file_hash(output/'prepared.json')):
        raise ValueError('Diagnostic source or prepared inputs changed')
    for path, expected in manifest['inputs_sha256'].items():
        if base.file_hash(Path(path)) != expected:
            raise ValueError('Frozen parent input changed: '+path)
    return manifest, base.read(output/'prepared.json')['questions']


def encoded_rows(rows, pad_id, device, max_len):
    if not rows or any(not row for row in rows) or max(map(len, rows)) > max_len:
        raise ValueError('Empty or oversized token sequence')
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, -len(row):] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, -len(row):] = 1
    positions = mask.cumsum(-1)-1
    positions.masked_fill_(mask == 0, 1)
    return {'input_ids': ids, 'attention_mask': mask, 'position_ids': positions}


@torch.inference_mode()
def capture_rows(model, rows, pad_id, max_len):
    inputs = encoded_rows(rows, pad_id, model.device, max_len)
    captured = []
    def capture(module, args, output):
        captured.append(output[:, -1, :].detach().clone())
    handle = model.model.norm.register_forward_hook(capture)
    try:
        output = model.model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1 or not torch.equal(output.last_hidden_state[:, -1, :], captured[0]):
        raise ValueError('Unexpected final normalization boundary')
    result = captured[0].float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite captured states')
    return result


@torch.inference_mode()
def collect_batch(engine, questions, reference, max_len):
    prompts = {}
    for kind in ['zero', 'icl_a']:
        encoded = engine.encode([p['prompts'][kind] for p in questions])
        prompts[kind] = [ids[mask.bool()].tolist() for ids, mask in
                        zip(encoded['input_ids'], encoded['attention_mask'])]
    if max(len(p) for group in prompts.values() for p in group)+128 > max_len:
        raise ValueError('Prompt and continuation exceed context')
    inputs = engine.encode([p['prompts']['zero'] for p in questions])
    inputs.pop('position_ids', None)
    eos = engine.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if any(type(t) is not int or t < 0 for t in eos):
        raise ValueError('Missing or invalid model EOS IDs')
    sequences = engine.model.generate(**inputs, max_new_tokens=128, do_sample=False,
        temperature=None, top_p=None, top_k=None, use_cache=True,
        pad_token_id=engine.tokenizer.pad_token_id)
    generated = sequences[:, inputs['input_ids'].shape[1]:].tolist()
    prefixes = [metrics.prefix_token_ids(ids, eos) for ids in generated]
    states = {'zero': [], 'icl_a': []}
    masks, hashes = [], {}
    for position in metrics.POSITIONS:
        zero, icl, available = metrics.paired_token_rows(prompts['zero'], prompts['icl_a'], prefixes, position)
        masks.append(available)
        hashes[str(position)] = {k: ltv.audit.digest(v) for k, v in [('zero', zero), ('icl_a', icl)]}
        for kind, rows in [('zero', zero), ('icl_a', icl)]:
            value = capture_rows(engine.model, rows, engine.tokenizer.pad_token_id, max_len)
            if position == 0:
                np.testing.assert_array_equal(value, reference[kind], err_msg='Original prompt capture mismatch')
            states[kind].append(value)
    arrays = {k: np.stack(v, axis=1) for k, v in states.items()}
    arrays['available'] = np.stack(masks, axis=1)
    record = {'problem_ids': [p['problem_id'] for p in questions], 'prompt_ids': prompts,
        'generated_token_ids': generated, 'prefix_ids': prefixes, 'eos_ids': eos,
        'paired_input_sha256': hashes}
    return arrays, record


def check_batch(path, questions, reference):
    record = base.read(path)
    if record['problem_ids'] != [p['problem_id'] for p in questions]:
        raise ValueError('Question order or extraction split changed')
    arrays_path = path.with_suffix('.npz')
    if record['arrays_sha256'] != base.file_hash(arrays_path):
        raise ValueError('Captured state file changed')
    prefixes = [metrics.prefix_token_ids(ids, record['eos_ids']) for ids in record['generated_token_ids']]
    if prefixes != record['prefix_ids'] or len(prefixes) != 4:
        raise ValueError('Stored prefix IDs changed')
    with np.load(arrays_path, allow_pickle=False) as packed:
        arrays = {k: packed[k] for k in ['zero', 'icl_a', 'available']}
    for kind in ['zero', 'icl_a']:
        if arrays[kind].shape != (4, 5, reference[kind].shape[1]) or not np.isfinite(arrays[kind]).all():
            raise ValueError('Invalid state array')
        np.testing.assert_array_equal(arrays[kind][:, 0], reference[kind])
    if arrays['available'].shape != (4, 5) or arrays['available'].dtype != np.bool_:
        raise ValueError('Invalid availability array')
    for j, position in enumerate(metrics.POSITIONS):
        zero, icl, mask = metrics.paired_token_rows(record['prompt_ids']['zero'],
                                                   record['prompt_ids']['icl_a'], prefixes, position)
        expected = {k: ltv.audit.digest(v) for k, v in [('zero', zero), ('icl_a', icl)]}
        if record['paired_input_sha256'][str(position)] != expected:
            raise ValueError('Paired token inputs changed')
        np.testing.assert_array_equal(mask, arrays['available'][:, j])
    return arrays


def collect(output):
    manifest, questions = verify(output)
    declaration = base.read(output/'declaration.json')
    if (declaration['manifest_sha256'] != base.file_hash(output/'manifest.json')
            or not declaration.get('parent_report_commit')):
        raise ValueError('Require declaration and published parent report before GPU loading')
    parent = Path(manifest['parent'])
    require_failed_parent(parent)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Use physical GPU 0 only')
    with np.load(parent/'extraction.npz', allow_pickle=False) as packed:
        reference = {k: packed[k] for k in ['zero', 'icl_a']}
    batches = output/'batches'; batches.mkdir(exist_ok=True)
    expected = {f'{start:03d}.json' for start in range(0, 128, 4)}
    if any(p.name not in expected for p in batches.glob('*.json')):
        raise ValueError('Unexpected diagnostic batch')
    engine = None
    for start in range(0, 128, 4):
        path = batches/f'{start:03d}.json'
        batch, ref = questions[start:start+4], {k: v[start:start+4] for k, v in reference.items()}
        if not path.exists():
            if engine is None:
                engine = ltv.backend(manifest['config'], {'splits': {'extract': questions}}, output)
            arrays, record = collect_batch(engine, batch, ref, manifest['config']['max_model_len'])
            ltv.arrays(path.with_suffix('.npz'), arrays)
            base.fixed(path, {**record, 'arrays_sha256': base.file_hash(path.with_suffix('.npz'))})
        check_batch(path, batch, ref)
        print(f'Extraction prefix pairs: {start+4}/128', flush=True)
    completion = {'n_extract': 128,
        'manifest_sha256': base.file_hash(output/'manifest.json'),
        'batch_files_sha256': {p.name: base.file_hash(p) for p in sorted(batches.glob('*.json'))}}
    if (output/'collection-complete.json').exists():
        previous = base.read(output/'collection-complete.json')
        if any(previous[k] != value for k, value in completion.items()):
            raise ValueError('Completed diagnostic changed')
    else:
        base.fixed(output/'collection-complete.json', {
            **completion, 'finished_at': datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'collect'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('runs/gsm8k-ltv-v1'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare': prepare(args.parent, args.output)
        else: collect(args.output)


if __name__ == '__main__':
    main()

from contextlib import redirect_stdout
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import gsm8k_published_screen as screen
from analysis import published_screen_backend as backend
from analysis import published_screen_recount as check
from analysis.paired_bootstrap_check import interval


class Tokenizer:
    def encode(self, value):
        return list(map(ord, value))

    def decode(self, values, **kwargs):
        return ''.join(chr(value) for value in values if value)


class Backend:
    runtime = {'physical_gpu': 0, 'frozen': True, 'dtype': 'torch.bfloat16', 'eos_token_ids': [0]}

    def records(self, prompts, config):
        result = []
        for prompt in prompts:
            kind, number = prompt.split()
            number = int(number)
            period = {'zero': 8, 'icl_original': 16, 'icl_complex': 32}[kind]
            text = 'The answer is '+('2' if number % period == 0 else '1')
            if number == 3:
                text = 'We have 1 apple.'
            if number == 4 and kind == 'icl_complex':
                text = 'The answer is 2'
            if number == 5 and kind == 'zero':
                text = 'We have 2 apples.'
            if number == 6:
                text = 'We have 1 apple.'+' '*1008
                ids = Tokenizer().encode(text)
                assert len(ids) == 1024
            else:
                ids = Tokenizer().encode(text)+[0]
            result.append(backend.record(ids, None, [0], Tokenizer(), 1024, 0.))
        return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


class RecountTests(unittest.TestCase):
    def test_nontrivial_production_report_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            config = screen.read(screen.CONFIG)
            plan = {name: [f'gsm8k:train:{i}' for i in range(start, stop)]
                    for name, start, stop in [('extract', 0, 256), ('validation', 256, 768), ('reserved', 768, 1280)]}
            data = {'plan': plan, 'splits': {}}
            for name, ids in plan.items():
                data['splits'][name] = [{'problem_id': pid, 'question': 'Question '+pid, 'answer': '1',
                    'prompts': {kind: f'{kind} {i}' for kind in screen.CONDITIONS}} for i, pid in enumerate(ids)]
            save(run/'prepared.json', data)
            save(run/'manifest.json', {'config': config, 'prepared_sha256': check.sha(run/'prepared.json'),
                'conditions': screen.CONDITIONS, 'fixed_target': 'icl_complex',
                'reserved_generation_supported': False, 'extraction_generation_supported': False})
            save(run/'declaration.json', {'manifest_sha256': check.sha(run/'manifest.json'), 'declaration_commit': 'fixture'})
            with patch.object(screen, 'verify', return_value=(config, data)), \
                    patch.object(screen, 'tokenizer', return_value=Tokenizer()), \
                    patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), redirect_stdout(io.StringIO()):
                screen.run(run, lambda _: Backend())
                packet = check.read(run/'validation-review-packet.json')
                annotations = {'packet_sha256': check.digest(packet), 'answers': [
                    {'response_id': item['response_id'], 'reviewed': True, 'rationale': 'Literal fixture number.',
                     'stated_answer': '2' if '2 apples' in item['response'] else '1'} for item in packet['items']]}
                save(run/'annotations.json', annotations)
                save(run/'review-freeze.json', {'packet_sha256': check.digest(packet),
                    'annotations_file_sha256': check.sha(run/'annotations.json'), 'annotations_commit': 'fixture'})
                screen.score(run, run/'annotations.json')

            @lru_cache(maxsize=None)
            def replay(differences, samples, seed):
                return interval(differences, samples, seed)

            with patch.object(check, 'interval', side_effect=lambda values, samples, seed: replay(tuple(values), samples, seed)):
                result = check.recount(run)
                self.assertTrue(result['screen_eligible'])
                self.assertEqual(result['counts'], {
                    'zero': {'primary': 445, 'audited': 446, 'unparsed': 3, 'truncated': 1},
                    'icl_original': {'primary': 478, 'audited': 479, 'unparsed': 2, 'truncated': 1},
                    'icl_complex': {'primary': 493, 'audited': 494, 'unparsed': 2, 'truncated': 1}})
                originals = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}

                def restore():
                    for name, content in originals.items():
                        (run/name).write_bytes(content)

                # Rehash mutated reports so semantic checks, not merely stale hashes,
                # must detect wrong counts, contrasts, adjusted p-values, and gates.
                for field in ['count', 'wins', 'ci', 'holm', 'gate']:
                    restore()
                    report = check.read(run/'screen-audit.json')
                    decision = check.read(run/'screen-selection.json')
                    if field == 'count':
                        report['summary']['icl_complex']['audited_completed_correct'] += 1
                    elif field == 'wins':
                        report['contrasts']['audited']['icl_complex-zero']['wins'] += 1
                    elif field == 'ci':
                        report['contrasts']['primary']['icl_original-zero']['ci95'][0] += .001
                    elif field == 'holm':
                        report['contrasts']['audited']['icl_complex-zero']['holm_p'] = .5
                    else:
                        decision['checks']['complex_gain'] = False
                        decision['eligible'] = False
                    save(run/'screen-audit.json', report)
                    decision['audit_sha256'] = check.sha(run/'screen-audit.json')
                    save(run/'screen-selection.json', decision)
                    with self.subTest(field=field), self.assertRaises(AssertionError):
                        check.recount(run)
                restore()
                batch = check.read(run/'batches/zero-0000.json')
                batch[0]['problem_id'] = plan['reserved'][0]
                save(run/'batches/zero-0000.json', batch)
                with self.assertRaises(AssertionError):
                    check.recount(run)
                restore()
                freeze = check.read(run/'review-freeze.json')
                freeze['annotations_commit'] = ''
                save(run/'review-freeze.json', freeze)
                with self.assertRaises(AssertionError):
                    check.recount(run)

    def test_exact_probability(self):
        self.assertEqual(check.probability(10, 0), 1/512)
        self.assertEqual(check.probability(0, 0), 1)
        self.assertEqual(check.probability(7, 7), 1)


if __name__ == '__main__':
    unittest.main()

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis import gsm8k_published_screen as screen
from analysis import published_screen_backend as backend


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(int(c)) for c in ids if int(c) != 0)

    def batch_decode(self, ids, **kwargs):
        return [self.decode(row) for row in ids]


class FakeBackend:
    def __init__(self, config):
        self.calls = 0
        self.runtime = {'physical_gpu': 0, 'frozen': True, 'dtype': 'torch.bfloat16', 'eos_token_ids': [0]}

    def records(self, prompts, config):
        self.calls += 1
        tok = CharacterTokenizer()
        return [backend.record(tok.encode('The answer is 1')+[0], None, [0], tok,
                               config['max_new_tokens'], 0.) for _ in prompts]


class PublishedScreenTests(unittest.TestCase):
    def test_boundaries_cut_later_answers_without_cutting_inline_mentions(self):
        for marker in ['Question:', 'q:', 'Problem:', 'Human:', 'User:',
                       'Given the following question,', 'Given the following problem ']:
            self.assertEqual(backend.solution('The answer is 7\n\t'+marker+'\nThe answer is 99'), 'The answer is 7')
            self.assertEqual(backend.solution(marker+' next'), '')
        self.assertEqual(backend.solution('I mention Question: within a sentence.'),
                         'I mention Question: within a sentence.')
        self.assertEqual(backend.solution('7\n'+' '*300+'Question: more'), '7')

    def test_boundary_padding_not_counted_and_short_length_stop_rejected(self):
        tok = CharacterTokenizer()
        raw = 'The answer is 7\nQuestion:'
        row = backend.record(tok.encode(raw)+[0]*20, len(raw), [0], tok, 1024, 0.)
        self.assertEqual(row['generated_tokens'], len(raw))
        self.assertEqual(row['solution_text'], 'The answer is 7')
        self.assertEqual(row['finish_reason'], 'next_question')
        with self.assertRaisesRegex(ValueError, 'length'):
            backend.record(tok.encode('unfinished'), None, [0], tok, 1024, 0.)

    def test_partition_is_disjoint_and_not_filtered_by_solution_length(self):
        config = dict(n_extract=2, n_screen=3, n_reserved=2, seed=3402)
        pool = [{'problem_id': f'gsm8k:train:{i}', 'calculation_annotations': 0} for i in range(7)]
        plan = screen.partition(pool, config)
        self.assertEqual([len(plan[k]) for k in ['extract', 'validation', 'reserved']], [2, 3, 2])
        self.assertEqual(set(sum(plan.values(), [])), {r['problem_id'] for r in pool})
        self.assertEqual(plan, screen.partition(pool, config))
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            screen.partition(pool[:-1], config)
        with self.assertRaisesRegex(ValueError, 'ordered'):
            screen.partition(pool[::-1], config)

    def test_prompts_keep_published_bytes_and_matched_query(self):
        banks = {'original': 'ORIGINAL\n', 'complex': 'COMPLEX\n'}
        prompts = screen.prompts('A question?', banks)
        query = "Question: A question?\nLet's think step by step\n"
        self.assertEqual(prompts, {'zero': query, 'icl_original': 'ORIGINAL\n\n'+query,
                                  'icl_complex': 'COMPLEX\n\n'+query})

    def fixture(self, output):
        config = screen.read(screen.CONFIG)
        problems = [{'problem_id': f'gsm8k:train:{i}', 'question': f'Question number {i}?', 'answer': '1',
                     'prompts': {k: f'{k} query {i}' for k in screen.CONDITIONS}} for i in range(512)]
        data = {'splits': {'validation': problems, 'extract': [{'problem_id': 'gsm8k:train:900'}],
                          'reserved': [{'problem_id': 'gsm8k:train:901'}]}}
        screen.fixed(output/'manifest.json', {'fixture': True})
        screen.fixed(output/'declaration.json', {'manifest_sha256': screen.sha(output/'manifest.json'),
                                               'declaration_commit': 'frozen-fixture'})
        return config, data

    def test_full_run_resume_preserves_batches_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            output = Path(tmp); config, data = self.fixture(output); fake = FakeBackend(config)
            with patch.object(screen, 'verify', return_value=(config, data)), \
                    patch.object(screen, 'tokenizer', return_value=CharacterTokenizer()), redirect_stdout(io.StringIO()):
                screen.run(output, lambda _: fake)
                self.assertEqual(fake.calls, 384)
                self.assertEqual(screen.read(output/'screen-complete.json')['rows'], 1536)
                before = {p.name: p.read_bytes() for p in (output/'batches').glob('*.json')}
                screen.run(output, lambda _: self.fail('Resume loaded a model'))
                self.assertEqual(before, {p.name: p.read_bytes() for p in (output/'batches').glob('*.json')})
                with self.assertRaises(FileNotFoundError):
                    screen.score(output, output/'annotations.json')
                packet = screen.read(output/'validation-review-packet.json')
                self.assertEqual(packet['items'], [])
                screen.fixed(output/'annotations.json', {'packet_sha256': screen.audit.digest(packet), 'answers': []})
                screen.fixed(output/'review-freeze.json', {'packet_sha256': screen.audit.digest(packet),
                    'annotations_file_sha256': screen.sha(output/'annotations.json'), 'annotations_commit': 'fixture'})
                screen.score(output, output/'annotations.json')
                self.assertFalse(screen.read(output/'screen-selection.json')['eligible'])
                report = screen.read(output/'screen-audit.json')
                self.assertEqual(len(report['contrasts']['audited']), 3)
                self.assertTrue(all(r['audited_completed_correct'] == 512 for r in report['summary'].values()))
                (output/'screen-selection.json').unlink()
                path = output/'batches/zero-0000.json'
                rows = screen.read(path); rows[0]['split'] = 'reserved'
                path.write_text(json.dumps(rows))
                with self.assertRaisesRegex(ValueError, 'Unexpected question'):
                    screen.run(output, lambda _: self.fail('Corrupt resume loaded a model'))

    def test_declaration_and_device_guards_precede_backend_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); config, data = self.fixture(output)
            with patch.object(screen, 'verify', return_value=(config, data)):
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
                    with self.assertRaisesRegex(ValueError, 'GPU 0'):
                        screen.run(output, lambda _: self.fail('Wrong GPU loaded'))
                (output/'declaration.json').write_text(json.dumps({'manifest_sha256': 'bad', 'declaration_commit': ''}))
                with self.assertRaisesRegex(ValueError, 'declaration'):
                    screen.run(output, lambda _: self.fail('Undeclared model loaded'))

    def test_row_checks_reject_followup_grade_and_token_accounting(self):
        config = screen.read(screen.CONFIG); tok = CharacterTokenizer()
        problem = {'problem_id': 'gsm8k:train:1', 'answer': '7', 'prompts': {'zero': 'query'}}
        raw = 'The answer is 7\nQuestion:'
        result = backend.record(tok.encode(raw), len(raw), [0], tok, 1024, 0.)
        row = {'problem_id': problem['problem_id'], 'split': 'validation', 'condition': 'zero',
               'prompt': 'query', 'answer': '7', **result, **screen.grade_answer(result['solution_text'], '7')}
        screen.check_batch([row], 'zero', [problem], config, {'eos_token_ids': [0]}, tok)
        for field, value, message in [('generated_tokens', 1, 'token accounting'),
                                      ('correct', False, 'grade'), ('solution_text', 'The answer is 99', 'boundary')]:
            changed = {**row, field: value}
            with self.assertRaisesRegex(ValueError, message):
                screen.check_batch([changed], 'zero', [problem], config, {'eos_token_ids': [0]}, tok)

    def test_fixed_target_gates_and_exact_tests(self):
        config = screen.read(screen.CONFIG)
        summary = {k: {'n': 512, 'primary_completed_correct': 400, 'truncated': 0} for k in screen.CONDITIONS}
        contrast = {'gain': 0.06, 'ci95': [0.01, 0.11], 'holm_p': 0.04}
        decision = screen.gates(summary, contrast, config)
        self.assertTrue(decision['eligible'])
        self.assertFalse(decision['reserved_generation_authorized'])
        self.assertFalse(decision['steering_generation_authorized'])
        self.assertFalse(screen.gates(summary, {**contrast, 'holm_p': 0.05}, config)['eligible'])
        self.assertFalse(screen.gates(summary, {**contrast, 'ci95': [0, 0.1]}, config)['eligible'])
        self.assertEqual(screen.exact_p(10, 0), 1/512)
        self.assertEqual(screen.holm({'a': .01, 'b': .03, 'c': .5}), {'a': .03, 'b': .06, 'c': .5})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU torch')
    def test_actual_stopping_callback_finishes_rows_independently(self):
        import torch
        tok = CharacterTokenizer()
        targets = [tok.encode('The answer is 7\nQuestion: ignored'),
                   tok.encode('The answer is 88888888888888888888')+[0]]
        seen_kwargs = {}

        def generate(input_ids, attention_mask, **kwargs):
            seen_kwargs.update(kwargs)
            finished = torch.zeros(2, dtype=torch.bool)
            for step in range(100):
                ids = [0 if finished[i] else targets[i][step] for i in range(2)]
                input_ids = torch.cat([input_ids, torch.tensor(ids).reshape(2, 1)], dim=1)
                finished |= kwargs['stopping_criteria'](input_ids, None) | torch.tensor([t == 0 for t in ids])
                if finished.all():
                    break
            return input_ids

        def encode(prompts):
            ids = torch.tensor([tok.encode('Question: prefix')]*2)
            return {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'position_ids': torch.zeros_like(ids)}

        instance = object.__new__(backend.PublishedBackend)
        instance.base = SimpleNamespace(encode=encode, model=SimpleNamespace(generate=generate))
        instance.tokenizer, instance.eos_ids = tok, [0]
        rows = instance.records(['a', 'b'], screen.read(screen.CONFIG))
        self.assertEqual([r['finish_reason'] for r in rows], ['next_question', 'eos'])
        self.assertEqual(rows[0]['text'], 'The answer is 7\nQuestion:')
        self.assertNotIn(0, rows[0]['token_ids'])
        self.assertEqual(rows[1]['token_ids'][-1], 0)
        self.assertFalse(seen_kwargs['do_sample'])


if __name__ == '__main__':
    unittest.main()

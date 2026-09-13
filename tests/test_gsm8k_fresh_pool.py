import json
from pathlib import Path
import tempfile
import unittest

from analysis.gsm8k_fresh_pool import digest, inventory, partition_inventory


class FreshPoolTests(unittest.TestCase):
    def test_prior_inventory_does_not_mark_unused_pool_as_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            pools = {'train': [{'problem_id': 'gsm8k:train:7', 'calculation_annotations': 4}], 'test': []}
            summary = {'no_sampling_or_generation': True, 'splits': {
                s: {'fresh': len(rows), 'identities_and_counts_sha256': digest(rows)}
                for s, rows in pools.items()}}
            (runs/'summary.json').write_text(json.dumps(summary))
            (runs/'fresh-pool.json').write_text(json.dumps(pools))
            (runs/'prepared.json').write_text(json.dumps({'problem_id': 'gsm8k:train:3'}))
            self.assertEqual(inventory(runs)[0], {'gsm8k:train:3'})
            pools['train'][0]['problem_id'] = 'gsm8k:train:8'
            (runs/'fresh-pool.json').write_text(json.dumps(pools))
            with self.assertRaisesRegex(ValueError, 'Invalid prior fresh-pool'):
                inventory(runs)

    def test_nested_ids_text_duplicates_and_reserved_exclusions(self):
        tables = {'train': [
            {'question': 'Previously extracted', 'answer': '<<1=1>>'},
            {'question': ' SHARED   question ', 'answer': '<<1=1>>'},
            {'question': 'Fresh question', 'answer': '<<1=1>><<2=2>><<3=3>><<4=4>>'},
            {'question': 'fresh QUESTION', 'answer': '<<1=1>>'}],
            'test': [{'question': 'Reserved', 'answer': '<<1=1>>'},
                     {'question': 'Fresh question', 'answer': '<<1=1>>'}]}
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            (runs/'inputs.json').write_text(json.dumps({'ids': ['gsm8k:train:0'],
                'bank': [{'question': 'shared question'}]}))
            (runs/'generations.jsonl').write_text(json.dumps({'problem_id': 'gsm8k:train:0',
                'condition': 'zero', 'text': 'not inspected', 'correct': False})+'\n')
            ids, questions, generated, files = inventory(runs)
            result, pools = partition_inventory(tables, ids, questions, generated,
                {'old': {'gsm8k:test:0'}}, prefix=0)
            self.assertEqual(len(files), 2)
            self.assertEqual(generated, {'gsm8k:train:0'})
            self.assertEqual(pools['train'], [{'problem_id': 'gsm8k:train:2', 'calculation_annotations': 4}])
            self.assertEqual(pools['test'], [])
            self.assertEqual(result['splits']['train']['fresh_at_least_four_annotations'], 1)
            self.assertEqual(result['splits']['train']['excluded']['fresh_text_duplicate'], 1)
            with self.assertRaisesRegex(ValueError, 'Reserved questions already generated'):
                partition_inventory(tables, ids, questions, generated,
                    {'bad': {'gsm8k:train:0'}}, prefix=0)
            with self.assertRaisesRegex(ValueError, 'absent from pinned dataset'):
                partition_inventory(tables, {'gsm8k:train:99'}, set(), set(), {}, prefix=0)

    def test_first_rows_and_their_cross_split_duplicates_excluded(self):
        tables = {'train': [{'question': 'prefix', 'answer': ''}, {'question': 'fresh', 'answer': ''}],
                  'test': [{'question': 'other prefix', 'answer': ''}, {'question': 'PREFIX', 'answer': ''}]}
        result, pools = partition_inventory(tables, set(), set(), set(), {}, prefix=1)
        self.assertEqual(len(pools['train']), 1)
        self.assertEqual(pools['test'], [])
        self.assertEqual(result['splits']['train']['calculation_annotation_histogram'], {0: 1})


if __name__ == '__main__':
    unittest.main()

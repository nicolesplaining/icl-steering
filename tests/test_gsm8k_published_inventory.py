import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import gsm8k_published_inventory as inventory


class PublishedInventoryTests(unittest.TestCase):
    def make_sources(self, root):
        reserved = {'old': ['gsm8k:train:153']}
        paths = [root/'local', root/'server']
        for path, index in zip(paths, [150, 151]):
            inventory.write_snapshot(path, {'ids': [f'gsm8k:train:{index}'],
                'generated_ids': [f'gsm8k:train:{index}'],
                'question_texts': ['train question 152']}, reserved, {'kind': 'single_checkout'})
        return paths

    def data(self):
        return {s: [{'question': f'{s} question {i}', 'answer': '<<1=1>>'}
                    for i in range(256)] for s in ['train', 'test']}

    def test_union_preserves_both_sources_and_text_and_reservations(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(inventory, 'tables', return_value=self.data()):
            root = Path(tmp)
            paths = self.make_sources(root)
            result = inventory.reconcile(paths, root/'union')
            pool = inventory.read(root/'union/fresh-pool.json')['train']
            ids = {r['problem_id'] for r in pool}
            self.assertEqual(len(ids), 124)
            self.assertFalse(ids & {f'gsm8k:train:{i}' for i in [150, 151, 152, 153]})
            self.assertEqual(result['generated_unique_ids'], 2)
            self.assertEqual(result['reservations']['old']['generated_overlap'], 0)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                inventory.reconcile(paths, root/'union')

    def test_changed_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(inventory, 'tables', return_value=self.data()):
            root = Path(tmp)
            paths = self.make_sources(root)
            file = paths[0]/'fresh-pool.json'
            pools = inventory.read(file)
            pools['train'][0]['problem_id'] = 'gsm8k:train:150'
            file.write_text(json.dumps(pools))
            with self.assertRaisesRegex(ValueError, 'pool does not replay'):
                inventory.reconcile(paths, root/'union')

    def test_reservation_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(inventory, 'tables', return_value=self.data()):
            root = Path(tmp)
            paths = self.make_sources(root)
            file = paths[1]/'reservations.json'
            reserved = {'old': ['gsm8k:train:154']}
            file.write_text(json.dumps(reserved))
            summary = inventory.read(paths[1]/'summary.json')
            summary['reservations_sha256'] = inventory.fresh.digest(reserved)
            (paths[1]/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'reservations disagree'):
                inventory.reconcile(paths, root/'union')


if __name__ == '__main__':
    unittest.main()

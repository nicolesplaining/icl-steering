import copy
import unittest

from analysis.complex_candidate_gates import CONDITIONS, NULL_CONTROLS, development_gate


def fixture():
    summary = {k: {'n': 256, 'primary_completed_correct': 180, 'audited_completed_correct': 210,
                   'unparsed': 30, 'truncated': 0} for k in CONDITIONS}
    summary['zero']['audited_completed_correct'] = 200
    summary['steered']['audited_completed_correct'] = 230
    return summary, {'gain': 30/256, 'ci95': [.06, .18]}


class CandidateGateTests(unittest.TestCase):
    def test_all_null_controls_and_prompt_only_must_be_strictly_worse(self):
        summary, contrast = fixture()
        good = development_gate(summary, contrast, True)
        self.assertTrue(good['eligible'])
        self.assertFalse(good['reserved_generation_authorized'])
        for control in [*NULL_CONTROLS, 'regularized_prefill']:
            with self.subTest(control=control):
                changed = copy.deepcopy(summary)
                changed[control]['audited_completed_correct'] = 230
                self.assertFalse(development_gate(changed, contrast, True)['eligible'])

    def test_each_text_cue_and_old_prefill_allow_ties_but_reject_losses(self):
        summary, contrast = fixture()
        for control in ['first', 'cot', 'prefill']:
            changed = copy.deepcopy(summary)
            changed[control]['audited_completed_correct'] = 230
            self.assertTrue(development_gate(changed, contrast, True)['eligible'])
            changed[control]['audited_completed_correct'] = 231
            self.assertFalse(development_gate(changed, contrast, True)['eligible'])

    def test_screen_gain_interval_parser_and_truncation_are_required(self):
        summary, contrast = fixture()
        self.assertFalse(development_gate(summary, contrast, False)['eligible'])
        self.assertFalse(development_gate(summary, {**contrast, 'ci95': [0, .18]}, True)['eligible'])
        changed = copy.deepcopy(summary)
        changed['zero']['audited_completed_correct'] = 223
        self.assertFalse(development_gate(changed, {'gain': 7/256, 'ci95': [.001, .1]}, True)['eligible'])
        changed = copy.deepcopy(summary)
        changed['steered']['primary_completed_correct'] = 179
        self.assertFalse(development_gate(changed, contrast, True)['eligible'])
        changed = copy.deepcopy(summary)
        changed['steered']['truncated'] = 12
        self.assertTrue(development_gate(changed, contrast, True)['eligible'])
        changed['steered']['truncated'] = 13
        self.assertFalse(development_gate(changed, contrast, True)['eligible'])

    def test_incomplete_or_inconsistent_evidence_is_rejected(self):
        summary, contrast = fixture()
        changed = copy.deepcopy(summary); del changed['permuted']
        with self.assertRaisesRegex(ValueError, '17 declared'):
            development_gate(changed, contrast, True)
        changed = copy.deepcopy(summary); changed['steered']['n'] = 255
        with self.assertRaisesRegex(ValueError, '256 development'):
            development_gate(changed, contrast, True)
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            development_gate(summary, {**contrast, 'gain': .5}, True)


if __name__ == '__main__':
    unittest.main()

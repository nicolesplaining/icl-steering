from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.gsm8k_prefix_controls import control_inputs
from replication.gsm8k_protocol import HEAD, matched_prompt


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [999] if text == " the" else list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i+1) for i in range(len(text))]}


def test_prefix_controls_preserve_query_positions_and_shuffle_only_demonstrations():
    tokenizer = CharacterTokenizer()
    bank = [{"question": "support one", "raw_response": " The answer is 12."},
            {"question": "support two", "raw_response": " The answer is 345."}]
    question = "What is the total?"
    variants, check = control_inputs(tokenizer, question, bank)
    prompt = matched_prompt(question, bank)
    start, end = len(HEAD), len(prompt)-len(f"Q: {question}\nA:")
    original = variants["icl_a"]
    for kind in ["token_shuffle", "length_filler"]:
        assert len(variants[kind]) == len(original)
        assert variants[kind][:start] == original[:start]
        assert variants[kind][end:] == original[end:]
    assert variants["token_shuffle"] != original
    assert Counter(variants["token_shuffle"]) == Counter(original)
    assert set(variants["length_filler"][start:end]) == {999}
    assert check["prefix_tokens_changed"] == end-start
    assert check["rotated_length_delta"] == 0
    assert control_inputs(tokenizer, question, bank)[0] == variants
    rotated_text = "".join(map(chr, variants["rotated_pairs"]))
    assert "Q: support one\nA: The answer is 345." in rotated_text

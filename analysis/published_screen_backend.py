"""Frozen inference with per-sequence boundaries for released Question: prompts."""

import os
import re
import time


BOUNDARY = re.compile(r'(?:^|\n)[ \t]*(?:Question:|Q:|Problem:|Human:|User:|'
                      r'Given the following (?:question|problem)[, ])', re.IGNORECASE)


def solution(text):
    return BOUNDARY.split(text, maxsplit=1)[0].rstrip()


def record(tokens, boundary_length, eos_ids, tokenizer, limit, seconds):
    end = next((i for i, t in enumerate(tokens) if t in eos_ids), None)
    if boundary_length is not None and (end is None or boundary_length <= end):
        actual, reason = tokens[:boundary_length], 'next_question'
    elif end is not None:
        actual, reason = tokens[:end+1], 'eos'
    else:
        actual, reason = tokens, 'length'
    if not actual or len(actual) > limit or (reason == 'length' and len(actual) != limit):
        raise ValueError('Invalid generation length')
    text = tokenizer.decode(actual, skip_special_tokens=True)
    if (reason == 'next_question') != bool(BOUNDARY.search(text)):
        raise ValueError('Boundary stop mismatch')
    return {'text': text, 'solution_text': solution(text), 'token_ids': actual,
            'generated_tokens': len(actual), 'finish_reason': reason,
            'truncated': reason == 'length', 'batch_seconds': seconds}


class PublishedBackend:
    def __init__(self, config):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('Use physical GPU 0 only')
        import torch
        import transformers
        from replication.gsm8k_activation import ActivationModel
        torch.manual_seed(config['seed'])
        self.base = ActivationModel(config['model'], config['revision'], config['max_model_len'])
        self.tokenizer = self.base.tokenizer
        eos = self.base.model.generation_config.eos_token_id
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        frozen = not any(p.requires_grad for p in self.base.model.parameters())
        if not frozen or self.base.model.training:
            raise ValueError('Require frozen evaluation model')
        self.runtime = {'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                        'eos_token_ids': self.eos_ids, 'pad_token_id': self.tokenizer.pad_token_id,
                        'dtype': str(self.base.model.dtype), 'frozen': frozen, 'physical_gpu': 0}

    def records(self, prompts, config):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        inputs = self.base.encode(prompts)
        inputs.pop('position_ids')
        width = inputs['input_ids'].shape[1]
        if width + config['max_new_tokens'] > config['max_model_len']:
            raise ValueError('Prompt plus answer exceeds context; no truncation')
        tokenizer = self.tokenizer
        lengths = [None]*len(prompts)

        class QuestionStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                # Full generated text preserves true line starts, including long
                # indentation. Prompt demonstrations never enter this search.
                text = tokenizer.batch_decode(input_ids[:, width:], skip_special_tokens=True)
                for i, value in enumerate(text):
                    if lengths[i] is None and BOUNDARY.search(value):
                        lengths[i] = input_ids.shape[1]-width
                return torch.tensor([n is not None for n in lengths], device=input_ids.device)

        start = time.monotonic()
        with torch.inference_mode():
            sequences = self.base.model.generate(**inputs, max_new_tokens=config['max_new_tokens'],
                do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([QuestionStop()]))
        seconds = time.monotonic()-start
        return [record(tokens, length, self.eos_ids, tokenizer, config['max_new_tokens'], seconds)
                for tokens, length in zip(sequences[:, width:].tolist(), lengths)]

"""Frozen-model inference and explicit, scoped decoder-block interventions."""

from contextlib import contextmanager
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def hidden_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


@contextmanager
def steering_hook(block, vector, strength, positions="prefill"):
    """A new context resets prefill detection for every generation batch.

    Last-position intervention is correct for left-padded prompts and cached decode.
    No other token positions, blocks, weights, or stored past keys are edited.
    """
    if positions not in {"prefill", "all"}:
        raise ValueError(positions)
    calls = 0

    def hook(_module, _inputs, output):
        nonlocal calls
        active = calls == 0 or positions == "all"
        calls += 1
        if not active or strength == 0:
            return output
        hidden = hidden_tensor(output).clone()
        delta = vector.to(device=hidden.device, dtype=hidden.dtype) * strength
        hidden[:, -1, :] += delta
        return replace_hidden(output, hidden)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class FrozenModel:
    def __init__(self, config):
        self.config = config
        torch.manual_seed(config["seed"])
        if not torch.cuda.is_available():
            raise RuntimeError("This runner expects a CUDA GPU. Data generation and scoring run on CPU.")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"], revision=config["revision"], torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0},
        ).eval()
        self.model.requires_grad_(False)
        self.blocks = self.model.model.layers
        for layer in config["layers"]:
            if not 0 <= layer < len(self.blocks):
                raise ValueError(f"Invalid decoder block index {layer}")

    def render(self, messages, first=False):
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.config.get("enable_thinking", False),
        )
        return rendered + ("First," if first else "")

    def encode(self, prompts):
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        lengths = encoded["attention_mask"].sum(1).tolist()
        if max(lengths) > self.config["max_prompt_tokens"]:
            raise ValueError(f"Prompt exceeds configured limit: {max(lengths)}; no silent truncation allowed")
        return encoded.to(self.model.device), lengths

    @torch.inference_mode()
    def activations(self, prompts):
        captured = {}
        handles = []
        for layer in self.config["layers"]:
            def capture(_m, _args, output, layer=layer):
                captured[layer] = hidden_tensor(output)[:, -1, :].float().cpu()
            handles.append(self.blocks[layer].register_forward_hook(capture))
        try:
            inputs, _ = self.encode(prompts)
            # Match generate's position IDs for left-padded batches.
            position_ids = inputs["attention_mask"].long().cumsum(-1) - 1
            position_ids.masked_fill_(inputs["attention_mask"] == 0, 1)
            # The backbone avoids materializing vocabulary logits for every prompt token.
            self.model.model(**inputs, position_ids=position_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    @torch.inference_mode()
    def generate(self, prompts, vector=None, layer=None, strength=0.0):
        inputs, lengths = self.encode(prompts)
        start = time.monotonic()
        kwargs = dict(max_new_tokens=self.config["max_new_tokens"], do_sample=False,
                      temperature=None, top_p=None, top_k=None, use_cache=True,
                      pad_token_id=self.tokenizer.pad_token_id)
        if vector is None:
            sequences = self.model.generate(**inputs, **kwargs)
        else:
            with steering_hook(self.blocks[layer], vector, strength, self.config["intervention_positions"]):
                sequences = self.model.generate(**inputs, **kwargs)
        elapsed = time.monotonic() - start
        generated = sequences[:, inputs["input_ids"].shape[1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos])
        outputs = []
        for ids, length in zip(generated, lengths):
            stop = next((i for i, token in enumerate(ids) if token in stop_ids), None)
            actual = ids if stop is None else ids[:stop+1]
            outputs.append({
                "text": self.tokenizer.decode(actual, skip_special_tokens=True),
                "generated_tokens": len(actual), "prompt_tokens": length,
                "truncated": stop is None and len(actual) >= self.config["max_new_tokens"],
                "batch_seconds": elapsed,
            })
        return outputs

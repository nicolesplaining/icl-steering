"""Measure and causally test an ICL-minus-zero activation direction on GSM8K.

The support bank is produced by ``gsm8k_joint_inference.py`` on the GSM8K
training split.  Questions used to fit the direction and questions used for
causal evaluation are disjoint test examples.  Activation tensors and model
weights stay in the run directory and are not repository artifacts.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import random
import re
from typing import Any

import torch


def load_joint_module() -> Any:
    path = Path(__file__).with_name("gsm8k_joint_inference.py")
    spec = importlib.util.spec_from_file_location("gsm8k_joint_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hidden_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


class ActivationModel:
    def __init__(self, model_name: str, revision: str, max_prompt_tokens: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()
        self.model.requires_grad_(False)
        self.blocks = self.model.model.layers
        self.max_prompt_tokens = max_prompt_tokens

    def encode(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        lengths = encoded["attention_mask"].sum(1)
        if int(lengths.max()) > self.max_prompt_tokens:
            raise ValueError(f"prompt exceeds {self.max_prompt_tokens} tokens")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        position_ids = encoded["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(encoded["attention_mask"] == 0, 1)
        encoded["position_ids"] = position_ids
        return encoded

    @torch.inference_mode()
    def activations(self, prompts: list[str], layers: list[int], batch_size: int) -> dict[int, torch.Tensor]:
        captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            handles = []
            for layer in layers:
                def capture(_module: Any, _args: Any, output: Any, layer: int = layer) -> None:
                    captured[layer].append(hidden_tensor(output)[:, -1, :].float().cpu())

                handles.append(self.blocks[layer].register_forward_hook(capture))
            try:
                inputs = self.encode(batch)
                self.model.model(**inputs, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
        return {layer: torch.cat(values, dim=0) for layer, values in captured.items()}

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int,
        layer: int | None = None,
        direction: torch.Tensor | None = None,
        strength: float = 0.0,
    ) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            inputs = self.encode(batch)
            # Let generate recompute cached decode positions; explicit positions
            # are only needed for the left-padded activation forward pass.
            inputs.pop("position_ids", None)
            hook_handle = None
            calls = 0
            if layer is not None and direction is not None and strength != 0:
                vector = direction.to(device=self.model.device, dtype=self.model.dtype)

                def steer(_module: Any, _args: Any, output: Any) -> Any:
                    nonlocal calls
                    active = calls == 0
                    calls += 1
                    if not active:
                        return output
                    hidden = hidden_tensor(output).clone()
                    hidden[:, -1, :] += vector * strength
                    return replace_hidden(output, hidden)

                hook_handle = self.blocks[layer].register_forward_hook(steer)
            try:
                sequences = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            finally:
                if hook_handle is not None:
                    hook_handle.remove()
            generated = sequences[:, inputs["input_ids"].shape[1] :]
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs


def pair_stats(differences: torch.Tensor) -> dict[str, float]:
    centered = differences - differences.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    return {
        "rank1_centered_energy": float((energy[0] / energy.sum()).item()) if energy.sum() else 0.0,
        "mean_difference_norm": float(differences.mean(dim=0).norm().item()),
        "mean_example_difference_norm": float(differences.norm(dim=1).mean().item()),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--revision", default="b101308fe89651ea5ce025f25317fea6fc07e96e")
    parser.add_argument("--fit-examples", type=int, default=64)
    parser.add_argument("--test-examples", type=int, default=32)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--layers", default="0,7,13,20,27")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--strengths", default="-2,-1,0,1,2")
    parser.add_argument("--tensor-output", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    joint = load_joint_module()
    from datasets import load_dataset

    support_record = json.loads(Path(args.support_json).read_text())
    support = [row for row in support_record["adaptation_support"] if row["formatted"]]
    if len(support) < args.shots:
        raise ValueError(f"support bank has only {len(support)} usable rows")
    dataset = load_dataset("gsm8k", "main")
    test_rows = joint.prepare_examples([dict(row) for row in dataset["test"]])
    fit_rows = test_rows[: args.fit_examples]
    eval_rows = test_rows[args.fit_examples : args.fit_examples + args.test_examples]
    rng = random.Random(args.seed)
    fit_zero = [joint.zero_prompt(row) for row in fit_rows]
    fit_icl = [joint.icl_prompt(support, row, args.shots, rng) for row in fit_rows]
    eval_zero = [joint.zero_prompt(row) for row in eval_rows]
    eval_icl = [joint.icl_prompt(support, row, args.shots, rng) for row in eval_rows]
    layers = [int(item) for item in args.layers.split(",") if item]
    strengths = [float(item) for item in args.strengths.split(",") if item]

    model = ActivationModel(args.model, args.revision, args.max_prompt_tokens)
    fit_zero_acts = model.activations(fit_zero, layers, args.batch_size)
    fit_icl_acts = model.activations(fit_icl, layers, args.batch_size)
    eval_zero_acts = model.activations(eval_zero, layers, args.batch_size)
    eval_icl_acts = model.activations(eval_icl, layers, args.batch_size)

    directions: dict[int, torch.Tensor] = {}
    stats: dict[str, Any] = OrderedDict()
    for layer in layers:
        differences = fit_icl_acts[layer] - fit_zero_acts[layer]
        direction = differences.mean(dim=0)
        directions[layer] = direction
        half = max(1, len(differences) // 2)
        stats[str(layer)] = {
            **pair_stats(differences),
            "split_mean_cosine": cosine(differences[:half].mean(0), differences[half:].mean(0)),
            "fit_zero_norm": float(fit_zero_acts[layer].norm(dim=1).mean().item()),
            "fit_icl_norm": float(fit_icl_acts[layer].norm(dim=1).mean().item()),
            "eval_zero_norm": float(eval_zero_acts[layer].norm(dim=1).mean().item()),
            "eval_icl_norm": float(eval_icl_acts[layer].norm(dim=1).mean().item()),
        }

    tensor_path = Path(args.tensor_output)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"directions": directions, "layers": layers, "fit_examples": args.fit_examples}, tensor_path)

    def accuracy(texts: list[str], rows: list[dict[str, Any]]) -> float:
        parsed = [joint.extract_answer(text) for text in texts]
        return sum(answer != "" and answer == row["answer"] for answer, row in zip(parsed, rows)) / len(rows)

    results: dict[str, Any] = {
        "args": vars(args),
        "fit_size": len(fit_rows),
        "test_size": len(eval_rows),
        "layers": stats,
        "baseline_accuracy": accuracy(model.generate(eval_zero, args.max_new_tokens, args.batch_size), eval_rows),
        "icl_accuracy": accuracy(model.generate(eval_icl, args.max_new_tokens, args.batch_size), eval_rows),
        "steering": {},
    }
    for layer in layers:
        direction = directions[layer]
        direction = direction / direction.norm()
        layer_results = {}
        for multiplier in strengths:
            # Scale by the fitted mean shift norm, so +/-1 means one observed ICL shift.
            scale = stats[str(layer)]["mean_difference_norm"] * multiplier
            texts = model.generate(eval_zero, args.max_new_tokens, args.batch_size, layer, direction, scale)
            layer_results[str(multiplier)] = {"accuracy": accuracy(texts, eval_rows), "scale": scale}
        results["steering"][str(layer)] = layer_results
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2), flush=True)

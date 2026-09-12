"""Extraction-only diagnostics for content and length in the ICL mean shift."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from replication.gsm8k_activation import ActivationModel, hidden_tensor
from replication.gsm8k_protocol import HEAD, matched_prompt
from replication.gsm8k_steering import geometry, save, source_hash


def control_inputs(tokenizer, question, bank, seed=907):
    prompt = matched_prompt(question, bank)
    query_start = len(prompt) - len(f"Q: {question}\nA:")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    # Tokens crossing a header/query boundary are kept unchanged.
    indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
               if len(HEAD) <= a < b <= query_start]
    if not indices:
        raise ValueError("No demonstration tokens found")
    values = [ids[i] for i in indices]
    random.Random(seed).shuffle(values)
    shuffled, filler = list(ids), list(ids)
    fill_id = tokenizer.encode(" the", add_special_tokens=False)
    if len(fill_id) != 1:
        raise ValueError("Filler must be exactly one token")
    for i, value in zip(indices, values):
        shuffled[i], filler[i] = value, fill_id[0]
    rotated = [dict(question=d["question"], raw_response=bank[(i+1) % len(bank)]["raw_response"])
               for i, d in enumerate(bank)]
    variants = {"zero": tokenizer.encode(matched_prompt(question), add_special_tokens=False),
                "icl_a": ids, "token_shuffle": shuffled, "length_filler": filler,
                "rotated_pairs": tokenizer.encode(matched_prompt(question, rotated), add_special_tokens=False)}
    return variants, {"prefix_tokens_changed": len(indices), "icl_tokens": len(ids),
                      "rotated_length_delta": len(variants["rotated_pairs"])-len(ids)}


@torch.inference_mode()
def capture_ids(model, sequences, layers, batch_size):
    captured = {layer: [] for layer in layers}
    for start in range(0, len(sequences), batch_size):
        inputs = model.tokenizer.pad({"input_ids": sequences[start:start+batch_size]},
                                     padding=True, return_tensors="pt", return_attention_mask=True)
        inputs = {k: v.to(model.model.device) for k, v in inputs.items()}
        positions = inputs["attention_mask"].long().cumsum(-1)-1
        positions.masked_fill_(inputs["attention_mask"] == 0, 1)
        inputs["position_ids"] = positions
        handles = []
        for layer in layers:
            def capture(_module, _args, output, layer=layer):
                captured[layer].append(hidden_tensor(output)[:, -1].float().cpu())
            handles.append(model.blocks[layer].register_forward_hook(capture))
        try:
            model.model.model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
    return {layer: torch.cat(rows) for layer, rows in captured.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "manifest.json").exists():
            raise ValueError("Control run already exists; inspect it before starting another")
        manifest = json.loads((args.run / "manifest.json").read_text())
        prepared_bytes = (args.run / "prepared.json").read_bytes()
        if sha256(prepared_bytes).hexdigest() != manifest["prepared_sha256"]:
            raise ValueError("Prepared data changed")
        if source_hash() != manifest["source_sha256"]:
            raise ValueError("Primary experiment source changed")
        data = json.loads(prepared_bytes)
        config = manifest["config"]
        reference_path = args.run / "directions.pt"
        reference_hash = sha256(reference_path.read_bytes()).hexdigest()
        if reference_hash != json.loads((args.run / "geometry.json").read_text())["tensor_sha256"]:
            raise ValueError("Reference directions changed")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable")
        torch.set_num_threads(8)
        model = ActivationModel(config["model"], config["revision"], config["max_model_len"])
        sequences, checks = {}, []
        ids = []
        for row in data["splits"]["extract"]:
            variants, check = control_inputs(model.tokenizer, row["question"], data["banks"]["icl_a"])
            ids.append(row["problem_id"])
            checks.append(check)
            for kind, tokens in variants.items():
                if len(tokens) > config["max_model_len"]:
                    raise ValueError("Control prompt exceeds context")
                sequences.setdefault(kind, []).append(tokens)
        save(args.output / "inputs.json", {"problem_ids": ids, "token_ids": sequences})
        provenance = {"model": config["model"], "revision": config["revision"],
                      "seed": 907, "n_extract": len(ids), "input_checks": checks,
                      "inputs_sha256": sha256((args.output / "inputs.json").read_bytes()).hexdigest(),
                      "primary_source_sha256": manifest["source_sha256"],
                      "prepared_sha256": manifest["prepared_sha256"],
                      "reference_tensor_sha256": reference_hash,
                      "control_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                      "gpu": torch.cuda.get_device_name(0)}
        save(args.output / "manifest.json", provenance)
        activations = {}
        for kind, tokens in sequences.items():
            activations[kind] = capture_ids(model, tokens, config["layers"], config["batch_size"])
            torch.save(activations[kind], args.output / f"{kind}.pt")
            print(f"Captured {kind}: {len(tokens)} extraction questions", flush=True)
        saved = torch.load(reference_path, map_location="cpu", weights_only=True)
        metrics, replay = {}, {}
        for layer in config["layers"]:
            real, real_stats = geometry(activations["icl_a"][layer], activations["zero"][layer])
            replay[layer] = (real-saved[layer]).abs().max().item()
            torch.testing.assert_close(real, saved[layer], atol=1e-3, rtol=1e-4)
            metrics[layer] = {"icl_a": real_stats}
            for kind in ("rotated_pairs", "token_shuffle", "length_filler"):
                mean, stat = geometry(activations[kind][layer], activations["zero"][layer])
                metrics[layer][kind] = {**stat,
                    "cosine_to_real_icl": torch.nn.functional.cosine_similarity(mean[None], real[None]).item(),
                    "norm_ratio_to_real": (mean.norm()/real.norm()).item(),
                    "relative_distance_to_real": ((mean-real).norm()/real.norm()).item()}
        save(args.output / "metrics.json", {"status": "extraction_controls_complete", "provenance": provenance,
             "primary_replay_max_abs_error": replay, "layers": metrics,
             "limitations": "Extraction geometry only; no accuracy inference or test labels used. "
             "Token shuffle and filler keep all query positions exactly fixed, but are unnatural "
             "contexts. Rotated pairs retain the question and solution inventory but break their "
             "correspondence; their token-length differences are recorded. These diagnostics "
             "cannot alone establish a causal mathematical benefit."})
        print("Extraction controls complete", flush=True)


if __name__ == "__main__":
    main()

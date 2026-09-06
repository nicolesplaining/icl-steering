"""Matched GSM8K activation experiment with validation and locked test stages."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import random
import time

import torch

from icl_steering.model import steering_hook
from icl_steering.report import paired_difference
from replication.gsm8k_activation import ActivationModel
from replication.gsm8k_protocol import (NEXT_QUESTION, digest, grade_answer, matched_prompt,
                                      partition, select_candidate, solution_text)


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def source_hash():
    base = Path(__file__).parents[1]
    paths = [*sorted((base / "replication").glob("*.py")),
             base / "src/icl_steering/model.py", base / "src/icl_steering/scoring.py",
             base / "src/icl_steering/report.py"]
    return digest({str(p.relative_to(base)): sha256(p.read_bytes()).hexdigest() for p in paths})


def prepare(config, support_path, output):
    """Prepare all identities and prompts before inference, including unseen test IDs."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    support_hash = sha256(support_path.read_bytes()).hexdigest()
    existing = output / "manifest.json"
    if existing.exists():
        manifest = json.loads(existing.read_text())
        if (manifest["requested_config"] != config or manifest["source_sha256"] != source_hash()
                or manifest["support_sha256"] != support_hash):
            raise ValueError("Inputs or code changed; create a new output directory")
        data_path = output / "prepared.json"
        if sha256(data_path.read_bytes()).hexdigest() != manifest["prepared_sha256"]:
            raise ValueError("Prepared data changed")
        return manifest["config"], json.loads(data_path.read_text())

    if config["revision"] == "main":
        raise ValueError("Pin the model revision")
    for key in ("max_new_tokens", "batch_size", "shots"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if any(a <= 0 for a in config["strengths"]):
        raise ValueError("Only positive directions enter the validation grid")
    if len(set(config["random_seeds"])) < 3:
        raise ValueError("Use at least three independent random control directions")
    if set(config["positions"]) - {"prefill", "all"}:
        raise ValueError("Unknown intervention scope")
    record = json.loads(support_path.read_text())
    if record["mode"] != "heldout" or record["model"] != config["model"]:
        raise ValueError("Require a train-only support bank from this model")
    if record["args"]["adaptation_examples"] > config["reserved_train"]:
        raise ValueError("Reserved training prefix does not cover support adaptation")
    support = [dict(question=x["question"], raw_response=solution_text(x["raw_response"]))
               for x in record["adaptation_support"] if x["formatted"]]
    if len({x["question"] for x in support}) != len(support):
        raise ValueError("Duplicate support examples")
    if len(support) < config["shots"] * 2:
        raise ValueError("Need two disjoint support banks")
    requested_config = dict(config)
    config = dict(config)
    if config["dataset_revision"] is None:
        config["dataset_revision"] = HfApi().dataset_info(config["dataset"]).sha
    tables, files = {}, {}
    for split in ("train", "test"):
        name = f"main/{split}-00000-of-00001.parquet"
        path = Path(hf_hub_download(config["dataset"], name, repo_type="dataset",
                                    revision=config["dataset_revision"]))
        tables[split] = pq.read_table(path).to_pylist()
        files[name] = sha256(path.read_bytes()).hexdigest()
    # Verify train provenance against the actual dataset, not just a mode label.
    train_support = {x["question"] for x in tables["train"][:config["reserved_train"]]}
    if any(x["question"] not in train_support for x in support):
        raise ValueError("Support question is outside the reserved training pool")
    plan = partition(tables["train"], tables["test"], [x["question"] for x in support], config)
    shuffled = list(support)
    random.Random(config["seed"]).shuffle(shuffled)
    banks = {"icl_a": shuffled[:config["shots"]], "icl_b": shuffled[config["shots"]:2*config["shots"]]}
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    prepared = {"splits": {}, "banks": banks, "plan": plan}
    for split, ids in plan.items():
        data = tables["test" if split == "test" else "train"]
        prepared["splits"][split] = []
        for i in ids:
            question = data[i]["question"]
            gold = data[i]["answer"].split("####")[-1].strip().replace(",", "")
            prompts = {"zero": matched_prompt(question),
                       "first": matched_prompt(question, suffix=" First,"),
                       "cot": matched_prompt(question, suffix=" Let's think step by step."),
                       **{key: matched_prompt(question, bank) for key, bank in banks.items()}}
            for prompt in prompts.values():
                n = len(tokenizer.encode(prompt, add_special_tokens=False))
                if n + config["max_new_tokens"] > config["max_model_len"]:
                    raise ValueError(f"Context overflow for {split}/{i}; no silent truncation")
            prepared["splits"][split].append({"problem_id": f"gsm8k:{'test' if split == 'test' else 'train'}:{i}",
                                             "question": question, "answer": gold, "prompts": prompts})
    save(output / "prepared.json", prepared)
    save(existing, {"requested_config": requested_config, "config": config,
                    "source_sha256": source_hash(), "support_sha256": support_hash,
                    "prepared_sha256": sha256((output / "prepared.json").read_bytes()).hexdigest(),
                    "data_sha256": files, "created_at": datetime.now(timezone.utc).isoformat(),
                    "selection_rule": "Validation only; test>=128 held back until selection.json is saved.",
                    "support_note": "Frozen legacy pseudo-label bank; not an exact replication of upstream UICL."})
    return config, prepared


class Backend(ActivationModel):
    @torch.inference_mode()
    def records(self, prompts, config, direction=None, layer=None, alpha=0., positions="prefill"):
        from contextlib import nullcontext
        from transformers import StoppingCriteria, StoppingCriteriaList

        inputs = self.encode(prompts)
        inputs.pop("position_ids")
        width = inputs["input_ids"].shape[1]
        if width + config["max_new_tokens"] > config["max_model_len"]:
            raise ValueError("Prompt plus output exceeds context")
        tokenizer = self.tokenizer

        class NextQuestionStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                # Mark each sequence finished independently. Decode only a short
                # tail; full text is retained and cut at the delimiter on scoring.
                tail = input_ids[:, max(width, input_ids.shape[1] - 32):]
                texts = tokenizer.batch_decode(tail, skip_special_tokens=True)
                return torch.tensor([bool(NEXT_QUESTION.search(t)) for t in texts],
                                    device=input_ids.device)

        context = (steering_hook(self.blocks[layer], direction, alpha, positions)
                   if direction is not None else nullcontext())
        start = time.monotonic()
        with context:
            sequences = self.model.generate(**inputs, max_new_tokens=config["max_new_tokens"],
                do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([NextQuestionStop()]))
        eos = self.model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos])
        results = []
        for tokens in sequences[:, width:].tolist():
            # Custom stopping can pad completed rows; count only actual output.
            end = next((i for i, token in enumerate(tokens) if token in stops or token == tokenizer.pad_token_id), None)
            actual = tokens if end is None else tokens[:end+1]
            raw = tokenizer.decode(actual, skip_special_tokens=True)
            next_question = NEXT_QUESTION.search(raw) is not None
            reason = "next_question" if next_question else ("eos" if end is not None else "length")
            results.append({"text": raw, "solution_text": solution_text(raw),
                            "token_ids": actual, "generated_tokens": len(actual),
                            "finish_reason": reason, "truncated": reason == "length",
                            "batch_seconds": time.monotonic()-start})
        return results


def geometry(a, b):
    delta = (a - b).float()
    mean = delta.mean(0)
    total = delta.square().sum().clamp_min(1e-12)
    unit = mean / mean.norm().clamp_min(1e-12)
    centered = delta - mean
    sv = torch.linalg.svdvals(centered).square()
    half = len(delta) // 2
    cos = torch.nn.functional.cosine_similarity
    return mean, {"mean_norm": mean.norm().item(),
        "mean_shift_fraction": (len(delta)*mean.square().sum()/total).item(),
        "energy_along_mean": ((delta @ unit).square().sum()/total).item(),
        "mean_example_cosine": cos(delta, mean[None]).mean().item(),
        "split_half_cosine": cos(delta[:half].mean(0)[None], delta[half:].mean(0)[None]).item(),
        "centered_rank1_energy": (sv[0]/sv.sum().clamp_min(1e-12)).item()}


def stats(rows):
    if not rows:
        raise ValueError("No evaluation rows")
    n = len(rows)
    return {"n": n, "accuracy": sum(r["correct"] for r in rows)/n,
            "completed_accuracy": sum(r["completed_correct"] for r in rows)/n,
            "truncation_rate": sum(r["truncated"] for r in rows)/n,
            "parse_rate": sum(r["parseable"] for r in rows)/n}


class Runner:
    def __init__(self, config, data, output):
        self.config, self.data, self.output = config, data, output
        self.rows = []
        path = output / "generations.jsonl"
        if path.exists():
            self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        keys = [(r["split"], r["problem_id"], r["condition"]) for r in self.rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate generation records")
        allowed = {(split, p["problem_id"]) for split, values in data["splits"].items() for p in values}
        if any((r["split"], r["problem_id"]) not in allowed for r in self.rows):
            raise ValueError("Unexpected generation question")
        self.done = set(keys)
        self.model = None

    def backend(self):
        if self.model is None:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU unavailable; prepared run remains reusable")
            config = self.config
            torch.manual_seed(config["seed"])
            self.model = Backend(config["model"], config["revision"], config["max_model_len"])
            save(self.output / "runtime.json", {"versions": {k: version(k) for k in
                 ("torch", "transformers", "accelerate")}, "gpu": torch.cuda.get_device_name(0),
                 "cuda": torch.version.cuda})
        return self.model

    def evaluate(self, split, condition, prompt_kind="zero", direction=None, layer=None, alpha=0., positions="prefill"):
        problems = self.data["splits"][split]
        pending = [p for p in problems if (split, p["problem_id"], condition) not in self.done]
        size = self.config["batch_size"]
        for start in range(0, len(pending), size):
            batch = pending[start:start+size]
            prompts = [p["prompts"][prompt_kind] for p in batch]
            outputs = self.backend().records(prompts, self.config, direction, layer, alpha, positions)
            with (self.output / "generations.jsonl").open("a") as handle:
                for p, prompt, result in zip(batch, prompts, outputs):
                    row = {"split": split, "condition": condition, "problem_id": p["problem_id"],
                           "prompt": prompt, "answer": p["answer"], **result,
                           **grade_answer(result["text"], p["answer"], result["truncated"]),
                           "layer": layer, "alpha": alpha, "positions": positions}
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    self.rows.append(row)
                    self.done.add((split, p["problem_id"], condition))
            print(f"{split} {condition}: {min(start+size, len(pending))}/{len(pending)} new", flush=True)
        return [r for r in self.rows if r["split"] == split and r["condition"] == condition]

    def extract(self):
        path = self.output / "directions.pt"
        if path.exists():
            expected = json.loads((self.output / "geometry.json").read_text())["tensor_sha256"]
            if sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Direction tensors changed")
            return torch.load(path, map_location="cpu", weights_only=True)
        model, config = self.backend(), self.config
        values = {}
        for kind in ("zero", "icl_a", "icl_b"):
            prompts = [r["prompts"][kind] for r in self.data["splits"]["extract"]]
            values[kind] = model.activations(prompts, config["layers"], config["batch_size"])
            print(f"extracted {kind}", flush=True)
        directions, diagnostics = {}, {}
        for layer in config["layers"]:
            d, stat = geometry(values["icl_a"][layer], values["zero"][layer])
            db, _ = geometry(values["icl_b"][layer], values["zero"][layer])
            diagnostics[layer] = {**stat, "cross_bank_cosine":
                torch.nn.functional.cosine_similarity(d[None], db[None]).item()}
            directions[layer] = d
        torch.save(directions, path)
        save(self.output / "geometry.json", {"layers": diagnostics,
             "tensor_sha256": sha256(path.read_bytes()).hexdigest(),
             "note": "Uncentered metrics test a common shift; centered PCA alone does not establish one."})
        return directions

    def validation(self):
        path = self.output / "selection.json"
        if path.exists():
            return json.loads(path.read_text())
        base = {}
        for kind in ("zero", "icl_a", "icl_b", "first", "cot"):
            base[kind] = stats(self.evaluate("validation", kind, prompt_kind=kind))
        if (any(base[k]["truncation_rate"] > self.config["max_truncation_rate"] for k in base)
                or min(base[k]["completed_accuracy"] for k in ("icl_a", "icl_b"))
                   - base["zero"]["completed_accuracy"] < self.config["min_icl_gain"]):
            result = {"eligible": False, "reason": "Matched ICL gain or truncation gate failed", "baselines": base}
            save(path, result)
            return result
        directions = self.extract()
        candidates = []
        for layer in self.config["layers"]:
            for alpha in self.config["strengths"]:
                for positions in self.config["positions"]:
                    name = f"steer_l{layer}_a{alpha:g}_{positions}"
                    rows = self.evaluate("validation", name, direction=directions[layer],
                                         layer=layer, alpha=alpha, positions=positions)
                    candidates.append({"condition": name, "layer": layer, "alpha": alpha,
                                       "positions": positions, **stats(rows)})
        choice = select_candidate(candidates, base["zero"]["completed_accuracy"], self.config["min_steering_gain"])
        choice.update(baselines=base, candidates=candidates)
        save(path, choice)
        return choice

    def test(self, selection):
        if not selection["eligible"]:
            return
        # Freeze selection AND direction hashes before any test inference.
        lock = {"selection_sha256": sha256((self.output / "selection.json").read_bytes()).hexdigest(),
                "directions_sha256": sha256((self.output / "directions.pt").read_bytes()).hexdigest()}
        lock_path = self.output / "test_lock.json"
        if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
            raise ValueError("Selection changed after test began")
        save(lock_path, lock)
        for kind in ("zero", "icl_a", "icl_b", "first", "cot"):
            self.evaluate("test", kind, prompt_kind=kind)
        c = selection["chosen"]
        d = self.extract()[c["layer"]]
        controls = {"steered": d, "reverse": -d}
        for seed in self.config["random_seeds"]:
            r = torch.randn(d.shape, generator=torch.Generator().manual_seed(seed))
            controls[f"random_{seed}"] = r/r.norm()*d.norm()
        for name, direction in controls.items():
            self.evaluate("test", name, direction=direction, layer=c["layer"],
                          alpha=c["alpha"], positions=c["positions"])
        save(self.output / "complete.json", {"status": "locked_test_complete",
                                           "completed_at": datetime.now(timezone.utc).isoformat()})

    def report(self):
        groups = defaultdict(list)
        for row in self.rows:
            groups[row["split"], row["condition"]].append(row)
        results = []
        for (split, condition), rows in sorted(groups.items()):
            entry = {"split": split, "condition": condition, **stats(rows)}
            controls = ["zero"] if split == "validation" else ["zero", "first", "cot", "reverse"] + [
                f"random_{s}" for s in self.config["random_seeds"]]
            if split == "validation" or condition in ("steered", "icl_a", "icl_b"):
                entry["paired"] = {}
                for control in controls:
                    ref = groups.get((split, control))
                    if condition == control or not ref or {r["problem_id"] for r in rows} != {r["problem_id"] for r in ref}:
                        continue
                    entry["paired"][control] = paired_difference(
                        [{**r, "correct": r["completed_correct"]} for r in rows],
                        [{**r, "correct": r["completed_correct"]} for r in ref],
                        self.config["bootstrap_samples"], self.config["seed"])
            results.append(entry)
        save(self.output / "summary.json", {"results": results,
            "test_complete": (self.output / "complete.json").exists(),
            "primary_metric": "Completed explicit-answer accuracy on locked held-out test questions.",
            "note": "Pairwise exploratory intervals are unadjusted. No automated scientific success claim."})
        lines = ["# Corrected GSM8K steering experiment", "",
                 "Matched prompts, frozen support, separate extraction/validation/test.",
                 "Completed accuracy excludes generations that hit the token limit.", "",
                 "| Split | Condition | n | Completed accuracy | Any explicit answer | Truncated |",
                 "|---|---|---:|---:|---:|---:|"]
        lines += [f"| {r['split']} | {r['condition']} | {r['n']} | {r['completed_accuracy']:.1%} | {r['accuracy']:.1%} | {r['truncation_rate']:.1%} |" for r in results]
        (self.output / "report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "validate", "test", "run", "report"))
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_steering.json"))
    parser.add_argument("--support-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    # Kernel lock releases on exit or crash; a stale filename is not a live run.
    import fcntl
    with (args.output / ".writer.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, data = prepare(config, args.support_json, args.output)
        if args.stage == "prepare":
            print("Prepared frozen inputs; model inference has not run.", flush=True)
            return
        runner = Runner(config, data, args.output)
        try:
            if args.stage in ("validate", "run"):
                selection = runner.validation()
            elif args.stage == "test":
                selection = json.loads((args.output / "selection.json").read_text())
            if args.stage in ("run", "test"):
                runner.test(selection)
        finally:
            runner.report()


if __name__ == "__main__":
    main()

"""Staged pilot with disjoint screening, vector fitting, selection, and final evaluation."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .data import make_dataset, demonstrations
from .prompts import messages
from .scoring import grade, method_signature
from .report import aggregate, build_report


BASELINES = ("zero", "cot", "first", "instruction", "icl_a", "icl_b")


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_digest():
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=True)
    if config["revision"] == "main":
        raise ValueError("Pin an immutable model revision before running")
    dataset = make_dataset(config)
    encoded = "".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in dataset)
    identity = {"config": config, "source_sha256": source_digest(),
                "dataset_sha256": sha256(encoded.encode()).hexdigest()}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous[k] != v for k, v in identity.items()):
            raise ValueError("Config, dataset, or source changed: use a new output directory")
    else:
        save_json(manifest_path, {**identity, "created_at": datetime.now(timezone.utc).isoformat()})
        (output / "dataset.jsonl").write_text(encoded)
    (output / "generations.jsonl").touch(exist_ok=True)
    return dataset


def choose_family(rows, families, minimum_gain):
    scores = {}
    for family in families:
        subset = [r for r in rows if r["split"] == "screen" and r["family"] == family]
        zero = aggregate([r for r in subset if r["condition"] == "zero"])
        icl = aggregate([r for r in subset if r["condition"] == "icl_a"])
        scores[family] = {"zero": zero["accuracy"], "icl_a": icl["accuracy"],
                          "gain": icl["accuracy"] - zero["accuracy"]}
    # Config order resolves a tie. Test data never enters family selection.
    best = max(families, key=lambda family: scores[family]["gain"])
    return {"scores": scores, "minimum_gain": minimum_gain,
            "selected_family": best if scores[best]["gain"] >= minimum_gain else None,
            "rule": "Largest screen ICL-A minus zero accuracy; ties use config family order."}


def choose_candidate(candidates):
    # Accuracy first; a tie uses the smaller intervention and then shallower block.
    return max(candidates, key=lambda c: (c["accuracy"], -c["strength"], -c["layer"]))


class Runner:
    def __init__(self, config, output):
        from .model import FrozenModel
        self.config, self.output = config, output
        self.dataset = prepare(config, output)
        self.rows = [json.loads(line) for line in (output / "generations.jsonl").read_text().splitlines() if line.strip()]
        self.completed = {(r["problem_id"], r["condition"]) for r in self.rows}
        if len(self.completed) != len(self.rows):
            raise ValueError("Duplicate generation records")
        self.model = FrozenModel(config)
        import torch
        import transformers
        save_json(output / "runtime.json", {"torch": torch.__version__,
                  "transformers": transformers.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0)})

    def problems(self, family, split):
        return [p for p in self.dataset if p.family == family and p.split == split]

    def prompts(self, problems, condition):
        return [self.model.render(messages(p, demonstrations(p, self.dataset, self.config["shots"],
                       self.config["seed"]), condition), first=condition == "first") for p in problems]

    def evaluate(self, family, split, condition, vector=None, layer=None, strength=0.0):
        problems = self.problems(family, split)
        pending = [p for p in problems if (p.id, condition) not in self.completed]
        prompt_condition = condition if condition in BASELINES else "zero"
        size = self.config["batch_size"]
        for start in range(0, len(pending), size):
            batch = pending[start:start + size]
            prompts = self.prompts(batch, prompt_condition)
            outputs = self.model.generate(prompts, vector, layer, strength)
            with (self.output / "generations.jsonl").open("a") as handle:
                for p, prompt, result in zip(batch, prompts, outputs):
                    # Restore the provided prefix for faithful solution auditing.
                    text = ("First," if condition == "first" else "") + result["text"]
                    row = {"problem_id": p.id, "family": family, "split": split,
                           "condition": condition, "prompt": prompt, "answer": p.answer,
                           **result, "text": text, **grade(text, p.answer),
                           "method_signature": method_signature(text, family),
                           "layer": layer, "strength": strength}
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    self.rows.append(row)
                    self.completed.add((p.id, condition))
            print(f"{split} {family} {condition}: {min(start + size, len(pending))}/{len(pending)} new", flush=True)
        return [r for r in self.rows if r["family"] == family and r["split"] == split and r["condition"] == condition]

    def screen(self):
        for family in self.config["families"]:
            for condition in ("zero", "icl_a", "icl_b"):
                self.evaluate(family, "screen", condition)
        decision = choose_family(self.rows, self.config["families"], self.config["min_screen_gain"])
        save_json(self.output / "screen_decision.json", decision)
        build_report(self.output)
        print(json.dumps(decision), flush=True)
        return decision["selected_family"]

    def extract(self, family):
        import torch
        path = self.output / "vectors.pt"
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        captures = {condition: {layer: [] for layer in self.config["layers"]}
                    for condition in ("zero", "icl_a", "icl_b")}
        problems = self.problems(family, "extract")
        size = self.config["batch_size"]
        for condition, layers in captures.items():
            for start in range(0, len(problems), size):
                activations = self.model.activations(self.prompts(problems[start:start + size], condition))
                for layer, values in activations.items():
                    layers[layer].append(values)
                print(f"extract {condition}: {min(start + size, len(problems))}/{len(problems)}", flush=True)
        stacked = {condition: {layer: torch.cat(values) for layer, values in layers.items()}
                   for condition, layers in captures.items()}
        vectors = {kind: {} for kind in ("icl", "method", "generic")}
        diagnostics = {}
        for layer in self.config["layers"]:
            zero, a, b = (stacked[c][layer] for c in ("zero", "icl_a", "icl_b"))
            for kind, delta in {"icl": a - zero, "method": a - b,
                                "generic": (a + b) / 2 - zero}.items():
                mean = delta.mean(0)
                vectors[kind][layer] = mean
                unit = mean / mean.norm().clamp_min(1e-12)
                diagnostics[f"{kind}_layer_{layer}"] = {
                    "mean_norm": mean.norm().item(),
                    "mean_delta_cosine": torch.nn.functional.cosine_similarity(delta, mean[None]).mean().item(),
                    "uncentered_energy_along_mean": ((delta @ unit).square().sum() / delta.square().sum().clamp_min(1e-12)).item(),
                    "note": "Descriptive geometry, not evidence that this direction causes useful ICL."}
        torch.save(vectors, path)
        save_json(self.output / "vector_diagnostics.json", diagnostics)
        return vectors

    def validate(self, family, vectors):
        path = self.output / "selection.json"
        if path.exists():
            return json.loads(path.read_text())
        for condition in BASELINES:
            self.evaluate(family, "validation", condition)
        candidates = {kind: [] for kind in ("icl", "method")}
        for kind in candidates:
            for layer in self.config["layers"]:
                for strength in self.config["strengths"]:
                    condition = f"val_{kind}_l{layer}_a{strength:g}"
                    rows = self.evaluate(family, "validation", condition, vectors[kind][layer], layer, strength)
                    candidates[kind].append({"kind": kind, "layer": layer,
                        "strength": strength, "accuracy": aggregate(rows)["accuracy"], "condition": condition})
        selection = {"family": family, "selected": {kind: choose_candidate(values) for kind, values in candidates.items()},
                     "candidates": candidates,
                     "rule": "Max validation accuracy for each direction; ties use smaller strength then shallower layer. Test always runs, even if no validation gain."}
        save_json(path, selection)
        build_report(self.output)
        return selection

    def test(self, family, vectors, selection):
        import torch
        for condition in BASELINES:
            self.evaluate(family, "test", condition)
        for kind, chosen in selection["selected"].items():
            layer, strength = chosen["layer"], chosen["strength"]
            self.evaluate(family, "test", f"steer_{kind}", vectors[kind][layer], layer, strength)
            self.evaluate(family, "test", f"negative_{kind}", vectors[kind][layer], layer, -strength)
            generic = vectors["generic"][layer]
            generic = generic / generic.norm().clamp_min(1e-12) * vectors[kind][layer].norm()
            self.evaluate(family, "test", f"generic_matched_{kind}", generic, layer, strength)
            for seed in self.config["random_seeds"]:
                random = torch.randn(vectors[kind][layer].shape, generator=torch.Generator().manual_seed(seed))
                random = random / random.norm() * vectors[kind][layer].norm()
                self.evaluate(family, "test", f"random_{kind}_{seed}", random, layer, strength)
        build_report(self.output)
        save_json(self.output / "complete.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
                                                  "status": "pilot_complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "screen", "run", "report"))
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "report":
        build_report(args.output)
        return
    config = json.loads(args.config.read_text())
    if args.stage == "prepare":
        prepare(config, args.output)
        return
    runner = Runner(config, args.output)
    family = runner.screen()
    if args.stage == "screen":
        return
    if family is None:
        save_json(args.output / "complete.json", {"status": "no_screened_icl_gain",
                  "note": "No family passed the prespecified screen. No steering claim is warranted."})
        return
    vectors = runner.extract(family)
    selection = runner.validate(family, vectors)
    runner.test(family, vectors, selection)


if __name__ == "__main__":
    main()

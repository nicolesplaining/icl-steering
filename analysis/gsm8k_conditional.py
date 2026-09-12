"""Fresh-data causal test of an extraction-fitted conditional activation shift."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from analysis import conditional_shift as shift
from analysis import gsm8k_answer_audit as audit
from analysis.gsm8k_comparison import compare
from replication.gsm8k_protocol import grade_answer, matched_prompt
from replication.gsm8k_steering import Runner, save, source_hash


ROOT = Path(__file__).parents[1]
BASELINES = ["zero", "icl_a", "icl_b", "first", "cot"]


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def code_hash():
    files = [Path(__file__), ROOT / "analysis/conditional_shift.py",
             ROOT / "analysis/gsm8k_answer_audit.py", ROOT / "analysis/gsm8k_comparison.py",
             ROOT / "research/gsm8k-conditional-protocol.md"]
    return audit.digest({"base": source_hash(), "files": {
        str(p.relative_to(ROOT)): file_hash(p) for p in files}})


def fixed(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Refuse changed input: {path}")
    save(path, value)


def specs(config, selection=None):
    result = {k: {"kind": "baseline", "prompt_kind": k, "layer": None,
                  "alpha": 0.0, "positions": "prefill"} for k in BASELINES}
    result["legacy_mean"] = {"kind": "legacy_mean", "layer": 7, "alpha": 0.5, "positions": "all"}
    if selection is None:
        for setting in shift.settings(config):
            for kind in ["ridge", "mean"]:
                result[shift.name(kind, setting)] = {"kind": kind, **setting}
    else:
        setting = {k: selection["chosen"][k] for k in ["layer", "alpha", "positions"]}
        for name, kind in [("steered", "ridge"), ("mean", "mean"), ("mean_norm", "mean_norm"),
                           ("scalar", "scalar"), ("permuted", "permuted"),
                           ("rotated_pairs", "rotated"), ("reverse", "reverse")]:
            result[name] = {"kind": kind, **setting}
        for seed in config["random_seeds"]:
            result[f"random_{seed}"] = {"kind": "random", "random_seed": seed, **setting}
    return result


def load_maps(output):
    with np.load(output / "maps.npz", allow_pickle=False) as packed:
        arrays = {k: packed[k] for k in packed.files}
    fitted = {}
    for key, value in arrays.items():
        if key == "legacy_mean":
            continue
        layer, kind, field = key.split("/", 2)
        fitted.setdefault((int(layer), kind), {})[field] = value
    return fitted, arrays["legacy_mean"]


def directions(spec, x, fitted, legacy):
    kind = spec["kind"]
    if kind == "legacy_mean":
        return np.broadcast_to(legacy, (len(x), len(legacy))).copy()
    real = fitted[spec["layer"], "real"]
    ridge = shift.predict(real, x)
    if kind == "ridge":
        return ridge
    if kind == "mean":
        return shift.predict(real, x, "mean")
    if kind == "reverse":
        return -ridge
    if kind == "mean_norm":
        control = shift.predict(real, x, "mean")
    elif kind == "scalar":
        control = shift.predict(real, x, "scalar")
    elif kind in {"permuted", "rotated"}:
        control = shift.predict(fitted[spec["layer"], kind], x)
    elif kind == "random":
        vector = np.random.default_rng(spec["random_seed"]).normal(size=x.shape[1])
        control = np.broadcast_to(vector, x.shape)
    else:
        raise ValueError(f"Unknown intervention kind: {kind}")
    return shift.match_norm(control, ridge)


def prepare(config, parent, controls, output):
    output.mkdir(parents=True, exist_ok=True)
    inputs = {"parent_manifest": parent / "manifest.json", "parent_prepared": parent / "prepared.json",
              "parent_selection": parent / "selection.json", "parent_directions": parent / "directions.pt",
              "control_inputs": controls / "inputs.json", "control_metrics": controls / "metrics.json",
              **{f"tensor_{k}": controls / f"{k}.pt" for k in ["zero", "icl_a", "rotated_pairs"]}}
    request = {"config": config, "code_sha256": code_hash(),
               "inputs_sha256": {k: file_hash(p) for k, p in inputs.items()}}
    if (output / "manifest.json").exists():
        manifest = read(output / "manifest.json")
        if manifest["request"] != request:
            raise ValueError("Conditional run inputs changed; use a new directory")
        return verify(output)
    if (output / "generations.jsonl").exists():
        raise ValueError("Unmanifested generation data")
    pm = read(parent / "manifest.json")
    old = read(parent / "prepared.json")
    metric = read(controls / "metrics.json")
    geometry = read(ROOT / "results/gsm8k-v2-conditional-geometry.json")
    if (pm["source_sha256"] != source_hash()
            or pm["prepared_sha256"] != file_hash(parent / "prepared.json")
            or metric["status"] != "extraction_controls_complete"
            or metric["provenance"]["prepared_sha256"] != pm["prepared_sha256"]
            or metric["provenance"]["reference_tensor_sha256"] != file_hash(parent / "directions.pt")):
        raise ValueError("Parent extraction provenance mismatch")
    for key in ["model", "revision", "dataset", "dataset_revision", "max_new_tokens", "max_model_len"]:
        if config[key] != pm["config"][key]:
            raise ValueError(f"Unexpected change from parent: {key}")
    old_choice = read(parent / "selection.json")["chosen"]
    if any(old_choice[k] != v for k, v in {"layer": 7, "alpha": 0.5, "positions": "all"}.items()):
        raise ValueError("Unexpected legacy mean parameters")
    ids = [r["problem_id"] for r in old["splits"]["extract"]]
    if len(ids) != 128 or read(controls / "inputs.json")["problem_ids"] != ids:
        raise ValueError("Extraction row identity mismatch")
    tensors = {}
    for kind in ["zero", "icl_a", "rotated_pairs"]:
        path = controls / f"{kind}.pt"
        if file_hash(path) != geometry["source_tensor_sha256"][kind]:
            raise ValueError("Extraction arrays differ from audited geometry")
        tensors[kind] = torch.load(path, map_location="cpu", weights_only=True)
    reference = torch.load(parent / "directions.pt", map_location="cpu", weights_only=True)
    arrays = {"legacy_mean": reference[7].numpy().astype(np.float64)}
    permutation = np.random.default_rng(config["seed"]).permutation(len(ids))
    fit_info = {"n_extract": len(ids), "permutation": permutation.tolist(), "fits": {}}
    for layer in config["layers"]:
        x = tensors["zero"][layer].numpy().astype(np.float64)
        delta = tensors["icl_a"][layer].numpy().astype(np.float64)-x
        if len(x) != len(ids) or not np.allclose(delta.mean(0), reference[layer].numpy(), atol=1e-3, rtol=1e-4):
            raise ValueError("Extraction dimensions or mean replay mismatch")
        targets = {"real": delta, "permuted": delta[permutation],
                   "rotated": tensors["rotated_pairs"][layer].numpy().astype(np.float64)-x}
        for kind, target in targets.items():
            fitted = shift.fit(x, target)
            arrays.update({f"{layer}/{kind}/{field}": value for field, value in fitted.items()})
            fit_info["fits"][f"{layer}/{kind}"] = {
                "width": x.shape[1], "mean_norm": float(np.linalg.norm(fitted["mean"])),
                "penalty": float(fitted["penalty"]),
                "relative_solver_residual": float(fitted["relative_solver_residual"])}
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tables, hashes = {}, {}
    for split in ["train", "test"]:
        path = Path(hf_hub_download(config["dataset"], f"main/{split}-00000-of-00001.parquet",
                                   repo_type="dataset", revision=config["dataset_revision"]))
        tables[split] = pq.read_table(path).to_pylist()
        hashes[split] = file_hash(path)
    plan = shift.fresh_ids(tables, old, config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    data = {"banks": old["banks"], "splits": {}, "plan": plan}
    for split, indices in plan.items():
        source = "test" if split == "test" else "train"
        data["splits"][split] = []
        for i in indices:
            q = tables[source][i]["question"]
            prompts = {"zero": matched_prompt(q), "first": matched_prompt(q, suffix=" First,"),
                       "cot": matched_prompt(q, suffix=" Let's think step by step."),
                       **{kind: matched_prompt(q, bank) for kind, bank in old["banks"].items()}}
            if any(len(tokenizer.encode(p, add_special_tokens=False))+config["max_new_tokens"] >
                   config["max_model_len"] for p in prompts.values()):
                raise ValueError("Fresh prompt would overflow context")
            data["splits"][split].append({"problem_id": f"gsm8k:{source}:{i}", "question": q,
                "answer": tables[source][i]["answer"].split("####")[-1].strip().replace(",", ""), "prompts": prompts})
    np.savez(output / "maps.npz", **arrays)
    save(output / "fit.json", fit_info)
    save(output / "prepared.json", data)
    save(output / "manifest.json", {"request": request, "config": config,
        "prepared_sha256": file_hash(output / "prepared.json"), "maps_sha256": file_hash(output / "maps.npz"),
        "fit_sha256": file_hash(output / "fit.json"), "data_sha256": hashes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_conditions": list(specs(config)), "test_generation_requires_selection": True})
    return verify(output)


def verify(output):
    manifest = read(output / "manifest.json")
    if manifest["request"]["code_sha256"] != code_hash():
        raise ValueError("Declared conditional source or protocol changed")
    for filename, key in [("prepared.json", "prepared_sha256"), ("maps.npz", "maps_sha256"), ("fit.json", "fit_sha256")]:
        if file_hash(output / filename) != manifest[key]:
            raise ValueError(f"Declared file changed: {filename}")
    return manifest["config"], read(output / "prepared.json")


def rows_from(output):
    path = output / "generations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def check_rows(rows, data, declared, split):
    questions = {p["problem_id"]: p for p in data["splits"][split]}
    for row in rows:
        if row["split"] != split:
            continue
        if row["condition"] not in declared or row["problem_id"] not in questions:
            raise ValueError("Unexpected generation identity")
        spec, question = declared[row["condition"]], questions[row["problem_id"]]
        if (row.get("intervention") != spec or row["answer"] != question["answer"]
                or row["prompt"] != question["prompts"][spec.get("prompt_kind", "zero")]
                or any(row[k] != spec[k] for k in ["layer", "alpha", "positions"])):
            raise ValueError("Generation does not match the declared question and intervention")
        grade = grade_answer(row["text"], row["answer"], row["truncated"])
        if any(row[k] != value for k, value in grade.items()):
            raise ValueError("Saved explicit grade mismatch")


class ConditionalRunner(Runner):
    def __init__(self, config, data, output):
        super().__init__(config, data, output)
        self.fitted, self.legacy = load_maps(output)
        self.queries = {}

    def backend(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ValueError("GPU inference must use physical GPU 0 only")
        return super().backend()

    def query_activations(self, split):
        if split in self.queries:
            return self.queries[split]
        path, meta = self.output / f"{split}-queries.npz", self.output / f"{split}-queries.json"
        ids = [r["problem_id"] for r in self.data["splits"][split]]
        provenance = {"prepared_sha256": file_hash(self.output / "prepared.json"), "problem_ids": ids,
                      "layers": self.config["layers"], "batch_size": self.config["batch_size"], "prompt_kind": "zero"}
        if meta.exists():
            value = read(meta)
            if value["provenance"] != provenance or value["arrays_sha256"] != file_hash(path):
                raise ValueError("Query activation cache changed")
        else:
            values = self.backend().activations([r["prompts"]["zero"] for r in self.data["splits"][split]],
                                               self.config["layers"], self.config["batch_size"])
            np.savez(path, **{str(k): v.numpy() for k, v in values.items()})
            save(meta, {"provenance": provenance, "arrays_sha256": file_hash(path)})
        with np.load(path, allow_pickle=False) as packed:
            values = {int(k): packed[k] for k in packed.files}
        for layer, value in values.items():
            if value.shape != (len(ids), len(self.fitted[layer, "real"]["mean"])) or not np.isfinite(value).all():
                raise ValueError("Invalid query activation cache")
        self.queries[split] = values
        return values

    def evaluate_spec(self, split, condition, spec):
        problems, size = self.data["splits"][split], self.config["batch_size"]
        baseline = spec["kind"] == "baseline"
        prompt_kind = spec.get("prompt_kind", "zero")
        for row in self.rows:
            if row["split"] == split and row["condition"] == condition and row.get("intervention") != spec:
                raise ValueError("Existing generation intervention mismatch")
        # Keep the original batches on resume even after a partially written batch.
        for start in range(0, len(problems), size):
            batch = problems[start:start+size]
            if all((split, p["problem_id"], condition) in self.done for p in batch):
                continue
            vectors = None
            if not baseline:
                x = self.query_activations(split)[spec["layer"]][start:start+size]
                vectors = directions(spec, x, self.fitted, self.legacy).astype(np.float32)
            outputs = self.backend().records([p["prompts"][prompt_kind] for p in batch], self.config,
                None if vectors is None else torch.from_numpy(vectors), spec["layer"], spec["alpha"], spec["positions"])
            if len(outputs) != len(batch):
                raise ValueError("Generation row count mismatch")
            with (self.output / "generations.jsonl").open("a") as handle:
                for i, (p, result) in enumerate(zip(batch, outputs)):
                    key = (split, p["problem_id"], condition)
                    if key in self.done:
                        continue
                    row = {"split": split, "condition": condition, "problem_id": p["problem_id"],
                           "prompt": p["prompts"][prompt_kind], "answer": p["answer"], **result,
                           **grade_answer(result["text"], p["answer"], result["truncated"]),
                           "intervention": spec, **{k: spec[k] for k in ["layer", "alpha", "positions"]},
                           "vector_sha256": sha256(vectors[i].tobytes()).hexdigest() if vectors is not None else None,
                           "vector_norm": float(np.linalg.norm(vectors[i])) if vectors is not None else 0.0}
                    handle.write(json.dumps(row)+"\n")
                    handle.flush()
                    self.rows.append(row)
                    self.done.add(key)
            print(f"{split} {condition}: {min(start+size, len(problems))}/{len(problems)}", flush=True)


def export(output, split, conditions):
    config, data = verify(output)
    packet = audit.make_packet(rows_from(output), data, split, conditions)
    fixed(output / f"{split}-review-packet.json", packet)
    print(f"Blinded {split} review: {len(packet['items'])} unique responses", flush=True)


def select(output, annotations):
    config, data = verify(output)
    if read(output / "validation-complete.json")["status"] != "awaiting_blinded_review":
        raise ValueError("Validation incomplete")
    rows = rows_from(output)
    if any(r["split"] == "test" for r in rows):
        raise ValueError("Cannot select after test generation")
    check_rows(rows, data, specs(config), "validation")
    packet, annotations = read(output / "validation-review-packet.json"), read(annotations)
    score = audit.score(rows, data, "validation", list(specs(config)), packet, annotations, config["bootstrap_samples"])
    choice = shift.choose(score["summary"], config)
    fixed(output / "validation-annotations.json", annotations)
    fixed(output / "validation-audit.json", score)
    choice.update(manifest_sha256=file_hash(output / "manifest.json"),
                  validation_rows_sha256=audit.digest(rows), packet_sha256=audit.digest(packet),
                  annotations_sha256=audit.digest(annotations), maps_sha256=file_hash(output / "maps.npz"),
                  validation_audit_sha256=file_hash(output / "validation-audit.json"),
                  validation_queries_sha256=file_hash(output / "validation-queries.npz"))
    fixed(output / "selection.json", choice)
    print(json.dumps(choice, indent=2), flush=True)


def test_selection(output):
    config, data = verify(output)
    selection = read(output / "selection.json")
    validation = [r for r in rows_from(output) if r["split"] == "validation"]
    check_rows(validation, data, specs(config), "validation")
    expected = {"manifest_sha256": file_hash(output / "manifest.json"),
                "validation_rows_sha256": audit.digest(validation),
                "packet_sha256": audit.digest(read(output / "validation-review-packet.json")),
                "annotations_sha256": audit.digest(read(output / "validation-annotations.json")),
                "maps_sha256": file_hash(output / "maps.npz"),
                "validation_audit_sha256": file_hash(output / "validation-audit.json"),
                "validation_queries_sha256": file_hash(output / "validation-queries.npz")}
    if any(selection[k] != value for k, value in expected.items()):
        raise ValueError("Selection provenance changed")
    calculated = shift.choose(read(output / "validation-audit.json")["summary"], config)
    if any(selection[k] != value for k, value in calculated.items()):
        raise ValueError("Selection rule mismatch")
    if not selection["eligible"]:
        raise ValueError("Validation gates failed; no test generation authorized by protocol")
    return selection


def report(output, annotations):
    config, data = verify(output)
    selection = test_selection(output)
    if read(output / "test-complete.json")["status"] != "awaiting_blinded_review":
        raise ValueError("Test incomplete")
    packet, annotations = read(output / "test-review-packet.json"), read(annotations)
    check_rows(rows_from(output), data, specs(config, selection), "test")
    if read(output / "test-lock.json") != {
        "selection_sha256": file_hash(output / "selection.json"),
        "manifest_sha256": file_hash(output / "manifest.json"), "maps_sha256": file_hash(output / "maps.npz")
    }:
        raise ValueError("Test lock changed")
    result = compare(rows_from(output), data, list(specs(config, selection)), packet, annotations,
                     config["bootstrap_samples"])
    result["selection"] = selection
    result["manifest_sha256"] = file_hash(output / "manifest.json")
    fixed(output / "test-annotations.json", annotations)
    fixed(output / "test-audit.json", result)
    print(json.dumps(result["comparisons"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "validate", "select", "test", "report"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gsm8k_conditional.json")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / ".writer.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            if args.parent is None or args.controls is None:
                parser.error("prepare requires --parent and --controls")
            config, _ = prepare(read(args.config), args.parent, args.controls, args.output)
            print(f"Prepared {config['n_validation']} validation and {config['n_test']} fresh test questions")
        elif args.stage in {"select", "report"}:
            if args.annotations is None:
                parser.error("review stage requires --annotations")
            (select if args.stage == "select" else report)(args.output, args.annotations)
        else:
            config, data = verify(args.output)
            split = "validation" if args.stage == "validate" else "test"
            selection = None
            if split == "validation" and (args.output / "selection.json").exists():
                raise ValueError("Validation is frozen after selection")
            if split == "test":
                selection = test_selection(args.output)
                fixed(args.output / "test-lock.json", {"selection_sha256": file_hash(args.output / "selection.json"),
                      "manifest_sha256": file_hash(args.output / "manifest.json"),
                      "maps_sha256": file_hash(args.output / "maps.npz")})
            with (args.output.parent / ".gpu0.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                runner = ConditionalRunner(config, data, args.output)
                try:
                    check_rows(runner.rows, data, specs(config, selection), split)
                    runner.query_activations(split)
                    for condition, spec in specs(config, selection).items():
                        runner.evaluate_spec(split, condition, spec)
                    export(args.output, split, list(specs(config, selection)))
                    save(args.output / f"{split}-complete.json", {"status": "awaiting_blinded_review",
                         "completed_at": datetime.now(timezone.utc).isoformat()})
                finally:
                    runner.report()


if __name__ == "__main__":
    main()

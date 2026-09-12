"""Test fixed prefix-control directions using the primary run's locked selection."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import torch

from replication.gsm8k_steering import Runner, save, source_hash


KINDS = ("rotated_pairs", "token_shuffle", "length_filler")


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def locked_selection(parent):
    path = parent / "selection.json"
    if not path.exists():
        return None
    selection = json.loads(path.read_text())
    if not selection["eligible"]:
        if (parent / "test_lock.json").exists():
            raise ValueError("Primary eligibility changed after test locking")
        return selection
    lock_path = parent / "test_lock.json"
    if not lock_path.exists():
        return None
    expected = {"selection_sha256": file_hash(path),
                "directions_sha256": file_hash(parent / "directions.pt")}
    if json.loads(lock_path.read_text()) != expected:
        raise ValueError("Primary selection or directions changed after locking")
    return selection


def matched_direction(control, zero, reference):
    delta = (control-zero).float().mean(0)
    if not torch.isfinite(delta).all() or delta.norm() <= 1e-12:
        raise ValueError("Invalid control direction")
    return delta / delta.norm() * reference.norm()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / ".writer.lock").open("a") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_manifest = json.loads((args.run / "manifest.json").read_text())
        if source_hash() != parent_manifest["source_sha256"]:
            raise ValueError("Primary source changed")
        if file_hash(args.run / "prepared.json") != parent_manifest["prepared_sha256"]:
            raise ValueError("Prepared data changed")
        metrics = json.loads((args.controls / "metrics.json").read_text())
        if metrics["status"] != "extraction_controls_complete":
            raise ValueError("Extraction controls incomplete")
        if metrics["provenance"]["prepared_sha256"] != parent_manifest["prepared_sha256"]:
            raise ValueError("Controls came from different questions")
        sources = {"parent_manifest": args.run / "manifest.json",
                   "parent_directions": args.run / "directions.pt",
                   "control_metrics": args.controls / "metrics.json",
                   "protocol": Path(__file__).parents[1] / "research/gsm8k-prefix-test-protocol.md",
                   **{kind: args.controls / f"{kind}.pt" for kind in ("zero", *KINDS)}}
        inputs = {kind: file_hash(path) for kind, path in sources.items()}
        if inputs["parent_directions"] != metrics["provenance"]["reference_tensor_sha256"]:
            raise ValueError("Controls reference different primary directions")
        request = {"inputs_sha256": inputs, "code_sha256": file_hash(Path(__file__)),
                   "conditions": [f"prefix_{kind}" for kind in KINDS],
                   "selection": "Use primary locked layer, strength, and scope without retuning.",
                   "normalization": "Scale every control mean to the real ICL mean norm at the selected layer.",
                   "n_test": parent_manifest["config"]["n_test"]}
        path = args.output / "manifest.json"
        if path.exists():
            if json.loads(path.read_text())["request"] != request:
                raise ValueError("Supplementary run inputs changed")
        else:
            if (args.run / "test_lock.json").exists():
                raise ValueError("New supplementary tests must be declared before primary test starts")
            save(path, {"request": request, "declared_at_utc": datetime.now(timezone.utc).isoformat(),
                        "primary_test_lock_present": False})
        print("Waiting for the primary selection and test lock", flush=True)
        while (selection := locked_selection(args.run)) is None:
            proc = Path(f"/proc/{args.parent_pid}/cmdline")
            if not proc.exists() or b"replication.gsm8k_steering" not in proc.read_bytes():
                selection = locked_selection(args.run)
                if selection is None:
                    raise RuntimeError("Primary process stopped without a locked selection; inspect its log")
                break
            time.sleep(10)
        if not selection["eligible"]:
            save(args.output / "complete.json", {"status": "skipped_parent_validation_gate"})
            return
        if any(file_hash(path) != inputs[kind] for kind, path in sources.items()):
            raise ValueError("Declared inputs changed while waiting")
        if file_hash(args.run / "prepared.json") != parent_manifest["prepared_sha256"]:
            raise ValueError("Prepared data changed while waiting")
        if source_hash() != parent_manifest["source_sha256"]:
            raise ValueError("Primary source changed while waiting")
        lock = {"parent_test_lock": json.loads((args.run / "test_lock.json").read_text()),
                "supplement_manifest_sha256": file_hash(path)}
        own_lock = args.output / "test_lock.json"
        if own_lock.exists() and json.loads(own_lock.read_text()) != lock:
            raise ValueError("Supplementary test selection changed")
        save(own_lock, lock)
        choice = selection["chosen"]
        layer = choice["layer"]
        reference = torch.load(args.run / "directions.pt", map_location="cpu", weights_only=True)[layer]
        zero = torch.load(args.controls / "zero.pt", map_location="cpu", weights_only=True)[layer]
        data = json.loads((args.run / "prepared.json").read_text())
        runner = Runner(parent_manifest["config"], data, args.output)
        try:
            for kind in KINDS:
                if locked_selection(args.run) != selection:
                    raise ValueError("Parent selection changed during supplementary evaluation")
                source = args.controls / f"{kind}.pt"
                if file_hash(source) != inputs[kind]:
                    raise ValueError("Control activations changed")
                values = torch.load(source, map_location="cpu", weights_only=True)[layer]
                direction = matched_direction(values, zero, reference)
                print(f"Evaluating prefix_{kind} with {choice['condition']}", flush=True)
                runner.evaluate("test", f"prefix_{kind}", direction=direction, layer=layer,
                                alpha=choice["alpha"], positions=choice["positions"])
            save(args.output / "complete.json", {"status": "prefix_test_complete",
                                                "completed_at": datetime.now(timezone.utc).isoformat()})
        finally:
            runner.report()


if __name__ == "__main__":
    main()

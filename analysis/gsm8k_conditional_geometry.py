"""Predict extraction activation shifts on questions excluded from each fit."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def crossfit_predictions(x, delta, folds=8, seed=907):
    x = x.detach().cpu().numpy().astype(np.float64)
    delta = delta.detach().cpu().numpy().astype(np.float64)
    if x.ndim != 2 or x.shape != delta.shape or not 2 <= folds <= len(x):
        raise ValueError("Require matched activation matrices and valid folds")
    if not np.isfinite(x).all() or not np.isfinite(delta).all():
        raise ValueError("Nonfinite activations")
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed)).numpy()
    assignment = np.empty(len(x), dtype=np.int64)
    predictions = {kind: np.empty_like(delta) for kind in ["mean", "scalar", "ridge"]}
    for fold, test in enumerate(np.array_split(order, folds)):
        assignment[test] = fold
        train = np.ones(len(x), dtype=bool)
        train[test] = False
        xm, ym = x[train].mean(0), delta[train].mean(0)
        xc, yc, query = x[train]-xm, delta[train]-ym, x[test]-xm
        predictions["mean"][test] = ym
        energy = np.square(xc).sum()
        coefficient = (xc*yc).sum()/energy if energy > 1e-12 else 0.0
        predictions["scalar"][test] = ym + coefficient*query
        gram = xc @ xc.T
        regularization = np.trace(gram)/len(gram)
        if regularization <= 1e-12:
            predictions["ridge"][test] = ym
        else:
            # The host's Torch CPU float64 matrix operations fail sanity checks.
            # Keep fitting and residual verification within NumPy.
            matrix = gram + regularization*np.eye(len(gram))
            weights = np.linalg.solve(matrix, yc)
            if np.linalg.norm(matrix @ weights-yc) > 1e-8 * max(np.linalg.norm(yc), 1.0):
                raise ValueError("Ridge solver failed its residual check")
            predictions["ridge"][test] = ym + (query @ xc.T) @ weights
    return {kind: torch.from_numpy(value) for kind, value in predictions.items()}, torch.from_numpy(assignment)


def summarize(delta, predictions, assignment):
    delta = delta.double()
    errors = {kind: (values-delta).square().sum(1) for kind, values in predictions.items()}
    baseline = errors["mean"].sum().item()
    total = delta.square().sum().item()
    result = {"mean_prediction_error": baseline, "delta_squared_energy": total,
              "mean_explained_energy": 1-baseline/total if total else None, "predictors": {}}
    for kind, values in errors.items():
        error = values.sum().item()
        result["predictors"][kind] = {
            "squared_error": error,
            "error_reduction_vs_mean": 1-error/baseline if baseline else None,
            "fold_error_reduction_vs_mean": [
                1-values[assignment == fold].sum().item()/errors["mean"][assignment == fold].sum().item()
                if errors["mean"][assignment == fold].sum() > 0 else None
                for fold in sorted(assignment.unique().tolist())]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.run / "manifest.json").read_text())
    controls = json.loads((args.controls / "manifest.json").read_text())
    metrics = json.loads((args.controls / "metrics.json").read_text())
    if (file_hash(args.run / "prepared.json") != manifest["prepared_sha256"]
            or controls["prepared_sha256"] != manifest["prepared_sha256"]
            or file_hash(args.controls / "inputs.json") != controls["inputs_sha256"]
            or metrics["status"] != "extraction_controls_complete"
            or file_hash(args.run / "directions.pt") != controls["reference_tensor_sha256"]):
        raise ValueError("Extraction inputs changed or are incomplete")
    inputs = json.loads((args.controls / "inputs.json").read_text())
    prepared = json.loads((args.run / "prepared.json").read_text())
    ids = [r["problem_id"] for r in prepared["splits"]["extract"]]
    if inputs["problem_ids"] != ids or len(set(ids)) != len(ids):
        raise ValueError("Extraction question identities or order changed")
    kinds = ["zero", "icl_a", "rotated_pairs", "token_shuffle", "length_filler"]
    hashes = {kind: file_hash(args.controls / f"{kind}.pt") for kind in kinds}
    tensors = {kind: torch.load(args.controls / f"{kind}.pt", map_location="cpu", weights_only=True)
               for kind in kinds}
    reference = torch.load(args.run / "directions.pt", map_location="cpu", weights_only=True)
    layers = {}
    assignments = None
    for layer in manifest["config"]["layers"]:
        x = tensors["zero"][layer]
        if len(x) != len(ids) or any(tensors[kind][layer].shape != x.shape for kind in kinds):
            raise ValueError("Activation shapes or row count changed")
        torch.testing.assert_close((tensors["icl_a"][layer]-x).mean(0), reference[layer], atol=1e-3, rtol=1e-4)
        layers[layer] = {}
        for kind in kinds[1:]:
            delta = tensors[kind][layer]-x
            predictions, assignments = crossfit_predictions(x, delta)
            layers[layer][kind] = summarize(delta, predictions, assignments)
    result = {"status": "extraction_conditional_geometry", "n_extract": len(ids), "folds": 8,
              "seed": 907, "fold_assignment": assignments.tolist(),
              "method": "Training-fold means, a fitted scalar rescaling of centered zero-shot activations, "
                        "and dual ridge regression with lambda=trace(Xc@Xc.T)/n_train. All settings fixed; "
                        "no outcome-based tuning. Predictions are evaluated only outside their training folds.",
              "layers": layers, "source_tensor_sha256": hashes,
              "prepared_sha256": manifest["prepared_sha256"], "inputs_sha256": controls["inputs_sha256"],
              "code_sha256": file_hash(Path(__file__)),
              "limitations": "Exploratory geometry on 128 extraction questions from one demonstration bank. "
                             "Cross-validation training folds overlap. Better activation prediction does not "
                             "establish better math accuracy or a causal intervention. The original mean direction, "
                             "selection, and ongoing final test are unchanged. No test generations are loaded."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError("Refuse to overwrite changed diagnostic")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({layer: {kind: {model: value["error_reduction_vs_mean"] for model, value in
                                  stats["predictors"].items()} for kind, stats in by_kind.items()}
                      for layer, by_kind in layers.items()}, indent=2))


if __name__ == "__main__":
    main()

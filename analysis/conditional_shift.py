"""NumPy-only fitting and prediction of query-dependent activation shifts."""

import numpy as np


def fit(x, delta):
    x, delta = np.asarray(x, dtype=np.float64), np.asarray(delta, dtype=np.float64)
    if x.ndim != 2 or x.shape != delta.shape or len(x) < 2:
        raise ValueError("Require matched extraction matrices")
    if not np.isfinite(x).all() or not np.isfinite(delta).all():
        raise ValueError("Nonfinite extraction activation")
    xm, mean = x.mean(0), delta.mean(0)
    xc, yc = x-xm, delta-mean
    gram = xc @ xc.T
    penalty = np.trace(gram)/len(x)
    weights = np.zeros_like(yc)
    residual = 0.0
    if penalty > 1e-12:
        matrix = gram + penalty*np.eye(len(x))
        weights = np.linalg.solve(matrix, yc)
        residual = np.linalg.norm(matrix @ weights-yc)/max(np.linalg.norm(yc), 1.0)
        if not np.isfinite(residual) or residual > 1e-8:
            raise ValueError("Ridge solve failed its residual check")
    energy = np.square(xc).sum()
    scalar = (xc*yc).sum()/energy if energy > 1e-12 else 0.0
    return {"x_mean": xm, "mean": mean, "x_centered": xc, "weights": weights,
            "scalar": np.asarray(scalar), "penalty": np.asarray(penalty),
            "relative_solver_residual": np.asarray(residual)}


def predict(fitted, x, kind="ridge"):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1:] != fitted["mean"].shape:
        raise ValueError("Query activation shape mismatch")
    if kind == "mean":
        result = np.broadcast_to(fitted["mean"], x.shape).copy()
    elif kind == "scalar":
        result = fitted["mean"] + fitted["scalar"]*(x-fitted["x_mean"])
    elif kind == "ridge":
        result = fitted["mean"] + ((x-fitted["x_mean"]) @ fitted["x_centered"].T) @ fitted["weights"]
    else:
        raise ValueError(f"Unknown predictor: {kind}")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite conditional shift")
    return result


def match_norm(vectors, reference):
    vectors, reference = np.asarray(vectors), np.asarray(reference)
    if vectors.shape != reference.shape or vectors.ndim != 2:
        raise ValueError("Norm matching requires matched batch matrices")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(reference).all() or (norms <= 1e-12).any():
        raise ValueError("Invalid norm-matching vector")
    return vectors/norms*np.linalg.norm(reference, axis=1, keepdims=True)


def fresh_ids(tables, old, config):
    """Exclude reserved prefixes and every prior question by ID and normalized text."""
    normalize = lambda value: " ".join(value.split())
    used_text = {normalize(r["question"]) for rows in old["splits"].values() for r in rows}
    used_text.update(normalize(r["question"]) for bank in old["banks"].values() for r in bank)
    used_text.update(normalize(r["question"]) for source in ["train", "test"]
                     for r in tables[source][:config[f"reserved_{source}"]])
    used_ids = {r["problem_id"] for rows in old["splits"].values() for r in rows}
    rng = np.random.default_rng(config["seed"])
    plans = {}
    for split, source, count in [("validation", "train", config["n_validation"]),
                                 ("test", "test", config["n_test"])]:
        available = range(config[f"reserved_{source}"], len(tables[source]))
        chosen = []
        for i in rng.permutation(list(available)).tolist():
            key, question = f"gsm8k:{source}:{i}", normalize(tables[source][i]["question"])
            if key in used_ids or question in used_text:
                continue
            chosen.append(i)
            used_ids.add(key)
            used_text.add(question)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Insufficient fresh {split} questions")
        plans[split] = sorted(chosen)
    return plans


def settings(config):
    return [{"layer": layer, "alpha": alpha, "positions": positions}
            for layer in config["layers"] for alpha in config["strengths"]
            for positions in config["positions"]]


def name(kind, setting):
    return f"{kind}_l{setting['layer']}_a{setting['alpha']:g}_{setting['positions']}"


def choose(summary, config):
    candidates = [{**setting, "condition": name("ridge", setting),
                   "audited_correct": summary[name("ridge", setting)]["audited_completed_correct"]}
                  for setting in settings(config)]
    chosen = min(candidates, key=lambda c: (-c["audited_correct"], c["positions"] != "prefill",
                                          c["alpha"], c["layer"]))
    n = config["n_validation"]
    zero = summary["zero"]["audited_completed_correct"]
    mean = summary[name("mean", chosen)]["audited_completed_correct"]
    strong = max(summary[k]["audited_completed_correct"] for k in ["first", "cot", "legacy_mean"])
    checks = {
        "icl_a_gain": (summary["icl_a"]["audited_completed_correct"]-zero)/n >= config["min_icl_gain"],
        "icl_b_gain": (summary["icl_b"]["audited_completed_correct"]-zero)/n >= config["min_icl_gain"],
        "zero_gain": (chosen["audited_correct"]-zero)/n >= config["min_steering_gain"],
        "beats_matched_mean": chosen["audited_correct"] > mean,
        "matches_strong_prompt_or_legacy": chosen["audited_correct"] >= strong,
        "truncation": all(summary[k]["truncated"]/n <= config["max_truncation_rate"]
                          for k in ["zero", "icl_a", "icl_b", chosen["condition"]]),
    }
    return {"eligible": all(checks.values()), "chosen": chosen,
            "checks": checks, "candidates": candidates}

"""Replicate published MATH prompting on a declared subset before fitting directions."""

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def load_upstream(path, expected_commit):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected_commit:
        raise ValueError(f"Upstream revision mismatch: {actual}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("Upstream tracked files must be unmodified")
    # Read prompt constants without importing the upstream GPU runner.
    tree = ast.parse((path / "cot_icl/runners/math.py").read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"PROBLEM_PROMPT", "DEMO_TEMPLATE"}:
                    constants[target.id] = ast.literal_eval(node.value)
    sys.path.insert(0, str(path.resolve()))
    grader = importlib.import_module("cot_icl.grading.math")
    return constants, grader


def render_content(templates, question, demos):
    block = "\n\n".join(templates["DEMO_TEMPLATE"].format(**d) for d in demos) + "\n\n"
    return templates["PROBLEM_PROMPT"].format(demo_prompt=block, problem=question)


def partition(train, test, n_demos, n_screen, seed):
    if len(train) < n_demos or len(test) < n_screen:
        raise ValueError("Insufficient data")
    indices = list(range(len(test)))
    random.Random(seed).shuffle(indices)
    selected = sorted(indices[:n_screen])
    demos = train[:n_demos]
    texts = {d["problem"].strip() for d in demos}
    if any(test[i]["problem"].strip() in texts for i in selected):
        raise ValueError("A screening query overlaps the demonstration pool")
    return {"demonstration_train_indices": list(range(n_demos)),
            "screen_test_indices": selected,
            "reserved_test_indices": sorted(indices[n_screen:]),
            "reserved_train_indices": list(range(n_demos, len(train)))}


def score_response(text, gold, grader):
    def one(part):
        box = grader.last_boxed_only_string(part)
        if box is None:
            return False, None
        parsed = grader.normalize_final_answer(grader.remove_boxed(box))
        return bool(grader.is_equiv(parsed, gold)), parsed
    correct, parsed = one(text)
    # Diagnostic: upstream also grades boxes inside unfinished thinking traces.
    final_correct, final_parsed = one(text.split("</think>", 1)[1]) if "</think>" in text else (False, None)
    return {"correct": correct, "parsed_answer": parsed, "boxed_answer_present": parsed is not None,
            "thinking_finished": "</think>" in text,
            "final_only_correct": final_correct, "final_only_parsed": final_parsed}


def prepare(config, output, upstream):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    templates, grader = load_upstream(upstream, config["upstream_commit"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    plan, jobs, files = {}, [], {}
    max_shots = max(config["shots"])
    for subject in config["subjects"]:
        tables = {}
        for split in ("train", "test"):
            filename = f"{subject}/{split}-00000-of-00001.parquet"
            path = Path(hf_hub_download(config["dataset"], filename, repo_type="dataset",
                                       revision=config["dataset_revision"]))
            files[filename] = sha256(path.read_bytes()).hexdigest()
            tables[split] = pq.read_table(path).to_pylist()
        train, test = tables["train"], tables["test"]
        plan[subject] = partition(train, test, max_shots, config["screen_questions"], config["seed"])
        for shots in config["shots"]:
            demos = train[:shots]
            for index in plan[subject]["screen_test_indices"]:
                question = test[index]
                box = grader.last_boxed_only_string(question["solution"])
                if box is None:
                    raise ValueError(f"Missing gold answer: {subject} test {index}")
                answer = grader.normalize_final_answer(grader.remove_boxed(box))
                content = render_content(templates, question["problem"], demos)
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=config["enable_thinking"])
                prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                if prompt_tokens + config["max_new_tokens"] > config["max_model_len"]:
                    raise ValueError(f"Context would overflow for {subject}, {shots} shots, {index}; no truncation allowed")
                jobs.append({"id": f"{subject}:test:{index}:shots:{shots}",
                    "problem_id": f"{subject}:test:{index}", "subject": subject,
                    "test_index": index, "shots": shots, "answer": answer,
                    "question_sha256": sha256(question["problem"].encode()).hexdigest(),
                    "prompt": prompt, "prompt_tokens": prompt_tokens})
    identity = {"config": config, "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "data_sha256": files, "plan": plan,
                "jobs_sha256": sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()}
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous[k] != v for k, v in identity.items()):
            raise ValueError("Replication inputs changed; use a new output directory")
    else:
        write_json(manifest, {**identity, "created_at": datetime.now(timezone.utc).isoformat()})
        with (output / "prompts.jsonl").open("w") as handle:
            for job in jobs:
                handle.write(json.dumps(job) + "\n")
        write_json(output / "prompt_lengths.json", {
            f"{s}:{n}": {"min": min(j["prompt_tokens"] for j in jobs if j["subject"] == s and j["shots"] == n),
                         "max": max(j["prompt_tokens"] for j in jobs if j["subject"] == s and j["shots"] == n)}
            for s in config["subjects"] for n in config["shots"]})
    (output / "generations.jsonl").touch(exist_ok=True)
    print(f"Prepared {len(jobs)} paired generation jobs", flush=True)
    return jobs, tokenizer, grader


def paired(rows, reference, config):
    import numpy as np
    a = {r["problem_id"]: r for r in rows}
    b = {r["problem_id"]: r for r in reference}
    if len(a) != len(rows) or len(b) != len(reference) or a.keys() != b.keys():
        raise ValueError("Paired rows must have unique, identical question IDs")
    d = np.array([int(a[k]["correct"]) - int(b[k]["correct"]) for k in sorted(a)])
    rng = np.random.default_rng(config["seed"])
    boot = d[rng.integers(0, len(d), size=(config["bootstrap_samples"], len(d)))].mean(1)
    return {"gain": float(d.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def report(output):
    config = json.loads((output / "manifest.json").read_text())["config"]
    rows = [json.loads(line) for line in (output / "generations.jsonl").read_text().splitlines() if line]
    groups = defaultdict(list)
    for row in rows:
        groups[row["subject"], row["shots"]].append(row)
    summaries = []
    lines = ["# MATH replication screen", "", "Fixed screening subset, not a full benchmark replication. No directions have been fitted on these questions.", "",
             "| Subject | Shots | n | Accuracy | Final-only accuracy | Truncated | Mean output tokens |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for (subject, shots), values in sorted(groups.items()):
        n = len(values)
        stats = {"subject": subject, "shots": shots, "n": n,
                 "accuracy": sum(r["correct"] for r in values) / n,
                 "final_only_accuracy": sum(r["final_only_correct"] for r in values) / n,
                 "truncation_rate": sum(r["truncated"] for r in values) / n,
                 "boxed_answer_rate": sum(r["boxed_answer_present"] for r in values) / n,
                 "mean_output_tokens": sum(r["generated_tokens"] for r in values) / n}
        stats["comparisons"] = {}
        for baseline in (0, 16):
            ref = groups.get((subject, baseline))
            if shots != baseline and ref and len(ref) == n == config["screen_questions"]:
                stats["comparisons"][str(baseline)] = paired(values, ref, config)
        summaries.append(stats)
        lines.append(f"| {subject} | {shots} | {n} | {stats['accuracy']:.1%} | {stats['final_only_accuracy']:.1%} | {stats['truncation_rate']:.1%} | {stats['mean_output_tokens']:.0f} |")
    lines += ["", "Accuracy uses the released upstream grader on the entire generated response. Final-only accuracy requires a completed thinking segment and grades only the subsequent answer.",
              "", "## Paired differences", "", "Percentile bootstrap intervals are exploratory, with no multiple-comparison correction.", ""]
    for stats in summaries:
        for baseline, diff in stats["comparisons"].items():
            lo, hi = diff["ci95"]
            lines.append(f"- {stats['subject']}, {stats['shots']} vs {baseline} shots: {diff['gain']:+.1%}, 95% interval [{lo:+.1%}, {hi:+.1%}].")
    expected = len(config["subjects"]) * len(config["shots"]) * config["screen_questions"]
    summary = {"complete": len(rows) == expected, "completed_generations": len(rows),
               "expected_generations": expected, "results": summaries}
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def run(config, output, upstream):
    jobs, tokenizer, grader = prepare(config, output, upstream)
    from importlib.metadata import version
    import torch
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in (output / "generations.jsonl").read_text().splitlines() if line]
    done = {r["id"] for r in rows}
    if len(done) != len(rows) or not done.issubset({j["id"] for j in jobs}):
        raise ValueError("Duplicate or unexpected results")
    pending = [j for j in jobs if j["id"] not in done]
    if not pending:
        report(output)
        return
    write_json(output / "runtime.json", {"versions": {name: version(name) for name in
        ("vllm", "torch", "transformers", "datasets", "sympy", "antlr4-python3-runtime")},
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda})
    engine_keys = ("dtype", "max_model_len", "gpu_memory_utilization", "enable_prefix_caching",
                  "enable_chunked_prefill", "max_num_batched_tokens", "max_num_seqs", "enforce_eager", "rope_scaling")
    llm = LLM(model=config["model"], revision=config["revision"], tokenizer_revision=config["revision"],
              seed=config["seed"], tensor_parallel_size=1, **{k: config[k] for k in engine_keys})
    sampling = SamplingParams(temperature=config["temperature"], top_p=config["top_p"],
        repetition_penalty=config["repetition_penalty"], max_tokens=config["max_new_tokens"],
        stop_token_ids=[tokenizer.eos_token_id], seed=config["seed"])
    # Keep each resumable batch within one subject and shot count.
    grouped = defaultdict(list)
    for job in pending:
        grouped[job["subject"], job["shots"]].append(job)
    for (subject, shots), group in grouped.items():
        for start in range(0, len(group), config["batch_size"]):
            batch = group[start:start + config["batch_size"]]
            begin = time.monotonic()
            outputs = llm.generate([j["prompt"] for j in batch], sampling_params=sampling)
            elapsed = time.monotonic() - begin
            with (output / "generations.jsonl").open("a") as handle:
                for job, result in zip(batch, outputs):
                    answer = result.outputs[0]
                    row = {k: v for k, v in job.items() if k != "prompt"}
                    row.update(text=answer.text, generated_tokens=len(answer.token_ids),
                               truncated=answer.finish_reason == "length", finish_reason=answer.finish_reason,
                               batch_seconds=elapsed, **score_response(answer.text, job["answer"], grader))
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
            print(f"{subject} {shots} shots: {min(start + len(batch), len(group))}/{len(group)} new, {elapsed:.1f}s", flush=True)
            report(output)
    summary = report(output)
    if summary["complete"]:
        write_json(output / "complete.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
                   "status": "replication_screen_complete", "next": "Inspect paired ICL gains before fitting steering directions."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--config", type=Path, default=Path("configs/math_replication.json"))
    parser.add_argument("--upstream", type=Path, default=Path("research/upstream/manyshot-cot-icl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "report":
        report(args.output)
        return
    config = json.loads(args.config.read_text())
    if args.stage == "prepare":
        prepare(config, args.output, args.upstream)
    else:
        run(config, args.output, args.upstream)


if __name__ == "__main__":
    main()

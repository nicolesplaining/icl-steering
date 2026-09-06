"""Small, auditable reproduction of the GSM8K unsupervised-ICL setup.

This follows Gadetsky et al. (ICLR 2025) closely: Qwen2.5-Math zero-shot CoT
answers initialize a support pool, then several rounds of self-generated
few-shot prompting refine the pool.  The paper-compatible mode uses the test
pool for both adaptation and scoring because that is what the released script
does.  The heldout mode adapts on train examples and scores on test examples.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from typing import Any


QWEN_ZERO_HEAD = (
    "Given the following question, reason step by step, and put your final "
    "answer within \\\\boxed{{}}. Your response should end with \"The final "
    "answer is [answer]\" where [answer] is the response to the problem.\n\n"
    "Problem: {question}\nAnswer: Let's think step by step."
)

ICL_HEAD = (
    "Given the following problem, reason and give a final answer to the problem. "
    "Your response should end with \"The answer is [answer]\" where [answer] "
    "is the response to the problem.\n\n"
)


def answer_from_gsm8k(text: str) -> int | float:
    match = re.search(r"#### *(-?[0-9%$]+(?:[,.]\d+)*)", text)
    if match is None:
        raise ValueError("GSM8K answer has no #### marker")
    raw = match.group(1).replace("$", "").replace("%", "").replace("g", "").replace(",", "")
    value = float(raw) if "." in raw else int(raw)
    return value


def prepare_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        answer = answer_from_gsm8k(row["answer"])
        reasoning = row["answer"].split("\n####", 1)[0]
        prepared.append({
            "question": row["question"],
            "answer": answer,
            "raw_response": reasoning + f"\nThe final answer is {answer}.",
        })
    return prepared


def extract_answer(response: str) -> int | float | str:
    match = re.search(r"The(?: final)? answer is .*?(\d[\d\\$€,(){}.]*|\d)", response)
    if match is None:
        return ""
    answer = match.group(1).rstrip(".")
    for char in ("$", "%", "g", ",", "{", "}", "(", ")"):
        answer = answer.replace(char, "")
    try:
        return float(answer) if "." in answer else int(answer)
    except ValueError:
        return ""


def zero_prompt(example: dict[str, Any]) -> str:
    return QWEN_ZERO_HEAD.format(question=example["question"])


def icl_prompt(
    support: list[dict[str, Any]],
    query: dict[str, Any],
    shots: int,
    rng: random.Random,
    exclude_query: bool = False,
) -> str:
    candidates = [x for x in support if not exclude_query or x["question"] != query["question"]]
    if len(candidates) < shots:
        raise ValueError(f"need {shots} support examples, found {len(candidates)}")
    chosen = rng.sample(candidates, shots)
    demonstrations = "".join(
        f"Q: {example['question']}\nA:{example['raw_response']}\n\n" for example in chosen
    )
    return ICL_HEAD + demonstrations + f"Q: {query['question']}\nA:"


def supervised_prompt(support: list[dict[str, Any]], query: dict[str, Any], shots: int) -> str:
    if len(support) < shots:
        raise ValueError(f"need {shots} supervised examples, found {len(support)}")
    demonstrations = "".join(
        f"Q: {example['question']}\nA:{example['raw_response']}\n\n" for example in support[:shots]
    )
    return ICL_HEAD + demonstrations + f"Q: {query['question']}\nA:"


def majority(outputs: list[str]) -> dict[str, Any]:
    parsed = [extract_answer(text) for text in outputs]
    valid = [answer for answer in parsed if answer != ""]
    if not valid:
        return {"raw_response": outputs[0], "answer": "", "formatted": False}
    answer = Counter(valid).most_common(1)[0][0]
    response = next(text for text, value in zip(outputs, parsed) if value == answer)
    return {"raw_response": response, "answer": answer, "formatted": True}


def score(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [pred["answer"] != "" and pred["answer"] == row["answer"] for row, pred in zip(rows, predictions)]
    return {
        "n": len(rows),
        "accuracy": sum(correct) / len(correct),
        "formatted_rate": sum(pred["formatted"] for pred in predictions) / len(predictions),
        "correct": correct,
    }


def generate(model: Any, prompts: list[str], sampling: Any, num_repeats: int) -> list[list[str]]:
    outputs = model.generate(prompts, sampling)
    flat = [item.text for result in outputs for item in result.outputs]
    return [flat[i * num_repeats : (i + 1) * num_repeats] for i in range(len(prompts) // num_repeats)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    dataset = load_dataset("gsm8k", "main")
    train = prepare_examples([dict(x) for x in dataset["train"]])
    test = prepare_examples([dict(x) for x in dataset["test"]])
    if args.mode == "paper":
        adaptation = test[: args.adaptation_examples]
        evaluation = test[: args.evaluation_examples]
    else:
        adaptation = train[: args.adaptation_examples]
        evaluation = test[: args.evaluation_examples]
    if len(adaptation) < args.shots:
        raise ValueError("adaptation pool must contain at least --shots examples")

    model = LLM(args.model, tensor_parallel_size=torch.cuda.device_count(), seed=args.seed)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        n=1,
        seed=args.seed,
        stop=["\n\nQ:"],
    )

    def ask(prompts: list[str], repeats: int = 1) -> list[list[str]]:
        expanded = [prompt for prompt in prompts for _ in range(repeats)]
        return generate(model, expanded, sampling, repeats)

    zero_preds = [majority(xs) for xs in ask([zero_prompt(x) for x in evaluation])]
    supervised_preds = [
        majority(xs)
        for xs in ask([supervised_prompt(train, x, args.shots) for x in evaluation])
    ]
    support_preds = [majority(xs) for xs in ask([zero_prompt(x) for x in adaptation])]
    support = [dict(x, **pred) for x, pred in zip(adaptation, support_preds) if pred["formatted"]]
    rounds = []
    for turn in range(args.turns):
        prompts = [icl_prompt(support, x, args.shots, rng, args.exclude_query) for x in adaptation]
        candidate_sets = ask(prompts, args.num_repeats)
        current = [majority(xs) for xs in candidate_sets]
        support = [dict(x, **pred) for x, pred in zip(adaptation, current) if pred["formatted"]]
        eval_prompts = [icl_prompt(support, x, args.shots, rng, False) for x in evaluation]
        eval_preds = [majority(xs) for xs in ask(eval_prompts, args.num_repeats)]
        rounds.append({"turn": turn + 1, "adaptation": score(adaptation, current), "evaluation": score(evaluation, eval_preds)})

    result = {
        "args": vars(args),
        "model": args.model,
        "mode": args.mode,
        "zero_shot": score(evaluation, zero_preds),
        "supervised_icl": score(evaluation, supervised_preds),
        "rounds": rounds,
        "gpu": torch.cuda.get_device_name(0),
    }
    if args.save_predictions:
        result["adaptation_support"] = support
        result["evaluation"] = [
            {"question": row["question"], "answer": row["answer"], "zero_shot": pred}
            for row, pred in zip(evaluation, zero_preds)
        ]
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--mode", choices=("paper", "heldout"), default="paper")
    parser.add_argument("--adaptation-examples", type=int, default=128)
    parser.add_argument("--evaluation-examples", type=int, default=128)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-query", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2), flush=True)

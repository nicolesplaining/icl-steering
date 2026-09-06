"""CPU checks shared by the corrected GSM8K experiment and the legacy audit."""

import hashlib
import json
import random
import re
from fractions import Fraction

from icl_steering.scoring import boxed_contents, numeric_value


HEAD = ('Given the following problem, reason and give a final answer to the problem. '
        'Your response should end with "The answer is [answer]" where [answer] '
        'is the response to the problem.\n\n')
NEXT_QUESTION = re.compile(r"\n\s*(?:Q:|Problem:|Given the following (?:question|problem)[, ])")
FINAL = re.compile(r"(?im)(?:\bthe\s+)?(?:final\s+)?answer\s*(?:is\b\s*:?[ \t]*|:[ \t]*)")
NUMBER = re.compile(r"[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def solution_text(text):
    """Never grade a subsequent generated demonstration as the query answer."""
    return NEXT_QUESTION.split(text, maxsplit=1)[0].rstrip()


def parse_answer(text):
    """Last explicit box, otherwise last answer marker; no arbitrary last-number fallback.

    Parsing never sees the gold answer. Unsupported or conflicting expressions
    fail closed instead of harvesting a convenient intermediate number.
    """
    text = solution_text(text)
    boxes = boxed_contents(text)
    candidate, source = (boxes[-1], "boxed") if boxes else (None, None)
    if candidate is None:
        markers = list(FINAL.finditer(text))
        if markers:
            tail = text[markers[-1].end():].split("\n", 1)[0].strip()
            tail = tail.replace(r"\(", "").replace(r"\)", "").replace("**", "")
            tail = tail.lstrip(" [$€£")
            number = NUMBER.match(tail)
            if number is None and not re.search(r"[\\(^+*/=]", tail):
                numbers = list(NUMBER.finditer(tail))
                if len(numbers) == 1:
                    number = numbers[0]
            if number:
                rest = tail[number.end():].lstrip()
                # Reject arithmetic, alternatives, units with numeric exponents,
                # and incomplete decimals rather than silently taking a prefix.
                if not re.search(r"\d|[+*/=]", rest) and not re.match(r"(?:or\b|and\b|-)", rest):
                    candidate, source = number.group(), "answer_marker"
    try:
        value = numeric_value(candidate) if candidate is not None else None
    except (ValueError, ZeroDivisionError):
        value = None
    return {"parsed_answer": str(value) if value is not None else None,
            "parseable": value is not None, "answer_source": source}


def grade_answer(text, gold, truncated=False):
    parsed = parse_answer(text)
    correct = parsed["parsed_answer"] is not None and Fraction(parsed["parsed_answer"]) == Fraction(str(gold))
    return {**parsed, "correct": correct, "completed_correct": correct and not truncated}


def matched_prompt(question, demonstrations=(), suffix=""):
    if any(d["question"].strip() == question.strip() for d in demonstrations):
        raise ValueError("Query appears in its demonstrations")
    demos = "".join(f"Q: {d['question']}\nA:{solution_text(d['raw_response'])}\n\n" for d in demonstrations)
    return HEAD + demos + f"Q: {question}\nA:" + suffix


def partition(train, test, support_questions, config):
    """Reserve all previously screened test questions and the support train pool."""
    norm = lambda x: " ".join(x.split())
    used = {norm(q) for q in support_questions}
    train_ids = list(range(config["reserved_train"], len(train)))
    test_ids = list(range(config["reserved_test"], len(test)))
    rng = random.Random(config["seed"])
    rng.shuffle(train_ids)
    rng.shuffle(test_ids)
    groups = {}
    for split, ids, count, data in (
        ("extract", train_ids[:config["n_extract"]], config["n_extract"], train),
        ("validation", train_ids[config["n_extract"]:config["n_extract"]+config["n_validation"]], config["n_validation"], train),
        ("test", test_ids[:config["n_test"]], config["n_test"], test),
    ):
        if len(ids) != count or count < 2:
            raise ValueError("Insufficient data for declared split sizes")
        for i in ids:
            q = norm(data[i]["question"])
            if q in used:
                raise ValueError("Duplicate question across support/extraction/validation/test")
            used.add(q)
        groups[split] = sorted(ids)
    return groups


def select_candidate(candidates, baseline_accuracy, min_gain):
    if not candidates:
        raise ValueError("No completed validation candidates")
    # Only positive ICL shifts enter tuning. Controls never enter selection.
    best = min(candidates, key=lambda r: (-r["completed_accuracy"], r["alpha"], r["layer"], r["positions"]))
    return {"chosen": best, "validation_gain": best["completed_accuracy"] - baseline_accuracy,
            "eligible": best["completed_accuracy"] - baseline_accuracy >= min_gain,
            "rule": "Max validation completed accuracy; ties use alpha, layer, positions in ascending order."}

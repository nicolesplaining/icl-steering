"""Conservative numeric grading. Never evaluate arbitrary model-generated code."""

from fractions import Fraction
import re


def boxed_contents(text):
    boxes = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    boxes.append(text[start:end])
                    break
    return boxes


def numeric_value(text):
    value = text.strip().replace(r"\,", "").replace(r"\!", "")
    value = value.replace("−", "-").replace(",", "").replace(" ", "")
    fraction = re.fullmatch(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", value)
    if fraction:
        return Fraction(int(fraction[1]), int(fraction[2]))
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", value):
        return Fraction(value)
    return None


def grade(text, answer):
    boxes = boxed_contents(text)
    candidate = boxes[-1] if boxes else None
    if candidate is None:
        finals = re.findall(r"(?im)^\s*(?:final answer|answer)\s*:\s*([^\n]+)", text)
        if finals:
            candidate = finals[-1].strip().rstrip(".")
    try:
        value = numeric_value(candidate) if candidate is not None else None
    except (ValueError, ZeroDivisionError):
        value = None
    return {"correct": value == answer if value is not None else False,
            "parsed_answer": str(value) if value is not None else None,
            "parseable": value is not None}


def method_signature(text, family):
    """A trace-labeling aid, not a validated judge or a causal claim."""
    lower = text.lower()
    if family == "quadratic_sum":
        a = bool(re.search(r"finite[- ]difference|second differences?|\\delta|binom\s*\{?n|binom\(n|newton", lower))
        b = bool(re.search(r"coefficients?|quadratic polynomial|an\^?2|a[nj]\^?2|a\s*\\?cdot\s*[nj]\^?2", lower))
    else:
        a = bool(re.search(r"recurren|p[_\{]?\s*[nk].*p[_\{]?\s*[nk].*[-−]\s*[12]|newton.s identities", lower, re.S))
        b = bool(re.search(r"binomial|quadratic formula|sqrt|\\sqrt", lower))
    return "both" if a and b else "a" if a else "b" if b else "unclear"

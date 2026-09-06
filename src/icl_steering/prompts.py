"""All target prompts ask for work; worked-example conditions differ in method only."""

METHOD_INSTRUCTIONS = {
    "quadratic_sum": "Use finite differences and the binomial-coefficient representation of the sequence to sum it.",
    "symmetric_power": "Use the recurrence P_j=(x+y)P_(j-1)-xy*P_(j-2), starting with P_0=2 and P_1=x+y.",
}


def messages(problem, demos=(), condition="zero"):
    prefix = "Solve the mathematics problem. Show your work and put the final answer in \\boxed{...}."
    if condition == "cot":
        prefix += " Think step by step."
    elif condition == "instruction":
        prefix += " " + METHOD_INSTRUCTIONS[problem.family]
    elif condition not in {"zero", "first", "icl_a", "icl_b"}:
        raise ValueError(condition)
    content = prefix + "\n\n"
    if condition in {"icl_a", "icl_b"}:
        method = "method_a" if condition == "icl_a" else "method_b"
        for demo in demos:
            content += f"Problem: {demo.question}\nSolution: {getattr(demo, method)}\n\n"
    content += f"Problem: {problem.question}\nSolution:"
    return [{"role": "user", "content": content}]

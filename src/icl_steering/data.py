"""Exact, procedural math instances with two independently checked solution paths."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import comb, isqrt
import random


FAMILIES = ("quadratic_sum", "symmetric_power")


@dataclass(frozen=True)
class Problem:
    id: str
    family: str
    split: str
    params: dict
    question: str
    answer: int
    method_a: str
    method_b: str

    def to_dict(self):
        return asdict(self)


def quadratic_sum(a, b, c, n, split, wording=0):
    terms = [a * j * j + b * j + c for j in range(1, 6)]
    u, d, dd = terms[0], terms[1] - terms[0], 2 * a
    answer = sum(a * j * j + b * j + c for j in range(1, n + 1))
    assert answer == n * u + comb(n, 2) * d + comb(n, 3) * dd
    displayed = ", ".join(map(str, terms))
    if wording == 0:
        q = f"The sequence begins {displayed}. Its nth term is a quadratic polynomial in n, starting at n=1. Find the sum of its first {n} terms."
    elif wording == 1:
        q = f"A quadratic sequence has values u_1 through u_5 equal to {displayed}, respectively. Calculate u_1 + u_2 + ... + u_{n}."
    else:
        q = f"A polynomial P of degree at most two satisfies P(1)={terms[0]}, P(2)={terms[1]}, P(3)={terms[2]}, P(4)={terms[3]}, and P(5)={terms[4]}. Evaluate the sum of P(j) for j=1,...,{n}."
    sol_a = (
        f"The first differences are {[terms[i+1]-terms[i] for i in range(4)]}; "
        f"the second difference is {dd}. Use the finite-difference representation "
        f"u_j={u}+(j-1)*({d})+binom(j-1,2)*({dd}). "
        f"Summing gives S_N=N*({u})+binom(N,2)*({d})+binom(N,3)*({dd}). "
        f"Thus S_{n}={n}*({u})+{comb(n,2)}*({d})+{comb(n,3)}*({dd})={answer}. "
        f"Final answer: \\boxed{{{answer}}}."
    )
    sol_b = (
        f"Write u_j=A*j^2+B*j+C. The first three terms give "
        f"A+B+C={terms[0]}, 4A+2B+C={terms[1]}, 9A+3B+C={terms[2]}. "
        f"Solving for the coefficients gives A={a}, B={b}, C={c}. "
        f"Use sum(j^2)=N*(N+1)*(2N+1)/6 and sum(j)=N*(N+1)/2. "
        f"S_{n}=({a})*{n*(n+1)*(2*n+1)//6}+({b})*{n*(n+1)//2}+({c})*{n}={answer}. "
        f"Final answer: \\boxed{{{answer}}}."
    )
    return Problem(f"quadratic_sum:{a}:{b}:{c}:{n}", "quadratic_sum", split,
                   {"a": a, "b": b, "c": c, "n": n}, q, answer, sol_a, sol_b)


def symmetric_power(s, p, k, split, wording=0):
    disc = s*s - 4*p
    assert disc > 0 and isqrt(disc)**2 != disc
    powers = [2, s]
    for j in range(2, k + 1):
        powers.append(s*powers[-1] - p*powers[-2])
    answer = powers[k]
    expansion_terms = [comb(k, 2*j) * s**(k-2*j) * disc**j for j in range(k//2+1)]
    numerator = sum(expansion_terms)
    assert numerator % (2**(k-1)) == 0
    assert answer == numerator // 2**(k-1)
    if wording == 0:
        q = f"Two real numbers x and y satisfy x+y={s} and xy={p}. Find x^{k}+y^{k}."
    elif wording == 1:
        q = f"The sum of two real numbers is {s} and their product is {p}. What is the sum of their {k}th powers?"
    else:
        q = f"Let r and t be the two real roots of z^2 - ({s})z + ({p}) = 0. Evaluate r^{k}+t^{k}."
    steps = "; ".join(f"P_{j}=({s})*({powers[j-1]})-({p})*({powers[j-2]})={powers[j]}" for j in range(2, k+1))
    sol_a = (
        f"Let P_j=x^j+y^j. Since x and y satisfy z^2-({s})z+({p})=0, "
        f"the recurrence is P_j=({s})P_(j-1)-({p})P_(j-2), with P_0=2 and P_1={s}. "
        f"Compute {steps}. Final answer: \\boxed{{{answer}}}."
    )
    sol_b = (
        f"Solve for the roots: x=({s}+sqrt({disc}))/2 and y=({s}-sqrt({disc}))/2. "
        f"Apply the binomial theorem to both {k}th powers. Terms with odd powers of the square root cancel. "
        f"The sum is 1/{2**(k-1)} times the sum of binom({k},2j)*({s})^({k}-2j)*({disc})^j "
        f"for j=0,...,{k//2}. The numerator terms are {expansion_terms}, summing to {numerator}. "
        f"The result is {numerator}/{2**(k-1)}={answer}. Final answer: \\boxed{{{answer}}}."
    )
    return Problem(f"symmetric_power:{s}:{p}:{k}", "symmetric_power", split,
                   {"s": s, "p": p, "k": k}, q, answer, sol_a, sol_b)


def make_dataset(config):
    """Generate splits together, rejecting repeated parameter tuples across all splits."""
    rng = random.Random(config["seed"])
    result = []
    seen = set()
    for family in config["families"]:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for split, count in config["counts"].items():
            i = 0
            while i < count:
                # Test contains an additional wording, with matching numerical difficulty.
                wording = i % (3 if split == "test" else 2)
                if family == "quadratic_sum":
                    item = quadratic_sum(rng.randint(1, 12), rng.randint(-20, 20),
                                         rng.randint(-30, 30), rng.randint(15, 75), split, wording)
                else:
                    s = rng.randint(3, 15)
                    p = rng.randint(-12, (s*s-1)//4)
                    if isqrt(s*s-4*p)**2 == s*s-4*p:
                        continue
                    item = symmetric_power(s, p, rng.randint(5, 9), split, wording)
                if item.id not in seen:
                    seen.add(item.id)
                    result.append(item)
                    i += 1
    return result


def demonstrations(problem, dataset, shots, seed):
    bank = [p for p in dataset if p.family == problem.family and p.split == "demonstration"]
    digest = sha256(f"{seed}:{problem.id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).sample(bank, shots)

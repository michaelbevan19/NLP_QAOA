"""
Classical baseline search methods (report Section 5 step 10, Section 6).

Design decision (confirmed with the user, since the report's wording --
"solving the identical QUBO or the identical underlying problem" -- was
ambiguous about whether these baselines call the real LLM):

    Random search, greedy removal, and simulated annealing (the "classical
    QUBO solver") ALL optimize the SAME static Q matrix numerically --
    bitstring_cost() from qubo.py, no LLM calls involved. This makes all
    three, plus standard QAOA and NLP-QAOA, alternative ALGORITHMS for
    minimizing the identical matrix, so evaluate.py's comparison table
    isolates search-method quality on a fixed landscape. QAOA's real
    per-iteration LLM calls (circuit.py, report Section 4.6) then become a
    genuine, measurable cost that QAOA pays and these three don't -- which
    is exactly what the research question's "bounded number of LLM calls"
    framing is asking about.

    Each function below therefore reports `llm_calls=0`: the only LLM
    calls in these methods happen once, later, in evaluate.py, when the
    winning bitstring from EVERY method (baselines and QAOA alike) is
    turned into a prompt and sent to the LLM once for final verification
    (O_optimized, report Section 4.7).

brute_force() is not one of the 5 compared methods -- it's the classical
validation checkpoint required by report Section 5 step 7 ("validate the
QUBO formulation using a classical solver before implementing the quantum
circuit"). It enumerates all 2^n bitstrings, which is only tractable
because n is capped at 8-12 (report Section 4.2).
"""

import math
import random

from qubo import bitstring_cost


def brute_force(n: int, Q: list[list[float]]) -> dict:
    """Enumerates all 2^n bitstrings and returns the true global optimum.
    Only used as a validation checkpoint (n <= 12 makes this trivial)."""
    best_bits, best_cost = None, math.inf
    for mask in range(2 ** n):
        bits = [(mask >> i) & 1 for i in range(n)]
        cost = bitstring_cost(bits, Q)
        if cost < best_cost:
            best_bits, best_cost = bits, cost
    return {"bitstring": best_bits, "cost": best_cost, "llm_calls": 0}


def random_search(n: int, Q: list[list[float]], budget: int = 200, seed: int | None = None) -> dict:
    """Samples `budget` uniformly random bitstrings, returns the best found."""
    rng = random.Random(seed)
    best_bits, best_cost = None, math.inf
    for _ in range(budget):
        bits = [rng.randint(0, 1) for _ in range(n)]
        cost = bitstring_cost(bits, Q)
        if cost < best_cost:
            best_bits, best_cost = bits, cost
    return {"bitstring": best_bits, "cost": best_cost, "llm_calls": 0}


def greedy_removal(n: int, Q: list[list[float]]) -> dict:
    """Starts from 'keep everything' and repeatedly removes whichever
    single currently-kept token most reduces cost, stopping when no
    single removal helps. Classic greedy hill-climbing on Q."""
    bits = [1] * n
    cost = bitstring_cost(bits, Q)
    improved = True
    while improved:
        improved = False
        best_i, best_cost = None, cost
        for i in range(n):
            if bits[i] == 0:
                continue
            trial = list(bits)
            trial[i] = 0
            trial_cost = bitstring_cost(trial, Q)
            if trial_cost < best_cost:
                best_i, best_cost = i, trial_cost
        if best_i is not None:
            bits[best_i] = 0
            cost = best_cost
            improved = True
    return {"bitstring": bits, "cost": cost, "llm_calls": 0}


def simulated_annealing(
    n: int,
    Q: list[list[float]],
    iterations: int = 1000,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.995,
    seed: int | None = None,
) -> dict:
    """
    The "classical QUBO solver" baseline. Standard single-bit-flip
    simulated annealing: propose flipping one random bit, accept
    downhill moves always, accept uphill moves with probability
    exp(-delta/T), cool T geometrically, track the best bitstring seen.
    """
    rng = random.Random(seed)
    bits = [rng.randint(0, 1) for _ in range(n)]
    cost = bitstring_cost(bits, Q)
    best_bits, best_cost = list(bits), cost

    temp = initial_temp
    for _ in range(iterations):
        i = rng.randrange(n)
        trial = list(bits)
        trial[i] = 1 - trial[i]
        trial_cost = bitstring_cost(trial, Q)
        delta = trial_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            bits, cost = trial, trial_cost
            if cost < best_cost:
                best_bits, best_cost = list(bits), cost

        temp *= cooling_rate

    return {"bitstring": best_bits, "cost": best_cost, "llm_calls": 0}


if __name__ == "__main__":
    from qubo import assemble_qubo

    # Same toy example as qubo.py's sanity check 2.
    toy_candidates = [{"text": "please"}, {"text": "classify"}, {"text": "categorize"}, {"text": "positive"}]
    toy_importance = [0.1, 0.9, 0.85, 0.95]
    toy_redundant_pairs = [(1, 2)]
    Q = assemble_qubo(toy_candidates, toy_importance, toy_redundant_pairs)
    n = len(toy_candidates)

    true_opt = brute_force(n, Q)
    print(f"brute_force (ground truth): {true_opt}")

    for name, result in [
        ("random_search", random_search(n, Q, budget=200, seed=0)),
        ("greedy_removal", greedy_removal(n, Q)),
        ("simulated_annealing", simulated_annealing(n, Q, iterations=500, seed=0)),
    ]:
        match = "MATCHES optimum" if result["cost"] == true_opt["cost"] else "does NOT match optimum"
        print(f"{name:22s}: {result}  [{match}]")

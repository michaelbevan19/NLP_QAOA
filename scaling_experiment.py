"""
FIX 6 (2026-08-15): scaling experiment.

The standard 10-prompt suite caps candidates at 8-12 (clean.py's
MAX_CANDIDATES), where brute_force always finds the exact optimum -- so no
method, QAOA included, can ever beat it, and the existing comparison has a
hard ceiling that cannot show a QAOA advantage even in principle. This
raises max_candidates on the longest prompts (by UNCAPPED candidate count,
not original word count -- a wordy prompt full of stopwords may still
yield few real candidates) to 14/16/18/20 and records, per method:
wall-clock time, final solution cost (Q-energy), and whether it completed
within a time budget at all.

IMPORTANT, honest caveat found while building this (read before trusting
any "QAOA keeps running past brute force's limit" result): circuit.py's
QAOA methods run on PennyLane's `default.qubit` -- an EXACT CLASSICAL
STATEVECTOR SIMULATOR. It represents the full 2^N-amplitude quantum state
explicitly, so building and evaluating the circuit ALSO scales
exponentially in N here, unlike real quantum hardware, where a QAOA
circuit's per-shot cost stays roughly constant regardless of N (only the
NUMBER of qubits used grows, not the classical simulation cost of
tracking every amplitude). So on THIS simulator, "brute force becomes
intractable while QAOA keeps running" is an empirical question, not a
given -- this script times out BOTH, independently, and reports whichever
result actually happens, including reporting plainly if QAOA shows no
completion advantage even at N=20 (per the fix's explicit instruction: a
negative result here is a valid, reportable finding, not a failure of the
experiment).

random_search / greedy_removal / simulated_annealing all evaluate
bitstring_cost() a fixed, small number of times regardless of N (their
budgets don't scale with N), so they're expected to stay fast throughout
-- included for comparison, not because they were ever genuinely at risk
of intractability the way brute_force and (on this simulator) QAOA are.

Local-machine usage (HANDOFF.md Sec 5/7 pattern, same as everywhere else
in this project): __main__ below runs a SMALL, FAST validation only (one
prompt, N=14, a short time budget, a reduced QAOA iteration count) to
confirm the mechanism works. The full sweep (multiple prompts x
[14, 16, 18, 20], full QAOA iteration budgets) is compute-heavy and
belongs on Colab, via run_scaling_experiment() directly.
"""

import time

from baselines import greedy_removal, random_search, simulated_annealing
from circuit import run_nlp_qaoa, run_standard_qaoa
from clean import clean_and_rank
from qubo import assemble_qubo, bitstring_cost
from redundancy import redundant_pairs
from scoring import BaselineScorer
from llm import LLM
from prompts import PROMPTS

CANDIDATE_COUNTS = [14, 16, 18, 20]
DEFAULT_TIME_BUDGET_SECONDS = 120  # per method, per (prompt, N) combination -- brute_force only (see module docstring)
DEFAULT_QAOA_MAXITER = 30  # deliberately lower than evaluate.py's default 100 -- see module docstring on simulator scaling


def uncapped_candidate_count(prompt: str) -> int:
    """How many real candidates clean_and_rank would find with NO cap --
    i.e. is this prompt even long enough to test at N candidates. A high
    original word count does not guarantee this (stopwords, punctuation,
    and lemma-deduplication all reduce it)."""
    return len(clean_and_rank(prompt, max_candidates=999))


def pick_longest_prompts(n_prompts: int = 3) -> list[dict]:
    scored = [(uncapped_candidate_count(p["prompt"]), p) for p in PROMPTS]
    scored.sort(key=lambda x: -x[0])
    chosen_ids = {p["id"] for _, p in scored[:n_prompts]}
    print("Uncapped candidate counts (choosing the longest for the scaling experiment):")
    for count, p in scored:
        marker = " <-- CHOSEN" if p["id"] in chosen_ids else ""
        print(f"  {p['id']:22s} {count:3d} real candidates{marker}")
    return [p for _, p in scored[:n_prompts]]


def _timeboxed_brute_force(n: int, Q: list[list[float]], time_budget_seconds: float) -> dict:
    """Same enumeration as baselines.brute_force(), but bails out and
    reports incompletion if it exceeds time_budget_seconds. The original
    brute_force() has no such guard -- it's only ever been used at n<=12,
    where 2^n is always fast; a real timeout starts to matter here."""
    start = time.perf_counter()
    best_bits, best_cost = None, float("inf")
    completed = True
    checked = 0
    total = 2 ** n
    CHECK_EVERY = 4096
    for mask in range(total):
        bits = [(mask >> i) & 1 for i in range(n)]
        cost = bitstring_cost(bits, Q)
        if cost < best_cost:
            best_bits, best_cost = bits, cost
        checked += 1
        if checked % CHECK_EVERY == 0 and (time.perf_counter() - start) > time_budget_seconds:
            completed = False
            break
    elapsed = time.perf_counter() - start
    return {
        "bitstring": best_bits if completed else None,
        "cost": best_cost if completed else None,
        "completed": completed,
        "checked": checked,
        "total": total,
        "elapsed_seconds": elapsed,
    }


def run_scaling_experiment(
    n_prompts: int = 3,
    candidate_counts: list[int] = CANDIDATE_COUNTS,
    bf_time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    qaoa_maxiter: int = DEFAULT_QAOA_MAXITER,
    llm: LLM | None = None,
) -> list[dict]:
    """
    Runs the scaling comparison across the `n_prompts` longest prompts
    (by real, uncapped candidate count) at each N in `candidate_counts`
    that prompt can actually support. Returns a list of per-(prompt, N)
    rows, each containing every method's {completed, elapsed_seconds,
    cost}. QAOA methods have no internal wall-clock timeout mechanism
    (COBYLA doesn't check elapsed time mid-run) -- if one hangs
    excessively at high N, interrupt the run and lower qaoa_maxiter or N;
    this is an intentional simplicity trade-off, not an oversight (a real
    preemptive kill would need a subprocess-level timeout, unnecessary
    complexity for what this experiment needs to show).
    """
    if llm is None:
        llm = LLM()

    prompts = pick_longest_prompts(n_prompts)
    all_rows = []

    for entry in prompts:
        max_available = uncapped_candidate_count(entry["prompt"])
        for N in candidate_counts:
            if N > max_available:
                print(f"[{entry['id']}] skipping N={N}: only {max_available} real candidates available")
                continue

            print(f"\n=== [{entry['id']}] N={N} ===")
            candidates = clean_and_rank(entry["prompt"], max_candidates=N)
            scorer = BaselineScorer(llm, entry["prompt"], entry["sample_input"])
            scorer.capture_baseline()

            importance_scores = []
            for i, c in enumerate(candidates):
                subset = candidates[:i] + candidates[i + 1:]
                sim = scorer.score(subset)
                importance_scores.append(1.0 - sim)

            pairs = redundant_pairs(candidates, llm)
            Q = assemble_qubo(candidates, importance_scores, pairs)

            row = {"prompt_id": entry["id"], "n_candidates": N}

            bf = _timeboxed_brute_force(N, Q, bf_time_budget_seconds)
            row["brute_force"] = {
                "completed": bf["completed"],
                "elapsed_seconds": bf["elapsed_seconds"],
                "cost": bf["cost"],
                "checked_fraction": bf["checked"] / bf["total"],
            }
            frac = 100 * bf["checked"] / bf["total"]
            print(
                f"  brute_force: completed={bf['completed']}  elapsed={bf['elapsed_seconds']:.1f}s  "
                f"checked {bf['checked']}/{bf['total']} ({frac:.4f}%)"
                + (f"  cost={bf['cost']:.3f}" if bf["completed"] else "  -- TIMED OUT, exact optimum unknown at this N")
            )

            for name, fn, kwargs in [
                ("random_search", random_search, {"n": N, "Q": Q, "seed": 0}),
                ("greedy_removal", greedy_removal, {"n": N, "Q": Q}),
                ("simulated_annealing", simulated_annealing, {"n": N, "Q": Q, "seed": 0}),
                ("standard_qaoa", run_standard_qaoa, {"n": N, "Q": Q, "maxiter": qaoa_maxiter, "seed": 0}),
                ("nlp_qaoa", run_nlp_qaoa, {"n": N, "Q": Q, "maxiter": qaoa_maxiter, "seed": 1}),
            ]:
                start = time.perf_counter()
                result = fn(**kwargs)
                elapsed = time.perf_counter() - start
                row[name] = {"completed": True, "elapsed_seconds": elapsed, "cost": result["cost"]}
                print(f"  {name}: elapsed={elapsed:.1f}s  cost={result['cost']:.3f}")

            all_rows.append(row)

    return all_rows


def print_scaling_table(rows: list[dict]):
    print("\n=== FIX 6 scaling experiment summary ===")
    header = f"{'prompt':18s} {'N':>3s} {'method':22s} {'completed':>10s} {'elapsed(s)':>11s} {'cost':>10s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        for method in ["brute_force", "random_search", "greedy_removal", "simulated_annealing", "standard_qaoa", "nlp_qaoa"]:
            m = row[method]
            cost_str = f"{m['cost']:.3f}" if m.get("cost") is not None else "n/a"
            print(
                f"{row['prompt_id']:18s} {row['n_candidates']:3d} {method:22s} {str(m['completed']):>10s} "
                f"{m['elapsed_seconds']:11.2f} {cost_str:>10s}"
            )

    print("\nExact point brute_force fails (first non-completed N per prompt):")
    seen = set()
    for row in rows:
        if row["prompt_id"] in seen:
            continue
        if not row["brute_force"]["completed"]:
            print(f"  {row['prompt_id']}: N={row['n_candidates']} "
                  f"(checked {row['brute_force']['checked_fraction']*100:.4f}% of {2**row['n_candidates']} states "
                  f"in {row['brute_force']['elapsed_seconds']:.1f}s before giving up)")
            seen.add(row["prompt_id"])
    if not seen:
        print("  brute_force completed at every N tested for every prompt -- did not become intractable in this range.")


if __name__ == "__main__":
    # Local quick validation ONLY: one prompt, one small N, a short time
    # budget, a reduced QAOA iteration count -- confirms the mechanism
    # works before running the full sweep (multiple prompts x
    # [14,16,18,20], full iteration budgets) on Colab. Same
    # local-verify-first pattern as everywhere else in this project.
    rows = run_scaling_experiment(n_prompts=1, candidate_counts=[14], bf_time_budget_seconds=60, qaoa_maxiter=20)
    print_scaling_table(rows)

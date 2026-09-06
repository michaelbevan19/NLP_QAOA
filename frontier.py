"""
frontier.py -- find the best prompt WITHOUT hand-set thresholds.

WHY THIS EXISTS
The existing pipeline minimises (-importance + length + redundancy) with
hand-set weights. Two of those three terms always push downward, and the
only one pushing back -- importance -- is a PROXY, not the thing we
actually care about (does the output still hold up?). So "drop almost
everything" is not a bug in the solver: it is the correct answer to the
question that objective asks. Every previous patch (FLOOR_PENALTY,
CARDINALITY_WEIGHT) was external scaffolding holding up an objective that
does not contain the real goal.

The proxy is weak in a specific, structural way: importance is measured by
removing ONE token from the full set, so it captures the marginal cost of
the FIRST removal and says nothing about the 5th simultaneous one. Removal
effects do not compose. The QUBO can therefore be solved perfectly and
still hand back a broken prompt.

THE REFRAME
Instead of weighting length against quality, CONSTRAIN length and MEASURE
quality: for each k from min_tokens..n, find the best k-token subset, then
spend exactly one real LLM call measuring what that subset's output
actually scores. That produces a trade-off curve instead of a single point,
and the chosen prompt is read off the curve's knee -- a criterion computed
from the data, not a threshold anybody picked.

WHAT FIXING k ELIMINATES (this is the point)
  LENGTH_PENALTY   -- at fixed k it contributes k*LENGTH_PENALTY, a
                      CONSTANT. Constants cannot change an argmin, so the
                      term drops out of the problem entirely. It is
                      omitted below rather than assigned a value.
  CARDINALITY_WEIGHT / CARDINALITY_TARGET_FRACTION
                   -- k is swept, never targeted. No such term exists.
  FLOOR_PENALTY    -- k >= min_tokens by construction. Collapse is not
                      penalised, it is unrepresentable.
  IMPORTANCE_SCALE -- only its RATIO to alpha matters (scaling all of Q
                      uniformly cannot change an argmin), so importance is
                      normalised to mean 1.0 and the scale disappears into
                      alpha_ratio.
  similarity thresholds
                   -- the knee replaces them as the DECISION input. They
                      remain useful for reporting, not for deciding.

WHAT SURVIVES, HONESTLY
  alpha_ratio -- one dimensionless number: what one flagged redundant pair
                 costs, in units of a typical token's importance.
                 alpha_ratio=1.0 means "one redundant pair is worth about
                 one average token". A genuine design choice, not pretended
                 away.
  redundancy.py's two thresholds -- they decide what COUNTS as redundant,
                 a measurement question calibrated against labelled
                 synonym/antonym pairs. Supervised fitting, not arbitrary
                 weighting, so it stays.

Eight hand-set knobs collapse to one free parameter plus the knee.

You cannot make "best" fully assumption-free -- "best" requires some
preference between brevity and fidelity. What you CAN do is make that
preference explicit and data-driven instead of burying it across eight
constants, and let the measured curve choose the point. That is what this
does. It does not claim to have removed the trade-off.

COST: 1 baseline + n removal tests + (n - min_tokens + 1) frontier points,
roughly 2n real LLM calls (about 24 for a 12-candidate prompt).
"""

import itertools

from clean import assemble_prompt, clean_and_rank, reading_order
from redundancy import redundant_pairs
from scoring import BaselineScorer
from llm import LLM

ALPHA_RATIO = 1.0   # the one surviving free parameter (see module docstring)
MIN_TOKENS = 2      # structural floor, not a penalty


def assemble_frontier_qubo(candidates, importance_scores, redundant_index_pairs,
                           alpha_ratio=ALPHA_RATIO):
    """
    Q for the fixed-k problem. Deliberately NOT qubo.assemble_qubo():
      - no length term (constant at fixed k -- see module docstring)
      - no cardinality term, no floor term (k is structural)
      - importance normalised to mean 1.0, so alpha_ratio is dimensionless
        and IMPORTANCE_SCALE has nothing left to do
    """
    n = len(candidates)
    mean_imp = (sum(importance_scores) / n) if n else 0.0
    if mean_imp > 0:
        norm = [v / mean_imp for v in importance_scores]
    else:
        # Degenerate: every removal left the output identical. Nothing to
        # rank by, so treat all tokens equally rather than crashing.
        norm = [0.0] * n

    Q = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Q[i][i] = -norm[i]          # negative = keeping is rewarded
    for (i, j) in redundant_index_pairs:
        assert i < j, "redundant_index_pairs must be (i, j) with i < j"
        Q[i][j] += alpha_ratio      # positive = co-selection discouraged
    return Q


def subset_cost(bits, Q):
    """Cost of one subset under Q. No floor penalty -- unlike
    qubo.bitstring_cost(), sub-minimum subsets simply never get built."""
    n = len(bits)
    total = 0.0
    for i in range(n):
        if not bits[i]:
            continue
        total += Q[i][i]
        for j in range(i + 1, n):
            if bits[j]:
                total += Q[i][j]
    return total


def best_subset_at_k(n, Q, k):
    """Exact best k-token subset, enumerating C(n, k) only. Summed over all
    k this is the same 2^n work as full brute force, which is free at
    n<=12; for larger n, swap in a heuristic solver restricted to |x|=k."""
    best_bits, best_cost = None, float("inf")
    for combo in itertools.combinations(range(n), k):
        bits = [0] * n
        for i in combo:
            bits[i] = 1
        c = subset_cost(bits, Q)
        if c < best_cost:
            best_bits, best_cost = bits, c
    return best_bits, best_cost


def find_knee(ks, sims):
    """
    Kneedle-style knee detection: min-max normalise both axes, then take
    the point sitting furthest ABOVE the straight chord joining the first
    and last points. For a curve that rises steeply then plateaus, that is
    the diminishing-returns elbow -- the smallest k still on the plateau.

    Returns (knee_k, info). Degenerate curves (fewer than 3 points, or no
    variation in similarity) return the smallest k with a stated reason,
    rather than inventing a knee the data does not support.
    """
    if len(ks) < 3:
        return ks[0], {"reason": "too few points for a knee; returning smallest k"}

    lo_s, hi_s = min(sims), max(sims)
    lo_k, hi_k = min(ks), max(ks)
    if hi_s - lo_s < 1e-9:
        return lo_k, {"reason": "similarity flat across all k; shortest is strictly best"}
    if hi_k == lo_k:
        return lo_k, {"reason": "single k value"}

    xs = [(k - lo_k) / (hi_k - lo_k) for k in ks]
    ys = [(s - lo_s) / (hi_s - lo_s) for s in sims]
    # vertical distance above the (0,0)->(1,1) chord after normalisation
    deltas = [y - x for x, y in zip(xs, ys)]
    best_i = max(range(len(deltas)), key=lambda i: deltas[i])
    return ks[best_i], {
        "reason": "max distance above endpoint chord (Kneedle)",
        "deltas": deltas,
    }


def run_frontier(prompt_entry, llm, alpha_ratio=ALPHA_RATIO, min_tokens=MIN_TOKENS):
    """
    Sweeps k, measures REAL similarity once per k, and reports the whole
    trade-off curve plus the auto-selected knee.
    """
    prompt, sample_input = prompt_entry["prompt"], prompt_entry["sample_input"]
    candidates = clean_and_rank(prompt)
    n = len(candidates)

    scorer = BaselineScorer(llm, prompt, sample_input)
    o_original = scorer.capture_baseline()

    importance = []
    for i in range(n):
        subset = candidates[:i] + candidates[i + 1:]
        importance.append(1.0 - scorer.score(subset))

    pairs = redundant_pairs(candidates, llm)
    Q = assemble_frontier_qubo(candidates, importance, pairs, alpha_ratio)

    rows = []
    for k in range(min_tokens, n + 1):
        bits, cost = best_subset_at_k(n, Q, k)
        kept = [candidates[i] for i in range(n) if bits[i]]
        text = assemble_prompt(kept)
        output = llm.generate(text, sample_input)
        sim = scorer.output_similarity(output)
        rows.append({
            "k": k,
            "bitstring": bits,
            "qubo_cost": cost,
            "prompt": text,
            "kept": [c["text"] for c in reading_order(kept)],
            "similarity": sim,
            "output": output,
        })

    knee_k, knee_info = find_knee([r["k"] for r in rows], [r["similarity"] for r in rows])

    return {
        "prompt_id": prompt_entry["id"],
        "original_prompt": prompt,
        "original_tokens": scorer.original_token_count,
        "o_original": o_original,
        "n_candidates": n,
        "redundant_pairs": pairs,
        "importance": importance,
        "alpha_ratio": alpha_ratio,
        "rows": rows,
        "knee_k": knee_k,
        "knee_info": knee_info,
        "llm_calls": 1 + n + len(rows),
    }


def print_frontier(result):
    print(f"[{result['prompt_id']}] original: {result['original_tokens']} tokens, "
          f"{result['n_candidates']} candidates, "
          f"{len(result['redundant_pairs'])} redundant pairs")
    print(f"alpha_ratio={result['alpha_ratio']}  (the ONLY hand-set weight here)")
    print(f"real LLM calls used: {result['llm_calls']}\n")

    header = f"{'k':>3s} {'similarity':>11s} {'qubo_cost':>10s}  prompt"
    print(header)
    print("-" * (len(header) + 24))
    for r in result["rows"]:
        marker = "  <== KNEE (auto-selected)" if r["k"] == result["knee_k"] else ""
        text = r["prompt"] if len(r["prompt"]) <= 46 else r["prompt"][:43] + "..."
        print(f"{r['k']:3d} {r['similarity']:11.4f} {r['qubo_cost']:10.3f}  {text!r}{marker}")

    knee_row = next(r for r in result["rows"] if r["k"] == result["knee_k"])
    print(f"\nselected k={result['knee_k']} by: {result['knee_info']['reason']}")
    print(f"  prompt:      {knee_row['prompt']!r}")
    print(f"  similarity:  {knee_row['similarity']:.4f}")
    print(f"  compression: {result['n_candidates']} candidates -> {result['knee_k']} kept")


if __name__ == "__main__":
    from prompts import PROMPTS

    llm = LLM()
    entry = next(p for p in PROMPTS if p["id"] == "control_sentiment")
    result = run_frontier(entry, llm)
    print_frontier(result)

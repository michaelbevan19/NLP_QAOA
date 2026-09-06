"""
Full 5-method comparison (report Section 6) + alpha sweep (report Section 5
step 12) + surviving-redundant-pairs check.

Wires together every module built so far into ONE per-prompt evaluation:
    clean.py -> scoring.py -> redundancy.py -> qubo.py
        -> baselines.{random_search, greedy_removal, simulated_annealing}
        -> circuit.{run_standard_qaoa, run_nlp_qaoa}
then, for the WINNING bitstring of every method, one final real LLM call to
get O_optimized (report Section 4.7).

FIX 1 (2026-08-15): all 5 methods -- including both QAOA variants now --
minimize the identical static Q matrix with ZERO LLM calls during search
(see circuit.py's module docstring for what changed and why). The ONLY
real LLM calls anywhere in this module are:
  (a) QUBO construction (prepare_prompt(), SHARED across every method:
      1 baseline capture + one removal-test call per candidate), and
  (b) each method's single final-verification call below (0 or 1,
      depending on whether that method's bitstring was valid).
This is FIX 4's "honest accounting": report qubo_construction_calls once
per prompt, and (search_calls, verification_calls, total_calls) per
method -- search_calls is now 0 for every method, which is itself the
point being demonstrated (the old comparison's 9-16 vs. 1 LLM-calls gap
was entirely a bookkeeping artifact of nlp_qaoa's now-removed per-iteration
LLM calls, not a real algorithmic cost). redundant_pairs()'s logprob()
reads (naturalness signal) are real model forward passes too, but a
qualitatively cheaper single-pass operation than generate() -- reported
separately (redundancy_logprob_calls), not folded into the same total, so
the numbers stay interpretable rather than conflating two very different
costs into one figure.

Local-machine usage note (HANDOFF.md Sec 5 / Sec 7): this module is built
to run the full 10-prompt suite and an alpha sweep, but that combination is
meant for Google Colab, not this laptop -- run it locally on 1-2 prompts
only (see __main__ below) to verify correctness, and run the full sweep
(all of PROMPTS x an ALPHA_SWEEP_VALUES list) on Colab once everything here
is confirmed working.

FIX 5 (2026-08-15): added two more rows, both alpha-INDEPENDENT so they're
computed once in prepare_prompt(), not re-run per alpha value:
  - brute_force: the exact optimum of Q (baselines.py); alpha-independent
    in the sense that it's re-run once per alpha inside _run_methods()
    just like the other Q-based methods (its answer DOES depend on
    alpha, since Q does) -- listed here only to note it uses Q, unlike
    naive_greedy_llm below.
  - naive_greedy_llm (naive_baseline.py): the "was the QUBO necessary"
    ablation. Uses NO Q at all, so its answer genuinely doesn't depend on
    alpha -- computed ONCE in prepare_prompt() and copied unchanged into
    every alpha value's results, exactly like the earlier alpha-sweep
    efficiency fix for baseline/importance/redundancy.
"""

from clean import assemble_prompt
from scoring import BaselineScorer
from redundancy import redundant_pairs
from qubo import ALPHA, assemble_qubo, bitstring_cost, kept_indices
from baselines import brute_force, greedy_removal, random_search, simulated_annealing
from circuit import run_nlp_qaoa, run_standard_qaoa
from naive_baseline import naive_greedy_llm
from clean import clean_and_rank
from llm import LLM

METHOD_NAMES = [
    "random_search", "greedy_removal", "simulated_annealing", "standard_qaoa", "nlp_qaoa",
    "naive_greedy_llm", "brute_force",
]
# The two FIX 5 additions aren't "compared methods" in the same sense as
# the original five (random/greedy/annealing/standard_qaoa/nlp_qaoa, all
# alternative search STRATEGIES over the identical Q): brute_force is the
# ground-truth reference point (how close does each method get to exact
# optimal), and naive_greedy_llm is the QUBO-necessity ablation. Printed
# tables label them distinctly rather than implying a fifth-vs-sixth
# apples-to-apples ranking.
REFERENCE_METHOD_NAMES = ["brute_force"]
ABLATION_METHOD_NAMES = ["naive_greedy_llm"]

# Sweep range re-centred 2026-09-04 alongside qubo.ALPHA's recalibration
# (4.0 -> 0.4). The old range [0.5 ... 16.0] sat entirely AT or ABOVE the
# value the magnitude audit identified as reasonable, so every point in it
# was in the regime where redundancy dominates the objective -- the sweep
# could only ever show degrees of over-penalisation, never the balanced
# region. This brackets 0.4 geometrically instead, keeping a top end near
# the old default so the previous (over-penalised) behaviour is still
# visible for comparison in the report.
ALPHA_SWEEP_VALUES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]

# FIX 1: standard_qaoa and nlp_qaoa are now two independent runs of the
# identical mechanism (circuit.py) -- giving them the same `seed` would
# make them literally bit-for-bit identical, which defeats the point of
# reporting them as two separate rows (see circuit.py's module docstring).
# This offset just needs to be large and fixed, not anything special.
NLP_QAOA_SEED_OFFSET = 10_000


def _final_verify(bitstring: list[int] | None, candidates: list[dict], llm: LLM, scorer: BaselineScorer) -> dict:
    """One real LLM call on `bitstring`'s assembled prompt (report Section
    4.7's O_optimized), scored against O_original. Returns None fields if
    the bitstring keeps fewer than 2 tokens or is missing entirely (a
    method that collapsed below the floor has nothing meaningful to
    verify) -- mirrors qubo.bitstring_cost()'s floor-penalty guard rather
    than spending a real call on a degenerate prompt. FIX 2: `kept_tokens`
    and `optimized_tokens` are made consistent by the CALLER (both forced
    to 0 together whenever this returns the "no valid bitstring" branch),
    not by this function alone -- see _run_methods().
    """
    if not bitstring or sum(bitstring) < 2:
        return {"prompt": None, "output": None, "similarity": None, "optimized_tokens": 0}
    kept_candidates = [candidates[i] for i in kept_indices(bitstring)]
    prompt = assemble_prompt(kept_candidates)
    output = llm.generate(prompt, scorer.sample_input)
    similarity = scorer.output_similarity(output)
    return {
        "prompt": prompt,
        "output": output,
        "similarity": similarity,
        "optimized_tokens": llm.token_count(prompt),
    }


def _method_result(name: str, bitstring: list[int] | None, cost: float, search_calls: int, candidates: list[dict], llm: LLM, scorer: BaselineScorer) -> dict:
    """Builds one method's row for the results table, uniformly across all
    5 methods (FIX 2: previously only nlp_qaoa guarded against a
    missing/degenerate bitstring; random_search/greedy_removal/
    simulated_annealing/standard_qaoa did not, which would have crashed on
    `sum(None)` now that circuit.py's QAOA functions can return
    bitstring=None). kept_tokens and optimized_tokens are FORCED to 0
    TOGETHER whenever the bitstring is missing/degenerate -- no more
    "1 kept / 0 opt_tok"-style mismatches."""
    valid = bool(bitstring) and sum(bitstring) >= 2
    verify = _final_verify(bitstring, candidates, llm, scorer)
    return {
        "bitstring": bitstring if valid else None,
        "kept_tokens": sum(bitstring) if valid else 0,
        "cost": cost,
        "search_calls": search_calls,
        "verification_calls": 1 if verify["prompt"] else 0,
        "llm_calls": search_calls + (1 if verify["prompt"] else 0),  # FIX 4: kept for backward-compat table printing
        **verify,
    }


def prepare_prompt(prompt_entry: dict, llm: LLM) -> dict:
    """
    Everything about `prompt_entry` that does NOT depend on alpha: cleaned
    candidates, O_original (captured once), real removal-test importance
    scores, and the two-signal redundancy check. Computed ONCE and reused
    across every alpha value in alpha_sweep() -- previously each alpha
    value redid all of this from scratch (roughly 13 real LLM calls + ~130
    logprob calls on a 12-candidate prompt like code_review, repeated
    needlessly per alpha). Only Q depends on alpha, so only Q assembly and
    the 5 methods need to be re-run per alpha; this function's output is
    passed into evaluate_prompt()/alpha_sweep() via their `prepared` param
    to skip re-deriving it.
    """
    prompt, sample_input = prompt_entry["prompt"], prompt_entry["sample_input"]

    candidates = clean_and_rank(prompt)

    scorer = BaselineScorer(llm, prompt, sample_input)
    scorer.capture_baseline()

    importance_scores = []
    for i, c in enumerate(candidates):
        subset = candidates[:i] + candidates[i + 1:]
        sim = scorer.score(subset)
        importance_scores.append(1.0 - sim)  # scoring.py + qubo.py convention: importance = 1 - similarity

    pairs = redundant_pairs(candidates, llm)
    n = len(candidates)

    # FIX 5: naive_greedy_llm uses NO Q at all, so its bitstring genuinely
    # doesn't depend on alpha -- run its real search + the same uniform
    # final-verification call ONCE here, not once per alpha value in
    # _run_methods() (which would just repeat the identical real LLM calls
    # for no reason, the same waste the earlier alpha-sweep fix removed
    # elsewhere). Its Q-based "cost" (informative for comparison even
    # though it never tried to minimize Q) IS alpha-dependent and gets
    # computed cheaply -- no new LLM calls, pure arithmetic -- per alpha
    # value in _run_methods() instead.
    ng_search = naive_greedy_llm(candidates, llm, scorer)
    ng_verify = _final_verify(ng_search["bitstring"], candidates, llm, scorer)
    naive_greedy_llm_result = {
        "bitstring": ng_search["bitstring"] if sum(ng_search["bitstring"]) >= 2 else None,
        "kept_tokens": sum(ng_search["bitstring"]) if sum(ng_search["bitstring"]) >= 2 else 0,
        "search_calls": ng_search["llm_calls"],
        "verification_calls": 1 if ng_verify["prompt"] else 0,
        **ng_verify,
    }

    return {
        "prompt_id": prompt_entry["id"],
        "original_prompt": prompt,
        "candidates": candidates,
        "scorer": scorer,
        "importance_scores": importance_scores,
        "redundant_pairs": pairs,
        "naive_greedy_llm_result": naive_greedy_llm_result,
        # FIX 4: real LLM cost of building the QUBO, shared across every
        # method (computed once regardless of which/how-many search
        # methods run afterward). generate()-based calls only -- see
        # redundancy_logprob_calls below for the separate, cheaper
        # logprob()-based cost. NOTE: does NOT include naive_greedy_llm's
        # own search_calls above -- that ablation intentionally bypasses
        # QUBO construction entirely (see naive_baseline.py), so its cost
        # is reported per-method, not folded into the shared number every
        # Q-based method pays.
        "qubo_construction_calls": 1 + len(candidates),  # 1 baseline capture + 1 removal-test call per candidate
        "redundancy_logprob_calls": n * (n - 1),  # is_redundant() reads naturalness in both directions per pair, C(n,2) pairs
    }


def _run_methods(prepared: dict, llm: LLM, alpha: float, qaoa_maxiter: int, seed: int | None) -> dict:
    """The alpha-DEPENDENT half of evaluate_prompt(): assemble Q at this
    alpha and run all methods that depend on it. FIX 1: standard_qaoa and
    nlp_qaoa now call the identical circuit.py mechanism with the SAME
    maxiter (no more reason to budget one lower -- neither pays a real
    per-iteration LLM cost), differing only by random seed (see
    NLP_QAOA_SEED_OFFSET). FIX 5: brute_force (exact optimum of THIS Q) is
    run fresh per alpha like the other Q-based methods; naive_greedy_llm's
    real search already ran once in prepare_prompt() (it doesn't use Q at
    all), so here we only recompute its Q-cost cheaply (no new LLM calls)
    for this specific alpha, for comparison against the Q-based methods."""
    candidates = prepared["candidates"]
    scorer = prepared["scorer"]
    n = len(candidates)

    Q = assemble_qubo(candidates, prepared["importance_scores"], prepared["redundant_pairs"], alpha=alpha)

    rs = random_search(n, Q, seed=seed)
    gr = greedy_removal(n, Q)
    sa = simulated_annealing(n, Q, seed=seed)
    sq = run_standard_qaoa(n, Q, maxiter=qaoa_maxiter, seed=seed)
    nq = run_nlp_qaoa(n, Q, maxiter=qaoa_maxiter, seed=(seed if seed is None else seed + NLP_QAOA_SEED_OFFSET))
    bf = brute_force(n, Q)

    methods = {}
    for name, result in [
        ("random_search", rs),
        ("greedy_removal", gr),
        ("simulated_annealing", sa),
        ("standard_qaoa", sq),
        ("nlp_qaoa", nq),
        ("brute_force", bf),
    ]:
        methods[name] = _method_result(
            name, result["bitstring"], result["cost"], result["llm_calls"], candidates, llm, scorer
        )

    # naive_greedy_llm: bitstring/search_calls/verification already
    # computed once in prepare_prompt(); just attach this alpha's Q-cost.
    ng = dict(prepared["naive_greedy_llm_result"])
    ng["cost"] = bitstring_cost(ng["bitstring"], Q) if ng["bitstring"] else None
    ng["llm_calls"] = ng["search_calls"] + ng["verification_calls"]  # backward-compat field, see _method_result()
    methods["naive_greedy_llm"] = ng

    return {"Q": Q, "methods": methods}


def evaluate_prompt(
    prompt_entry: dict,
    llm: LLM,
    alpha: float = ALPHA,
    qaoa_maxiter: int = 100,
    seed: int | None = 0,
    prepared: dict | None = None,
) -> dict:
    """
    Runs the full pipeline once for `prompt_entry` (a dict from prompts.py,
    e.g. PROMPTS[0]) across all 5 methods. Returns a dict with the prompt's
    candidates/importance/redundancy/Q (for inspection/debugging) plus
    `methods`: {method_name -> {bitstring, kept_tokens, optimized_tokens,
    cost, prompt, output, similarity, search_calls, verification_calls,
    llm_calls}}, plus FIX 4's shared call counts:
    qubo_construction_calls (generate()-based, shared across every method)
    and redundancy_logprob_calls (logprob()-based, reported separately --
    see module docstring for why these aren't folded into one number).

    `qaoa_maxiter` (FIX 1) applies to BOTH standard_qaoa and nlp_qaoa now
    -- there's no longer a real-LLM-cost reason to budget them
    differently, see circuit.py's module docstring.

    Pass `prepared` (prepare_prompt()'s output) if you already have it --
    e.g. main.py's run_one() calls prepare_prompt() once and shares it
    between this and alpha_sweep() to avoid preparing the same prompt
    twice.
    """
    if prepared is None:
        prepared = prepare_prompt(prompt_entry, llm)

    run = _run_methods(prepared, llm, alpha, qaoa_maxiter, seed)

    return {
        "prompt_id": prepared["prompt_id"],
        "original_prompt": prepared["original_prompt"],
        "original_tokens": prepared["scorer"].original_token_count,
        "candidates": prepared["candidates"],
        "importance_scores": prepared["importance_scores"],
        "redundant_pairs": prepared["redundant_pairs"],
        "qubo_construction_calls": prepared["qubo_construction_calls"],
        "redundancy_logprob_calls": prepared["redundancy_logprob_calls"],
        "Q": run["Q"],
        "methods": run["methods"],
    }


def surviving_redundant_pairs(bitstring: list[int], redundant_index_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Of the flagged redundant pairs, which still have BOTH tokens kept in
    `bitstring` -- i.e. the QUBO failed to separate them (report Section 6's
    'surviving redundant pairs' check)."""
    return [(i, j) for (i, j) in redundant_index_pairs if bitstring[i] and bitstring[j]]


def print_comparison_table(result: dict):
    print(f"[{result['prompt_id']}] original: {result['original_tokens']} tokens -- {result['original_prompt']!r}")
    # FIX 4: honest 3-part LLM-call accounting. qubo_construction_calls is
    # ONE number for the whole prompt (shared across the Q-based methods,
    # printed once here, not per-row) -- search_calls/verification_calls/
    # total are per method. search_calls is 0 for every Q-based method
    # post-FIX-1; that's the point being shown, not an omission.
    # FIX 5: naive_greedy_llm does NOT pay qubo_construction_calls -- it
    # bypasses QUBO construction entirely by design (naive_baseline.py) --
    # so its `total` below is search+verify only, not padded with a cost
    # it never actually incurred.
    print(f"QUBO construction: {result['qubo_construction_calls']} generate() calls (shared across Q-based methods) "
          f"+ {result['redundancy_logprob_calls']} logprob() reads (shared, reported separately -- see module docstring)")
    # FIX 5: brute_force is the exact optimum of Q -- gap_to_optimal shows
    # how close every OTHER Q-based method's cost is to it. Not meaningful
    # for naive_greedy_llm (never tried to minimize Q in the first place).
    bf_cost = result["methods"]["brute_force"]["cost"]
    header = f"{'method':22s} {'kept':>5s} {'opt_tok':>8s} {'similarity':>11s} {'cost':>9s} {'gap':>8s} {'search':>7s} {'verify':>7s} {'total':>6s}"
    print(header)
    print("-" * len(header))
    for name in METHOD_NAMES:
        m = result["methods"][name]
        sim_str = f"{m['similarity']:.4f}" if m["similarity"] is not None else "n/a"
        cost_str = f"{m['cost']:.3f}" if m["cost"] is not None else "n/a"
        if name in ABLATION_METHOD_NAMES:
            gap_str = "n/a"
            total = m["search_calls"] + m["verification_calls"]
        else:
            gap_str = f"{m['cost'] - bf_cost:+.3f}" if m["cost"] is not None else "n/a"
            total = result["qubo_construction_calls"] + m["search_calls"] + m["verification_calls"]
        tag = " (ref)" if name in REFERENCE_METHOD_NAMES else (" (ablation)" if name in ABLATION_METHOD_NAMES else "")
        print(f"{name+tag:22s} {m['kept_tokens']:5d} {m['optimized_tokens']:8d} {sim_str:>11s} "
              f"{cost_str:>9s} {gap_str:>8s} {m['search_calls']:7d} {m['verification_calls']:7d} {total:6d}")

    print(f"\n{len(result['redundant_pairs'])} redundant pairs flagged. Surviving per method (both kept -> QUBO failed to separate):")
    for name in METHOD_NAMES:
        m = result["methods"][name]
        if not m["bitstring"]:
            print(f"  {name}: n/a (bitstring collapsed below floor)")
            continue
        survivors = surviving_redundant_pairs(m["bitstring"], result["redundant_pairs"])
        pair_texts = [f"{result['candidates'][i]['text']}/{result['candidates'][j]['text']}" for i, j in survivors]
        print(f"  {name}: {pair_texts if pair_texts else '(none)'}")


def alpha_sweep(
    prompt_entry: dict,
    llm: LLM,
    alphas: list[float] = ALPHA_SWEEP_VALUES,
    qaoa_maxiter: int = 100,
    seed: int | None = 0,
    prepared: dict | None = None,
) -> list[dict]:
    """Report Section 5 step 12: re-run the alpha-dependent half of the
    pipeline (Q assembly + the 5 methods) at each alpha value, tracking how
    many originally-flagged redundant pairs still survive (both kept) in
    each method's winning bitstring, and how many tokens each method
    keeps. HANDOFF.md Sec 2.5's finding (a fixed alpha too small relative
    to importance never breaks a redundant pair apart) predicts
    surviving-pair counts should fall as alpha rises -- this makes that
    visible across a real range instead of the one fixed default value.

    Prepares the prompt (clean/baseline/importance/redundancy) ONCE, not
    once per alpha -- none of that depends on alpha, only Q's off-diagonal
    scaling does. Pass `prepared` if you already have it (see
    evaluate_prompt()'s docstring).
    """
    if prepared is None:
        prepared = prepare_prompt(prompt_entry, llm)

    sweep_results = []
    for alpha in alphas:
        run = _run_methods(prepared, llm, alpha, qaoa_maxiter, seed)
        row = {"alpha": alpha, "n_redundant_pairs": len(prepared["redundant_pairs"])}
        for name in METHOD_NAMES:
            m = run["methods"][name]
            survivors = surviving_redundant_pairs(m["bitstring"], prepared["redundant_pairs"]) if m["bitstring"] else []
            row[f"{name}_surviving"] = len(survivors)
            row[f"{name}_kept_tokens"] = m["kept_tokens"]
            # Recording similarity per alpha (added 2026-09-04) is what makes
            # "pick the alpha just before similarity falls off a cliff"
            # answerable at all. The sweep previously tracked only
            # kept_tokens and surviving pairs, so any alpha recommendation
            # based on output quality was unsupported by the data we
            # actually collected.
            row[f"{name}_similarity"] = m["similarity"]
        sweep_results.append(row)
    return sweep_results


def print_alpha_sweep_table(prompt_id: str, sweep_results: list[dict]):
    print(f"\n[{prompt_id}] alpha sweep")
    header = f"{'alpha':>6s} {'redundant':>10s} | " + " | ".join(f"{n[:12]:>12s}" for n in METHOD_NAMES)
    print(header)
    print("-" * len(header))
    for row in sweep_results:
        cells = []
        for n in METHOD_NAMES:
            sim = row.get(f"{n}_similarity")
            sim_str = f"{sim:.2f}" if sim is not None else " n/a"
            cells.append(f"{row[f'{n}_kept_tokens']:>2d}k/{row[f'{n}_surviving']:>2d}s/{sim_str}")
        print(f"{row['alpha']:6.2f} {row['n_redundant_pairs']:10d} | " + " | ".join(cells))
    print("  (cells: <kept tokens>k / <surviving redundant pairs>s / <output similarity>)")
    print("  Similarity is the column to watch when choosing alpha -- the useful value is")
    print("  the largest alpha BEFORE similarity drops, not the one that compresses most.")


if __name__ == "__main__":
    from prompts import PROMPTS

    llm = LLM()

    # Local smoke test: ONE fast prompt only (per HANDOFF.md Sec 5/7 -- the
    # full PROMPTS x ALPHA_SWEEP_VALUES sweep belongs on Colab).
    entry = next(p for p in PROMPTS if p["id"] == "control_sentiment")

    print("=== single-alpha comparison (default ALPHA) ===")
    result = evaluate_prompt(entry, llm, qaoa_maxiter=100, seed=0)
    print_comparison_table(result)

    print("\n=== alpha sweep (small local range) ===")
    sweep = alpha_sweep(entry, llm, alphas=[0.5, 2.0, 8.0], qaoa_maxiter=100, seed=0)
    print_alpha_sweep_table(entry["id"], sweep)

"""
Full 5-method comparison (report Section 6) + alpha sweep (report Section 5
step 12) + surviving-redundant-pairs check.

Wires together every module built so far into ONE per-prompt evaluation:
    clean.py -> scoring.py -> redundancy.py -> qubo.py
        -> baselines.{random_search, greedy_removal, simulated_annealing}
        -> circuit.{run_standard_qaoa, run_nlp_qaoa}
then, for the WINNING bitstring of every method, one final real LLM call to
get O_optimized (report Section 4.7). Per HANDOFF.md Sec 2.2: the three
classical baselines and standard_qaoa all optimize the identical static Q
matrix with zero LLM calls during search, so this one post-hoc verification
call is their ONLY LLM cost. nlp_qaoa additionally pays its real
per-iteration search calls (circuit.py) on top of this same verification
call, which is exactly what makes the "LLM calls" column in the comparison
table below a real, meaningful cost NLP-QAOA pays that the other four
don't.

Local-machine usage note (HANDOFF.md Sec 5 / Sec 7): this module is built
to run the full 10-prompt suite and an alpha sweep, but that combination is
meant for Google Colab, not this laptop -- run it locally on 1-2 prompts
only (see __main__ below) to verify correctness, and run the full sweep
(all of PROMPTS x an ALPHA_SWEEP_VALUES list) on Colab once everything here
is confirmed working.
"""

from clean import assemble_prompt
from scoring import BaselineScorer
from redundancy import redundant_pairs
from qubo import ALPHA, assemble_qubo, kept_indices
from baselines import greedy_removal, random_search, simulated_annealing
from circuit import run_nlp_qaoa, run_standard_qaoa
from clean import clean_and_rank
from llm import LLM

METHOD_NAMES = ["random_search", "greedy_removal", "simulated_annealing", "standard_qaoa", "nlp_qaoa"]

# A modest default sweep for local smoke-testing; report Section 5 step 12's
# real sweep should cover a wider/finer range on Colab once this is verified.
ALPHA_SWEEP_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def _final_verify(bitstring: list[int] | None, candidates: list[dict], llm: LLM, scorer: BaselineScorer) -> dict:
    """One real LLM call on `bitstring`'s assembled prompt (report Section
    4.7's O_optimized), scored against O_original. Returns None fields if
    the bitstring keeps fewer than 2 tokens or is missing entirely (a
    method that collapsed below the floor has nothing meaningful to
    verify) -- mirrors qubo.bitstring_cost()'s floor-penalty guard rather
    than spending a real call on a degenerate prompt."""
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

    return {
        "prompt_id": prompt_entry["id"],
        "original_prompt": prompt,
        "candidates": candidates,
        "scorer": scorer,
        "importance_scores": importance_scores,
        "redundant_pairs": pairs,
    }


def _run_methods(prepared: dict, llm: LLM, alpha: float, nlp_qaoa_maxiter: int, seed: int | None) -> dict:
    """The alpha-DEPENDENT half of evaluate_prompt(): assemble Q at this
    alpha, run all 5 methods, and do the one final verification call per
    method (report Section 4.7's O_optimized)."""
    candidates = prepared["candidates"]
    scorer = prepared["scorer"]
    n = len(candidates)

    Q = assemble_qubo(candidates, prepared["importance_scores"], prepared["redundant_pairs"], alpha=alpha)

    rs = random_search(n, Q, seed=seed)
    gr = greedy_removal(n, Q)
    sa = simulated_annealing(n, Q, seed=seed)
    sq = run_standard_qaoa(n, Q, seed=seed)
    nq = run_nlp_qaoa(candidates, Q, llm, scorer, maxiter=nlp_qaoa_maxiter, seed=seed)

    methods = {}
    for name, result in [
        ("random_search", rs),
        ("greedy_removal", gr),
        ("simulated_annealing", sa),
        ("standard_qaoa", sq),
    ]:
        verify = _final_verify(result["bitstring"], candidates, llm, scorer)
        methods[name] = {
            "bitstring": result["bitstring"],
            "kept_tokens": sum(result["bitstring"]),
            "cost": result["cost"],
            "llm_calls": result["llm_calls"] + (1 if verify["prompt"] else 0),
            **verify,
        }

    # nlp_qaoa already paid real LLM calls during search (circuit.py); this
    # extra call still runs so every method's O_optimized/similarity comes
    # from the SAME uniform post-hoc measurement point, kept separate from
    # circuit.py's internal "best observed during search" bookkeeping.
    verify = _final_verify(nq["bitstring"], candidates, llm, scorer)
    methods["nlp_qaoa"] = {
        "bitstring": nq["bitstring"],
        "kept_tokens": sum(nq["bitstring"]) if nq["bitstring"] else 0,
        "cost": nq["cost"],
        "llm_calls": nq["llm_calls"] + (1 if verify["prompt"] else 0),
        **verify,
    }

    return {"Q": Q, "methods": methods}


def evaluate_prompt(
    prompt_entry: dict,
    llm: LLM,
    alpha: float = ALPHA,
    nlp_qaoa_maxiter: int = 15,
    seed: int | None = 0,
    prepared: dict | None = None,
) -> dict:
    """
    Runs the full pipeline once for `prompt_entry` (a dict from prompts.py,
    e.g. PROMPTS[0]) across all 5 methods. Returns a dict with the prompt's
    candidates/importance/redundancy/Q (for inspection/debugging) plus
    `methods`: {method_name -> {bitstring, kept_tokens, optimized_tokens,
    cost, prompt, output, similarity, llm_calls}}.

    Pass `prepared` (prepare_prompt()'s output) if you already have it --
    e.g. main.py's run_one() calls prepare_prompt() once and shares it
    between this and alpha_sweep() to avoid preparing the same prompt
    twice.
    """
    if prepared is None:
        prepared = prepare_prompt(prompt_entry, llm)

    run = _run_methods(prepared, llm, alpha, nlp_qaoa_maxiter, seed)

    return {
        "prompt_id": prepared["prompt_id"],
        "original_prompt": prepared["original_prompt"],
        "original_tokens": prepared["scorer"].original_token_count,
        "candidates": prepared["candidates"],
        "importance_scores": prepared["importance_scores"],
        "redundant_pairs": prepared["redundant_pairs"],
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
    header = f"{'method':22s} {'kept':>5s} {'opt_tok':>8s} {'similarity':>11s} {'llm_calls':>10s}"
    print(header)
    print("-" * len(header))
    for name in METHOD_NAMES:
        m = result["methods"][name]
        sim_str = f"{m['similarity']:.4f}" if m["similarity"] is not None else "n/a"
        print(f"{name:22s} {m['kept_tokens']:5d} {m['optimized_tokens']:8d} {sim_str:>11s} {m['llm_calls']:10d}")

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
    nlp_qaoa_maxiter: int = 15,
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
        run = _run_methods(prepared, llm, alpha, nlp_qaoa_maxiter, seed)
        row = {"alpha": alpha, "n_redundant_pairs": len(prepared["redundant_pairs"])}
        for name in METHOD_NAMES:
            m = run["methods"][name]
            survivors = surviving_redundant_pairs(m["bitstring"], prepared["redundant_pairs"]) if m["bitstring"] else []
            row[f"{name}_surviving"] = len(survivors)
            row[f"{name}_kept_tokens"] = m["kept_tokens"]
        sweep_results.append(row)
    return sweep_results


def print_alpha_sweep_table(prompt_id: str, sweep_results: list[dict]):
    print(f"\n[{prompt_id}] alpha sweep")
    header = f"{'alpha':>6s} {'redundant':>10s} | " + " | ".join(f"{n[:12]:>12s}" for n in METHOD_NAMES)
    print(header)
    print("-" * len(header))
    for row in sweep_results:
        cells = " | ".join(
            f"{row[f'{n}_kept_tokens']:>3d}kept/{row[f'{n}_surviving']:>2d}surv"
            for n in METHOD_NAMES
        )
        print(f"{row['alpha']:6.2f} {row['n_redundant_pairs']:10d} | {cells}")


if __name__ == "__main__":
    from prompts import PROMPTS

    llm = LLM()

    # Local smoke test: ONE fast prompt only (per HANDOFF.md Sec 5/7 -- the
    # full PROMPTS x ALPHA_SWEEP_VALUES sweep belongs on Colab).
    entry = next(p for p in PROMPTS if p["id"] == "control_sentiment")

    print("=== single-alpha comparison (default ALPHA) ===")
    result = evaluate_prompt(entry, llm, nlp_qaoa_maxiter=10, seed=0)
    print_comparison_table(result)

    print("\n=== alpha sweep (small local range) ===")
    sweep = alpha_sweep(entry, llm, alphas=[0.5, 2.0, 8.0], nlp_qaoa_maxiter=5, seed=0)
    print_alpha_sweep_table(entry["id"], sweep)

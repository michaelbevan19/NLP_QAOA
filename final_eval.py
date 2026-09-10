"""
final_eval.py -- the consolidated, paper-ready evaluation.

    python final_eval.py --dataset prompts  --n 10 --restarts 5 --out prompts_final.json
    python final_eval.py --dataset dolly    --n 10 --restarts 5 --out dolly_final.json
    python final_eval.py --dataset wildchat --n 10 --restarts 5 --out wildchat_final.json

WHY THIS REPLACES THE EIGHT-ROW TABLE
The accumulated method set had become unreadable and, in two places,
actively misleading:
  * nlp_qaoa and standard_qaoa are THE SAME FUNCTION with different seeds
    (circuit.py, post-FIX-1). Reporting them as two methods invited reading
    seed variance as a method difference.
  * random_search / greedy_removal / simulated_annealing / brute_force all
    consume the identical NLP-derived Q, so calling only one row "nlp_*"
    implied an NLP advantage the others supposedly lacked. They all had it.
  * At n<=12 brute_force finds the exact optimum, so several rows were
    simply tied at the ceiling, carrying no information.

What actually deserves comparing is ONE question: at a given prompt length,
does QAOA search find as good a token subset as a near-exact classical
search, and does either produce better real output? So this file reports
exactly two rows:

  classical -- best k-token subset by exact enumeration of C(n, k)
               (frontier.best_subset_at_k). At these sizes this IS the
               optimum, so it is both a method and the ceiling.
  qaoa      -- the QAOA circuit constrained to select exactly k tokens,
               best of --restarts independent seeds by QUBO cost.

Both arms use the IDENTICAL Q and the IDENTICAL cost function, so any
difference between them is search quality alone -- not objective, not
seeding, not LLM-call accounting.

HOW QAOA IS CONSTRAINED TO EXACTLY k (and why not by penalty)
circuit.py's _run_qaoa samples freely across all 2^n states with no
cardinality constraint. Rather than rewrite the circuit, this reuses its
real components (build_cost_hamiltonian, _make_probs_qnode) and restricts
the MEASUREMENT to the feasible subspace: the circuit's own probability
distribution is renormalised over just those basis states with exactly k
bits set, then sampled. That guarantees |x| = k exactly, leaves the circuit
untouched, and is the standard treatment for constrained QAOA. A soft
penalty term was rejected because it only biases toward k and would let
infeasible bitstrings through, making the two arms non-comparable.

KNEE SELECTION IS PER METHOD
Each method sweeps k, spends one real LLM call per k measuring its own
subset's actual output similarity, and then runs frontier.find_knee on ITS
OWN curve. Neither method is handed the other's chosen length.

COST per prompt: 1 baseline + n removal tests + 2 calls per k
(~3n real generate calls, about 37 for a 12-candidate prompt). QAOA
simulation adds (k values x restarts x maxiter) circuit evaluations, which
is compute rather than LLM cost but is the dominant runtime at n=12 --
hence --qaoa-maxiter defaults to 50 here rather than circuit.py's 100.

REUSE -- nothing is reimplemented:
  importance / redundancy / Q  <- evaluate.prepare_prompt, frontier.assemble_frontier_qubo
  classical search             <- frontier.best_subset_at_k
  cost function                <- frontier.subset_cost
  QAOA circuit                 <- circuit.build_cost_hamiltonian / _make_probs_qnode
  knee detection               <- frontier.find_knee
  loaders                      <- dataset_eval / wildchat_eval / prompts.py
  stats + win-rate             <- dataset_eval._mean_sd, report.win_rates
Existing files are untouched.
"""

import argparse
import itertools
import json
import textwrap

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

import dataset_eval
import report as report_mod
from circuit import P_LAYERS_DEFAULT, _bitstring_from_index, _make_probs_qnode, build_cost_hamiltonian
from clean import assemble_prompt
from evaluate import prepare_prompt
from frontier import MIN_TOKENS, assemble_frontier_qubo, best_subset_at_k, find_knee, subset_cost
from llm import LLM

METHODS = ["classical", "qaoa"]
QAOA_MAXITER = 50   # lower than circuit.py's 100: this sweeps every k, so sim cost multiplies


def _register_methods():
    """report.win_rates iterates report.METHOD_NAMES at call time; point it
    at this file's two-method set rather than editing report.py."""
    report_mod.METHOD_NAMES = list(METHODS)


# ---------------------------------------------------------------------
# QAOA constrained to exactly k tokens
# ---------------------------------------------------------------------
def _feasible_indices(n, k):
    """Basis-state indices with exactly k bits set -- the feasible subspace."""
    return [idx for idx in range(2 ** n) if bin(idx).count("1") == k]


def qaoa_best_subset_at_k(n, Q, k, restarts=5, p_layers=P_LAYERS_DEFAULT, maxiter=QAOA_MAXITER):
    """
    Runs the real QAOA circuit (circuit.py's Hamiltonian + qnode), measuring
    only within the |x| = k subspace, for `restarts` independent seeds.
    Returns (best_bits, best_cost, per_restart_costs) where best_* is the
    restart whose chosen subset had the LOWEST QUBO cost.
    """
    cost_h = build_cost_hamiltonian(Q)
    mixer_h = qml.qaoa.x_mixer(range(n))
    probs_qnode = _make_probs_qnode(n, cost_h, mixer_h, p_layers)
    feasible = _feasible_indices(n, k)

    best_bits, best_cost = None, float("inf")
    per_restart = []

    for r in range(restarts):
        rng = np.random.default_rng(r)
        local = {"bits": None, "cost": float("inf")}

        def objective(params):
            gammas, betas = params[:p_layers], params[p_layers:]
            pr = np.asarray(probs_qnode(gammas, betas))
            sub = pr[feasible]
            total = sub.sum()
            # renormalise over the feasible subspace; if the circuit puts
            # ~zero mass there, fall back to uniform over feasible states
            sub = (sub / total) if total > 1e-12 else np.full(len(feasible), 1.0 / len(feasible))
            idx = feasible[int(rng.choice(len(feasible), p=sub))]
            bits = _bitstring_from_index(idx, n)
            c = subset_cost(bits, Q)
            if c < local["cost"]:
                local["bits"], local["cost"] = bits, c
            return c

        x0 = rng.uniform(0.0, np.pi / 2, size=2 * p_layers)
        minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})
        per_restart.append(local["cost"])
        if local["cost"] < best_cost:
            best_bits, best_cost = local["bits"], local["cost"]

    return best_bits, best_cost, per_restart


# ---------------------------------------------------------------------
# one prompt
# ---------------------------------------------------------------------
def run_prompt(entry, llm, restarts=5, min_tokens=MIN_TOKENS,
               p_layers=P_LAYERS_DEFAULT, qaoa_maxiter=QAOA_MAXITER):
    prepared = prepare_prompt(entry, llm)
    candidates = prepared["candidates"]
    scorer = prepared["scorer"]
    n = len(candidates)

    Q = assemble_frontier_qubo(candidates, prepared["importance_scores"],
                               prepared["redundant_pairs"])

    curves = {m: [] for m in METHODS}
    for k in range(min_tokens, n + 1):
        cls_bits, cls_cost = best_subset_at_k(n, Q, k)
        qa_bits, qa_cost, per_restart = qaoa_best_subset_at_k(
            n, Q, k, restarts=restarts, p_layers=p_layers, maxiter=qaoa_maxiter)

        for name, bits, cost in (("classical", cls_bits, cls_cost), ("qaoa", qa_bits, qa_cost)):
            kept = [candidates[i] for i in range(n) if bits[i]]
            text = assemble_prompt(kept)
            output = llm.generate(text, scorer.sample_input)
            curves[name].append({
                "k": k, "bitstring": bits, "cost": cost, "prompt": text,
                "output": output, "similarity": scorer.output_similarity(output),
                # classical enumerates C(n,k) exactly, so it IS the optimum
                # at this k; qaoa's gap is measured against it
                "gap": 0.0 if name == "classical" else cost - cls_cost,
                "restart_costs": per_restart if name == "qaoa" else None,
            })

    methods, knees = {}, {}
    for m in METHODS:
        kk, info = find_knee([r["k"] for r in curves[m]], [r["similarity"] for r in curves[m]])
        row = next(r for r in curves[m] if r["k"] == kk)
        knees[m] = {"k": kk, "reason": info["reason"]}
        methods[m] = {
            "bitstring": row["bitstring"],
            "kept_tokens": sum(row["bitstring"]),
            "optimized_tokens": llm.token_count(row["prompt"]),
            "cost": row["cost"],
            "gap": row["gap"],
            "prompt": row["prompt"],
            "output": row["output"],
            "similarity": row["similarity"],
            "knee_k": kk,
            "knee_reason": info["reason"],
            # every sweep point is a real measured call; no separate
            # verification call is needed because the sweep already measured
            # this exact subset's real output
            "llm_calls": len(curves[m]),
        }

    return {
        "prompt_id": entry["id"],
        "original_prompt": prepared["original_prompt"],
        "original_tokens": scorer.original_token_count,
        "candidates": candidates,
        "redundant_pairs": prepared["redundant_pairs"],
        "qubo_construction_calls": prepared["qubo_construction_calls"],
        "redundancy_logprob_calls": prepared["redundancy_logprob_calls"],
        "methods": methods,
        "knees": knees,
        "curves": curves,
        "o_original": scorer.o_original,
    }


# ---------------------------------------------------------------------
# printing -- same visual style as the existing scripts
# ---------------------------------------------------------------------
def print_prompt_block(res):
    print(f"[{res['prompt_id']}] original: {res['original_tokens']} tokens -- {res['original_prompt']!r}")
    print(f"QUBO construction: {res['qubo_construction_calls']} generate() calls (shared) "
          f"+ {res['redundancy_logprob_calls']} logprob() reads (shared, reported separately)")

    header = (f"{'method':12s} {'kept':>5s} {'opt_tok':>8s} {'similarity':>11s} "
              f"{'cost':>9s} {'gap':>8s} {'calls':>6s}  knee")
    print(header)
    print("-" * (len(header) + 22))
    total_shared = res["qubo_construction_calls"]
    for m in METHODS:
        d = res["methods"][m]
        print(f"{m:12s} {d['kept_tokens']:5d} {d['optimized_tokens']:8d} {d['similarity']:11.4f} "
              f"{d['cost']:9.3f} {d['gap']:+8.3f} {total_shared + d['llm_calls']:6d}  "
              f"k={d['knee_k']} ({d['knee_reason']})")

    pairs = res["redundant_pairs"]
    print(f"\n{len(pairs)} redundant pairs flagged. Surviving per method "
          f"(both kept -> QUBO failed to separate):")
    for m in METHODS:
        bits = res["methods"][m]["bitstring"]
        surv = [(i, j) for (i, j) in pairs if bits[i] and bits[j]]
        txt = [f"{res['candidates'][i]['text']}/{res['candidates'][j]['text']}" for i, j in surv]
        print(f"  {m}: {txt if txt else '(none)'}")


def print_metrics(results, stats):
    bar = "=" * 92
    print("\n" + bar)
    print("  EVALUATION METRICS")
    print(bar)
    print(f"  dataset          {stats['dataset']}")
    print(f"  sampled          {stats['sampled']} prompts "
          f"(from {stats['eligible_rows']} eligible of {stats['total_rows']} total rows)")
    print(f"  prompt length    {stats['min_words']}-{stats['max_words']} words")
    print(f"  similarity       raw cosine (scoring.py), win-rate floor = "
          f"{report_mod.DEFAULT_SIMILARITY_FLOOR}")
    print(f"  methods          classical = exact best k-subset; "
          f"qaoa = k-constrained circuit, best of {stats.get('restarts', '?')} restarts")
    print(bar)

    wins = report_mod.win_rates(
        [{"prompt_id": r["prompt_id"], "comparison": r} for r in results],
        report_mod.DEFAULT_SIMILARITY_FLOOR)

    header = (f"  {'method':12s} {'similarity mean+-sd':>22s} {'compression mean+-sd':>23s} "
              f"{'wins':>5s} {'total calls':>12s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in METHODS:
        sims = [r["methods"][m]["similarity"] for r in results]
        comps = [r["methods"][m]["optimized_tokens"] / r["original_tokens"]
                 for r in results if r["original_tokens"]]
        calls = sum(r["qubo_construction_calls"] + r["methods"][m]["llm_calls"] for r in results)
        s_mean, s_sd = dataset_eval._mean_sd(sims)
        c_mean, c_sd = dataset_eval._mean_sd(comps)
        s_txt = f"{s_mean:.4f} +- {s_sd:.4f}" if s_mean is not None else "n/a"
        c_txt = f"{c_mean*100:.1f}% +- {c_sd*100:.1f}%" if c_mean is not None else "n/a"
        print(f"  {m:12s} {s_txt:>22s} {c_txt:>23s} {wins.get(m, 0):>5d} {calls:>12d}")
    print("  " + "-" * (len(header) - 2))
    print("  compression = optimized_tokens / original_tokens (LOWER = shorter prompt)")
    print(f"  wins = prompts where that method had the fewest tokens among methods "
          f"scoring >= {report_mod.DEFAULT_SIMILARITY_FLOOR}")

    gaps = [r["methods"]["qaoa"]["gap"] for r in results]
    matched = sum(1 for g in gaps if abs(g) < 1e-9)
    print(f"\n  QAOA search quality: matched the exact optimum on {matched}/{len(gaps)} prompts; "
          f"mean gap {sum(gaps)/len(gaps):+.4f} (0 = found the optimum at its chosen k)")
    print(bar)


def print_worked_examples(results, how_many=4, width=88):
    def wrap(t, indent="      "):
        return textwrap.fill(" ".join(str(t).split()), width=width,
                             initial_indent=indent, subsequent_indent=indent)

    # `how_many - 1` divides by zero at --examples 1; dataset_eval.py guards
    # this with `if how_many > 1 else 1` and this dropped it. Caught by
    # rendering a real checkpoint with how_many=1.
    if len(results) <= how_many:
        picks = results
    elif how_many <= 1:
        picks = results[:1]
    else:
        step = (len(results) - 1) / (how_many - 1)
        picks = [results[round(i * step)] for i in range(how_many)]

    bar = "=" * width
    print("\n" + bar)
    print("  WORKED EXAMPLES -- original vs optimized, both methods")
    print(bar)
    for r in picks:
        print(f"\n  [{r['prompt_id']}]  {r['original_tokens']} original tokens")
        print("  " + "-" * (width - 2))
        print("    ORIGINAL PROMPT:")
        print(wrap(r["original_prompt"]))
        print("    OUTPUT FROM ORIGINAL:")
        print(wrap(r["o_original"]))
        for m in METHODS:
            d = r["methods"][m]
            print(f"\n    {m.upper()} -- k={d['knee_k']}, "
                  f"{r['original_tokens']} -> {d['optimized_tokens']} tokens, "
                  f"similarity {d['similarity']*100:.1f}%")
            print("    optimized prompt:")
            print(wrap(d["prompt"]))
            print("    output:")
            print(wrap(d["output"]))
        print("  " + "-" * (width - 2))
    print(bar)


# ---------------------------------------------------------------------
def load_prompts_builtin(n=None, seed=0):
    import random as _random

    from prompts import PROMPTS
    entries = list(PROMPTS)
    if n and n < len(entries):
        entries = _random.Random(seed).sample(entries, n)
    wc = [len(e["prompt"].split()) for e in entries]
    return entries, {e["id"]: e["domain"] for e in entries}, {
        "dataset": "prompts.py (hand-written set)", "total_rows": len(PROMPTS),
        "eligible_rows": len(PROMPTS), "sampled": len(entries),
        "min_words": min(wc), "max_words": max(wc),
    }


def main():
    p = argparse.ArgumentParser(description="Final two-method evaluation: classical vs k-constrained QAOA.")
    p.add_argument("--dataset", choices=["prompts", "dolly", "wildchat"], default="prompts")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--qaoa-maxiter", type=int, default=QAOA_MAXITER)
    args = p.parse_args()

    out_path = args.out or f"{args.dataset}_final_results.json"
    _register_methods()

    if args.dataset == "prompts":
        entries, groups, stats = load_prompts_builtin(args.n, args.seed)
    elif args.dataset == "dolly":
        entries, groups, stats = dataset_eval.load_dataset_prompts(args.n, seed=args.seed)
    else:
        import wildchat_eval
        entries, groups, stats = wildchat_eval.load_wildchat_prompts(args.n, seed=args.seed)
        wildchat_eval.print_scan_stats(stats)
    stats["restarts"] = args.restarts

    if not entries:
        print("No eligible prompts found.")
        return
    print(f"\nsampled {len(entries)} prompts from {stats['dataset']}")

    llm = LLM()
    results = []
    for i, entry in enumerate(entries, 1):
        print(f"\n########## [{i}/{len(entries)}] {entry['id']} "
              f"({groups.get(entry['id'], '?')}) ##########")
        res = run_prompt(entry, llm, restarts=args.restarts, qaoa_maxiter=args.qaoa_maxiter)
        res["category"] = groups.get(entry["id"], "unknown")
        print_prompt_block(res)
        results.append(res)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[checkpoint] {len(results)}/{len(entries)} saved to {out_path}")

    print_metrics(results, stats)
    print_worked_examples(results, how_many=args.examples)
    print(f"\nfull results: {out_path}")


if __name__ == "__main__":
    main()

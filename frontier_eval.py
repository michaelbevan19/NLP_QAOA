"""
frontier_eval.py -- dataset evaluation WITH frontier.py's measured-knee
length selection added as an extra method row.

    python frontier_eval.py --dataset dolly    --n 10 --out dolly_frontier.json
    python frontier_eval.py --dataset wildchat --n 10 --out wildchat_frontier.json

WHY THIS FILE EXISTS
dataset_eval.py and wildchat_eval.py both route through evaluate_prompt(),
which selects length as an EMERGENT side effect of the blended QUBO
objective (IMPORTANCE_SCALE / LENGTH_PENALTY / ALPHA / FLOOR_PENALTY /
CARDINALITY_WEIGHT all competing in one score, with no real output measured
during selection). frontier.py -- built and validated separately -- instead
CONSTRAINS length, measures real output similarity at every k, and picks
the knee of that measured curve. The dataset runs never used it.

This file closes that gap WITHOUT modifying any existing file.

WHY AN EXTRA ROW RATHER THAN A REPLACEMENT
frontier.run_frontier() does not produce a method comparison at all: it
never calls random_search / greedy_removal / simulated_annealing / either
QAOA, and returns {rows, knee_k, ...} rather than a `methods` dict. So it
cannot simply be swapped in for evaluate_prompt() -- it answers a different
question ("what length actually preserves the output?" vs "which search
method solves the QUBO best?"). Running BOTH per prompt keeps the existing
table intact and adds the frontier as a directly comparable competitor,
which also answers the question that matters most given the Dolly result
(brute_force 0.6332 < random_search 0.6588 -- i.e. solving the QUBO better
produced WORSE real output): does measuring beat optimising a proxy?

COST NOTE -- the frontier is cheaper here than running frontier.py alone.
run_frontier() re-derives the baseline, the n removal tests and the
redundancy check itself. Those are exactly what prepare_prompt() has
already paid for, so this file reuses `prepared` and pays only the
(n - min_tokens + 1) sweep calls on top: about n extra generate calls per
prompt rather than 2n.

REUSE
  loaders          <- dataset_eval.load_dataset_prompts / wildchat_eval.load_wildchat_prompts
  per-prompt run   <- evaluate.prepare_prompt + evaluate.evaluate_prompt
  frontier logic   <- frontier.assemble_frontier_qubo / best_subset_at_k / find_knee
  output format    <- dataset_eval.print_metrics_block / print_worked_examples
No scoring, QUBO, solver or knee logic is reimplemented here.
"""

import argparse
import json

import dataset_eval
import report as report_mod
from dataset_eval import print_metrics_block, print_worked_examples
from evaluate import evaluate_prompt, prepare_prompt, print_comparison_table
from frontier import ALPHA_RATIO, MIN_TOKENS, assemble_frontier_qubo, best_subset_at_k, find_knee
from clean import assemble_prompt
from llm import LLM

FRONTIER_METHOD = "frontier_knee"


def _register_frontier_row():
    """
    print_metrics_block() and report.win_rates() both iterate their module's
    METHOD_NAMES, looked up at CALL time. Appending to those lists here adds
    the frontier row to the existing output without editing either file --
    which the brief requires be left untouched. Idempotent.

    NOTE evaluate.METHOD_NAMES is deliberately NOT patched, so the
    per-prompt table stays at 7 rows while the summary block shows 8. That
    is not an oversight: the per-prompt table prints cost and gap-to-optimum,
    and frontier_knee's cost comes from a DIFFERENT matrix
    (assemble_frontier_qubo -- normalised, no length or cardinality terms),
    so listing it in that column would compare two incompatible objectives.
    Similarity and compression ARE measured identically for every method,
    which is why the frontier does appear in the summary block.
    """
    for mod in (dataset_eval, report_mod):
        if FRONTIER_METHOD not in mod.METHOD_NAMES:
            mod.METHOD_NAMES = list(mod.METHOD_NAMES) + [FRONTIER_METHOD]


def frontier_from_prepared(prepared, llm, alpha_ratio=ALPHA_RATIO, min_tokens=MIN_TOKENS):
    """
    frontier.py's sweep, reusing prepare_prompt()'s already-computed
    candidates / importance / redundancy instead of re-deriving them.
    Returns (method_row, sweep_rows, knee_k, knee_info) where method_row has
    the same keys evaluate.py builds for every other method, so it drops
    straight into the results table.
    """
    candidates = prepared["candidates"]
    scorer = prepared["scorer"]
    n = len(candidates)

    Q = assemble_frontier_qubo(candidates, prepared["importance_scores"],
                               prepared["redundant_pairs"], alpha_ratio)

    sweep_rows = []
    for k in range(min_tokens, n + 1):
        bits, cost = best_subset_at_k(n, Q, k)
        kept = [candidates[i] for i in range(n) if bits[i]]
        text = assemble_prompt(kept)
        output = llm.generate(text, scorer.sample_input)
        sim = scorer.output_similarity(output)
        sweep_rows.append({
            "k": k, "bitstring": bits, "qubo_cost": cost,
            "prompt": text, "similarity": sim, "output": output,
        })

    knee_k, knee_info = find_knee([r["k"] for r in sweep_rows],
                                  [r["similarity"] for r in sweep_rows])
    knee = next(r for r in sweep_rows if r["k"] == knee_k)

    method_row = {
        "bitstring": knee["bitstring"],
        "kept_tokens": sum(knee["bitstring"]),
        "optimized_tokens": llm.token_count(knee["prompt"]),
        "cost": knee["qubo_cost"],
        "prompt": knee["prompt"],
        "output": knee["output"],
        "similarity": knee["similarity"],
        # every sweep point is a real generate() call; there is no separate
        # verification call because the sweep already measured this exact
        # subset's real output -- that is the whole point of the method
        "search_calls": len(sweep_rows),
        "verification_calls": 0,
        "llm_calls": len(sweep_rows),
    }
    return method_row, sweep_rows, knee_k, knee_info


def run_eval(entries, groups, llm, out_path, qaoa_maxiter=100, alpha_ratio=ALPHA_RATIO):
    """Same loop shape as dataset_eval.run_dataset_eval(), plus the frontier
    sweep merged in as an extra method. Checkpoints after every prompt."""
    results = []
    for i, entry in enumerate(entries, 1):
        print(f"\n########## [{i}/{len(entries)}] {entry['id']} "
              f"({groups.get(entry['id'], '?')}) ##########")

        prepared = prepare_prompt(entry, llm)                      # paid once
        result = evaluate_prompt(entry, llm, qaoa_maxiter=qaoa_maxiter, prepared=prepared)

        row, sweep_rows, knee_k, knee_info = frontier_from_prepared(prepared, llm, alpha_ratio)
        result["methods"][FRONTIER_METHOD] = row                    # 8th row

        print_comparison_table(result)
        print(f"  frontier sweep: k={min(r['k'] for r in sweep_rows)}.."
              f"{max(r['k'] for r in sweep_rows)}  ->  knee k={knee_k} "
              f"({knee_info['reason']}), similarity {row['similarity']:.4f}, "
              f"{row['search_calls']} measured points")

        results.append({
            "prompt_id": entry["id"],
            "comparison": result,
            "category": groups.get(entry["id"], "unknown"),
            "sample_input": entry["sample_input"],
            "o_original": prepared["scorer"].o_original,
            "frontier_sweep": sweep_rows,
            "frontier_knee_k": knee_k,
            "frontier_knee_reason": knee_info["reason"],
        })

        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"[checkpoint] {len(results)}/{len(entries)} saved to {out_path}")

    return results


def print_frontier_summary(results):
    """The frontier-specific block: the measured curve is the whole point,
    so show where each prompt's knee landed and how it compares with what
    the blended QUBO objective chose."""
    bar = "=" * 92
    print("\n" + bar)
    print("  FRONTIER SUMMARY -- measured-knee length selection vs blended-objective length")
    print(bar)
    header = (f"  {'prompt':26s} {'n':>3s} {'knee k':>7s} {'knee sim':>9s} "
              f"{'nlp_qaoa k':>11s} {'nlp_qaoa sim':>13s} {'sim delta':>10s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    deltas = []
    for r in results:
        m = r["comparison"]["methods"]
        fk, nq = m[FRONTIER_METHOD], m["nlp_qaoa"]
        n = len(r["comparison"]["candidates"])
        if fk["similarity"] is None or nq["similarity"] is None:
            continue
        d = fk["similarity"] - nq["similarity"]
        deltas.append(d)
        print(f"  {r['prompt_id'][:26]:26s} {n:>3d} {r['frontier_knee_k']:>7d} "
              f"{fk['similarity']:>9.4f} {nq['kept_tokens']:>11d} "
              f"{nq['similarity']:>13.4f} {d:>+10.4f}")
    print("  " + "-" * (len(header) - 2))
    if deltas:
        better = sum(1 for d in deltas if d > 0)
        print(f"  frontier scored higher on {better}/{len(deltas)} prompts, "
              f"mean delta {sum(deltas)/len(deltas):+.4f}")
    print("  (delta > 0 = measuring the curve beat optimising the blended proxy)")
    print(bar)


def main():
    p = argparse.ArgumentParser(description="Dataset evaluation with frontier measured-knee selection.")
    p.add_argument("--dataset", choices=["dolly", "wildchat"], default="dolly",
                   help="which dataset loader to use (default: %(default)s)")
    p.add_argument("--n", type=int, default=10, help="prompts to sample (default: %(default)s)")
    p.add_argument("--out", default=None, help="output JSON (default: <dataset>_frontier_results.json)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--qaoa-maxiter", type=int, default=100)
    p.add_argument("--alpha-ratio", type=float, default=ALPHA_RATIO,
                   help="frontier's single free parameter (default: %(default)s)")
    args = p.parse_args()

    out_path = args.out or f"{args.dataset}_frontier_results.json"
    _register_frontier_row()

    if args.dataset == "dolly":
        entries, groups, stats = dataset_eval.load_dataset_prompts(args.n, seed=args.seed)
    else:
        import wildchat_eval
        entries, groups, stats = wildchat_eval.load_wildchat_prompts(args.n, seed=args.seed)
        wildchat_eval.print_scan_stats(stats)

    if not entries:
        print("No eligible rows found.")
        return
    print(f"\nsampled {len(entries)} prompts from {stats['dataset']}")

    llm = LLM()
    results = run_eval(entries, groups, llm, out_path,
                       qaoa_maxiter=args.qaoa_maxiter, alpha_ratio=args.alpha_ratio)

    print_metrics_block(results, stats)
    print_frontier_summary(results)
    print_worked_examples(results, how_many=args.examples)
    print(f"\nfull results: {out_path}")


if __name__ == "__main__":
    main()

"""
Post-hoc metrics + plots + a full readable output dump, computed purely
from a main.py --out JSON results file. No LLM calls here at all -- this
is pure post-processing, so it runs identically on 1-2 local prompts or
the full 10-prompt Colab run, and re-running it after tweaking a plot is
instant.

Usage:
    python report.py results.json
    python report.py results.json --out-dir figs/ --similarity-floor 0.8

Produces, in --out-dir (default: alongside results.json):
    all_outputs.txt   -- every prompt's original text + every method's
                          optimized prompt + real O_optimized output, in
                          full, for reading (not just summary numbers).
    tradeoff.png       -- report Section 6's core figure: compression (%
                          of original tokens kept) vs. output similarity
                          retained, one point per (prompt, method).
    method_bars.png    -- three bar charts (mean compression, mean
                          similarity, mean LLM calls), one bar per method.
    alpha_sweep_<id>.png -- one per prompt that has alpha_sweep data:
                          surviving redundant pairs vs. alpha, per method
                          (report Section 5 step 12's figure).
A summary table (with a win-rate column) also prints to stdout.

Win-rate note: deliberately NOT based on qubo "cost", even though FIX 1
(2026-08-15) made cost directly comparable across all Q-based methods now
(nlp_qaoa minimizes the identical bitstring_cost(Q) as everyone else --
see circuit.py's module docstring). Token count and similarity stay the
fairer comparison anyway, since naive_greedy_llm (FIX 5) never tries to
minimize Q at all, so its "cost" number, while computed for reference, was
never the thing it optimized for -- comparing it on cost would understate
it unfairly. Token count and similarity are both measured the same way
(one real final verification LLM call, evaluate.py's _final_verify()) for
EVERY method including naive_greedy_llm and brute_force, so those ARE
comparable across all 7: a method "wins" a prompt if it produced the
fewest optimized tokens among methods that stayed at or above
--similarity-floor.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed locally or on Colab when run as a script
import matplotlib.pyplot as plt

# FIX 5 (2026-08-15): added naive_greedy_llm (the QUBO-necessity ablation)
# and brute_force (the exact-optimum reference row) -- both now appear in
# main.py --out's JSON, so report.py needs to know about them too.
METHOD_NAMES = [
    "random_search", "greedy_removal", "simulated_annealing", "standard_qaoa", "nlp_qaoa",
    "naive_greedy_llm", "brute_force",
]
METHOD_COLORS = {  # fixed, consistent across every plot
    "random_search": "#9e9e9e",
    "greedy_removal": "#4c78a8",
    "simulated_annealing": "#f58518",
    "standard_qaoa": "#54a24b",
    "nlp_qaoa": "#b279a2",
    "naive_greedy_llm": "#e45756",
    "brute_force": "#000000",
}
# 0.85 on the RAW COSINE scale -- scoring.py stopped mapping cosine through
# (cos+1)/2 on 2026-09-04, so this constant's VALUE is unchanged but its
# MEANING is now much stricter (old 0.85-mapped was merely cosine 0.70,
# about the midpoint of the achievable range). Comparing win-rates across
# results files produced before and after that date is invalid.
DEFAULT_SIMILARITY_FLOOR = 0.85


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def per_prompt_rows(results: list[dict]) -> list[dict]:
    """Flattens results into one row per (prompt, method) for stats/plots."""
    rows = []
    for entry in results:
        comp = entry["comparison"]
        original_tokens = comp["original_tokens"]
        for method in METHOD_NAMES:
            m = comp["methods"][method]
            optimized_tokens = m["optimized_tokens"]
            rows.append(
                {
                    "prompt_id": comp["prompt_id"],
                    "method": method,
                    "original_tokens": original_tokens,
                    "optimized_tokens": optimized_tokens,
                    "compression_ratio": (optimized_tokens / original_tokens) if original_tokens else None,
                    "similarity": m["similarity"],
                    "llm_calls": m["llm_calls"],
                }
            )
    return rows


def aggregate_by_method(rows: list[dict]) -> dict:
    """Mean/median compression ratio, similarity, and LLM calls per
    method, across every prompt in `rows`."""
    agg = {}
    for method in METHOD_NAMES:
        method_rows = [r for r in rows if r["method"] == method]
        valid = [r for r in method_rows if r["similarity"] is not None]
        ratios = [r["compression_ratio"] for r in valid if r["compression_ratio"] is not None]
        sims = [r["similarity"] for r in valid]
        calls = [r["llm_calls"] for r in method_rows]
        agg[method] = {
            "n_valid": len(valid),
            "n_total": len(method_rows),
            "mean_compression_ratio": statistics.mean(ratios) if ratios else None,
            "median_compression_ratio": statistics.median(ratios) if ratios else None,
            "mean_similarity": statistics.mean(sims) if sims else None,
            "median_similarity": statistics.median(sims) if sims else None,
            "total_llm_calls": sum(calls),
            "mean_llm_calls": statistics.mean(calls) if calls else None,
        }
    return agg


def win_rates(results: list[dict], similarity_floor: float = DEFAULT_SIMILARITY_FLOOR) -> dict:
    """Per prompt, which method(s) produced the FEWEST optimized tokens
    among methods whose similarity stayed >= similarity_floor. See module
    docstring for why this -- not qubo cost -- is the fair cross-method
    comparison. Ties count for every tied method."""
    wins = {m: 0 for m in METHOD_NAMES}
    for entry in results:
        comp = entry["comparison"]
        eligible = {
            m: comp["methods"][m]["optimized_tokens"]
            for m in METHOD_NAMES
            if comp["methods"][m]["similarity"] is not None and comp["methods"][m]["similarity"] >= similarity_floor
        }
        if not eligible:
            continue
        best = min(eligible.values())
        for m, tok in eligible.items():
            if tok == best:
                wins[m] += 1
    return wins


def print_summary(results: list[dict], similarity_floor: float = DEFAULT_SIMILARITY_FLOOR):
    rows = per_prompt_rows(results)
    agg = aggregate_by_method(rows)
    wins = win_rates(results, similarity_floor)

    print(f"=== Summary across {len(results)} prompt(s) ===\n")
    header = f"{'method':22s} {'valid':>7s} {'mean_compress':>14s} {'mean_sim':>9s} {'mean_calls':>11s} {'wins':>5s}"
    print(header)
    print("-" * len(header))
    for method in METHOD_NAMES:
        a = agg[method]
        compress = f"{a['mean_compression_ratio']*100:.1f}%" if a["mean_compression_ratio"] is not None else "n/a"
        sim = f"{a['mean_similarity']:.4f}" if a["mean_similarity"] is not None else "n/a"
        calls = f"{a['mean_llm_calls']:.1f}" if a["mean_llm_calls"] is not None else "n/a"
        print(f"{method:22s} {a['n_valid']:3d}/{a['n_total']:<3d} {compress:>14s} {sim:>9s} {calls:>11s} {wins[method]:5d}")

    print(f"\nmean_compression_ratio = optimized_tokens/original_tokens (lower = more compression)")
    print(f"wins = # prompts where this method had the fewest tokens among methods with similarity >= {similarity_floor}")


def dump_all_outputs(results: list[dict], out_path: str):
    """Every prompt's original text + every method's optimized prompt +
    real O_optimized output, written in full -- the 'let me actually read
    everything it produced' view, not just summary numbers."""
    lines = []
    for entry in results:
        comp = entry["comparison"]
        lines.append("=" * 100)
        lines.append(f"PROMPT: {comp['prompt_id']}  ({comp['original_tokens']} tokens)")
        lines.append(f"original: {comp['original_prompt']!r}")
        lines.append("")
        for method in METHOD_NAMES:
            m = comp["methods"][method]
            lines.append(f"--- {method} ---")
            if not m["bitstring"] or m["prompt"] is None:
                lines.append("  (collapsed below the 2-token floor / no valid result)")
            else:
                lines.append(
                    f"  kept_tokens={m['kept_tokens']}  optimized_tokens={m['optimized_tokens']}  "
                    f"similarity={m['similarity']:.4f}  llm_calls={m['llm_calls']}"
                )
                lines.append(f"  optimized_prompt: {m['prompt']!r}")
                lines.append(f"  O_optimized: {m['output']!r}")
            lines.append("")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"full output dump written to {out_path}")


def plot_tradeoff(rows: list[dict], out_path: str):
    """The core figure: token compression (x) vs. output similarity
    retained (y), one point per (prompt, method). Further down-left is
    strictly better -- fewer tokens kept, similarity still high."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHOD_NAMES:
        pts = [
            r for r in rows
            if r["method"] == method and r["similarity"] is not None and r["compression_ratio"] is not None
        ]
        if not pts:
            continue
        xs = [r["compression_ratio"] * 100 for r in pts]
        ys = [r["similarity"] for r in pts]
        ax.scatter(xs, ys, label=method, color=METHOD_COLORS[method], alpha=0.75, s=60)
    ax.set_xlabel("optimized / original tokens (%) -- lower = more compression")
    ax.set_ylabel("output similarity to O_original -- higher = better preserved")
    ax.set_title("Compression vs. fidelity trade-off, all prompts x methods")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"trade-off plot written to {out_path}")


def plot_method_bars(agg: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    methods = METHOD_NAMES
    colors = [METHOD_COLORS[m] for m in methods]

    compress = [(agg[m]["mean_compression_ratio"] or 0) * 100 for m in methods]
    axes[0].bar(methods, compress, color=colors)
    axes[0].set_ylabel("mean optimized/original tokens (%)")
    axes[0].set_title("Compression (lower = better)")
    axes[0].tick_params(axis="x", rotation=30)

    sims = [(agg[m]["mean_similarity"] or 0) for m in methods]
    axes[1].bar(methods, sims, color=colors)
    axes[1].set_ylabel("mean similarity to O_original")
    axes[1].set_title("Fidelity (higher = better)")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=30)

    calls = [(agg[m]["mean_llm_calls"] or 0) for m in methods]
    axes[2].bar(methods, calls, color=colors)
    axes[2].set_ylabel("mean LLM calls per prompt")
    axes[2].set_title("Cost")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"per-method bar charts written to {out_path}")


def plot_alpha_sweep(results: list[dict], out_dir: Path):
    """One line plot per prompt that has alpha_sweep data: surviving
    redundant pairs (y) vs. alpha (x), one line per method."""
    for entry in results:
        sweep = entry.get("alpha_sweep")
        if not sweep:
            continue
        prompt_id = entry["prompt_id"]
        fig, ax = plt.subplots(figsize=(6, 4))
        alphas = [row["alpha"] for row in sweep]
        for method in METHOD_NAMES:
            surviving = [row[f"{method}_surviving"] for row in sweep]
            ax.plot(alphas, surviving, marker="o", label=method, color=METHOD_COLORS[method])
        ax.set_xlabel("alpha (redundancy penalty weight)")
        ax.set_ylabel("surviving redundant pairs")
        ax.set_title(f"Alpha sweep: {prompt_id}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"alpha_sweep_{prompt_id}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"alpha sweep plot written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc metrics + plots from a main.py results JSON file.")
    parser.add_argument("results_json", help="Path to a JSON file written by main.py's --out.")
    parser.add_argument("--out-dir", default=None, help="Directory for plots/dumps (default: alongside results_json).")
    parser.add_argument(
        "--similarity-floor", type=float, default=DEFAULT_SIMILARITY_FLOOR,
        help=f"Minimum similarity for a method to be 'eligible' in the win-rate count (default {DEFAULT_SIMILARITY_FLOOR}).",
    )
    args = parser.parse_args()

    results = load_results(args.results_json)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = per_prompt_rows(results)
    agg = aggregate_by_method(rows)

    print_summary(results, args.similarity_floor)
    dump_all_outputs(results, str(out_dir / "all_outputs.txt"))
    plot_tradeoff(rows, str(out_dir / "tradeoff.png"))
    plot_method_bars(agg, str(out_dir / "method_bars.png"))
    plot_alpha_sweep(results, out_dir)


if __name__ == "__main__":
    main()

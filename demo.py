"""
demo.py -- the human-readable "before and after" view.

Everything else in this project reports matrices, bitstrings, costs and
per-method tables. This file answers the one question a reader actually
asks first: what did the prompt look like before, what does it look like
now, and how much of the output survived?

Usage:
    python demo.py                      # code_review (most illustrative)
    python demo.py --prompt control_sql # any prompts.py entry
    python demo.py --all                # every prompt + a summary table
    python demo.py --alpha-ratio 2.0    # sweep the one free parameter

Selection is done by frontier.py, so there is no similarity threshold or
target length behind these numbers -- the length is chosen at the measured
knee of the quality curve. See frontier.py's module docstring.

COST: roughly 2n real LLM calls per prompt (~24 for a 12-candidate
prompt). --all runs that for all 10 prompts, so use it on a GPU.
"""

import argparse
import textwrap

from frontier import ALPHA_RATIO, run_frontier
from llm import LLM
from prompts import PROMPTS

RULE = "=" * 78


def _wrap(text, indent="    "):
    return textwrap.fill(text, width=78, initial_indent=indent, subsequent_indent=indent)


def demo_one(prompt_entry, llm, alpha_ratio=ALPHA_RATIO, show_outputs=True):
    """Runs the frontier for one prompt and prints the before/after view.
    Returns a small summary dict so --all can tabulate across prompts."""
    result = run_frontier(prompt_entry, llm, alpha_ratio=alpha_ratio)
    knee = next(r for r in result["rows"] if r["k"] == result["knee_k"])

    original_prompt = result["original_prompt"]
    minimised_prompt = knee["prompt"]
    orig_tokens = llm.token_count(original_prompt)
    min_tokens = llm.token_count(minimised_prompt)

    similarity_pct = knee["similarity"] * 100.0
    reduction_pct = (1.0 - min_tokens / orig_tokens) * 100.0 if orig_tokens else 0.0

    print(RULE)
    print(f"  {result['prompt_id']}")
    print(RULE)
    print(f"\nORIGINAL PROMPT  ({orig_tokens} tokens)")
    print(_wrap(original_prompt))
    print(f"\nMINIMISED PROMPT  ({min_tokens} tokens)")
    print(_wrap(minimised_prompt))

    print(f"\n{'-' * 78}")
    print(f"  TOKEN REDUCTION    {orig_tokens} -> {min_tokens} tokens "
          f"({reduction_pct:.1f}% fewer)")
    print(f"  OUTPUT SIMILARITY  {similarity_pct:.1f}%")
    print(f"  length chosen      k={result['knee_k']} of {result['n_candidates']} candidates, "
          f"by {result['knee_info']['reason']}")
    print(f"  real LLM calls     {result['llm_calls']}")
    print(f"{'-' * 78}")

    if show_outputs:
        # The similarity number above is a cosine between these two exact
        # texts. Printing both lets a reader judge whether it matches their
        # own sense of "close enough" rather than trust it blind.
        # result["o_original"] is the REAL baseline -- the untouched
        # original prompt's output -- not a stand-in.
        print(f"\nOUTPUT FROM ORIGINAL PROMPT:")
        print(_wrap(result["o_original"]))
        print(f"\nOUTPUT FROM MINIMISED PROMPT:")
        print(_wrap(knee["output"]))

    print()
    return {
        "prompt_id": result["prompt_id"],
        "orig_tokens": orig_tokens,
        "min_tokens": min_tokens,
        "reduction_pct": reduction_pct,
        "similarity_pct": similarity_pct,
        "k": result["knee_k"],
        "n_candidates": result["n_candidates"],
        "llm_calls": result["llm_calls"],
        "minimised_prompt": minimised_prompt,
    }


def print_summary(summaries):
    print(RULE)
    print("  SUMMARY -- all prompts")
    print(RULE)
    header = (f"{'prompt':22s} {'tokens':>14s} {'reduction':>10s} "
              f"{'similarity':>11s} {'k':>8s}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s['prompt_id']:22s} {s['orig_tokens']:>5d} -> {s['min_tokens']:<5d} "
              f"{s['reduction_pct']:>9.1f}% {s['similarity_pct']:>10.1f}% "
              f"{s['k']:>4d}/{s['n_candidates']:<3d}")
    n = len(summaries)
    if n:
        print("-" * len(header))
        print(f"{'MEAN':22s} {'':>14s} "
              f"{sum(s['reduction_pct'] for s in summaries)/n:>9.1f}% "
              f"{sum(s['similarity_pct'] for s in summaries)/n:>10.1f}%")
        print(f"\ntotal real LLM calls: {sum(s['llm_calls'] for s in summaries)}")


def main():
    parser = argparse.ArgumentParser(description="Before/after view of prompt minimisation.")
    parser.add_argument("--prompt", default="code_review",
                        choices=[p["id"] for p in PROMPTS],
                        help="Which prompts.py entry to demo (default: code_review).")
    parser.add_argument("--all", action="store_true",
                        help="Run every prompt and print a summary table (GPU recommended: "
                             "~2n LLM calls per prompt, ~200 total).")
    parser.add_argument("--alpha-ratio", type=float, default=ALPHA_RATIO,
                        help="The single free parameter: cost of one redundant pair in units "
                             "of an average token's importance (default: %(default)s).")
    parser.add_argument("--no-outputs", action="store_true",
                        help="Hide the full LLM outputs, show only the numbers.")
    args = parser.parse_args()

    llm = LLM()

    if args.all:
        summaries = []
        for entry in PROMPTS:
            summaries.append(demo_one(entry, llm, args.alpha_ratio,
                                      show_outputs=not args.no_outputs))
        print_summary(summaries)
    else:
        entry = next(p for p in PROMPTS if p["id"] == args.prompt)
        demo_one(entry, llm, args.alpha_ratio, show_outputs=not args.no_outputs)


if __name__ == "__main__":
    main()

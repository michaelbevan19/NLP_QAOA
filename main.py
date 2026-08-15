"""
Entry point tying the whole pipeline together (HANDOFF.md Sec 6 step 9).

Usage:
    python main.py                              # local dev: fastest prompt (control_sentiment), single alpha
    python main.py --prompt code_review         # local dev: a specific prompts.py entry
    python main.py --prompt code_review --alpha-sweep   # + a small local alpha sweep
    python main.py --full                       # ALL 10 prompts x the full alpha sweep -- Colab only,
                                                  #   NOT meant to run on the local laptop (HANDOFF.md Sec 5:
                                                  #   this machine has no CUDA GPU, and the full 10-prompt x
                                                  #   5-method x alpha-sweep evaluation is explicitly reserved
                                                  #   for Colab's free T4 GPU).

Results print as comparison tables (evaluate.py's formatting) and, if --out
is given, are additionally written to a JSON file for the report's figures
(Section 6).
"""

import argparse
import json

from evaluate import alpha_sweep, evaluate_prompt, prepare_prompt, print_alpha_sweep_table, print_comparison_table
from llm import LLM
from prompts import PROMPTS


def run_one(prompt_id: str, llm: LLM, do_alpha_sweep: bool, nlp_qaoa_maxiter: int) -> dict:
    entry = next(p for p in PROMPTS if p["id"] == prompt_id)
    # prepare_prompt() (clean/baseline/importance/redundancy) is shared
    # between the single-alpha comparison and the sweep below -- neither
    # depends on alpha, so there's no reason to redo it twice.
    prepared = prepare_prompt(entry, llm)
    result = evaluate_prompt(entry, llm, nlp_qaoa_maxiter=nlp_qaoa_maxiter, prepared=prepared)
    print_comparison_table(result)
    output = {"prompt_id": prompt_id, "comparison": result}
    if do_alpha_sweep:
        sweep = alpha_sweep(entry, llm, nlp_qaoa_maxiter=min(nlp_qaoa_maxiter, 5), prepared=prepared)
        print_alpha_sweep_table(prompt_id, sweep)
        output["alpha_sweep"] = sweep
    return output


def run_full(llm: LLM, nlp_qaoa_maxiter: int, out_path: str | None = None) -> list[dict]:
    """The full report Section 6 table: all 10 prompts x 5 methods, plus
    the full alpha sweep (Section 5 step 12). Colab only -- see module
    docstring / HANDOFF.md Sec 5.

    If `out_path` is given, results are written to it after EVERY prompt
    (not just once at the end) -- a long Colab session that disconnects
    partway through (free-tier sessions can be reclaimed after ~90 min
    idle or a ~12hr cap) still leaves every completed prompt's results on
    disk instead of losing the whole run, the same failure mode that
    interrupted the original Mac session (HANDOFF.md Sec 3/4). Write
    `out_path` to a Google-Drive-mounted path, not local Colab VM disk,
    which is wiped on disconnect.
    """
    all_results = []
    for entry in PROMPTS:
        prepared = prepare_prompt(entry, llm)
        result = evaluate_prompt(entry, llm, nlp_qaoa_maxiter=nlp_qaoa_maxiter, prepared=prepared)
        print_comparison_table(result)
        sweep = alpha_sweep(entry, llm, nlp_qaoa_maxiter=nlp_qaoa_maxiter, prepared=prepared)
        print_alpha_sweep_table(entry["id"], sweep)
        all_results.append({"prompt_id": entry["id"], "comparison": result, "alpha_sweep": sweep})

        if out_path:
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"[checkpoint] {len(all_results)}/{len(PROMPTS)} prompts saved to {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="NLP-QAOA: find a shorter version of one prompt whose LLM output stays close to the original."
    )
    parser.add_argument(
        "--prompt", default="control_sentiment", choices=[p["id"] for p in PROMPTS],
        help="Which prompts.py entry to run locally (default: control_sentiment, the fastest to iterate on).",
    )
    parser.add_argument("--alpha-sweep", action="store_true", help="Also run a small local alpha sweep for --prompt.")
    parser.add_argument(
        "--nlp-qaoa-maxiter", type=int, default=15,
        help="COBYLA iteration budget for NLP-QAOA (1 real LLM call each, report Section 4.6).",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run ALL 10 prompts x the full alpha sweep. Colab only -- do NOT run this on the "
             "local laptop (HANDOFF.md Sec 5: no CUDA GPU here, and this combination is explicitly "
             "reserved for Colab's free T4 GPU).",
    )
    parser.add_argument("--out", default=None, help="Optional path to write results as JSON.")
    args = parser.parse_args()

    if args.full:
        print(
            "!!! --full runs 10 prompts x 5 methods x an alpha sweep -- this is meant for "
            "Google Colab, not a local laptop (HANDOFF.md Sec 5). Proceeding since --full was "
            "passed explicitly, but Ctrl+C now if this is the local machine.\n"
        )

    llm = LLM()

    if args.full:
        # run_full() checkpoints to args.out after EVERY prompt, not just
        # once at the end -- see its docstring.
        results = run_full(llm, args.nlp_qaoa_maxiter, out_path=args.out)
        if args.out:
            print(f"\nfinal results confirmed at {args.out} (checkpointed incrementally during the run)")
    else:
        results = [run_one(args.prompt, llm, args.alpha_sweep, args.nlp_qaoa_maxiter)]
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()

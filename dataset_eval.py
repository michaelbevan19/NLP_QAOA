"""
dataset_eval.py -- run the EXISTING pipeline against an external dataset.

NEW DEPENDENCY (deliberately not added to the existing install cell):
    pip install datasets

WHAT THIS DOES NOT DO
It modifies nothing. No scoring, QUBO, redundancy or solver logic is
reimplemented here: it imports evaluate.prepare_prompt / evaluate_prompt --
the exact pair main.py's run_full() calls for each prompt -- and
report.win_rates for the win metric. Every number below therefore comes
from the same code path as the existing 10-prompt results, so the two are
directly comparable.

DATASET CHOICE: databricks/databricks-dolly-15k
Kept as requested, and it does fit: its rows are
{instruction, context, response, category}, which maps cleanly onto
prompts.py's {id, domain, prompt, sample_input}:
    prompt        <- instruction
    sample_input  <- context  (empty string when the row has none;
                     llm.generate() already treats "" as "no input", see
                     its user_content line, so this needs no special case)
    domain        <- category (kept for the per-category breakdown)

FILTERING, AND WHY IT IS NOT OPTIONAL
Measured on the real 15,011 rows:
    instruction length: min 1 word, MEDIAN 10, max 841
    rows with non-empty context: 4,467 / 15,011 (30%)
    rows with instruction >= 12 words: 5,555
Half the dataset is shorter than "When did Virgin Australia start
operating?". clean.py would reduce such a row to 2-4 candidates, at which
point k is pinned near the 2-token floor and every method returns the same
answer -- the comparison would measure nothing. Sampling is therefore
restricted to instructions of MIN_WORDS..MAX_WORDS, chosen so the sampled
prompts sit in the same length band as the existing hand-written ones
(prompts.py ranges 6-37 words). Long contexts are truncated to
MAX_CONTEXT_CHARS to keep generation time bounded and comparable.

This filtering IS a selection bias and is reported in the output header
rather than hidden: results describe "dolly rows long enough to compress",
not "dolly".

COST: roughly (2 * n_candidates + 8) real generate() calls per prompt plus
n*(n-1) logprob reads -- about 32 generate calls for a 12-candidate
prompt. --n 10 is therefore ~320 generate calls: minutes on a GPU, an hour
plus on CPU. Checkpoints after every prompt so a disconnect loses at most
one prompt's work.
"""

import argparse
import json
import random
import statistics

from evaluate import METHOD_NAMES, evaluate_prompt, prepare_prompt, print_comparison_table
from report import DEFAULT_SIMILARITY_FLOOR, win_rates
from llm import LLM

DATASET_NAME = "databricks/databricks-dolly-15k"
MIN_WORDS = 12          # below this, clean.py yields too few candidates to compare anything
MAX_WORDS = 60          # keeps sampled prompts near prompts.py's 6-37 word band
MAX_CONTEXT_CHARS = 400  # bound generation time; long wiki contexts dominate otherwise

CLASSICAL_BASELINES = ["random_search", "greedy_removal", "simulated_annealing"]


# ---------------------------------------------------------------------
# dataset -> prompts.py entry shape
# ---------------------------------------------------------------------
def load_dataset_prompts(n, seed=0, dataset_name=DATASET_NAME,
                         min_words=MIN_WORDS, max_words=MAX_WORDS,
                         max_context_chars=MAX_CONTEXT_CHARS):
    """Samples n rows and converts each into the exact dict shape
    prompts.py uses. Returns (entries, stats) where stats records what the
    filter removed, so the selection bias is reportable."""
    from datasets import load_dataset  # imported here so the module loads without `datasets`

    ds = load_dataset(dataset_name, split="train")
    total = len(ds)

    eligible = []
    for i, row in enumerate(ds):
        instruction = (row.get("instruction") or "").strip()
        wc = len(instruction.split())
        if wc < min_words or wc > max_words:
            continue
        eligible.append(i)

    rng = random.Random(seed)
    picked = rng.sample(eligible, min(n, len(eligible)))

    entries, categories = [], {}
    n_with_context = 0
    for rank, idx in enumerate(picked):
        row = ds[idx]
        context = (row.get("context") or "").strip()
        if context:
            n_with_context += 1
            if len(context) > max_context_chars:
                context = context[:max_context_chars].rsplit(" ", 1)[0] + " ..."
        category = row.get("category") or "unknown"
        pid = f"dolly_{idx}"
        entries.append({
            "id": pid,
            "domain": category,          # same field name prompts.py uses
            "prompt": row["instruction"].strip(),
            "sample_input": context,     # "" when absent -- generate() handles it
        })
        categories[pid] = category

    stats = {
        "dataset": dataset_name,
        "total_rows": total,
        "eligible_rows": len(eligible),
        "sampled": len(entries),
        "with_context": n_with_context,
        "without_context": len(entries) - n_with_context,
        "min_words": min_words,
        "max_words": max_words,
    }
    return entries, categories, stats


# ---------------------------------------------------------------------
# run -- identical call sequence to main.py's run_full(), per prompt
# ---------------------------------------------------------------------
def run_dataset_eval(entries, categories, llm, out_path, qaoa_maxiter=100):
    """Same prepare_prompt -> evaluate_prompt sequence run_full() uses.
    The alpha sweep is deliberately skipped: it is a per-prompt parameter
    study, not part of the method comparison, and running it would multiply
    LLM cost ~7x for numbers this evaluation does not use."""
    results = []
    for i, entry in enumerate(entries, 1):
        print(f"\n########## [{i}/{len(entries)}] {entry['id']} "
              f"({categories.get(entry['id'], '?')}) ##########")
        prepared = prepare_prompt(entry, llm)
        result = evaluate_prompt(entry, llm, qaoa_maxiter=qaoa_maxiter, prepared=prepared)
        print_comparison_table(result)

        results.append({
            "prompt_id": entry["id"],           # report.py-compatible shape
            "comparison": result,
            "category": categories.get(entry["id"], "unknown"),
            "sample_input": entry["sample_input"],
            # captured from the scorer we already own; evaluate_prompt does
            # not return it, and we are not modifying evaluate.py to add it
            "o_original": prepared["scorer"].o_original,
        })

        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"[checkpoint] {len(results)}/{len(entries)} saved to {out_path}")

    return results


# ---------------------------------------------------------------------
# metrics block
# ---------------------------------------------------------------------
def _mean_sd(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, sd


def collect_per_method(results):
    """{method: {'similarity': [...], 'compression': [...]}} across prompts."""
    acc = {m: {"similarity": [], "compression": []} for m in METHOD_NAMES}
    for entry in results:
        comp = entry["comparison"]
        orig = comp["original_tokens"]
        for m in METHOD_NAMES:
            md = comp["methods"][m]
            acc[m]["similarity"].append(md["similarity"])
            if md["similarity"] is not None and orig:
                acc[m]["compression"].append(md["optimized_tokens"] / orig)
            else:
                acc[m]["compression"].append(None)
    return acc


def print_metrics_block(results, stats, similarity_floor=DEFAULT_SIMILARITY_FLOOR):
    bar = "=" * 92
    print("\n" + bar)
    print("  EVALUATION METRICS -- external dataset")
    print(bar)
    print(f"  dataset          {stats['dataset']}")
    print(f"  sampled          {stats['sampled']} prompts "
          f"(from {stats['eligible_rows']} eligible of {stats['total_rows']} total rows)")
    print(f"  filter           instruction length {stats['min_words']}-{stats['max_words']} words "
          f"-- SELECTION BIAS: results describe dolly rows long enough to compress, not dolly")
    print(f"  sample_input     {stats['with_context']} rows had context, "
          f"{stats['without_context']} ran with empty input")
    print(f"  similarity       raw cosine (scoring.py), floor for win-rate = {similarity_floor}")
    print(bar)

    acc = collect_per_method(results)
    wins = win_rates(results, similarity_floor)

    header = (f"  {'method':22s} {'similarity mean+-sd':>22s} {'compression mean+-sd':>23s} "
              f"{'valid':>7s} {'wins':>5s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in METHOD_NAMES:
        s_mean, s_sd = _mean_sd(acc[m]["similarity"])
        c_mean, c_sd = _mean_sd(acc[m]["compression"])
        n_valid = sum(1 for v in acc[m]["similarity"] if v is not None)
        s_txt = f"{s_mean:.4f} +- {s_sd:.4f}" if s_mean is not None else "n/a"
        c_txt = f"{c_mean*100:.1f}% +- {c_sd*100:.1f}%" if c_mean is not None else "n/a"
        print(f"  {m:22s} {s_txt:>22s} {c_txt:>23s} "
              f"{n_valid:>4d}/{len(results):<2d} {wins[m]:>5d}")
    print("  " + "-" * (len(header) - 2))
    print("  compression = optimized_tokens / original_tokens (LOWER = shorter prompt)")
    print(f"  wins = prompts where that method had the fewest tokens among methods "
          f"scoring >= {similarity_floor}")

    _print_category_breakdown(results, acc)
    print(bar)


def _print_category_breakdown(results, acc):
    """Per-category similarity, nlp_qaoa vs the strongest classical baseline
    (strongest = highest mean similarity across the sample, chosen from the
    data rather than assumed)."""
    means = {}
    for m in CLASSICAL_BASELINES:
        mean, _ = _mean_sd(acc[m]["similarity"])
        if mean is not None:
            means[m] = mean
    if not means:
        print("\n  (no classical baseline produced a valid result; skipping breakdown)")
        return
    strongest = max(means, key=means.get)

    by_cat = {}
    for entry in results:
        cat = entry.get("category", "unknown")
        comp = entry["comparison"]["methods"]
        by_cat.setdefault(cat, {"nlp_qaoa": [], strongest: []})
        by_cat[cat]["nlp_qaoa"].append(comp["nlp_qaoa"]["similarity"])
        by_cat[cat][strongest].append(comp[strongest]["similarity"])

    print(f"\n  PER-CATEGORY SIMILARITY -- nlp_qaoa vs {strongest} "
          f"(strongest classical by mean similarity, {means[strongest]:.4f})")
    header = f"  {'category':24s} {'n':>3s} {'nlp_qaoa':>18s} {strongest:>18s} {'delta':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cat in sorted(by_cat):
        nq_mean, nq_sd = _mean_sd(by_cat[cat]["nlp_qaoa"])
        cb_mean, cb_sd = _mean_sd(by_cat[cat][strongest])
        n = len(by_cat[cat]["nlp_qaoa"])
        if nq_mean is None or cb_mean is None:
            print(f"  {cat:24s} {n:>3d} {'n/a':>18s} {'n/a':>18s} {'n/a':>9s}")
            continue
        print(f"  {cat:24s} {n:>3d} {nq_mean:>10.4f}+-{nq_sd:<6.4f} "
              f"{cb_mean:>10.4f}+-{cb_sd:<6.4f} {nq_mean - cb_mean:>+9.4f}")
    print("  (delta > 0 means nlp_qaoa preserved more of the output on that category)")
    print("  NOTE: per-category n is small at this sample size -- read these as "
          "indicative, not significant.")


# ---------------------------------------------------------------------
# worked examples
# ---------------------------------------------------------------------
def print_worked_examples(results, how_many=4, width=88):
    """Side-by-side before/after for a spread of prompts: original prompt,
    its real output, the nlp_qaoa-optimized prompt, ITS real output, and
    the similarity between the two outputs."""
    import textwrap

    def wrap(text, indent="      "):
        return textwrap.fill(" ".join(str(text).split()), width=width,
                             initial_indent=indent, subsequent_indent=indent)

    # spread the picks across the length range so examples are varied
    usable = [r for r in results if r["comparison"]["methods"]["nlp_qaoa"]["similarity"] is not None]
    usable.sort(key=lambda r: r["comparison"]["original_tokens"])
    if not usable:
        print("\n(no prompt produced a valid nlp_qaoa result; no worked examples)")
        return
    if len(usable) <= how_many:
        picks = usable
    else:
        step = (len(usable) - 1) / (how_many - 1) if how_many > 1 else 1
        picks = [usable[round(i * step)] for i in range(how_many)]

    bar = "=" * width
    print("\n" + bar)
    print("  WORKED EXAMPLES -- original vs nlp_qaoa-optimized")
    print(bar)
    for r in picks:
        comp = r["comparison"]
        nq = comp["methods"]["nlp_qaoa"]
        print(f"\n  [{r['prompt_id']}]  category={r.get('category','?')}  "
              f"{comp['original_tokens']} -> {nq['optimized_tokens']} tokens  "
              f"(similarity {nq['similarity']*100:.1f}%)")
        print("  " + "-" * (width - 2))
        print("    ORIGINAL PROMPT:")
        print(wrap(comp["original_prompt"]))
        if r.get("sample_input"):
            print("    SAMPLE INPUT:")
            print(wrap(r["sample_input"]))
        print("    OPTIMIZED PROMPT (nlp_qaoa):")
        print(wrap(nq["prompt"]))
        print("\n    OUTPUT FROM ORIGINAL:")
        print(wrap(r["o_original"]))
        print("    OUTPUT FROM OPTIMIZED:")
        print(wrap(nq["output"]))
        print(f"\n    >>> OUTPUT SIMILARITY: {nq['similarity']*100:.1f}%")
        print("  " + "-" * (width - 2))
    print(bar)


def main():
    p = argparse.ArgumentParser(description="Evaluate the existing pipeline on an external dataset.")
    p.add_argument("--n", type=int, default=20, help="number of prompts to sample (default: %(default)s)")
    p.add_argument("--out", default="dataset_results.json", help="output JSON path (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default: %(default)s)")
    p.add_argument("--examples", type=int, default=4, help="worked examples to print (default: %(default)s)")
    p.add_argument("--qaoa-maxiter", type=int, default=100, help="COBYLA budget for both QAOA variants")
    p.add_argument("--dataset", default=DATASET_NAME, help="HF dataset name (default: %(default)s)")
    args = p.parse_args()

    entries, categories, stats = load_dataset_prompts(args.n, seed=args.seed, dataset_name=args.dataset)
    print(f"sampled {len(entries)} prompts from {stats['dataset']} "
          f"({stats['eligible_rows']} eligible of {stats['total_rows']})")

    llm = LLM()
    results = run_dataset_eval(entries, categories, llm, args.out, qaoa_maxiter=args.qaoa_maxiter)

    print_metrics_block(results, stats)
    print_worked_examples(results, how_many=args.examples)
    print(f"\nfull results: {args.out}")


if __name__ == "__main__":
    main()

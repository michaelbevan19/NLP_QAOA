"""
wildchat_eval.py -- the dataset_eval.py evaluation, run against
allenai/WildChat-1M instead of Dolly-15k.

NEW DEPENDENCY (same one dataset_eval.py needs):
    pip install datasets

WHY A SECOND DATASET
Dolly is concise, professionally-written instructions: median instruction
length 10 words, and the sampled rows were things like "Classify each of
the following as either a US state or a country: Illinois, Arizona, Iran,
..." where every token is load-bearing. Compression necessarily damages
them, because there is no padding to remove. WildChat is the opposite --
real user chat prompts, measured median 39.5 words on the English subset
(max 2260), full of the hedging and restatement this pipeline was designed
to exploit. Running both is the honest way to show whether the method
needs redundancy to work.

WHAT IS REUSED, NOT REWRITTEN
Everything except the loader is imported from dataset_eval.py:
    run_dataset_eval    -- the per-prompt prepare_prompt -> evaluate_prompt
                           loop with checkpointing (itself the exact pair
                           main.py's run_full calls)
    print_metrics_block -- mean+-sd similarity/compression, win-rate,
                           per-group breakdown
    print_worked_examples
    MIN_WORDS / MAX_WORDS -- SAME word band as the Dolly run, so the two
                           datasets are comparable on prompt length
No pipeline file is modified, and no scoring/QUBO/solver logic exists here.

STRUCTURE (verified against the live dataset, not assumed)
Each row has a "conversation" list of turns; each turn carries
{content, role, language, toxic, redacted, ...}. Only the FIRST user turn
is used: later turns depend on earlier context and would not be a fair
standalone prompt. Language is spelled out ("English", not "en").

FILTERS, ALL REPORTED RATHER THAN HIDDEN
  english     -- first user turn's language == English
  word band   -- MIN_WORDS..MAX_WORDS, identical to the Dolly run
  toxic       -- EXCLUDED, and this one is evidence-based rather than
                 squeamish: the Dolly run produced a similarity of -0.12 on
                 a prompt whose ORIGINAL output was a model refusal while
                 the compressed prompt answered normally, i.e. the
                 optimizer was penalised for outperforming its own
                 baseline. Any prompt likely to trigger a refusal makes the
                 similarity metric meaningless, and toxic rows are the
                 largest single source of those.
Measured yield on a 300-row scan: 162 English first-user turns, 54 inside
the word band (~18%), so a bounded stream easily supplies a sample.

GROUPING: WildChat has no task-category field like Dolly's "category". The
per-group breakdown therefore groups by the "model" that produced the
conversation (gpt-3.5 / gpt-4 variants), which is the most meaningful
grouping actually present. It is a PROXY, not a task taxonomy -- read it as
"did prompts collected from different models behave differently", not as a
per-task-type result.

COST: same as dataset_eval.py, roughly (2*n_candidates + 8) generate calls
per prompt. Checkpoints after every prompt.
"""

import argparse
import itertools
import random

# every non-loader piece comes from dataset_eval.py unchanged
from dataset_eval import (
    MAX_WORDS,
    MIN_WORDS,
    print_metrics_block,
    print_worked_examples,
    run_dataset_eval,
)
from llm import LLM

DATASET_NAME = "allenai/WildChat-1M"
SCAN_MULTIPLIER = 60   # rows to stream per prompt wanted; ~18% pass the filters


def load_wildchat_prompts(n, seed=0, dataset_name=DATASET_NAME,
                          min_words=MIN_WORDS, max_words=MAX_WORDS,
                          scan_limit=None, exclude_toxic=True):
    """
    Streams the dataset (never downloads all ~1M rows) and converts
    qualifying rows into the exact dict shape prompts.py uses. Returns
    (entries, groups, stats) matching dataset_eval's contract so its
    printing functions work unchanged.
    """
    from datasets import load_dataset

    if scan_limit is None:
        scan_limit = max(400, n * SCAN_MULTIPLIER)

    ds = load_dataset(dataset_name, split="train", streaming=True)

    scanned = 0
    no_user_turn = 0
    non_english = 0
    toxic_skipped = 0
    out_of_band = 0
    eligible = []

    for row in itertools.islice(ds, scan_limit):
        scanned += 1
        conv = row.get("conversation") or []
        first_user = next((t for t in conv if t.get("role") == "user"), None)
        if first_user is None:
            no_user_turn += 1
            continue
        if (first_user.get("language") or "").strip().lower() != "english":
            non_english += 1
            continue
        if exclude_toxic and (first_user.get("toxic") or row.get("toxic")):
            toxic_skipped += 1
            continue

        content = (first_user.get("content") or "").strip()
        wc = len(content.split())
        if wc < min_words or wc > max_words:
            out_of_band += 1
            continue

        eligible.append({
            "content": content,
            "model": row.get("model") or "unknown",
            "hash": row.get("conversation_hash") or f"row{scanned}",
        })

    rng = random.Random(seed)
    picked = rng.sample(eligible, min(n, len(eligible)))

    entries, groups = [], {}
    for item in picked:
        pid = f"wildchat_{item['hash'][:10]}"
        entries.append({
            "id": pid,
            "domain": item["model"],      # same field name prompts.py uses
            "prompt": item["content"],
            # WildChat has no separate context field; every prompt runs with
            # an empty sample_input, exactly as dataset_eval.py already does
            # for Dolly rows lacking context (llm.generate treats "" as none)
            "sample_input": "",
        })
        groups[pid] = item["model"]

    stats = {
        "dataset": dataset_name,
        "total_rows": scanned,          # rows STREAMED, not the full 1M
        "eligible_rows": len(eligible),
        "sampled": len(entries),
        "with_context": 0,              # WildChat has no context field at all
        "without_context": len(entries),
        "min_words": min_words,
        "max_words": max_words,
        "_scan": {
            "scanned": scanned,
            "no_user_turn": no_user_turn,
            "non_english": non_english,
            "toxic_skipped": toxic_skipped,
            "out_of_band": out_of_band,
            "eligible": len(eligible),
        },
    }
    return entries, groups, stats


def print_scan_stats(stats):
    s = stats["_scan"]
    print("\n  STREAM SCAN (WildChat is ~1M rows; only a bounded window is streamed)")
    print(f"    rows streamed            {s['scanned']}")
    print(f"    no user turn             -{s['no_user_turn']}")
    print(f"    not English              -{s['non_english']}")
    print(f"    toxic (excluded)         -{s['toxic_skipped']}   "
          f"(refused baselines make similarity meaningless -- see module docstring)")
    print(f"    outside {stats['min_words']}-{stats['max_words']} words"
          f"{'':<10}-{s['out_of_band']}")
    print(f"    ELIGIBLE                 {s['eligible']}  -> sampled {stats['sampled']}")


def main():
    p = argparse.ArgumentParser(description="Run the existing pipeline against WildChat-1M.")
    p.add_argument("--n", type=int, default=20, help="prompts to sample (default: %(default)s)")
    p.add_argument("--out", default="wildchat_results.json", help="output JSON (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default: %(default)s)")
    p.add_argument("--examples", type=int, default=4, help="worked examples to print (default: %(default)s)")
    p.add_argument("--qaoa-maxiter", type=int, default=100, help="COBYLA budget for both QAOA variants")
    p.add_argument("--scan-limit", type=int, default=None,
                   help="rows to stream (default: max(400, 60*n))")
    p.add_argument("--include-toxic", action="store_true",
                   help="do NOT filter toxic rows (expect refused baselines to distort similarity)")
    args = p.parse_args()

    entries, groups, stats = load_wildchat_prompts(
        args.n, seed=args.seed, scan_limit=args.scan_limit,
        exclude_toxic=not args.include_toxic,
    )
    print_scan_stats(stats)
    if not entries:
        print("\nNo eligible rows found -- try raising --scan-limit.")
        return
    print(f"\nsampled {len(entries)} prompts from {stats['dataset']}")

    llm = LLM()
    results = run_dataset_eval(entries, groups, llm, args.out, qaoa_maxiter=args.qaoa_maxiter)

    print_metrics_block(results, stats)
    print_scan_stats(stats)
    print_worked_examples(results, how_many=args.examples)
    print(f"\nfull results: {args.out}")
    print("NOTE: the per-category block groups by originating MODEL -- WildChat has no "
          "task-category field. It is a proxy grouping, not a task taxonomy.")


if __name__ == "__main__":
    main()

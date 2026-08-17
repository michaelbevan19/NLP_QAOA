"""
naive_greedy_llm -- FIX 5 (2026-08-15): the ablation a reviewer will ask
for. Every other search method in this project (baselines.py's three
classical solvers, circuit.py's two QAOA variants) optimizes the SAME
QUBO matrix Q, built from the two-signal importance/redundancy pipeline
(clean.py -> scoring.py -> redundancy.py -> qubo.py). That machinery is
the actual novelty being tested -- so a reviewer's first question is fair:
was any of it necessary, or does a simple loop that just asks the real LLM
"is this still good enough?" after every deletion do just as well?

naive_greedy_llm answers that directly. It uses NO QUBO, NO importance
scores, NO redundancy check -- only clean.py's candidate list and a real
LLM call per deletion attempt:
    1. Start with every candidate token kept.
    2. Walk the candidates once, in their existing (POS-priority) order --
       deliberately NOT re-ranked by any importance signal, since the
       whole point is to see how a method with zero linguistic guidance
       performs.
    3. For each still-kept token, tentatively remove it, assemble the
       resulting prompt, and send ONE real LLM call. If the output's
       similarity to O_original stays >= similarity_threshold, keep the
       removal; otherwise put the token back.
    4. Never drop below 2 kept tokens (same floor convention as
       qubo.bitstring_cost() everywhere else in this project).

Its LLM-call count is real and reported honestly -- expected to be high
(one real call per token ATTEMPTED, roughly n-1 for an n-candidate
prompt), comparable to what the pre-FIX-1 nlp_qaoa design used to cost.
That's the point of the comparison, not a flaw to explain away: if this
much simpler, QUBO-free loop matches the full pipeline's compression and
similarity at a similar or lower LLM-call cost, that is a significant
finding about whether the QUBO/QAOA machinery earns its complexity, and
must be reported prominently rather than buried (per the fix's explicit
instruction).
"""

from clean import assemble_prompt


def naive_greedy_llm(
    candidates: list[dict],
    llm,
    scorer,
    similarity_threshold: float = 0.85,
) -> dict:
    """
    Returns {"bitstring": list[int], "llm_calls": int}. `candidates` is
    clean.py's ranked candidate list (same input every other method
    receives); `llm`/`scorer` mirror every other method's real-LLM wiring
    (scorer is a scoring.BaselineScorer with capture_baseline() already
    called, so O_original is available for comparison).
    """
    n = len(candidates)
    bits = [1] * n
    llm_calls = 0

    for i in range(n):
        if bits[i] == 0:
            continue
        trial = list(bits)
        trial[i] = 0
        if sum(trial) < 2:
            continue  # never drop below the floor -- same convention as qubo.bitstring_cost()

        kept_candidates = [candidates[j] for j in range(n) if trial[j]]
        prompt = assemble_prompt(kept_candidates)
        output = llm.generate(prompt, scorer.sample_input)
        llm_calls += 1
        similarity = scorer.output_similarity(output)

        if similarity >= similarity_threshold:
            bits = trial  # keep the deletion

    return {"bitstring": bits, "llm_calls": llm_calls}


if __name__ == "__main__":
    from clean import clean_and_rank
    from scoring import BaselineScorer
    from llm import LLM
    from prompts import PROMPTS

    llm = LLM()
    entry = next(p for p in PROMPTS if p["id"] == "control_sentiment")
    candidates = clean_and_rank(entry["prompt"])

    scorer = BaselineScorer(llm, entry["prompt"], entry["sample_input"])
    o_original = scorer.capture_baseline()
    print(f"O_original: {o_original!r}")
    print(f"candidates: {[c['text'] for c in candidates]}\n")

    result = naive_greedy_llm(candidates, llm, scorer)
    kept = [candidates[i]["text"] for i in range(len(candidates)) if result["bitstring"][i]]
    print(f"bitstring: {result['bitstring']}")
    print(f"kept: {kept}")
    print(f"llm_calls: {result['llm_calls']}")

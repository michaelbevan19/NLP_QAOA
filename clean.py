"""
Token cleaning: prompt -> ranked list of candidate content tokens for the
QUBO (report Section 4.2).

Pipeline:
    1. Tokenize + POS-tag with spaCy.
    2. Drop stopwords and punctuation/whitespace tokens.
    3. Deduplicate by lemma (keep the first surface form seen) -- this
       catches literal repeats, e.g. the same word used twice.
    4. RANK by part-of-speech priority -- nouns/proper nouns first, then
       verbs, then adjectives/adverbs, then anything else -- before
       taking the top N. This ordering is required, not cosmetic: taking
       the first N tokens in SENTENCE order instead would keep polite
       openers ("I would really appreciate it if you could please...")
       ahead of the actual content words, and the content words are what
       everything downstream (importance scoring, redundancy, QUBO) needs
       to see.
    5. Cap at MAX_CANDIDATES tokens.

What this stage deliberately does NOT do: merge synonyms. "bugs",
"errors", and "mistakes" all survive as distinct candidates here --
recognising that they're doing the same job is a semantic judgement left
entirely to redundancy.py's two-signal check (report Section 4.4). This
stage only strips stopwords/punctuation and literal lemma repeats.

spaCy POS tags are used ONLY for this ranking and for descriptive
reporting tables later -- never to drive a penalty decision directly.
There is deliberately no ROLE/TASK/LABEL/CONSTRAINT taxonomy here (report
Section 4.4's "Why no word lists or role categories" box) -- an earlier
version of this project had one and it was removed.
"""

import spacy

MAX_CANDIDATES = 12

# Lower number = kept first when the candidate list is truncated to
# MAX_CANDIDATES. Content-bearing tokens (what/who) outrank action tokens
# (verbs), which outrank descriptive tokens (adjectives/adverbs).
_POS_PRIORITY = {
    "PROPN": 0,
    "NOUN": 0,
    "VERB": 1,
    "ADJ": 2,
    "ADV": 2,
}
_DEFAULT_PRIORITY = 3  # anything else that survives stopword filtering

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_md")
    return _nlp


def clean_and_rank(prompt: str, max_candidates: int = MAX_CANDIDATES) -> list[dict]:
    """
    Returns a list of candidate-token dicts, ranked best-first:
        {"text": str, "lemma": str, "pos": str, "position": int}

    `position` is the token's index in the ORIGINAL prompt (spaCy doc
    order), kept so later stages (qubo.py / circuit.py) can reassemble a
    chosen subset back into reading order rather than POS-priority order.
    """
    nlp = _get_nlp()
    doc = nlp(prompt)

    candidates = []
    seen_lemmas = set()
    for i, tok in enumerate(doc):
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.lower()
        if lemma in seen_lemmas:
            continue  # literal repeat / same-lemma duplicate
        seen_lemmas.add(lemma)
        candidates.append(
            {"text": tok.text, "lemma": lemma, "pos": tok.pos_, "position": i}
        )

    candidates.sort(
        key=lambda c: (_POS_PRIORITY.get(c["pos"], _DEFAULT_PRIORITY), c["position"])
    )
    return candidates[:max_candidates]


def reading_order(candidates: list[dict]) -> list[dict]:
    """Re-sorts a (sub)set of candidates back into original sentence order
    -- used whenever we need to assemble actual prompt text, since
    clean_and_rank's output order is POS-priority, not reading order."""
    return sorted(candidates, key=lambda c: c["position"])


def assemble_prompt(candidates: list[dict]) -> str:
    """Reassembles a (sub)set of candidate tokens into prompt text, in
    original reading order. Used by scoring.py (removal-test / bitstring
    scoring) and circuit.py (candidate prompt for each measured bitstring)."""
    return " ".join(c["text"] for c in reading_order(candidates))


if __name__ == "__main__":
    from prompts import PROMPTS

    for p in PROMPTS:
        cands = clean_and_rank(p["prompt"])
        kept = [f"{c['text']}({c['pos']})" for c in reading_order(cands)]
        print(f"[{p['id']}]")
        print(f"  original ({len(p['prompt'].split())} words): {p['prompt']}")
        print(f"  candidates ({len(cands)}, shown in reading order): {', '.join(kept)}")
        print()

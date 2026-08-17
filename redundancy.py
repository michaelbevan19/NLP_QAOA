"""
Two-signal redundancy check (report Section 4.4 / Figure 2).

A candidate token pair (w_i, w_j) is flagged as redundant ONLY when BOTH
signals agree:

  Signal 1 -- semantic closeness
      spaCy word-vector cosine similarity. Answers: do these two words
      mean roughly the same thing?

  Signal 2 -- sequence naturalness
      -log P(w_j | w_i) under the local LLM (llm.py's logprob(), a pure
      forward-pass logit read -- see that file's docstring). Answers: do
      these two words read awkwardly placed next to each other?

Why both are required (do not simplify to one signal):
  - Semantic closeness ALONE wrongly flags antonym pairs such as
    "positive"/"negative": antonyms sit close together in vector space
    because they appear in near-identical contexts during training.
  - Naturalness ALONE is not a redundancy measure at all: it flags any
    pair that reads oddly, including unrelated words like
    "diagnose"/"spreadsheet", which are simply unconnected, not redundant.
  - Requiring agreement resolves both failure modes with no special-case
    rules: a close-but-natural pair is a legitimate collocation (left
    alone); a distant-but-awkward pair is just unrelated (left alone);
    only close-AND-awkward -- one person saying the same thing twice --
    gets penalised.

Both thresholds below are set from REAL measured values, not guessed from
theory (report Section 7, limitation #1). See diagnose_thresholds() at the
bottom of this file, which prints signal values for a hand-picked set of
synonym / antonym / unrelated pairs so the cut-offs can be read off actual
data before being hardcoded here.

spaCy is used ONLY for the word-vector similarity below (Signal 1) and,
separately, for POS tagging in clean.py / reporting. Neither use drives a
category-based penalty rule -- there is deliberately no ROLE/TASK/LABEL
taxonomy in this design (see clean.py's docstring).

Uses en_core_web_lg here (NOT en_core_web_md, unlike clean.py's POS
tagging, which doesn't need vectors and stays on md). Discovered on the
Sec 5 step 7 checkpoint (code_review, Windows run 2026-08-13): md's vector
table is PRUNED -- 684,830 words hashed onto only 20,000 unique 300-dim
vectors (confirmed via nlp.vocab.vectors.shape) -- so unrelated words can
collide onto the identical vector and read as an exact sim=1.000. This
isn't a threshold-calibration problem: a real run on code_review's 12
candidates found "error"/"mistake" (genuine synonym, must be flagged) and
"look"/"know" (unrelated verbs, must NOT be flagged) both landing on
sim=1.000 with nearly identical naturalness_penalty too (14.4 vs 14.0) --
no threshold on either signal can tell those two pairs apart when they're
bitwise identical inputs. en_core_web_lg gives every word its own unique
vector (no pruning), which removes the collision at the root.
"""

import spacy

from llm import LLM

VECTOR_MODEL = "en_core_web_lg"

# --- thresholds -------------------------------------------------------
# Calibrated from the real `diagnose_thresholds()` table (Windows run,
# Qwen2.5-1.5B-Instruct + en_core_web_lg, 2026-08-13). An earlier pass used
# en_core_web_md and got SIMILARITY_THRESHOLD=0.45 from a table that turned
# out corrupted by md's pruned vector table (684,830 words hashed onto only
# 20,000 unique vectors -- confirmed via nlp.vocab.vectors.shape), which
# made unrelated word pairs collide onto an identical vector and read as an
# exact sim=1.000 indistinguishable from genuine synonyms (a REAL prompt's
# candidate pairs -- code_review, same checkpoint run -- turned up
# "error"/"mistake", a true synonym pair, and "look"/"know", two unrelated
# verbs, both landing on sim=1.000 with near-identical naturalness_penalty
# too: no threshold on either signal could ever separate them). Switched
# Signal 1 to en_core_web_lg (full unique vectors, no pruning) to remove
# the collision at the root, then recalibrated both thresholds from a fresh
# table under lg:
#
#   pair                   category                         sim    nat_pen
#   errors/mistakes        synonym                         0.692     17.282
#   bugs/errors            synonym                         0.456     14.963
#   review/examine         synonym                         0.380     14.120
#   classify/categorize    synonym                         0.840      9.829
#   assess/evaluate        synonym                         0.906     10.361
#   extract/identify       synonym                         0.330     15.260
#   positive/negative      antonym (must NOT be flagged)   0.814      6.429
#   urgent/routine         antonym (must NOT be flagged)   0.313     12.781
#   python/function        related-but-distinct (unclear)  0.244     10.533
#   diagnose/spreadsheet   unrelated (must NOT be flagged)  0.141     12.403
#   the/cat                unrelated (must NOT be flagged)  0.234     10.743
#
# These lg numbers land close to the reference values from prior spaCy
# experiments (errors/mistakes ~0.73, bugs/errors ~0.45, python/function
# ~0.30) that the earlier md-based table couldn't reproduce -- confirming
# those reference values came from full (non-pruned) vectors all along.
#
# Under lg, positive/negative is no longer a vector collision -- it's a
# GENUINE 0.814, antonyms really do sit close in real vector space, exactly
# as this module's docstring predicts. Its similarity (0.814) is now close
# enough to classify/categorize's (0.840) that no similarity threshold
# could safely separate them (razor-thin, fragile margin) -- so, same as
# before, naturalness is what excludes it: positive/negative's
# naturalness_penalty (6.429) is the lowest of all 11 pairs, well below
# every synonym pair's (9.8+).
#
# NATURALNESS_THRESHOLD = 8.0 (unchanged from the md pass -- naturalness
# comes from llm.py's Qwen logits, not spaCy, so it wasn't affected by the
# vector-model switch) sits strictly between positive/negative's 6.429
# (must exclude) and classify/categorize's 9.829 (must include -- the pair
# qubo.py's Figure-3 replication already depends on being flagged
# redundant, see HANDOFF.md Sec 2.5).
#
# SIMILARITY_THRESHOLD = 0.35 sits strictly between urgent/routine's 0.313
# (must exclude) and review/examine's 0.380 (want to include), and excludes
# both unrelated pairs (<=0.234).
#
# Net effect on the 11-pair table at SIMILARITY_THRESHOLD=0.35: correctly
# flags errors/mistakes, bugs/errors, review/examine, classify/categorize,
# assess/evaluate as redundant (5/6 synonym pairs); correctly excludes both
# antonym pairs and both unrelated pairs. extract/identify (0.330, just
# under 0.35) is the one synonym pair NOT flagged -- a real limitation at
# this margin, not a bug; recall was never a hard requirement, only
# correctly excluding antonyms/unrelated pairs is (report Section 4.4).
#
# --- FIX 3 (2026-08-15): SIMILARITY_THRESHOLD raised 0.35 -> 0.40 -------
# A real diagnostic run on medical_triage (12 candidates, 66 pairs) found
# naturalness essentially saturated -- 131/132 direction-readings exceeded
# NATURALNESS_THRESHOLD -- so at 0.35 the flagged set (35/66, 53%) was
# effectively driven by similarity alone, with naturalness contributing no
# real filtering ("a check that fires on nearly everything carries no
# information"). Re-ran the 11-pair calibration set with BOTH naturalness
# directions computed (the original table above only checked forward) and
# tested candidate SIMILARITY_THRESHOLD values against it plus
# medical_triage, combined with the is_redundant() both-directions change
# above:
#
#   threshold   calib synonyms kept        medical_triage flag rate
#   0.35        5/6                        52%  (bugs/errors margin 0.121)
#   0.40        4/6  (loses review/examine) 30%  (bugs/errors margin 0.056)
#   0.45        4/6  (loses review/examine) 23%  (bugs/errors margin 0.006 -- too fragile)
#   0.50        3/6  (loses bugs/errors)    15%
#
# No threshold in this range ever wrongly flags an antonym/unrelated pair
# (naturalness alone already excludes urgent/routine at 0.313 and
# positive/negative regardless of similarity -- both directions fail
# naturalness for positive/negative: 6.429 and 4.920, both < 8.0).
#
# Picked 0.40: keeps both pairs this project repeatedly leans on as
# canonical examples -- classify/categorize (qubo.py's Figure-3 replication
# depends on it being flagged, HANDOFF.md Sec 2.5) and bugs/errors (the
# running naturalness example throughout llm.py/redundancy.py's own
# docstrings) -- with a real margin (0.056), not 0.45's razor-thin 0.006
# margin on bugs/errors. Cuts medical_triage's over-flagging nearly in
# half (52% -> 30%) despite naturalness alone barely moving that number
# (52% -> 52% from the both-directions change alone, at the OLD 0.35
# threshold) -- confirming similarity was always the real lever here, not
# naturalness. control_sentiment's false positives (sentiment/positive,
# sentiment/negative) are fixed by the both-directions change alone, at
# every threshold tested (0 pairs flagged from 0.35 through 0.55) -- not
# by this threshold change.
SIMILARITY_THRESHOLD = 0.40     # Signal 1: similarity above this = "close"
NATURALNESS_THRESHOLD = 8.0     # Signal 2: -logprob above this = "unnatural" (unchanged by FIX 3)
# ------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(VECTOR_MODEL)
    return _nlp


def semantic_closeness(word_i: str, word_j: str) -> float:
    """Signal 1: spaCy word-vector cosine similarity, roughly in [-1, 1]
    (usually [0, 1] in practice for common-word vectors)."""
    nlp = _get_nlp()
    doc = nlp(f"{word_i} {word_j}")
    if len(doc) < 2 or not doc[0].has_vector or not doc[1].has_vector:
        return 0.0
    return doc[0].similarity(doc[1])


def naturalness_penalty(word_i: str, word_j: str, llm: LLM) -> float:
    """Signal 2: -log P(word_j | word_i). Higher = more surprising /
    unnatural continuation. This calls llm.logprob(), which is a single
    forward-pass logit read -- never generation (see llm.py)."""
    return -llm.logprob(word_i, word_j)


def is_redundant(
    word_i: str,
    word_j: str,
    llm: LLM,
    sim_threshold: float = SIMILARITY_THRESHOLD,
    nat_threshold: float = NATURALNESS_THRESHOLD,
) -> dict:
    """
    Evaluates both signals for the pair (word_i, word_j) and returns:
        {"redundant": bool, "similarity": float,
         "naturalness_fwd": float, "naturalness_bwd": float}

    `redundant` is True only if similarity > sim_threshold AND naturalness
    exceeds nat_threshold in BOTH reading orders (word_i->word_j and
    word_j->word_i) -- changed from "either order" in FIX 3 (2026-08-15):
    a real diagnostic run (medical_triage, control_sentiment) found that
    requiring only one direction let non-antonym, topically-related pairs
    slip through -- "sentiment"/"positive" and "sentiment"/"negative" on
    control_sentiment each failed naturalness in ONE direction only (7.520
    and 7.245 respectively) but passed in the other (8.946 and 9.291), so
    the old "either" rule flagged both anyway. The true antonym case this
    design exists to protect, "positive"/"negative", is unaffected by this
    change -- it already failed naturalness in BOTH directions (6.429 and
    4.920), so requiring both changes nothing there. See redundant_pairs()
    for the full before/after numbers.
    """
    sim = semantic_closeness(word_i, word_j)
    nat_fwd = naturalness_penalty(word_i, word_j, llm)
    nat_bwd = naturalness_penalty(word_j, word_i, llm)
    redundant = (sim > sim_threshold) and (nat_fwd > nat_threshold) and (nat_bwd > nat_threshold)
    return {"redundant": redundant, "similarity": sim, "naturalness_fwd": nat_fwd, "naturalness_bwd": nat_bwd}


def redundant_pairs(candidates: list[dict], llm: LLM) -> list[tuple[int, int]]:
    """
    Given clean.py's candidate list, checks every unordered pair and
    returns the (i, j) index pairs (into `candidates`) flagged redundant.

    FIX 3 (2026-08-15): a real diagnostic run on medical_triage (12
    candidates, 66 pairs) found naturalness essentially non-discriminating
    in single-direction form -- 131/132 direction-readings exceeded
    NATURALNESS_THRESHOLD, so the flagged set (35/66, 53%) was effectively
    just "similarity > threshold" alone, with naturalness contributing no
    real filtering. Two changes were made together (both re-verified
    against the diagnostic data before landing here, see the module's
    threshold comment block and redundancy_fix3_notes below):
      1. is_redundant() now requires BOTH reading orders to exceed
         NATURALNESS_THRESHOLD (previously either order was enough) --
         this alone fixed control_sentiment completely (2 false-positive
         pairs -> 0) without affecting the antonym case it must protect.
      2. SIMILARITY_THRESHOLD raised 0.35 -> 0.40 (see threshold comment
         block) -- this is what actually brings medical_triage's rate down
         (52% -> 30% at the both-directions rule), since naturalness alone
         wasn't doing enough filtering to fix it on its own.
    """
    flagged = []
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            w_i, w_j = candidates[i]["lemma"], candidates[j]["lemma"]
            result = is_redundant(w_i, w_j, llm)
            if result["redundant"]:
                flagged.append((i, j))
    return flagged


# --- threshold calibration script --------------------------------------

_DIAGNOSTIC_PAIRS = [
    # (word_i, word_j, expected category)
    ("errors", "mistakes", "synonym"),
    ("bugs", "errors", "synonym"),
    ("review", "examine", "synonym"),
    ("classify", "categorize", "synonym"),
    ("assess", "evaluate", "synonym"),
    ("extract", "identify", "synonym"),
    ("positive", "negative", "antonym (must NOT be flagged)"),
    ("urgent", "routine", "antonym (must NOT be flagged)"),
    ("python", "function", "related-but-distinct (unclear)"),
    ("diagnose", "spreadsheet", "unrelated (must NOT be flagged)"),
    ("the", "cat", "unrelated (must NOT be flagged)"),
]


def diagnose_thresholds(llm: LLM | None = None):
    """Prints Signal 1 and Signal 2 values for a hand-picked set of pairs
    so thresholds can be set from real numbers rather than guessed."""
    if llm is None:
        llm = LLM()
    print(f"{'pair':35s} {'category':32s} {'sim':>8s} {'nat_pen':>10s}")
    print("-" * 90)
    for w_i, w_j, category in _DIAGNOSTIC_PAIRS:
        sim = semantic_closeness(w_i, w_j)
        nat = naturalness_penalty(w_i, w_j, llm)
        print(f"{w_i + '/' + w_j:35s} {category:32s} {sim:8.3f} {nat:10.3f}")


if __name__ == "__main__":
    diagnose_thresholds()

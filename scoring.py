"""
Output-similarity scoring against a captured baseline (report Sections
4.1, 4.3, 4.6, 4.7).

BaselineScorer wraps ONE original prompt + its sample_input:
  1. capture_baseline() sends the untouched original prompt to the LLM
     and records the output as O_original (Section 4.1) -- the reference
     point every candidate is compared against.
  2. score(subset) reassembles a candidate token subset into prompt text,
     sends THAT to the LLM with the same sample_input, and returns the
     embedding cosine similarity between its output and O_original.

Embedding cosine similarity is used deliberately instead of exact string
matching (do NOT simplify this away): the same LLM call, run twice on
semantically identical prompts, routinely produces different wording.
Exact-match or token-overlap scoring would read that wording variation as
a large behavioural change and make every score noisy and untrustworthy.

A separate small embedding model (sentence-transformers,
all-MiniLM-L6-v2) is used for this -- NOT the Qwen model itself and NOT
spaCy word vectors. Those are reserved for different jobs: spaCy vectors
measure word-to-word semantic closeness for the redundancy check
(redundancy.py); Qwen's own logits measure sequence naturalness (also
redundancy.py). This module measures whole-OUTPUT similarity, a
different granularity, so it gets its own dedicated model.
"""

from sentence_transformers import SentenceTransformer, util as st_util

from clean import assemble_prompt
from llm import LLM

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


class BaselineScorer:
    def __init__(self, llm: LLM, prompt: str, sample_input: str):
        self.llm = llm
        self.prompt = prompt
        self.sample_input = sample_input
        self.o_original = None
        self.o_original_embedding = None
        self.original_token_count = llm.token_count(prompt)

    def capture_baseline(self) -> str:
        """Section 4.1: one LLM call on the untouched original prompt."""
        self.o_original = self.llm.generate(self.prompt, self.sample_input)
        embedder = _get_embedder()
        self.o_original_embedding = embedder.encode(self.o_original, convert_to_tensor=True)
        return self.o_original

    def output_similarity(self, output_text: str) -> float:
        """Cosine similarity between `output_text` and O_original, mapped
        from [-1, 1] to [0, 1] (1 = identical meaning, 0 = opposite)."""
        if self.o_original is None:
            raise RuntimeError("capture_baseline() must be called first")
        embedder = _get_embedder()
        emb = embedder.encode(output_text, convert_to_tensor=True)
        cos_sim = st_util.cos_sim(emb, self.o_original_embedding).item()
        return (cos_sim + 1.0) / 2.0

    def score(self, candidate_tokens: list[dict]) -> float:
        """
        Assembles `candidate_tokens` (a subset of clean.py's candidate
        dicts) into a prompt, runs it through the LLM with the same
        sample_input used for the baseline, and returns output similarity
        to O_original in [0, 1].
        """
        candidate_prompt = assemble_prompt(candidate_tokens)
        if not candidate_prompt.strip():
            return 0.0  # empty prompt: no output can meaningfully match
        output = self.llm.generate(candidate_prompt, self.sample_input)
        return self.output_similarity(output)


if __name__ == "__main__":
    from clean import clean_and_rank

    llm = LLM()
    prompt = "Classify sentiment as positive or negative"
    sample_input = "The battery died after three hours."

    scorer = BaselineScorer(llm, prompt, sample_input)
    baseline_output = scorer.capture_baseline()
    print(f"O_original: {baseline_output!r}")
    print(f"original_token_count: {scorer.original_token_count}")

    candidates = clean_and_rank(prompt)
    print(f"candidates: {[c['text'] for c in candidates]}")

    print("\n--- removal test: score with each single token removed ---")
    for i, c in enumerate(candidates):
        subset = candidates[:i] + candidates[i + 1:]
        sim = scorer.score(subset)
        print(f"remove {c['text']!r:12s} -> similarity={sim:.4f}")

    print("\n--- full candidate set (sanity check: should score high) ---")
    print(f"similarity={scorer.score(candidates):.4f}")

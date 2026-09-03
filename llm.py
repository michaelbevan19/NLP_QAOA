"""
Local LLM wrapper for NLP-QAOA.

Loads Qwen/Qwen2.5-1.5B-Instruct via `transformers` (not an API), because
the redundancy check's naturalness signal (report Section 4.4) needs raw
token log-probabilities from a forward pass, which hosted chat APIs do not
expose.

Two entry points, deliberately kept separate:

    generate(prompt, input_text)  -> str
        Full text generation via .generate(). Used for the baseline call
        (Section 4.1), the removal-test calls (Section 4.3), and the final
        optimized-prompt call (Section 4.7).

    logprob(word_i, word_j)  -> float
        A single forward pass reading logits directly off the model head.
        NO sampling, NO decoding, NO .generate() call anywhere in this
        method. This is the sequence-naturalness signal for the redundancy
        check (Section 4.4) and must stay a pure logit read.

Runs on CPU/MPS on a laptop (no CUDA available there) and on CUDA (T4) on
Colab, picked automatically.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LLM:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or _pick_device()
        print(f"[llm.py] loading {model_name} on device={self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

        # Qwen ships sampling defaults (temperature=0.7, top_p=0.8,
        # top_k=20) in its generation_config.json. Every generate() call
        # here is greedy (do_sample=False), which IGNORES all three -- but
        # transformers warns loudly about each one on every single call,
        # which buried real output in noise across whole-suite runs.
        # Clearing them at the source is purely cosmetic: greedy decoding
        # never read these values in the first place, so no generation
        # behaviour changes.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        print("[llm.py] model loaded.")

    # ------------------------------------------------------------------
    # Text generation (used everywhere EXCEPT the naturalness signal)
    # ------------------------------------------------------------------
    def generate(self, prompt: str, input_text: str = "", max_new_tokens: int = 150) -> str:
        """
        Runs `prompt` (optionally followed by `input_text`) through the
        model's chat template and returns the decoded response text.

        Greedy decoding (do_sample=False) is used deliberately: the removal
        test in Section 4.3 compares outputs across many candidate prompts,
        and sampling noise would be indistinguishable from a genuine effect
        of removing a token. Wording variation between calls is still
        possible (different prompt -> different greedy path), which is
        exactly why output comparison uses embedding cosine similarity
        (scoring.py) rather than exact string matching.
        """
        user_content = prompt if not input_text else f"{prompt}\n\n{input_text}"
        messages = [{"role": "user", "content": user_content}]

        # `return_dict=True` + ** unpacking, NOT a bare positional tensor.
        # transformers 4.5x changed apply_chat_template's default so that
        # return_tensors="pt" hands back a BatchEncoding (dict-like), not a
        # tensor. Passing that positionally made generate() evaluate
        # `input_ids.shape`, which falls through BatchEncoding.__getattr__
        # to self.data['shape'] -> KeyError: 'shape', surfacing as the
        # AttributeError crash at generation/utils.py:2498 on Colab.
        # Requesting the dict explicitly and unpacking it works on both
        # 4.46.x (this machine) and 4.5x (Colab).
        #
        # It also passes a real attention_mask, which the old code never
        # did -- hence the "attention mask is not set and cannot be
        # inferred because pad token is same as eos token" warning on
        # every call. That warning mattered: Qwen's chat template embeds
        # <|im_end|> tokens that ARE the eos/pad token, so transformers'
        # fallback guess could mask genuine prompt positions. Supplying
        # the correct mask removes that ambiguity.
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Raw logit read (the naturalness signal — never generation)
    # ------------------------------------------------------------------
    def logprob(self, word_i: str, word_j: str) -> float:
        """
        Returns the mean log P(token | context) for word_j's tokens,
        conditioned on word_i immediately preceding it — i.e. how natural
        word_j reads right after word_i under the base model.

        This is a single forward pass; logits are read directly off
        `self.model(...).logits`. There is no call to .generate() and no
        sampling anywhere in this method (report requirement: the
        naturalness signal must be a pure forward-pass logit read).

        Returns a value <= 0 (log-probability). Closer to 0 = more natural.
        More negative = more surprising / unnatural continuation.
        Mean-over-tokens (not sum) is used so multi-token words aren't
        penalised just for having more sub-word pieces.
        """
        # NOTE (checked during the transformers 4.5x BatchEncoding fix in
        # generate() above): this method does NOT have that bug and is
        # deliberately left as-is. It calls the plain tokenizer (whose
        # return type did not change across versions) and explicitly
        # indexes ["input_ids"][0] to get a bare 1-D tensor, which is then
        # passed to self.model(...) as a forward pass -- never to
        # .generate(), and never as a BatchEncoding in a positional slot.
        # Leaving it untouched is also load-bearing: these exact values are
        # what redundancy.py's NATURALNESS_THRESHOLD (8.0) was calibrated
        # against in FIX 3, so changing the arithmetic here would silently
        # invalidate that calibration.
        text = f"{word_i} {word_j}"
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0].to(self.device)

        # word_i's own tokenisation gives us the split point between the
        # "context" tokens and the "word_j" tokens we're scoring.
        prefix_ids = self.tokenizer(word_i, return_tensors="pt")["input_ids"][0]
        split = len(prefix_ids)

        if split >= len(input_ids):
            # Degenerate input (e.g. empty/whitespace word_j) — no tokens
            # to score. Treat as neutral rather than crashing.
            return 0.0

        with torch.no_grad():
            logits = self.model(input_ids.unsqueeze(0)).logits[0]  # [seq_len, vocab]

        log_probs = F.log_softmax(logits, dim=-1)

        # logits[p] predicts the token at position p+1, so the score for
        # the token at position `pos` is read from logits[pos - 1].
        token_logprobs = [
            log_probs[pos - 1, input_ids[pos]].item()
            for pos in range(split, len(input_ids))
        ]
        return sum(token_logprobs) / len(token_logprobs)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def token_count(self, text: str) -> int:
        """Real token count via the model's own tokenizer (not word count)."""
        return len(self.tokenizer(text)["input_ids"])


if __name__ == "__main__":
    llm = LLM()

    print("\n--- generate() sanity check ---")
    out = llm.generate(
        "Classify sentiment as positive or negative",
        "The battery died after three hours.",
    )
    print(f"output: {out!r}")

    print("\n--- logprob() sanity check (natural vs unnatural pairs) ---")
    pairs = [
        ("bugs", "errors"),      # near-synonyms, plausibly read oddly back-to-back
        ("positive", "negative"),  # antonyms, but a very natural collocation
        ("diagnose", "spreadsheet"),  # unrelated, should read unnaturally
        ("the", "cat"),          # ordinary natural continuation
    ]
    for w_i, w_j in pairs:
        lp = llm.logprob(w_i, w_j)
        print(f"logprob({w_i!r:12s}, {w_j!r:12s}) = {lp:.4f}")

    print("\n--- token_count() sanity check ---")
    for t in ["hello", "Classify sentiment as positive or negative"]:
        print(f"token_count({t!r}) = {llm.token_count(t)}")

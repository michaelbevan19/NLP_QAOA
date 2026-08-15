"""
Curated test prompts for NLP-QAOA evaluation.

Each entry is a dict with:
    id            short identifier
    domain        task domain (spread across several, per the project report)
    prompt        the original, already-written prompt (the thing we optimize)
    sample_input  the input the prompt is applied to; without this there is
                  no output to compare, so the removal test (Section 4.3 of
                  the report) has nothing to measure against.

Design notes (see report Section 5, step 1):
- Most prompts deliberately contain politeness openers, hedging, and
  REDUNDANT SYNONYM GROUPS (e.g. "bugs or errors or mistakes"). Without
  these, the two-signal redundancy check (Section 4.4) never fires and we
  get no evidence it works.
- Three prompts are deliberately concise CONTROLS. NLP-QAOA should leave
  these largely intact; if it strips them down further, that's a real
  finding to report (Section 7), not a bug to paper over.
- One control ("control_sentiment") intentionally pairs "positive" and
  "negative" — an antonym pair that sits close in embedding space. This is
  the exact case Section 4.4 says similarity-alone would wrongly flag as
  redundant. It's a deliberate stress test for the two-signal check, not
  an oversight.
- Lengths vary: several ~15-token controls, several ~35-45 token bloated
  prompts, so the length-penalty term has something to bite on across a
  range.
"""

PROMPTS = [
    # ---- software / code review ----
    {
        "id": "code_review",
        "domain": "software",
        "prompt": (
            "I would really appreciate it if you could please take a careful "
            "look at the following Python function and let me know about any "
            "bugs or errors or mistakes you happen to notice in it"
        ),
        "sample_input": "def divide(a, b): return a / b",
    },

    # ---- summarisation ----
    {
        "id": "summarise_article",
        "domain": "summarisation",
        "prompt": (
            "Could you kindly go ahead and read through the article provided "
            "below and then give me a short summary or overview or recap of "
            "the main points that are discussed in it"
        ),
        "sample_input": (
            "The city council approved a new transit line yesterday. "
            "Construction begins in March and is expected to take four "
            "years. Funding comes from a mix of federal grants and "
            "municipal bonds."
        ),
    },

    # ---- support-ticket routing ----
    {
        "id": "ticket_routing",
        "domain": "support-ticket routing",
        "prompt": (
            "Please read this customer support message carefully and figure "
            "out or determine or decide which department it should be sent "
            "to, either technical support or billing or general enquiries"
        ),
        "sample_input": (
            "My card was charged twice for the same subscription last month."
        ),
    },

    # ---- medical triage ----
    {
        "id": "medical_triage",
        "domain": "medical triage",
        "prompt": (
            "Based on the symptoms described below, could you please assess "
            "and evaluate and judge whether this case should be treated as "
            "urgent or as routine, taking care to consider all the details "
            "mentioned"
        ),
        "sample_input": (
            "Patient reports chest tightness and shortness of breath "
            "beginning two hours ago."
        ),
    },

    # ---- legal clause review ----
    {
        "id": "legal_clause",
        "domain": "legal clause review",
        "prompt": (
            "I would like you to go through the following contract clause "
            "thoroughly and tell me whether it represents a risk or a "
            "liability or a danger to the company, or whether it is "
            "acceptable as written"
        ),
        "sample_input": (
            "The vendor shall not be liable for any damages arising from "
            "service interruption of any duration."
        ),
    },

    # ---- data extraction ----
    {
        "id": "data_extraction",
        "domain": "data extraction",
        "prompt": (
            "Please have a look at the text below and extract or pull out or "
            "identify all of the dates and names and locations that appear "
            "anywhere within it"
        ),
        "sample_input": (
            "Maria Chen met with the Osaka delegation on 14 March 2024 at "
            "the Berlin office."
        ),
    },

    # ---- news categorisation ----
    {
        "id": "news_categorisation",
        "domain": "news categorisation",
        "prompt": (
            "Would you be so kind as to read the following news snippet and "
            "classify or categorise or label it into the single most "
            "appropriate topic"
        ),
        "sample_input": (
            "Central banks raised interest rates again this week in an "
            "effort to curb inflation."
        ),
    },

    # ---- concise controls: system should leave these largely intact ----
    {
        "id": "control_translate",
        "domain": "translation",
        "prompt": "Translate the following text to French",
        "sample_input": "The library closes at six on weekdays.",
    },
    {
        "id": "control_sql",
        "domain": "software",
        "prompt": "Explain what this SQL query does",
        "sample_input": "SELECT name FROM users WHERE created_at > '2024-01-01';",
    },
    {
        "id": "control_sentiment",
        "domain": "sentiment classification",
        # Deliberate stress test: "positive"/"negative" are antonyms that
        # sit close together in embedding space (Section 4.4). A
        # similarity-only redundancy check would wrongly flag this pair;
        # the two-signal check should leave it alone because the pair
        # reads perfectly naturally together.
        "prompt": "Classify sentiment as positive or negative",
        "sample_input": "The battery died after three hours.",
    },
]


if __name__ == "__main__":
    for p in PROMPTS:
        word_count = len(p["prompt"].split())
        print(f"[{p['id']:22s}] domain={p['domain']:24s} words={word_count:3d}  "
              f"prompt={p['prompt']!r}")

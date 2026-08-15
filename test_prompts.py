PROMPTS = [
    {
        "id": "code_review",
        "domain": "software",
        "prompt": "I would really appreciate it if you could please take a "
                  "careful look at the following Python function and let me "
                  "know about any bugs or errors or mistakes you happen to "
                  "notice in it",
        "sample_input": "def divide(a, b): return a / b"
    },
    {
        "id": "summarise_article",
        "domain": "summarisation",
        "prompt": "Could you kindly go ahead and read through the article "
                  "provided below and then give me a short summary or "
                  "overview or recap of the main points that are discussed "
                  "in it",
        "sample_input": "The city council approved a new transit line "
                        "yesterday. Construction begins in March and is "
                        "expected to take four years. Funding comes from a "
                        "mix of federal grants and municipal bonds."
    },
    {
        "id": "ticket_routing",
        "domain": "support",
        "prompt": "Please read this customer support message carefully and "
                  "figure out or determine or decide which department it "
                  "should be sent to, either technical or billing or general",
        "sample_input": "My card was charged twice for the same subscription "
                        "last month."
    },
    {
        "id": "medical_triage",
        "domain": "medical",
        "prompt": "Based on the symptoms described below could you please "
                  "assess and evaluate whether this case should be treated "
                  "as urgent or as routine, taking care to consider all the "
                  "details mentioned",
        "sample_input": "Patient reports chest tightness and shortness of "
                        "breath beginning two hours ago."
    },
    {
        "id": "legal_clause",
        "domain": "legal",
        "prompt": "I would like you to go through the following contract "
                  "clause thoroughly and tell me whether it represents a "
                  "risk or a liability or a danger to the company or whether "
                  "it is acceptable",
        "sample_input": "The vendor shall not be liable for any damages "
                        "arising from service interruption of any duration."
    },
    {
        "id": "data_extraction",
        "domain": "extraction",
        "prompt": "Please have a look at the text below and extract or pull "
                  "out or identify all of the dates and names and locations "
                  "that appear anywhere within it",
        "sample_input": "Maria Chen met with the Osaka delegation on 14 March "
                        "2024 at the Berlin office."
    },
    {
        "id": "email_tone",
        "domain": "writing",
        "prompt": "Would you be so kind as to review and examine and check "
                  "the email below and tell me if the tone comes across as "
                  "professional or unprofessional",
        "sample_input": "Hey — need those numbers by EOD, been waiting all "
                        "week on this."
    },

    # ---- concise controls: system should leave these largely intact ----
    {
        "id": "control_translate",
        "domain": "translation",
        "prompt": "Translate the following text to French",
        "sample_input": "The library closes at six on weekdays."
    },
    {
        "id": "control_sql",
        "domain": "software",
        "prompt": "Explain what this SQL query does",
        "sample_input": "SELECT name FROM users WHERE created_at > '2024-01-01';"
    },
    {
        "id": "control_sentiment",
        "domain": "classification",
        "prompt": "Classify sentiment: positive or negative",
        "sample_input": "The battery died after three hours."
    },
]
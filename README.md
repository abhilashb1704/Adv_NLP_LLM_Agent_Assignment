# Adv_NLP_LLM_Agent_Assignment
# Research Report Generator — Multi-Step LLM Agent

A six-step LLM pipeline that turns any research topic into a polished Markdown report.
Built with the **Grok API** (xAI) and **DuckDuckGo** as the external tool.

---

## What the agent does

Given a free-text research topic (e.g. *"the environmental impact of electric vehicles"*),
the agent runs six sequential steps where every step feeds its output directly into the next:

| Step | Type | Input | Output |
|------|------|-------|--------|
| 1 | **LLM** — Decompose | Raw topic string | JSON list of 3–4 sub-questions |
| 2 | **TOOL** — Web search | Sub-questions (Step 1) | Raw search snippets + URLs |
| 3 | **LLM** — Extract facts | Search snippets (Step 2) | JSON list of key facts with sources |
| 4 | **LLM** — Synthesise | Key facts (Step 3) | Markdown draft report |
| 5 | **LLM** — Critique | Draft + key facts (Steps 3 & 4) | JSON critique object |
| 6 | **LLM** — Revise | Draft + critique (Steps 4 & 5) | Final polished Markdown report |

No step can be skipped — each one is an explicit function that reads from and writes
to a shared `state` dictionary.

---

## Installation

```bash
# 1. Clone / unzip the repo
cd research_agent

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Grok API key
export GROQ_API_KEY="gsk_..."   # Windows: set GROQ_API_KEY=gsk_...
```

---

## Running the agent

```bash
python agent.py "the environmental impact of electric vehicles"
```

The agent will print progress for each step to the terminal.
Two output files are written to the `reports/` directory:

- `<slug>_<timestamp>.md` — the final polished Markdown report
- `<slug>_<timestamp>_state.json` — the full accumulated state (for inspection / grading)

---

## What inputs it expects

A single quoted string — the research topic. It can be broad or narrow:

```bash
python agent.py "quantum computing applications in cryptography"
python agent.py "rising antibiotic resistance in South Asia"
python agent.py "impact of remote work on urban housing markets"
```

Avoid extremely short (< 4 words) or vague inputs like *"AI"* — the decomposition step
will produce better sub-questions from a topic with some context.

---

## Chain structure

Each step is an isolated function `stepN_name(state: dict) -> dict`.
The `state` dictionary is the single source of truth:

```
state = {
  "topic":           str,          # user input
  "sub_questions":   list[str],    # Step 1 → Step 2
  "search_results":  dict,         # Step 2 → Step 3
  "key_facts":       list[dict],   # Step 3 → Steps 4 & 5
  "draft":           str,          # Step 4 → Steps 5 & 6
  "critique":        dict,         # Step 5 → Step 6
  "final_report":    str,          # Step 6 → saved to disk
}
```

---

## Error handling

- If DuckDuckGo returns zero results for a sub-question, the step records an empty list
  and the agent continues — Step 3 will note the gap.
- The `parse_json_block()` helper strips Markdown fences and locates the outermost
  JSON object/array, making LLM output parsing robust to minor formatting variation.
- A `SEARCH_DELAY` constant (default 1 s) prevents DuckDuckGo rate-limiting.

---

## Model

Default: `llama-3.3-70b-versatile` (fast, low-cost). Change the `MODEL` constant in `agent.py`
to `llama-3.1-70b-versatile` for higher quality output at greater cost.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `openai` | Grok API client (OpenAI-compatible) |
| `duckduckgo-search` | Free web search — no API key required |

No LangChain, LlamaIndex, or agent frameworks are used.

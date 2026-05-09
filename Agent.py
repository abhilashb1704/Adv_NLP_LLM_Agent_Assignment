"""
Research Report Generator — Multi-Step LLM Agent
Uses Groq API for LLM calls + DuckDuckGo (via ddgs) as external tool.

Pipeline:
  Step 1 (LLM)  — Decompose the topic into focused sub-questions
  Step 2 (TOOL) — Web-search each sub-question with DuckDuckGo
  Step 3 (LLM)  — Extract & structure key facts from raw search results
  Step 4 (LLM)  — Synthesise facts into a coherent narrative draft
  Step 5 (LLM)  — Critique the draft for gaps, bias, unsupported claims
  Step 6 (LLM)  — Revise draft using the critique → final polished report

State object accumulates everything; final report written to reports/<slug>.md
"""

import os
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI          # Groq is OpenAI-compatible
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"  # free on Groq; fast & capable
MAX_SEARCH_RESULTS = 4             # results per sub-question
SEARCH_DELAY = 1.0                 # seconds between DDG calls (rate-limit safety)

client = OpenAI(
    api_key=GROQ_API_KEY or "placeholder",  # validated at runtime in run_agent()
    base_url="https://api.groq.com/openai/v1",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def llm(system: str, user: str, label: str = "") -> str:
    """Single LLM call; returns content string."""
    tag = f"[LLM:{label}] " if label else "[LLM] "
    print(f"{tag}calling {MODEL}…", flush=True)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    content = response.choices[0].message.content
    print(f"{tag}done ({len(content)} chars)\n", flush=True)
    return content


def parse_json_block(text: str) -> any:
    """Extract the first JSON object/array from an LLM response."""
    # Strip markdown fences
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    # Find outermost { } or [ ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = clean.find(start_char)
        if start != -1:
            depth = 0
            for i, ch in enumerate(clean[start:], start):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        return json.loads(clean[start:i+1])
    raise ValueError(f"No JSON found in:\n{text[:300]}")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


# ---------------------------------------------------------------------------
# Step 1 — Decompose topic into sub-questions (LLM)
# ---------------------------------------------------------------------------
SYSTEM_DECOMPOSE = """You are a senior research editor.
Your job is to decompose a broad research topic into 3–4 focused, searchable sub-questions.
Each sub-question must be specific enough that a web search would return useful results.
Respond ONLY with a JSON object in this exact shape — no preamble, no markdown fences:
{
  "topic": "<the original topic>",
  "sub_questions": ["<q1>", "<q2>", "<q3>"]
}"""

def step1_decompose(state: dict) -> dict:
    print("=" * 60)
    print("STEP 1 — Decompose topic into sub-questions")
    print("=" * 60)
    raw = llm(
        system=SYSTEM_DECOMPOSE,
        user=f"Research topic: {state['topic']}",
        label="decompose",
    )
    parsed = parse_json_block(raw)
    state["sub_questions"] = parsed["sub_questions"]
    state["step1_raw"] = raw
    print(f"  Sub-questions identified: {len(state['sub_questions'])}")
    for i, q in enumerate(state["sub_questions"], 1):
        print(f"  {i}. {q}")
    print()
    return state


# ---------------------------------------------------------------------------
# Step 2 — Web search (TOOL)
# ---------------------------------------------------------------------------
def step2_web_search(state: dict) -> dict:
    print("=" * 60)
    print("STEP 2 — Web search (DuckDuckGo)")
    print("=" * 60)
    results_by_question = {}
    ddgs = DDGS()

    for q in state["sub_questions"]:
        print(f"  Searching: {q!r}")
        try:
            hits = list(ddgs.text(q, max_results=MAX_SEARCH_RESULTS))
            results_by_question[q] = [
                {"title": h.get("title", ""), "snippet": h.get("body", ""), "url": h.get("href", "")}
                for h in hits
            ]
            print(f"    → {len(hits)} results")
        except Exception as exc:
            print(f"    ⚠ Search failed for {q!r}: {exc}")
            results_by_question[q] = []
        time.sleep(SEARCH_DELAY)

    state["search_results"] = results_by_question
    total = sum(len(v) for v in results_by_question.values())
    print(f"\n  Total snippets collected: {total}\n")
    return state


# ---------------------------------------------------------------------------
# Step 3 — Extract & structure key facts (LLM)
# ---------------------------------------------------------------------------
SYSTEM_EXTRACT = """You are a meticulous research analyst.
You receive raw web-search snippets for several sub-questions about a topic.
Extract the most important, verifiable facts. Remove duplicates and irrelevant material.
Respond ONLY with a JSON object — no preamble, no markdown fences:
{
  "topic": "<topic>",
  "key_facts": [
    {
      "sub_question": "<sub-question this fact answers>",
      "fact": "<one clear factual statement>",
      "source_url": "<URL or empty string>"
    }
  ]
}
Include 6–12 key facts total across all sub-questions."""

def step3_extract_facts(state: dict) -> dict:
    print("=" * 60)
    print("STEP 3 — Extract structured facts from search results")
    print("=" * 60)

    # Build a compact representation of search results for the prompt
    snippets_text = ""
    for q, results in state["search_results"].items():
        snippets_text += f"\n### Sub-question: {q}\n"
        if not results:
            snippets_text += "  [No results found]\n"
        for r in results:
            snippets_text += f"- **{r['title']}** ({r['url']})\n  {r['snippet']}\n"

    user_msg = f"Topic: {state['topic']}\n\nSearch Results:\n{snippets_text}"
    raw = llm(system=SYSTEM_EXTRACT, user=user_msg, label="extract")
    parsed = parse_json_block(raw)
    state["key_facts"] = parsed["key_facts"]
    state["step3_raw"] = raw
    print(f"  Key facts extracted: {len(state['key_facts'])}\n")
    return state


# ---------------------------------------------------------------------------
# Step 4 — Synthesise draft report (LLM)
# ---------------------------------------------------------------------------
SYSTEM_SYNTHESISE = """You are an expert science and technology writer.
Using the structured facts provided, write a coherent, well-organised research report draft.
Requirements:
- 400–600 words
- Use Markdown headings (## for sections)
- Cover each sub-question in a dedicated section
- Cite source URLs inline as [source](url) where available
- Write in a clear, neutral, informative tone
- End with a brief "## Summary" paragraph
Do NOT add anything beyond the Markdown report."""

def step4_synthesise(state: dict) -> dict:
    print("=" * 60)
    print("STEP 4 — Synthesise draft report")
    print("=" * 60)
    facts_text = json.dumps(state["key_facts"], indent=2)
    user_msg = (
        f"Topic: **{state['topic']}**\n\n"
        f"Sub-questions addressed:\n" +
        "\n".join(f"- {q}" for q in state["sub_questions"]) +
        f"\n\nStructured facts:\n```json\n{facts_text}\n```"
    )
    draft = llm(system=SYSTEM_SYNTHESISE, user=user_msg, label="synthesise")
    state["draft"] = draft
    print(f"  Draft length: {len(draft.split())} words\n")
    return state


# ---------------------------------------------------------------------------
# Step 5 — Critique the draft (LLM)
# ---------------------------------------------------------------------------
SYSTEM_CRITIQUE = """You are a rigorous peer reviewer for a research publication.
Read the draft report and the original key facts, then provide a structured critique.
Respond ONLY with a JSON object — no preamble, no markdown fences:
{
  "overall_quality": "<brief 1-sentence verdict>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": [
    {
      "issue": "<description of the problem>",
      "suggestion": "<concrete fix>"
    }
  ],
  "missing_topics": ["<topic not covered that should be>"],
  "unsupported_claims": ["<claim in draft not backed by the facts provided>"]
}"""

def step5_critique(state: dict) -> dict:
    print("=" * 60)
    print("STEP 5 — Critique the draft")
    print("=" * 60)
    facts_text = json.dumps(state["key_facts"], indent=2)
    user_msg = (
        f"ORIGINAL TOPIC: {state['topic']}\n\n"
        f"KEY FACTS:\n{facts_text}\n\n"
        f"DRAFT REPORT:\n{state['draft']}"
    )
    raw = llm(system=SYSTEM_CRITIQUE, user=user_msg, label="critique")
    state["critique"] = parse_json_block(raw)
    state["step5_raw"] = raw
    print(f"  Weaknesses found: {len(state['critique'].get('weaknesses', []))}")
    print(f"  Missing topics: {state['critique'].get('missing_topics', [])}\n")
    return state


# ---------------------------------------------------------------------------
# Step 6 — Revise into final polished report (LLM)
# ---------------------------------------------------------------------------
SYSTEM_REVISE = """You are an expert editor finalising a research report.
You have a draft and a detailed critique. Produce the final polished version.
Requirements:
- Address every weakness and missing topic listed in the critique
- Remove or qualify any unsupported claims
- 500–700 words
- Use Markdown: # for title, ## for sections, ### for sub-sections if needed
- Keep source citations ([source](url))
- Add a "## References" section at the end listing all cited URLs
- Professional, neutral, publication-ready tone
Output ONLY the final Markdown report."""

def step6_revise(state: dict) -> dict:
    print("=" * 60)
    print("STEP 6 — Revise into final polished report")
    print("=" * 60)
    critique_text = json.dumps(state["critique"], indent=2)
    user_msg = (
        f"TOPIC: {state['topic']}\n\n"
        f"DRAFT:\n{state['draft']}\n\n"
        f"CRITIQUE:\n{critique_text}"
    )
    final = llm(system=SYSTEM_REVISE, user=user_msg, label="revise")
    state["final_report"] = final
    print(f"  Final report length: {len(final.split())} words\n")
    return state


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------
def save_output(state: dict) -> Path:
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    slug = slugify(state["topic"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"{slug}_{ts}.md"
    json_path = out_dir / f"{slug}_{ts}_state.json"

    # Write final report
    md_path.write_text(state["final_report"], encoding="utf-8")

    # Write full state (minus raw LLM noise) for inspection
    export_state = {k: v for k, v in state.items()
                    if not k.endswith("_raw") and k != "draft"}
    export_state["draft_preview"] = state.get("draft", "")[:500]
    json_path.write_text(json.dumps(export_state, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"✅ Final report   → {md_path}")
    print(f"✅ Full state     → {json_path}")
    return md_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_agent(topic: str) -> dict:
    if not GROQ_API_KEY:
        print("ERROR: GROK_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    state = {
        "topic": topic,
        "started_at": datetime.now().isoformat(),
    }

    print(f"\n🔬 Research Agent — Topic: {topic!r}\n")
    state = step1_decompose(state)
    state = step2_web_search(state)
    state = step3_extract_facts(state)
    state = step4_synthesise(state)
    state = step5_critique(state)
    state = step6_revise(state)

    state["finished_at"] = datetime.now().isoformat()
    save_output(state)
    return state


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py \"<research topic>\"")
        sys.exit(1)
    topic_arg = " ".join(sys.argv[1:])
    run_agent(topic_arg)

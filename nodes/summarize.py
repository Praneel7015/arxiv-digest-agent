"""
Produce the structured executive briefing via Groq.
"""
from __future__ import annotations

import json
from state import AgentState
import llm

BRIEFING_SYSTEM = """You are a research analyst writing an executive briefing of an academic paper.
Return ONLY a single JSON object with exactly these keys:
- title (string)
- authors (array of strings)
- arxiv_id (string)
- publish_date (string)
- link (string)
- why_it_matters (string, 1 plain-English paragraph)
- problem_statement (string)
- method (array of short bullet strings)
- key_results (array of short bullet strings)
- limitations (array of short bullet strings)  # REQUIRED - never empty; if the paper is vague, say what is missing
- suggested_followups (array of 3-5 reader questions)

Rules:
- Base claims ONLY on the provided paper text.
- Do not invent experiments, numbers, or affiliations.
- limitations must be explicit and non-empty.
- No markdown, no prose outside the JSON object.
"""


def _context_blob(state: AgentState, max_chars: int = 14_000) -> str:
    paper = state.selected_paper
    parts: list[str] = []
    if paper:
        parts.append(f"TITLE: {paper.title}")
        parts.append(f"AUTHORS: {', '.join(paper.authors)}")
        parts.append(f"ARXIV_ID: {paper.arxiv_id}")
        parts.append(f"PUBLISHED: {paper.published}")
        parts.append(f"LINK: https://arxiv.org/abs/{paper.arxiv_id}")
        parts.append(f"ABSTRACT: {paper.abstract}")

    preferred = [
        "introduction",
        "method",
        "methodology",
        "approach",
        "results",
        "experiments",
        "evaluation",
        "limitations",
        "discussion",
        "conclusion",
    ]
    sections = state.parsed_sections or {}
    included = set()
    for key in preferred:
        if key in sections:
            parts.append(f"## {key.upper()}\n{sections[key][:2500]}")
            included.add(key)

    # Include remaining sections (abstract, preamble, etc.) until budget is hit.
    for key, text in sections.items():
        if key not in included and key != "preamble":
            parts.append(f"## {key.upper()}\n{text[:1500]}")

    blob = "\n\n".join(parts)
    if len(blob) > max_chars:
        blob = blob[:max_chars] + "\n\n[truncated]"
    return blob


def summarize(state: AgentState) -> AgentState:
    paper = state.selected_paper
    if paper is None:
        state.warnings.append("summarize called with no selected_paper.")
        return state

    user = (
        f"Parse status: {state.parse_status}\n"
        f"Warnings so far: {state.warnings[-3:]}\n\n"
        f"Paper text:\n{_context_blob(state)}"
    )

    try:
        briefing = llm.chat_json(BRIEFING_SYSTEM, user, temperature=0.1)
    except Exception as e:
        state.warnings.append(f"LLM briefing failed ({e}); emitting metadata-only stub.")
        briefing = {
            "title": paper.title,
            "authors": paper.authors,
            "arxiv_id": paper.arxiv_id,
            "publish_date": paper.published,
            "link": f"https://arxiv.org/abs/{paper.arxiv_id}",
            "why_it_matters": paper.abstract[:400],
            "problem_statement": "Unavailable — LLM call failed; see abstract.",
            "method": [],
            "key_results": [],
            "limitations": [
                "Briefing generation failed; limitations could not be extracted automatically."
            ],
            "suggested_followups": [
                "What is the core method?",
                "What datasets or benchmarks are used?",
                "What are the stated limitations?",
            ],
        }

    # Enforce required keys / non-empty limitations even if the model skimps.
    briefing.setdefault("title", paper.title)
    briefing.setdefault("authors", paper.authors)
    briefing.setdefault("arxiv_id", paper.arxiv_id)
    briefing.setdefault("publish_date", paper.published)
    briefing.setdefault("link", f"https://arxiv.org/abs/{paper.arxiv_id}")
    lim = briefing.get("limitations")
    if not lim:
        briefing["limitations"] = [
            "Paper text did not clearly enumerate limitations; treat claims cautiously."
        ]
    kr = briefing.get("key_results")
    if not kr:
        briefing["key_results"] = [
            "Key results could not be extracted — the results section may have been truncated or absent."
        ]
    sf = briefing.get("suggested_followups")
    if not sf:
        briefing["suggested_followups"] = [
            "What datasets or benchmarks are used?",
            "How does this compare to prior work?",
            "What are the main limitations?",
        ]

    if state.parse_status == "failed":
        briefing["limitations"] = list(briefing.get("limitations") or []) + [
            "PDF parse failed — this briefing is based primarily on the abstract."
        ]
        state.warnings.append("Briefing generated from abstract fallback due to parse failure.")

    state.briefing = briefing
    return state


def format_briefing_markdown(briefing: dict) -> str:
    """Pretty CLI / README rendering of the JSON briefing."""
    lines = [
        f"# {briefing.get('title', 'Untitled')}",
        "",
        f"**Authors:** {', '.join(briefing.get('authors') or [])}",
        f"**arXiv:** {briefing.get('arxiv_id')}  |  **Date:** {briefing.get('publish_date')}",
        f"**Link:** {briefing.get('link')}",
        "",
        "## Why it matters",
        briefing.get("why_it_matters") or "",
        "",
        "## Problem",
        briefing.get("problem_statement") or "",
        "",
        "## Method",
        *([f"- {m}" for m in briefing.get("method") or []] or ["- (none)"]),
        "",
        "## Key results",
        *([f"- {m}" for m in briefing.get("key_results") or []] or ["- (none)"]),
        "",
        "## Limitations",
        *([f"- {m}" for m in briefing.get("limitations") or []] or ["- (none)"]),
        "",
        "## Suggested follow-ups",
        *([f"- {m}" for m in briefing.get("suggested_followups") or []] or ["- (none)"]),
    ]
    return "\n".join(lines)

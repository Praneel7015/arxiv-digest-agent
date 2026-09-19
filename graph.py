"""
The agent graph, built as an explicit Python state machine rather than a
framework DSL — the routing logic below IS the graph. Each node is a plain
function: (AgentState) -> AgentState. Edges are decided by small router
functions that inspect state and return the name of the next node.
"""
from __future__ import annotations

import json
import sys
from state import AgentState
from nodes.query_understanding import query_understanding
from nodes.arxiv_retrieval import arxiv_retrieval
from nodes.selection_ranking import selection_ranking
from nodes.fetch_and_parse import fetch_and_parse
from nodes.chunk_and_embed import chunk_and_embed
from nodes.summarize import summarize, format_briefing_markdown
from nodes.qa_loop import answer_question


def _log(msg: str) -> None:
    """Pipeline progress indicator — gives the user confidence that things are happening."""
    print(f"  \u2192 {msg}", flush=True)


def error_handler(state: AgentState) -> AgentState:
    """Surfaces collected warnings instead of crashing. Terminal node for empty retrieval."""
    for w in state.warnings:
        print(f"[warning] {w}")
    if not state.warnings:
        print("[warning] Reached error_handler with no recorded warning - investigate.")
    return state


def route_after_retrieval(state: AgentState) -> str:
    """Failure case (§5): zero candidates -> error_handler. One -> skip ranking. Many -> rank."""
    if not state.candidates:
        return "error_handler"
    if len(state.candidates) == 1:
        return "fetch_and_parse"
    return "selection_ranking"


def run_pipeline(query: str) -> AgentState:
    state = AgentState(query=query)

    _log("Understanding query...")
    state = query_understanding(state)
    if state.intent == "topic_search" and state.expanded_queries:
        _log(f"Expanded to {len(state.expanded_queries)} search queries")

    _log(f"Fetching papers from arXiv ({state.intent})...")
    state = arxiv_retrieval(state)
    _log(f"Found {len(state.candidates)} candidate paper(s)")

    next_step = route_after_retrieval(state)
    if next_step == "error_handler":
        return error_handler(state)

    if next_step == "selection_ranking":
        _log(f"Ranking {len(state.candidates)} candidates (hybrid BM25 + cosine, two-stage)...")
        state = selection_ranking(state)
        _log(f"Selected: {state.selected_paper.title}")

    _log("Downloading and parsing PDF...")
    state = fetch_and_parse(state)
    _log(f"Parse status: {state.parse_status} ({len(state.parsed_sections)} sections)")

    _log("Chunking and embedding into Chroma...")
    state = chunk_and_embed(state)
    _log(f"{len(state.chunks)} chunks stored")

    _log("Generating structured briefing via Groq...")
    state = summarize(state)
    _log("Briefing complete")

    return state


def run_qa(state: AgentState, question: str) -> AgentState:
    return answer_question(state, question)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "KV-cache compression for LLMs"
    result = run_pipeline(q)
    print(f"\nquery:      {result.query}")
    print(f"intent:     {result.intent}")
    print(f"candidates: {len(result.candidates)}")
    for c in result.candidates:
        print(f"  - [{c.arxiv_id}] {c.title}")
    if result.briefing:
        print("\n--- briefing (markdown) ---")
        print(format_briefing_markdown(result.briefing))
        print("\n--- briefing (json) ---")
        print(json.dumps(result.briefing, indent=2))
    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings:
            print(f"  - {w}")

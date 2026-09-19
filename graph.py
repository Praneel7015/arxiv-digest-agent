"""
The agent graph, built as an explicit Python state machine rather than a
framework DSL — the routing logic below IS the graph. Each node is a plain
function: (AgentState) -> AgentState. Edges are decided by small router
functions that inspect state and return the name of the next node.
"""
from __future__ import annotations

import json
from state import AgentState
from nodes.query_understanding import query_understanding
from nodes.arxiv_retrieval import arxiv_retrieval
from nodes.selection_ranking import selection_ranking
from nodes.fetch_and_parse import fetch_and_parse
from nodes.chunk_and_embed import chunk_and_embed
from nodes.summarize import summarize, format_briefing_markdown
from nodes.qa_loop import answer_question


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
    state = query_understanding(state)
    state = arxiv_retrieval(state)

    next_step = route_after_retrieval(state)
    if next_step == "error_handler":
        return error_handler(state)

    if next_step == "selection_ranking":
        state = selection_ranking(state)

    state = fetch_and_parse(state)
    state = chunk_and_embed(state)
    state = summarize(state)
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

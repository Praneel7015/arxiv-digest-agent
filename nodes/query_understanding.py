"""
Query understanding: intent classification + LLM-powered query expansion.

For topic_search, the LLM rewrites the user's vague query into 5-6 targeted
arXiv search strings, improving recall. If Groq fails, we silently fall back
to the raw query (no crash).
"""
from __future__ import annotations

from state import AgentState
from arxiv_client import ARXIV_ID_PATTERN
import llm

EXPAND_SYSTEM = """You expand a research topic into 5-6 targeted arXiv search queries.
Return ONLY a JSON object with exactly these keys:
- queries (array of 5-6 short search strings suitable for the arXiv API, each phrased differently)
- time_hint (string or null: if the user implies recency like "latest" or "recent", output "recent"; otherwise null)

Rules:
- Each query should be 3-8 words, varied vocabulary (synonyms, related terms).
- Do not repeat the original query verbatim as one of the variants.
- Do not add explanations outside the JSON.
"""


def query_understanding(state: AgentState) -> AgentState:
    """Decide intent: does the query name a specific paper, or describe a topic?"""
    has_id = bool(ARXIV_ID_PATTERN.search(state.query))
    looks_like_url = "arxiv.org" in state.query.lower()
    state.intent = "paper_lookup" if (has_id or looks_like_url) else "topic_search"

    if state.intent == "topic_search":
        state.expanded_queries = _expand_query(state.query)

    return state


def _expand_query(raw_query: str) -> list[str]:
    """Use Groq to generate 5-6 search query variants. Falls back to [raw_query] on any error."""
    try:
        result = llm.chat_json(
            EXPAND_SYSTEM,
            f"Topic: {raw_query}",
            temperature=0.4,
        )
        queries = result.get("queries", [])
        if not queries or not isinstance(queries, list):
            return [raw_query]
        # Always include the original query as the first entry for BM25 grounding.
        cleaned = [raw_query] + [q.strip() for q in queries if q.strip()]
        return cleaned[:7]  # cap at original + 6 expansions
    except Exception:
        return [raw_query]

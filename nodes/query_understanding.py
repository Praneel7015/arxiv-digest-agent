from state import AgentState
from arxiv_client import ARXIV_ID_PATTERN


def query_understanding(state: AgentState) -> AgentState:
    """Decide intent: does the query name a specific paper, or describe a topic?"""
    has_id = bool(ARXIV_ID_PATTERN.search(state.query))
    looks_like_url = "arxiv.org" in state.query.lower()
    state.intent = "paper_lookup" if (has_id or looks_like_url) else "topic_search"
    return state

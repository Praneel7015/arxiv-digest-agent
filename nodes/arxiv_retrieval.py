"""
arXiv retrieval: fetch candidate papers.

For topic_search: loops over expanded_queries (from LLM expansion),
fetches ~10 papers per query, deduplicates by arxiv_id.
For paper_lookup: fetches exactly one paper by ID.
"""
from state import AgentState
import arxiv_client


def arxiv_retrieval(state: AgentState) -> AgentState:
    """
    Fetch candidates from arXiv.
    - paper_lookup: fetch exactly that one paper (or none, if the id is bad).
    - topic_search: use expanded_queries for wider recall, deduplicate by arxiv_id.
    Zero/failed results are recorded as warnings, not exceptions —
    the graph decides what to do with an empty candidate list.
    """
    if state.intent == "paper_lookup":
        try:
            paper = arxiv_client.fetch_by_id(state.query)
        except ValueError as e:
            state.warnings.append(str(e))
            paper = None

        if paper is None:
            state.warnings.append(f"No arXiv paper found for '{state.query}'.")
            state.candidates = []
        else:
            state.candidates = [paper]
            state.selected_paper = paper  # only one candidate, no ranking needed
    else:
        # Use expanded queries if available; fall back to raw query.
        queries = state.expanded_queries if state.expanded_queries else [state.query]
        results = arxiv_client.search_multi_queries(queries, per_query=10)
        state.candidates = results
        if not results:
            state.warnings.append(
                f"No arXiv papers found for topic '{state.query}'. "
                "Try rephrasing or being more specific."
            )
        else:
            state.warnings.append(
                f"Fetched {len(results)} unique candidates from "
                f"{len(queries)} expanded queries."
            )
    return state

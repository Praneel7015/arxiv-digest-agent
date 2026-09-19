from state import AgentState
import arxiv_client


def arxiv_retrieval(state: AgentState) -> AgentState:
    """
    Fetch candidates from arXiv.
    - paper_lookup: fetch exactly that one paper (or none, if the id is bad).
    - topic_search: fetch up to 5 candidates for ranking downstream.
    Zero/failed results are recorded as warnings, not exceptions -
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
        results = arxiv_client.search_by_topic(state.query, max_results=5)
        state.candidates = results
        if not results:
            state.warnings.append(
                f"No arXiv papers found for topic '{state.query}'. "
                "Try rephrasing or being more specific."
            )
    return state

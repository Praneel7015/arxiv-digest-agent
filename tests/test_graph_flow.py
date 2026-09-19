from state import AgentState, PaperMeta
import arxiv_client
from nodes.query_understanding import query_understanding
from nodes.arxiv_retrieval import arxiv_retrieval
from graph import route_after_retrieval


def _fake_paper(i=1):
    return PaperMeta(
        arxiv_id=f"2401.1234{i}",
        title=f"Fake Paper {i}",
        authors=["A. Author"],
        abstract="An abstract.",
        pdf_url="http://arxiv.org/pdf/fake",
        published="2024-01-01T00:00:00Z",
        categories=["cs.LG"],
    )


def test_paper_lookup_intent():
    state = query_understanding(AgentState(query="2401.12345"))
    assert state.intent == "paper_lookup"


def test_topic_search_intent():
    state = query_understanding(AgentState(query="KV-cache compression for LLMs"))
    assert state.intent == "topic_search"


def test_url_is_paper_lookup():
    state = query_understanding(AgentState(query="https://arxiv.org/abs/2401.12345"))
    assert state.intent == "paper_lookup"


def test_zero_results_routes_to_error(monkeypatch):
    monkeypatch.setattr(arxiv_client, "search_by_topic", lambda q, max_results=5: [])
    state = arxiv_retrieval(
        AgentState(query="an extremely obscure nonsense topic", intent="topic_search")
    )
    assert route_after_retrieval(state) == "error_handler"
    assert state.warnings


def test_single_result_skips_ranking(monkeypatch):
    monkeypatch.setattr(arxiv_client, "fetch_by_id", lambda pid: _fake_paper())
    state = arxiv_retrieval(AgentState(query="2401.12345", intent="paper_lookup"))
    assert route_after_retrieval(state) == "fetch_and_parse"
    assert state.selected_paper is not None


def test_many_results_routes_to_ranking(monkeypatch):
    monkeypatch.setattr(
        arxiv_client,
        "search_by_topic",
        lambda q, max_results=5: [_fake_paper(1), _fake_paper(2)],
    )
    state = arxiv_retrieval(AgentState(query="kv cache compression", intent="topic_search"))
    assert route_after_retrieval(state) == "selection_ranking"

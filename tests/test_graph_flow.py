from state import AgentState, PaperMeta
import arxiv_client
import llm
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
    # Paper lookup should NOT expand queries.
    assert state.expanded_queries == []


def test_topic_search_intent_with_expansion(monkeypatch):
    """Topic search should call LLM to expand queries."""
    monkeypatch.setattr(
        llm,
        "chat_json",
        lambda *a, **k: {
            "queries": [
                "KV cache compression transformers",
                "key-value cache optimization LLMs",
                "memory-efficient attention inference",
            ],
            "time_hint": None,
        },
    )
    state = query_understanding(AgentState(query="KV-cache compression for LLMs"))
    assert state.intent == "topic_search"
    assert len(state.expanded_queries) >= 2
    # Original query should always be first.
    assert state.expanded_queries[0] == "KV-cache compression for LLMs"


def test_topic_search_expansion_fallback(monkeypatch):
    """If LLM expansion fails, fall back to original query."""
    monkeypatch.setattr(
        llm,
        "chat_json",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API down")),
    )
    state = query_understanding(AgentState(query="KV-cache compression for LLMs"))
    assert state.intent == "topic_search"
    assert state.expanded_queries == ["KV-cache compression for LLMs"]


def test_url_is_paper_lookup():
    state = query_understanding(AgentState(query="https://arxiv.org/abs/2401.12345"))
    assert state.intent == "paper_lookup"


def test_zero_results_routes_to_error(monkeypatch):
    monkeypatch.setattr(
        arxiv_client,
        "search_multi_queries",
        lambda queries, per_query=10: [],
    )
    state = AgentState(
        query="an extremely obscure nonsense topic",
        intent="topic_search",
        expanded_queries=["an extremely obscure nonsense topic"],
    )
    state = arxiv_retrieval(state)
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
        "search_multi_queries",
        lambda queries, per_query=10: [_fake_paper(1), _fake_paper(2), _fake_paper(3)],
    )
    state = AgentState(
        query="kv cache compression",
        intent="topic_search",
        expanded_queries=["kv cache compression", "KV-cache optimization transformers"],
    )
    state = arxiv_retrieval(state)
    assert route_after_retrieval(state) == "selection_ranking"
    assert len(state.candidates) == 3


def test_multi_query_dedup(monkeypatch):
    """search_multi_queries should deduplicate by arxiv_id."""
    call_count = [0]

    def fake_search(query, max_results=10):
        call_count[0] += 1
        # Both queries return overlapping results.
        return [_fake_paper(1), _fake_paper(2)]

    monkeypatch.setattr(arxiv_client, "search_by_topic", fake_search)

    results = arxiv_client.search_multi_queries(
        ["query A", "query B"], per_query=10
    )
    assert call_count[0] == 2  # Both queries were executed.
    # Dedup: only 2 unique papers despite 4 total results.
    assert len(results) == 2

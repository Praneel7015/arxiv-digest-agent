import numpy as np
from state import AgentState, PaperMeta
import embeddings
from nodes.selection_ranking import selection_ranking, _hybrid_rank, _bm25_scores, _normalize


def _paper(i, abstract=None):
    return PaperMeta(
        arxiv_id=f"id{i}",
        title=f"Paper {i}",
        authors=[],
        abstract=abstract or f"abstract about topic {i}",
        pdf_url="",
        published="",
        categories=[],
    )


def test_bm25_scores_basic():
    """BM25 should score higher when query terms appear in the document."""
    docs = [
        "transformer attention mechanism in neural networks",
        "cooking recipes for pasta and pizza",
        "self-attention layers in transformer architecture",
    ]
    scores = _bm25_scores("transformer attention", docs)
    # First and third docs mention the query terms; second does not.
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_normalize_identical_values():
    """When all values are the same, normalize should return zeros."""
    arr = np.array([0.5, 0.5, 0.5])
    result = _normalize(arr)
    np.testing.assert_array_almost_equal(result, np.zeros(3))


def test_normalize_spread():
    """Min-max normalization should map to [0, 1]."""
    arr = np.array([1.0, 3.0, 5.0])
    result = _normalize(arr)
    np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])


def test_hybrid_rank_combines_both_signals(monkeypatch):
    """Hybrid ranking should combine semantic and lexical scores."""
    abstracts = [
        "neural network transformer attention",
        "pizza cooking recipe Italian food",
        "transformer model self-attention mechanism",
    ]

    # Mock embeddings so that doc 2 (index 2) has highest cosine with query.
    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    doc_vecs = np.array([
        [0.7, 0.3],   # mid cosine
        [0.0, 1.0],   # low cosine
        [0.95, 0.05], # high cosine
    ], dtype=np.float32)

    def fake_embed(texts):
        if len(texts) == 1:
            return np.array([query_vec])
        return doc_vecs

    monkeypatch.setattr(embeddings, "embed", fake_embed)

    scores = _hybrid_rank("transformer attention", abstracts, alpha=0.6)
    # Doc at index 2 should score highest (high cosine + BM25 hit)
    # Doc at index 1 (pizza) should score lowest
    assert scores[2] > scores[1]
    assert scores[0] > scores[1]


def test_selection_ranking_two_stage(monkeypatch):
    """With many candidates, selection_ranking should run two stages."""
    candidates = [_paper(i, abstract=f"topic about neural nets paper {i}") for i in range(15)]
    # Make paper 7's abstract very relevant.
    candidates[7] = _paper(7, abstract="transformer attention mechanism self-attention layers")

    # Use real BM25 but mock embeddings so paper 7 has highest cosine.
    n = len(candidates)
    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    doc_vecs = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        doc_vecs[i] = [0.1 + 0.01 * i, 0.9 - 0.01 * i]
    doc_vecs[7] = [0.99, 0.01]  # paper 7 is most similar

    call_count = [0]

    def fake_embed(texts):
        call_count[0] += 1
        if len(texts) == 1:
            return np.array([query_vec])
        # Return subset of doc_vecs matching current batch size.
        m = len(texts)
        if m == n:
            return doc_vecs
        # Stage 2: return high scores for first few, highest for paper 7 if present.
        vecs = np.zeros((m, 2), dtype=np.float32)
        for j in range(m):
            vecs[j] = [0.3, 0.7]
        # Find paper 7 in this batch.
        for j, txt in enumerate(texts):
            if "transformer attention" in txt:
                vecs[j] = [0.99, 0.01]
        return vecs

    monkeypatch.setattr(embeddings, "embed", fake_embed)

    state = AgentState(
        query="transformer attention",
        candidates=candidates,
    )
    state = selection_ranking(state)

    assert state.selected_paper is not None
    assert state.selected_paper.arxiv_id == "id7"
    # Two stages: at least 4 embed calls (query + docs for stage1, query + docs for stage2).
    assert call_count[0] >= 4


def test_selection_ranking_single_candidate():
    """Single candidate should be auto-selected without ranking."""
    paper = _paper(0, abstract="some paper")
    state = AgentState(query="irrelevant", candidates=[paper])
    state = selection_ranking(state)
    assert state.selected_paper is not None
    assert state.selected_paper.arxiv_id == "id0"


def test_selection_ranking_handles_empty_candidates():
    state = selection_ranking(AgentState(query="irrelevant", candidates=[]))
    assert state.selected_paper is None
    assert state.warnings

import numpy as np
from state import AgentState, PaperMeta
import embeddings
from nodes.selection_ranking import selection_ranking


def _paper(i):
    return PaperMeta(
        arxiv_id=f"id{i}",
        title=f"Paper {i}",
        authors=[],
        abstract=f"abstract {i}",
        pdf_url="",
        published="",
        categories=[],
    )


def test_selection_ranking_picks_highest_similarity(monkeypatch):
    candidates = [_paper(0), _paper(1), _paper(2)]
    query_vec = np.array([1.0, 0.0])
    doc_vecs = np.array(
        [
            [0.0, 1.0],
            [0.9, 0.1],
            [-1.0, 0.0],
        ]
    )

    def fake_embed(texts):
        return np.array([query_vec]) if len(texts) == 1 else doc_vecs

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    state = selection_ranking(AgentState(query="irrelevant", candidates=candidates))
    assert state.selected_paper.arxiv_id == "id1"


def test_selection_ranking_handles_empty_candidates():
    state = selection_ranking(AgentState(query="irrelevant", candidates=[]))
    assert state.selected_paper is None
    assert state.warnings

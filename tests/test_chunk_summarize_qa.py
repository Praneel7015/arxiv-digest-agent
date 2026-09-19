import numpy as np
from state import AgentState, PaperMeta, Chunk
from nodes.chunk_and_embed import chunk_and_embed, _chunk_text
from nodes.summarize import summarize
from nodes.qa_loop import answer_question
import llm


def test_chunk_text_overlaps():
    text = "word " * 800  # long enough for multiple chunks
    chunks = _chunk_text("method", text, start_idx=0)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id.startswith("method_")


def test_chunk_and_embed_upserts(monkeypatch, tmp_path):
    paper = PaperMeta(
        arxiv_id="2401.11111",
        title="T",
        authors=["A"],
        abstract="abs",
        pdf_url="http://x",
        published="2024",
    )
    state = AgentState(
        query="q",
        selected_paper=paper,
        parsed_sections={"method": "A" * 2500, "results": "B" * 2500},
        parse_status="ok",
    )

    monkeypatch.setattr(
        "embeddings.embed",
        lambda texts: np.ones((len(texts), 8), dtype=np.float32),
    )

    captured = {}

    def fake_upsert(arxiv_id, chunks, persist_dir=None):
        captured["n"] = len(chunks)
        captured["id"] = arxiv_id
        return "paper_2401_11111"

    monkeypatch.setattr("vectorstore.upsert_chunks", fake_upsert)
    state = chunk_and_embed(state)
    assert state.collection_name == "paper_2401_11111"
    assert len(state.chunks) >= 2
    assert captured["n"] == len(state.chunks)


def test_summarize_enforces_limitations(monkeypatch):
    paper = PaperMeta(
        arxiv_id="2401.22222",
        title="Cool Paper",
        authors=["Ada"],
        abstract="We invent cool stuff.",
        pdf_url="http://x",
        published="2024-02-01",
    )
    state = AgentState(
        query="q",
        selected_paper=paper,
        parsed_sections={"abstract": paper.abstract, "method": "We use math."},
        parse_status="ok",
    )

    monkeypatch.setattr(
        llm,
        "chat_json",
        lambda *a, **k: {
            "title": "Cool Paper",
            "authors": ["Ada"],
            "arxiv_id": "2401.22222",
            "publish_date": "2024-02-01",
            "link": "https://arxiv.org/abs/2401.22222",
            "why_it_matters": "It matters.",
            "problem_statement": "Hard problem.",
            "method": ["Do math"],
            "key_results": ["It works"],
            "limitations": [],  # model tried to skip — node must fill
            "suggested_followups": ["Why?"],
        },
    )
    state = summarize(state)
    assert state.briefing["limitations"]


def test_qa_grounded_not_found(monkeypatch):
    state = AgentState(query="q", collection_name="paper_x")
    monkeypatch.setattr("vectorstore.query_chunks", lambda *a, **k: [])
    state = answer_question(state, "What is the answer to life?")
    assert "couldn't find" in state.qa_history[-1][1].lower()


def test_qa_uses_retrieved_chunks(monkeypatch):
    state = AgentState(query="q", collection_name="paper_x")
    monkeypatch.setattr(
        "vectorstore.query_chunks",
        lambda *a, **k: [
            {
                "chunk_id": "method_0",
                "section": "method",
                "text": "We quantize KV caches to 4 bits.",
                "distance": 0.1,
            }
        ],
    )
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "They use 4-bit quantization [method_0].")
    state = answer_question(state, "What precision is used?")
    q, a, ids = state.qa_history[-1]
    assert "4-bit" in a
    assert ids == ["method_0"]


def test_state_save_load_roundtrip(tmp_path):
    from state import AgentState, PaperMeta, Chunk

    path = tmp_path / "s.json"
    state = AgentState(
        query="topic",
        intent="topic_search",
        expanded_queries=["topic", "related topic query"],
        candidates=[
            PaperMeta("1", "t", ["a"], "abs", "u", "d", ["cs.LG"]),
        ],
        selected_paper=PaperMeta("1", "t", ["a"], "abs", "u", "d", ["cs.LG"]),
        chunks=[Chunk("c0", "method", "hello")],
        qa_history=[("q", "a", ["c0"])],
        warnings=["w"],
    )
    state.save(str(path))
    loaded = AgentState.load(str(path))
    assert loaded.selected_paper.arxiv_id == "1"
    assert loaded.chunks[0].text == "hello"
    assert loaded.qa_history[0][0] == "q"
    assert loaded.expanded_queries == ["topic", "related topic query"]

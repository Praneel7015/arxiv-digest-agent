import numpy as np
from state import AgentState, PaperMeta, Chunk
from nodes.chunk_and_embed import chunk_and_embed, _chunk_text
from nodes.summarize import summarize
from nodes.qa_loop import answer_question, _expand_qa_question, _hybrid_rerank, _retrieve_multi_query, _detect_section_hint
import llm
import vectorstore
import embeddings


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


def test_section_hint_detection():
    """_detect_section_hint should map question keywords to paper sections."""
    assert "method" in _detect_section_hint("What method did they use?")
    assert "results" in _detect_section_hint("What were the evaluation results?")
    assert "results" in _detect_section_hint("How was the performance?")
    assert "conclusion" in _detect_section_hint("What is the conclusion?")
    assert len(_detect_section_hint("Tell me about this paper")) == 0


def test_section_boost_in_reranking(monkeypatch):
    """Chunks from a question-relevant section should get a metadata boost."""
    hits = [
        {"chunk_id": "intro_0", "section": "introduction", "text": "attention is a mechanism for weighting"},
        {"chunk_id": "method_0", "section": "method", "text": "attention is used as the core mechanism"},
        {"chunk_id": "results_0", "section": "results", "text": "attention heads show varied patterns"},
        {"chunk_id": "method_1", "section": "method", "text": "the approach uses multi-head attention"},
        {"chunk_id": "conclusion_0", "section": "conclusion", "text": "attention replaced recurrence entirely"},
    ]

    # Make all embeddings identical so cosine scores are equal — only BM25 + section boost differ.
    monkeypatch.setattr(
        embeddings, "embed",
        lambda texts: np.ones((len(texts), 4), dtype=np.float32),
    )

    # Ask a "method" question — method chunks should be boosted.
    result = _hybrid_rerank("What method did they use for attention?", hits, top_k=3)
    result_sections = [h["section"] for h in result]
    # At least one method chunk should appear (boosted by section hint).
    assert "method" in result_sections


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
    """QA should gracefully say 'couldn't find' when no chunks are retrieved."""
    state = AgentState(query="q", collection_name="paper_x")
    # Mock expansion to skip Groq call
    monkeypatch.setattr(
        "nodes.qa_loop._expand_qa_question",
        lambda q: [q],
    )
    monkeypatch.setattr(
        "vectorstore.query_chunks",
        lambda *a, **k: [],
    )
    state = answer_question(state, "What is the answer to life?")
    assert "couldn't find" in state.qa_history[-1][1].lower()


def test_qa_uses_retrieved_chunks(monkeypatch):
    """QA should use hybrid-reranked chunks and return proper chunk ids."""
    state = AgentState(query="q", collection_name="paper_x")

    # Mock expansion
    monkeypatch.setattr(
        "nodes.qa_loop._expand_qa_question",
        lambda q: [q],
    )

    fake_hits = [
        {
            "chunk_id": "method_0",
            "section": "method",
            "text": "We quantize KV caches to 4 bits.",
            "distance": 0.1,
        }
    ]
    monkeypatch.setattr(
        "vectorstore.query_chunks",
        lambda *a, **k: fake_hits,
    )
    # With only 1 hit, reranker won't filter anything (hits <= top_k).
    monkeypatch.setattr(
        embeddings,
        "embed",
        lambda texts: np.ones((len(texts), 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *a, **k: "They use 4-bit quantization [method_0].",
    )
    state = answer_question(state, "What precision is used?")
    q, a, ids = state.qa_history[-1]
    assert "4-bit" in a
    assert ids == ["method_0"]


def test_qa_expand_fallback(monkeypatch):
    """If LLM expansion fails, _expand_qa_question returns [original question]."""
    monkeypatch.setattr(
        llm,
        "chat_json",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API down")),
    )
    result = _expand_qa_question("What is attention?")
    assert result == ["What is attention?"]


def test_qa_hybrid_rerank(monkeypatch):
    """Hybrid reranking should keep top_k best chunks by BM25+cosine."""
    hits = [
        {"chunk_id": f"c{i}", "section": "s", "text": t}
        for i, t in enumerate([
            "transformer attention mechanism self-attention heads",
            "pizza recipe Italian food cooking",
            "multi-head attention scaled dot product query key value",
            "gardening tips for spring flowers",
            "attention weights softmax normalization",
        ])
    ]

    # Mock embeddings: items 0, 2, 4 are relevant, 1, 3 are not.
    def fake_embed(texts):
        n = len(texts)
        vecs = np.zeros((n, 4), dtype=np.float32)
        for i, t in enumerate(texts):
            if "attention" in t:
                vecs[i] = [0.9, 0.1, 0.0, 0.0]
            else:
                vecs[i] = [0.0, 0.0, 0.9, 0.1]
        return vecs

    monkeypatch.setattr(embeddings, "embed", fake_embed)

    result = _hybrid_rerank("attention mechanism", hits, top_k=3)
    # Should keep the 3 attention-related chunks and drop pizza + gardening.
    result_ids = {h["chunk_id"] for h in result}
    assert "c0" in result_ids
    assert "c2" in result_ids
    assert "c4" in result_ids
    assert "c1" not in result_ids
    assert "c3" not in result_ids


def test_qa_multi_query_dedup(monkeypatch):
    """Multi-query retrieval should deduplicate chunks by chunk_id."""
    call_log = []

    def fake_query(collection_name, question, top_k=4):
        call_log.append(question)
        return [
            {"chunk_id": "shared_0", "section": "s", "text": "shared chunk", "distance": 0.1},
            {"chunk_id": f"unique_{len(call_log)}", "section": "s", "text": f"unique {len(call_log)}", "distance": 0.2},
        ]

    monkeypatch.setattr(vectorstore, "query_chunks", fake_query)

    results = _retrieve_multi_query("col", ["q1", "q2", "q3"], top_k=4)
    assert len(call_log) == 3  # all 3 queries executed
    ids = [r["chunk_id"] for r in results]
    # shared_0 appears once despite 3 queries returning it
    assert ids.count("shared_0") == 1
    # 3 unique chunks + 1 shared = 4 total
    assert len(results) == 4


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

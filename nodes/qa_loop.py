"""
Grounded QA over the selected paper's Chroma collection.

Enhanced with:
- Hybrid BM25 + cosine reranking of retrieved chunks
- Multi-query RAG: expand user question into 2-3 variants for wider recall
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from state import AgentState
import embeddings
import llm
import vectorstore

QA_SYSTEM = """You answer questions about ONE academic paper using ONLY the provided context chunks.
Rules:
- If the answer is not supported by the context, say exactly: "I couldn't find that in the paper."
- Do not use outside knowledge.
- Keep answers concise (3-8 sentences unless asked for detail).
- Cite chunk ids inline like [chunk_id] when you use evidence.
"""

QA_EXPAND_SYSTEM = """You rephrase a question about an academic paper into 2-3 alternative phrasings.
Return ONLY a JSON object with key "queries" containing an array of 2-3 rephrased questions.
Each rephrasing should use different vocabulary but preserve the original meaning.
Do not add explanations outside the JSON."""

RERANK_ALPHA = 0.6       # semantic weight; (1-alpha) = BM25 weight
SECTION_BOOST = 0.15     # bonus for chunks from a question-relevant section

# Map question keywords to likely paper sections.
_SECTION_HINTS: dict[str, list[str]] = {
    "method":       ["method", "methods", "methodology", "approach", "model", "architecture"],
    "results":      ["results", "result", "experiments", "experiment", "evaluation", "performance"],
    "abstract":     ["abstract", "summary", "overview", "contribution"],
    "introduction": ["introduction", "background", "motivation", "related"],
    "conclusion":   ["conclusion", "conclusions", "discussion", "future"],
    "limitations":  ["limitation", "limitations", "weakness", "drawback"],
}


def _detect_section_hint(question: str) -> set[str]:
    """Detect which paper sections a question likely targets, using keyword matching."""
    q_lower = question.lower()
    matched: set[str] = set()
    for section, keywords in _SECTION_HINTS.items():
        if any(kw in q_lower for kw in keywords):
            matched.add(section)
    return matched


def _expand_qa_question(question: str) -> list[str]:
    """Use Groq to generate 2-3 alternative phrasings of the question.
    Falls back to [question] on any error."""
    try:
        result = llm.chat_json(
            QA_EXPAND_SYSTEM,
            f"Question: {question}",
            temperature=0.3,
        )
        queries = result.get("queries", [])
        if not queries or not isinstance(queries, list):
            return [question]
        return [question] + [q.strip() for q in queries if q.strip()][:2]
    except Exception:
        return [question]


def _retrieve_multi_query(
    collection_name: str, questions: list[str], top_k: int
) -> list[dict]:
    """Retrieve chunks for multiple query variants and deduplicate."""
    seen: dict[str, dict] = {}
    for q in questions:
        hits = vectorstore.query_chunks(collection_name, q, top_k=top_k)
        for h in hits:
            cid = h["chunk_id"]
            if cid not in seen:
                seen[cid] = h
    return list(seen.values())


def _hybrid_rerank(question: str, hits: list[dict], top_k: int) -> list[dict]:
    """Rerank retrieved chunks using BM25 + cosine hybrid scoring + section metadata boost."""
    if len(hits) <= top_k:
        return hits

    texts = [h["text"] for h in hits]

    # Cosine similarity
    q_vec = embeddings.embed([question])[0]
    doc_vecs = embeddings.embed(texts)
    cosine_scores = embeddings.cosine_sim_matrix(q_vec, doc_vecs)

    # BM25
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = np.array(bm25.get_scores(question.lower().split()), dtype=np.float32)

    # Normalize
    def norm(a: np.ndarray) -> np.ndarray:
        lo, hi = a.min(), a.max()
        return np.zeros_like(a) if (hi - lo) < 1e-9 else (a - lo) / (hi - lo)

    combined = RERANK_ALPHA * norm(cosine_scores) + (1 - RERANK_ALPHA) * norm(bm25_scores)

    # Section metadata boost: if the question implies a specific section,
    # give matching chunks a small score bump.
    hint_sections = _detect_section_hint(question)
    if hint_sections:
        for i, h in enumerate(hits):
            chunk_section = h.get("section", "").lower()
            if any(hint in chunk_section for hint in hint_sections):
                combined[i] += SECTION_BOOST

    order = np.argsort(-combined)[:top_k]
    return [hits[i] for i in order]


def answer_question(state: AgentState, question: str, *, top_k: int = 4) -> AgentState:
    if not state.collection_name:
        state.warnings.append("qa_loop: no vector collection — run chunk_and_embed first.")
        answer = "I couldn't find that in the paper."
        state.qa_history.append((question, answer, []))
        return state

    # Multi-query expansion: retrieve chunks for 2-3 question variants
    expanded = _expand_qa_question(question)
    raw_hits = _retrieve_multi_query(state.collection_name, expanded, top_k=top_k * 2)

    if not raw_hits:
        answer = "I couldn't find that in the paper."
        state.qa_history.append((question, answer, []))
        return state

    # Hybrid rerank and keep top-k
    hits = _hybrid_rerank(question, raw_hits, top_k=top_k)

    context_blocks = []
    chunk_ids = []

    # Inject paper metadata so the LLM can answer basic metadata questions
    # (author, date, title) that aren't in the PDF text chunks.
    paper = state.selected_paper
    if paper:
        meta_block = (
            f"[PAPER_METADATA]\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"arXiv ID: {paper.arxiv_id}\n"
            f"Published: {paper.published}\n"
            f"Categories: {', '.join(paper.categories)}\n"
            f"Abstract: {paper.abstract}"
        )
        context_blocks.append(meta_block)

    for h in hits:
        chunk_ids.append(h["chunk_id"])
        context_blocks.append(
            f"[{h['chunk_id']}] (section={h['section']})\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    user = (
        f"Question: {question}\n\n"
        f"Context chunks from the paper:\n{context}\n\n"
        "Answer using only the context above."
    )

    try:
        answer = llm.chat(QA_SYSTEM, user, temperature=0.0).strip()
    except Exception as e:
        answer = f"I couldn't find that in the paper. (LLM error: {e})"
        state.warnings.append(f"qa_loop LLM failed: {e}")

    state.qa_history.append((question, answer, chunk_ids))
    return state

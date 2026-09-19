"""
Hybrid BM25 + semantic ranking with two-stage filtering.

Stage 1: Score all candidates with weighted BM25 + cosine, keep top 10.
Stage 2: Re-rank those 10, pick the single best paper.

Alpha weight (0.6 semantic, 0.4 lexical) balances keyword-exact and
paraphrase matches. Both scores are min-max normalized before combining.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from state import AgentState
import embeddings

ALPHA = 0.6          # weight for semantic (cosine) score
STAGE1_KEEP = 10     # candidates to keep after first pass
STAGE2_KEEP = 1      # final pick (1 = single best paper)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if all values are identical."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _bm25_scores(query: str, documents: list[str]) -> np.ndarray:
    """Compute BM25 scores for query against a list of documents."""
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.lower().split())
    return np.array(scores, dtype=np.float32)


def _hybrid_rank(
    query: str,
    abstracts: list[str],
    alpha: float = ALPHA,
) -> np.ndarray:
    """Return combined hybrid scores (higher = better)."""
    # Semantic scores via embeddings
    query_vec = embeddings.embed([query])[0]
    doc_vecs = embeddings.embed(abstracts)
    cosine_scores = _normalize(embeddings.cosine_sim_matrix(query_vec, doc_vecs))

    # Lexical scores via BM25
    bm25_scores = _normalize(_bm25_scores(query, abstracts))

    return alpha * cosine_scores + (1.0 - alpha) * bm25_scores


def selection_ranking(state: AgentState) -> AgentState:
    """Two-stage hybrid ranking: 50 -> 10 -> 1."""
    if not state.candidates:
        state.warnings.append("selection_ranking called with no candidates.")
        return state

    # If only 1 candidate, skip ranking entirely.
    if len(state.candidates) == 1:
        state.selected_paper = state.candidates[0]
        return state

    abstracts = [c.abstract for c in state.candidates]

    # --- Stage 1: coarse rank, keep top STAGE1_KEEP ---
    stage1_scores = _hybrid_rank(state.query, abstracts)
    stage1_order = np.argsort(-stage1_scores)  # descending
    stage1_top = stage1_order[:STAGE1_KEEP]

    stage1_candidates = [state.candidates[i] for i in stage1_top]
    stage1_abstracts = [abstracts[i] for i in stage1_top]

    # Log stage 1 results
    s1_preview = "; ".join(
        f"{state.candidates[i].arxiv_id}={stage1_scores[i]:.3f}"
        for i in stage1_top[:5]
    )
    state.warnings.append(
        f"Stage 1: ranked {len(state.candidates)} candidates (hybrid BM25+cosine), "
        f"kept top {len(stage1_candidates)}. Scores: {s1_preview}"
    )

    # --- Stage 2: fine rank the shortlist ---
    if len(stage1_candidates) <= STAGE2_KEEP:
        state.selected_paper = stage1_candidates[0]
    else:
        stage2_scores = _hybrid_rank(state.query, stage1_abstracts)
        best_idx = int(np.argmax(stage2_scores))
        state.selected_paper = stage1_candidates[best_idx]

        s2_preview = "; ".join(
            f"{stage1_candidates[i].arxiv_id}={stage2_scores[i]:.3f}"
            for i in np.argsort(-stage2_scores)[:3]
        )
        state.warnings.append(
            f"Stage 2: re-ranked {len(stage1_candidates)} shortlisted papers, "
            f"picked '{state.selected_paper.title}' "
            f"(score={stage2_scores[best_idx]:.3f}). Top: {s2_preview}"
        )

    return state

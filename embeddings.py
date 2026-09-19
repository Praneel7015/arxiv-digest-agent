"""
Local embeddings via sentence-transformers - no API key, no cost, runs on CPU.
Used in selection_ranking, chunk_and_embed, and qa_loop - one model, one code path.

Model choice: all-MiniLM-L6-v2
- ~80MB, fast on CPU, good enough for abstract ranking + paper-chunk RAG
- No paid API; reviewers can run offline after first download from Hugging Face
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    # Lazy import so pytest collection doesn't require torch unless embed() runs.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_sim_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    # Normalized vectors => cosine similarity is a plain dot product.
    return doc_vecs @ query_vec

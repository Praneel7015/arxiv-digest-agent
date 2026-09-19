"""
Local Chroma vector store keyed by arXiv ID.

Why Chroma over FAISS for this assessment:
- Persistent collections on disk => QA can resume after briefing without re-embedding
- Built-in metadata filters (section, chunk_id) for grounded citations
- Still fully local / free — no cloud account
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

import embeddings
from state import Chunk

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent / "data" / "chroma"


def _client(persist_dir: Path | str = DEFAULT_PERSIST_DIR) -> chromadb.PersistentClient:
    path = Path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def collection_name_for(arxiv_id: str) -> str:
    # Chroma collection names: 3-63 chars, alnum + _ -
    safe = re_sub_id(arxiv_id)
    return f"paper_{safe}"[:63]


def re_sub_id(arxiv_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in arxiv_id)


def upsert_chunks(
    arxiv_id: str,
    chunks: list[Chunk],
    *,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
) -> str:
    """Embed + upsert chunks. Returns the collection name used."""
    name = collection_name_for(arxiv_id)
    client = _client(persist_dir)
    # Replace any previous run for this paper so re-runs stay idempotent.
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name=name, metadata={"arxiv_id": arxiv_id})

    if not chunks:
        return name

    texts = [c.text for c in chunks]
    vectors = embeddings.embed(texts)
    col.add(
        ids=[c.chunk_id for c in chunks],
        documents=texts,
        embeddings=vectors.tolist(),
        metadatas=[{"section": c.section, "arxiv_id": arxiv_id} for c in chunks],
    )
    return name


def query_chunks(
    collection_name: str,
    question: str,
    *,
    top_k: int = 4,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
) -> list[dict]:
    """Return top-k hits as {chunk_id, section, text, distance}."""
    client = _client(persist_dir)
    col = client.get_collection(collection_name)
    qvec = embeddings.embed([question])[0].tolist()
    result = col.query(query_embeddings=[qvec], n_results=min(top_k, max(col.count(), 1)))

    hits: list[dict] = []
    if not result["ids"] or not result["ids"][0]:
        return hits

    for i, chunk_id in enumerate(result["ids"][0]):
        hits.append(
            {
                "chunk_id": chunk_id,
                "section": (result["metadatas"][0][i] or {}).get("section", ""),
                "text": result["documents"][0][i] or "",
                "distance": (result["distances"][0][i] if result.get("distances") else None),
            }
        )
    return hits


def collection_exists(
    collection_name: str,
    *,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
) -> bool:
    client = _client(persist_dir)
    names = {c.name for c in client.list_collections()}
    return collection_name in names

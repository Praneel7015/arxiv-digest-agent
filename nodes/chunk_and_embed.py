"""
Chunk parsed sections into overlapping windows and upsert into Chroma.
"""
from __future__ import annotations

from state import AgentState, Chunk
import vectorstore

# ~500 tokens ≈ 2000 chars for English scientific prose; 200-char overlap keeps
# sentence boundaries from splitting key claims across chunks.
TARGET_CHARS = 2000
OVERLAP_CHARS = 200


def _chunk_text(section: str, text: str, start_idx: int) -> list[Chunk]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[Chunk] = []
    i = 0
    local = 0
    while i < len(text):
        end = min(i + TARGET_CHARS, len(text))
        piece = text[i:end]
        chunks.append(
            Chunk(
                chunk_id=f"{section}_{start_idx + local}",
                section=section,
                text=piece,
            )
        )
        local += 1
        if end >= len(text):
            break
        i = max(end - OVERLAP_CHARS, i + 1)
    return chunks


def chunk_and_embed(state: AgentState) -> AgentState:
    paper = state.selected_paper
    if paper is None:
        state.warnings.append("chunk_and_embed called with no selected_paper.")
        return state

    sections = state.parsed_sections or {}
    if not sections and paper.abstract:
        sections = {"abstract": paper.abstract}

    built: list[Chunk] = []
    for section, text in sections.items():
        built.extend(_chunk_text(section, text, start_idx=len(built)))

    state.chunks = built
    if not built:
        state.warnings.append("No text available to chunk/embed.")
        return state

    name = vectorstore.upsert_chunks(paper.arxiv_id, built)
    state.collection_name = name
    state.warnings.append(
        f"Stored {len(built)} chunks in Chroma collection '{name}'."
    )
    return state

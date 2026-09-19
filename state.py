"""
Shared state that flows through every node in the graph.

Using a dataclass instead of a plain dict gives us:
- IDE autocomplete + type checking on every node's input/output
- Free JSON serialization for session persistence (save/load), which
  directly answers "how does state persist between summarize and QA?"
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional
import json


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    published: str
    categories: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    section: str
    text: str


@dataclass
class AgentState:
    query: str
    intent: Optional[Literal["topic_search", "paper_lookup"]] = None

    candidates: list[PaperMeta] = field(default_factory=list)
    selected_paper: Optional[PaperMeta] = None

    parsed_sections: dict[str, str] = field(default_factory=dict)
    parse_status: Literal["ok", "partial", "failed", "pending"] = "pending"

    chunks: list[Chunk] = field(default_factory=list)
    collection_name: Optional[str] = None  # Chroma collection id for this paper

    briefing: Optional[dict] = None

    # (question, answer, chunk_ids_used_as_evidence)
    qa_history: list[tuple[str, str, list[str]]] = field(default_factory=list)

    # non-fatal issues collected along the way (surfaced to the user, not raised)
    warnings: list[str] = field(default_factory=list)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "AgentState":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        # Rebuild nested dataclasses (plain json.load only gives dicts/lists).
        raw["candidates"] = [PaperMeta(**c) for c in raw.get("candidates", [])]
        if raw.get("selected_paper"):
            raw["selected_paper"] = PaperMeta(**raw["selected_paper"])
        raw["chunks"] = [Chunk(**c) for c in raw.get("chunks", [])]
        raw["qa_history"] = [tuple(item) for item in raw.get("qa_history", [])]
        return cls(**raw)

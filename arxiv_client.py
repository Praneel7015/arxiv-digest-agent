"""
Thin wrapper around the official arXiv API (Atom feed) - no scraping.
Docs: https://info.arxiv.org/help/api/user-manual.html
"""
from __future__ import annotations

import re
import requests
import feedparser
from state import PaperMeta

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def normalize_id(raw: str) -> str:
    """Pull a clean arXiv id out of a raw string, URL, or bare id."""
    match = ARXIV_ID_PATTERN.search(raw)
    if not match:
        raise ValueError(f"Could not find an arXiv id in: {raw!r}")
    return match.group(0)


def search_by_topic(query: str, max_results: int = 5) -> list[PaperMeta]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API_BASE, params=params, timeout=15)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    return [_entry_to_meta(e) for e in feed.entries]


def fetch_by_id(raw_id: str) -> PaperMeta | None:
    clean_id = normalize_id(raw_id)
    resp = requests.get(ARXIV_API_BASE, params={"id_list": clean_id}, timeout=15)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    if not feed.entries:
        return None
    return _entry_to_meta(feed.entries[0])


def _entry_to_meta(entry) -> PaperMeta:
    pdf_url = next(
        (l.href for l in entry.links if getattr(l, "type", "") == "application/pdf"),
        None,
    )
    arxiv_id = entry.id.split("/abs/")[-1]
    return PaperMeta(
        arxiv_id=arxiv_id,
        title=" ".join(entry.title.split()),
        authors=[a.name for a in entry.authors],
        abstract=" ".join(entry.summary.split()),
        pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
        published=entry.published,
        categories=[t.term for t in entry.tags] if hasattr(entry, "tags") else [],
    )

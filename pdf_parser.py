"""
PDF fetch + parse. Handles the two failure modes called out in the brief:
- broken/scanned PDFs -> suspiciously little extracted text -> status='failed',
  caller falls back to abstract-only summarization instead of crashing.
- huge papers -> text is truncated with a warning so it doesn't blow the
  LLM's context window later.
"""
from __future__ import annotations

import re
import requests
import fitz  # PyMuPDF

MIN_TEXT_LENGTH = 500
MAX_TEXT_LENGTH = 40_000

SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)?(abstract|introduction|related work|background|method(?:ology)?|"
    r"approach|experiments?|results?|evaluation|discussion|limitations?|conclusion|"
    r"references|acknowledge?ments?)\s*$",
    re.IGNORECASE,
)


def download_pdf(pdf_url: str, timeout: int = 30) -> bytes:
    resp = requests.get(pdf_url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def extract_text(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def split_into_sections(full_text: str) -> dict[str, str]:
    current, buffer, sections = "preamble", [], {}
    for line in full_text.splitlines():
        match = SECTION_HEADING_PATTERN.match(line.strip())
        if match:
            sections[current] = "\n".join(buffer).strip()
            current, buffer = match.group(1).lower(), []
        else:
            buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return {k: v for k, v in sections.items() if v}


def fetch_and_parse_pdf(pdf_url: str) -> tuple[dict[str, str], str, list[str]]:
    """Never raises. Returns (sections, status, warnings) - status in {ok, partial, failed}."""
    try:
        pdf_bytes = download_pdf(pdf_url)
    except requests.RequestException as e:
        return {}, "failed", [f"Could not download PDF: {e}"]

    try:
        full_text = extract_text(pdf_bytes)
    except Exception as e:
        return {}, "failed", [
            f"Could not extract text from PDF (corrupt or unsupported): {e}"
        ]

    if len(full_text.strip()) < MIN_TEXT_LENGTH:
        return {}, "failed", [
            "Extracted text is suspiciously short - likely a scanned/image-based PDF "
            "or broken layout. Falling back to abstract-only summarization."
        ]

    warnings: list[str] = []
    if len(full_text) > MAX_TEXT_LENGTH:
        warnings.append(
            f"Paper is large ({len(full_text)} chars); truncated to {MAX_TEXT_LENGTH} "
            "chars to stay within LLM context limits."
        )
        full_text = full_text[:MAX_TEXT_LENGTH]

    sections = split_into_sections(full_text)
    status = "ok" if len(sections) > 1 else "partial"
    if status == "partial":
        warnings.append(
            "Could not detect clear section headings; treating the paper as one block."
        )
    return sections, status, warnings

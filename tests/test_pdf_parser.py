import requests
import pdf_parser
from state import AgentState, PaperMeta
from nodes.fetch_and_parse import fetch_and_parse


def _make_pdf_bytes(sections: dict[str, str]) -> bytes:
    """Build a tiny real PDF in-memory so we test actual PyMuPDF extraction.

    Uses insert_textbox (wraps within a rect) rather than insert_text, which
    draws along a single line and silently clips anything past the page edge.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for heading, body in sections.items():
        rect = fitz.Rect(72, y, 523, y + 150)
        page.insert_textbox(rect, f"{heading}\n{body}")
        y += 160
        if y > 720:
            page = doc.new_page()
            y = 72
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_split_into_sections_detects_headings():
    pdf_bytes = _make_pdf_bytes(
        {
            "Abstract": "We propose a new method for compressing KV caches.",
            "Introduction": "Large language models are expensive to serve.",
            "Method": "Our approach quantizes the cache to 4 bits.",
            "Limitations": "This does not work well for streaming inference.",
        }
    )
    text = pdf_parser.extract_text(pdf_bytes)
    sections = pdf_parser.split_into_sections(text)
    assert "abstract" in sections
    assert "kv caches" in sections["abstract"].lower()
    assert "limitations" in sections


def test_fetch_and_parse_pdf_success(monkeypatch):
    pdf_bytes = _make_pdf_bytes({"Abstract": "A" * 300, "Introduction": "B" * 300})
    monkeypatch.setattr(pdf_parser, "download_pdf", lambda url, timeout=30: pdf_bytes)
    sections, status, warnings = pdf_parser.fetch_and_parse_pdf("http://fake/pdf")
    assert status == "ok"
    assert "abstract" in sections


def test_fetch_and_parse_pdf_short_text_marked_failed(monkeypatch):
    pdf_bytes = _make_pdf_bytes({"Abstract": "too short"})
    monkeypatch.setattr(pdf_parser, "download_pdf", lambda url, timeout=30: pdf_bytes)
    sections, status, warnings = pdf_parser.fetch_and_parse_pdf("http://fake/pdf")
    assert status == "failed"
    assert warnings


def test_fetch_and_parse_pdf_download_error(monkeypatch):
    def _boom(url, timeout=30):
        raise requests.RequestException("network down")

    monkeypatch.setattr(pdf_parser, "download_pdf", _boom)
    sections, status, warnings = pdf_parser.fetch_and_parse_pdf("http://fake/pdf")
    assert status == "failed"
    assert "download" in warnings[0].lower()


def test_fetch_and_parse_node_falls_back_to_abstract(monkeypatch):
    paper = PaperMeta(
        arxiv_id="2401.99999",
        title="T",
        authors=["A"],
        abstract="Fallback abstract text that is long enough.",
        pdf_url="http://fake/pdf",
        published="2024-01-01",
    )

    def fake_parse(url):
        return {}, "failed", ["scanned pdf"]

    monkeypatch.setattr("nodes.fetch_and_parse.fetch_and_parse_pdf", fake_parse)
    state = fetch_and_parse(AgentState(query="x", selected_paper=paper))
    assert state.parse_status == "failed"
    assert state.parsed_sections["abstract"] == paper.abstract

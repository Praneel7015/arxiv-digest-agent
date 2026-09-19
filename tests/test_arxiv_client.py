import feedparser
from arxiv_client import _entry_to_meta

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <published>2024-01-22T00:00:00Z</published>
    <title>A Sample Paper on   KV Cache Compression</title>
    <summary>  This paper studies compression of KV caches in transformer models.  </summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2" rel="related" type="application/pdf"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


def test_entry_to_meta_parses_correctly():
    feed = feedparser.parse(SAMPLE_FEED)
    meta = _entry_to_meta(feed.entries[0])
    assert meta.arxiv_id == "2401.12345v2"
    assert meta.title == "A Sample Paper on KV Cache Compression"
    assert meta.authors == ["Jane Doe", "John Smith"]
    assert meta.pdf_url == "http://arxiv.org/pdf/2401.12345v2"
    assert "KV caches" in meta.abstract
    assert meta.categories == ["cs.LG"]

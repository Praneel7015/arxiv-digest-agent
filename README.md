# arXiv Paper Digest & QA Agent

Autonomous agent that takes a **research topic** or **arXiv ID/URL**, retrieves the paper via the official arXiv API, parses the PDF, builds a local vector index, produces a **structured executive briefing**, and supports **grounded follow-up QA** (RAG).

Built for the 8byte AI Intern assessment as an explicit **stateful graph** (nodes + edges + shared state)—not a single monolithic prompt.

> **New to this codebase?** Start here → [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)  
> Beginner-friendly walkthrough with Mermaid diagrams, node-by-node explanations, and interview talking points.

## Architecture

```
                    ┌─────────────────────┐
 query ───────────► │ query_understanding │
                    └──────────┬──────────┘
                               │ intent: topic_search | paper_lookup
                    ┌──────────▼──────────┐
                    │   arxiv_retrieval   │ ◄── official Atom API
                    └──────────┬──────────┘
               ┌───────────────┼───────────────┐
               │ 0 results     │ 1 result      │ many results
               ▼               ▼               ▼
         error_handler   fetch_and_parse  selection_ranking
               │               ▲               │
               │               └───────────────┘
               │               │
               │      ┌────────▼────────┐
               │      │ chunk_and_embed │ ◄── sentence-transformers + Chroma
               │      └────────┬────────┘
               │      ┌────────▼────────┐
               │      │    summarize    │ ◄── Groq (structured JSON briefing)
               │      └────────┬────────┘
               │      ┌────────▼────────┐
               └─────►│     qa_loop     │ ◄── retrieve top-k chunks → grounded answer
                      └─────────────────┘
```

### Shared state (`AgentState`)

| Field | Role |
|---|---|
| `query`, `intent` | Raw input + routing decision |
| `candidates`, `selected_paper` | arXiv metadata |
| `parsed_sections`, `parse_status` | PDF extract (`ok` / `partial` / `failed`) |
| `chunks`, `collection_name` | Chunk list + Chroma collection for this paper |
| `briefing` | Structured executive briefing (dict/JSON) |
| `qa_history` | `(question, answer, evidence_chunk_ids)` |
| `warnings` | Non-fatal issues (never silent failures) |

State is a dataclass with `save()` / `load()` so a session can pause after the briefing and resume QA later without re-parsing.

### Failure cases (§5)

| Case | Behavior |
|---|---|
| Zero arXiv hits | Route to `error_handler`, ask to rephrase (warning) |
| Many topic hits | Embed query + abstracts, cosine-rank, pick top-1 (scores logged) |
| PDF parse failure | `parse_status=failed`, fall back to **abstract-only** briefing |
| Answer not in paper | QA replies: `I couldn't find that in the paper.` |

## Stack (locked choices)

| Concern | Choice | Why |
|---|---|---|
| Orchestration | Custom Python state machine | Rubric wants justified graph design; no framework lock-in |
| LLM | Groq `openai/gpt-oss-20b` (default) | Free tier; swap to `openai/gpt-oss-120b` or `qwen/qwen3.8-27b` in `.env` if available |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` | Local, free, ~80MB, good enough for ranking + RAG |
| Vector DB | Chroma (persistent, local) | Resume QA without re-embedding; metadata for citations |
| PDF | PyMuPDF | Fast, reliable text extract |
| arXiv | Official Atom API | No scraping |
| Python | 3.10–3.12 (developed on 3.12) | |

No paid API keys are required. You need a free Groq key.

## Setup

```bash
# Python 3.12 recommended
py -3.12 -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # then paste GROQ_API_KEY from https://console.groq.com
```

First run downloads the MiniLM model from Hugging Face (~80MB).

### Rate limits to expect when reviewing

Groq free tier is generous but not unlimited. If you hit 429s, wait a minute and retry. If a model 404s, list models with the Groq SDK and set `GROQ_MODEL` in `.env` to one you have access to (e.g. `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`).

## Run

```bash
# Topic search → briefing → interactive QA
python main.py "KV-cache compression for LLMs"

# Specific paper
python main.py 1706.03762
python main.py https://arxiv.org/abs/1706.03762

# Briefing only
python main.py --no-qa 1706.03762

# Resume QA from a saved session
python main.py --session sessions/1706.03762.json
python main.py --session sessions/1706.03762.json --question "What are the limitations?"
```

Sessions are written under `sessions/` as JSON. Chroma data lives under `data/chroma/`.

## Tests

```bash
pytest -q
```

Network / Hugging Face / Groq are **mocked** in unit tests so CI-like environments pass without live keys. Live end-to-end is verified locally with `python main.py ...`.

## Example run

### Briefing (paper lookup by arXiv ID)

```text
$ python main.py --no-qa 1706.03762
Running pipeline for: '1706.03762'

# Attention Is All You Need

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
**arXiv:** 1706.03762v7  |  **Date:** 2017-06-12T17:57:34Z
**Link:** https://arxiv.org/abs/1706.03762v7

## Why it matters
The paper introduces the Transformer, a novel neural architecture that replaces recurrent and
convolutional layers with self-attention, enabling far greater parallelism during training and
inference. This breakthrough yields state-of-the-art results on benchmark machine-translation
tasks while dramatically reducing training time and computational cost.

## Problem
Existing sequence transduction models rely on recurrent or convolutional networks, whose
inherently sequential computation limits parallelism and increases training time.

## Method
- Propose the Transformer architecture composed solely of multi-head self-attention and
  position-wise feed-forward layers.
- Replace recurrence with self-attention in both encoder and decoder, adding residual
  connections and layer normalization.
- Introduce positional encodings to inject sequence order information.
- Train on WMT 2014 English-to-German and English-to-French translation tasks.
- Conduct ablation studies varying depth, width, dropout, and attention heads.

## Key results
- 28.4 BLEU on WMT 2014 English-to-German, surpassing prior best by over 2 BLEU.
- 41.8 BLEU on WMT 2014 English-to-French as a single model.
- Trains in 3.5 days on eight P100 GPUs (< 1/4 the cost of previous SOTA).

## Limitations
- Evaluation limited to two translation benchmarks and one parsing task.
- No exploration of very long input sequences or memory-efficient local attention.
- Training still requires substantial GPU resources (8 P100 GPUs).
- No theoretical convergence analysis for a purely attention-based architecture.
- No ablation on positional encoding alternatives beyond sinusoidal.

## Suggested follow-ups
- How does the Transformer perform on summarization or speech recognition?
- What are the trade-offs for very long-range dependencies?
- Would learned or relative positional encodings improve results?

Session saved to sessions/1706.03762v7.json
```

### QA: grounded answer

```text
$ python main.py --session sessions/1706.03762v7.json --question "What is multi-head attention?"

Q: What is multi-head attention?
A: Multi-head attention runs several scaled dot-product attention layers in parallel. Each head
projects queries, keys and values into a lower-dimensional space (dk = dv = dmodel/h), computes
attention independently, then all head outputs are concatenated and linearly transformed back to
the model dimension. This lets the model attend to different representation subspaces at once
while keeping computational cost similar to single-head attention. [background_6] [background_8]
```

### QA: refusing out-of-paper question

```text
$ python main.py --session sessions/1706.03762v7.json --question "What is the capital of France?"

Q: What is the capital of France?
A: I couldn't find that in the paper.
```

## Design decisions & tradeoffs

**Custom graph vs LangGraph.** A plain Python state machine keeps every edge visible in ~50 lines of `graph.py`. For an assessment that grades reasoning, that clarity beats framework fluency. With more time I would add typed edge enums and optional LangGraph export for visualization only.

**Top-1 auto-select for topics.** Showing a picker UI is nicer UX but out of scope (CLI only). Ranking scores are logged in `warnings` so a reviewer can see *why* a paper was chosen. Next step: `ask> pick 2` interactive selection.

**Chroma over FAISS.** FAISS is lighter, but Chroma gives durable collections + metadata with almost no code. A single-paper session does not need Postgres; the dataclass `save()` / `load()` is enough for process restarts, while Chroma holds the vectors.

**Abstract fallback on parse failure.** Scanned PDFs are common. Dying on extract would fail the rubric’s failure-case requirement. Abstract-only briefing is honest (and labeled in limitations) rather than hallucinating full-paper claims.

**Grounding.** QA retrieves top-k chunks, prompts “answer only from context”, and records evidence chunk ids. This is basic RAG—not citation-faithful decoding—but it reliably refuses out-of-paper questions in practice.

**Known limitations / next with more time**

- Heading heuristics miss weird layouts; layout-aware parsers (or arXiv HTML) would help.
- No OCR path for scanned PDFs.
- Single-paper focus (by design); multi-paper compare would be a new graph branch.
- No eval set measuring faithfulness; would add a small golden Q/A suite.

## Video reflection outline (≤ 4 min)

1. Problem + graph sketch (45s)
2. Failure cases & state persistence (60s)
3. Live demo: topic → briefing → 2 QA turns including a “not in paper” (90s)
4. Tradeoffs & what I’d do next (45s)

## Project layout

```
arxiv-digest-agent/
  main.py              # CLI
  graph.py             # nodes + edges
  state.py             # AgentState
  arxiv_client.py
  embeddings.py
  pdf_parser.py
  llm.py
  vectorstore.py
  nodes/               # one file per stage
  tests/
  sessions/            # saved runs
  data/chroma/         # local vector DB
```

## License

Assessment submission — all rights reserved by author unless otherwise noted.

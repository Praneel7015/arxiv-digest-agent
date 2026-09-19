"""
CLI entrypoint for the arXiv Paper Digest & QA Agent.

Examples:
  python main.py "KV-cache compression for LLMs"
  python main.py 2401.12345
  python main.py --session sessions/last.json --question "What are the limitations?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph import run_pipeline, run_qa
from nodes.summarize import format_briefing_markdown
from state import AgentState


def _configure_stdout() -> None:
    """Avoid UnicodeEncodeError on Windows cp1252 consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"


def _default_session_path(state: AgentState) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    paper = state.selected_paper
    name = paper.arxiv_id.replace("/", "_") if paper else "session"
    return SESSIONS_DIR / f"{name}.json"


def _print_briefing(state: AgentState, *, debug: bool = False) -> None:
    if not state.briefing:
        print("No briefing produced.")
        return
    print("\n" + format_briefing_markdown(state.briefing))
    if debug:
        print("\n--- raw JSON ---")
        print(json.dumps(state.briefing, indent=2))


def _interactive_qa(state: AgentState, session_path: Path) -> AgentState:
    print("\nQA mode — ask about the paper. Type 'exit' / 'quit' to stop.\n")
    while True:
        try:
            q = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        state = run_qa(state, q)
        answer, chunk_ids = state.qa_history[-1][1], state.qa_history[-1][2]
        print(f"\n{answer}")
        if chunk_ids:
            print(f"(evidence chunks: {', '.join(chunk_ids)})\n")
        state.save(str(session_path))
    return state


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="arXiv Paper Digest & QA Agent")
    parser.add_argument("query", nargs="*", help="Topic or arXiv ID/URL")
    parser.add_argument("--session", help="Load a saved session JSON and skip retrieval")
    parser.add_argument("--question", help="One-shot QA question (non-interactive)")
    parser.add_argument("--no-qa", action="store_true", help="Stop after briefing")
    parser.add_argument("--save", help="Explicit path to save session JSON")
    parser.add_argument("--debug", action="store_true", help="Print raw JSON briefing and all warnings")
    args = parser.parse_args(argv)

    if args.session:
        state = AgentState.load(args.session)
        session_path = Path(args.session)
        print(f"Loaded session from {session_path}")
        if state.briefing:
            _print_briefing(state, debug=args.debug)
    else:
        query = " ".join(args.query).strip()
        if not query:
            parser.error("Provide a topic/arXiv id, or --session path")
        print(f"Running pipeline for: {query!r}")
        state = run_pipeline(query)
        if not state.selected_paper:
            print("Pipeline stopped early (no paper selected).")
            for w in state.warnings:
                print(f"  [warning] {w}")
            return 1
        _print_briefing(state, debug=args.debug)
        session_path = Path(args.save) if args.save else _default_session_path(state)
        state.save(str(session_path))
        print(f"\nSession saved to {session_path}")

    # Only show warnings that are user-relevant (skip internal ranking scores unless debug)
    user_warnings = [
        w for w in state.warnings
        if not any(skip in w for skip in ("Stage 1:", "Stage 2:", "Fetched ", "Stored "))
    ] if not args.debug else state.warnings

    if user_warnings:
        print("\nWarnings:")
        for w in user_warnings:
            print(f"  ⚠  {w}")

    if args.question:
        state = run_qa(state, args.question)
        print(f"\nQ: {args.question}\nA: {state.qa_history[-1][1]}")
        state.save(str(session_path if args.session else _default_session_path(state)))
        return 0

    if not args.no_qa and state.collection_name:
        session_path = Path(args.session) if args.session else (
            Path(args.save) if args.save else _default_session_path(state)
        )
        _interactive_qa(state, session_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

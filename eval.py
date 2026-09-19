#!/usr/bin/env python
"""
eval.py - Lightweight synthetic evaluation of the retrieval + QA pipeline.

NOT part of the main pipeline. This is a standalone script that:
1. Runs the full pipeline on 2–3 known papers.
2. Generates synthetic Q/A pairs from the briefing using Groq.
3. Runs each question through qa_loop.
4. Measures:
   (a) chunk retrieval hit rate (did relevant chunks surface?)
   (b) answer quality (does the answer mention expected key terms?)
5. Prints a summary report.

Usage:
    python eval.py                   # run on default papers
    python eval.py 1706.03762        # run on a specific paper

Requires a valid GROQ_API_KEY in .env.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

from graph import run_pipeline, run_qa
import llm


# Well-known papers that are always on arXiv and have clean PDFs.
DEFAULT_PAPERS = [
    "1706.03762",   # Attention Is All You Need
    "2005.14165",   # Language Models are Few-Shot Learners (GPT-3)
]

SYNTH_QA_SYSTEM = """You generate evaluation questions for an academic paper.
Given a structured briefing of a paper, create exactly 5 question/answer pairs.
Return ONLY a JSON object with key "pairs", where each pair has:
- "question": a factual question answerable from the paper
- "answer": the gold answer (1-2 sentences)
- "key_terms": list of 3-5 important terms the answer MUST contain

Rules:
- Questions should vary: cover methods, results, limitations, comparisons.
- key_terms are lowercase keywords used to check if the model's answer is grounded.
- Do not ask subjective or opinion questions.
"""


@dataclass
class EvalResult:
    paper_id: str
    title: str
    total_questions: int = 0
    retrieval_hits: int = 0        # question where >=1 relevant chunk surfaced
    answer_term_matches: int = 0   # question where answer contains >=50% key terms
    details: list[dict] = field(default_factory=list)


def generate_synthetic_qa(briefing: dict) -> list[dict]:
    """Use Groq to generate 5 Q/A pairs from the briefing."""
    briefing_text = json.dumps(briefing, indent=2, default=str)
    prompt = f"Paper briefing:\n{briefing_text}\n\nGenerate 5 evaluation Q/A pairs."
    try:
        result = llm.chat_json(SYNTH_QA_SYSTEM, prompt, temperature=0.3)
        pairs = result.get("pairs", [])
        if not isinstance(pairs, list) or len(pairs) == 0:
            return []
        return pairs
    except Exception as e:
        print(f"  [!] Synthetic QA generation failed: {e}")
        return []


def check_answer_quality(answer: str, key_terms: list[str]) -> float:
    """Return fraction of key_terms found in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    if not key_terms:
        return 1.0
    hits = sum(1 for t in key_terms if t.lower() in answer_lower)
    return hits / len(key_terms)


def evaluate_paper(paper_id: str) -> EvalResult:
    """Run full pipeline on one paper, generate synthetic QA, measure quality."""
    print(f"\n{'='*60}")
    print(f"Evaluating paper: {paper_id}")
    print(f"{'='*60}")

    # Run pipeline
    print("  [1/3] Running pipeline...")
    t0 = time.time()
    state = run_pipeline(paper_id)
    t_pipeline = time.time() - t0
    print(f"  Pipeline completed in {t_pipeline:.1f}s")

    if not state.selected_paper:
        print(f"  [!] No paper selected for {paper_id}, skipping.")
        return EvalResult(paper_id=paper_id, title="(not found)")

    result = EvalResult(
        paper_id=paper_id,
        title=state.selected_paper.title,
    )

    if not state.briefing:
        print("  [!] No briefing generated, skipping QA eval.")
        return result

    # Generate synthetic Q/A
    print("  [2/3] Generating synthetic Q/A pairs...")
    pairs = generate_synthetic_qa(state.briefing)
    if not pairs:
        print("  [!] No synthetic Q/A pairs generated, skipping.")
        return result

    print(f"  Generated {len(pairs)} Q/A pairs")

    # Run QA and measure
    print("  [3/3] Running QA evaluation...")
    result.total_questions = len(pairs)

    for i, pair in enumerate(pairs):
        question = pair.get("question", "")
        gold_answer = pair.get("answer", "")
        key_terms = pair.get("key_terms", [])

        if not question:
            continue

        state = run_qa(state, question)
        _, actual_answer, chunk_ids = state.qa_history[-1]

        # Measure retrieval: did we get chunks?
        got_chunks = len(chunk_ids) > 0
        if got_chunks:
            result.retrieval_hits += 1

        # Measure answer quality: key term overlap
        term_match = check_answer_quality(actual_answer, key_terms)
        if term_match >= 0.5:
            result.answer_term_matches += 1

        detail = {
            "question": question,
            "gold_answer": gold_answer,
            "actual_answer": actual_answer[:200],
            "key_terms": key_terms,
            "term_match_ratio": round(term_match, 2),
            "chunks_retrieved": len(chunk_ids),
            "retrieval_hit": got_chunks,
        }
        result.details.append(detail)

        status = "PASS" if (got_chunks and term_match >= 0.5) else "FAIL"
        print(f"    Q{i+1}: [{status}] chunks={len(chunk_ids)}, "
              f"terms={term_match:.0%} - {question[:60]}...")

    return result


def print_report(results: list[EvalResult]) -> None:
    """Print a summary report of all evaluated papers."""
    print(f"\n{'='*60}")
    print("EVALUATION REPORT")
    print(f"{'='*60}")

    total_q = sum(r.total_questions for r in results)
    total_retrieval = sum(r.retrieval_hits for r in results)
    total_term = sum(r.answer_term_matches for r in results)

    for r in results:
        if r.total_questions == 0:
            print(f"\n  {r.paper_id} ({r.title}): NO DATA")
            continue
        print(f"\n  {r.paper_id} ({r.title}):")
        print(f"    Retrieval hit rate:  {r.retrieval_hits}/{r.total_questions} "
              f"({r.retrieval_hits/r.total_questions:.0%})")
        print(f"    Answer quality rate: {r.answer_term_matches}/{r.total_questions} "
              f"({r.answer_term_matches/r.total_questions:.0%})")

    if total_q > 0:
        print(f"\n  OVERALL ({total_q} questions across {len(results)} papers):")
        print(f"    Retrieval hit rate:  {total_retrieval}/{total_q} "
              f"({total_retrieval/total_q:.0%})")
        print(f"    Answer quality rate: {total_term}/{total_q} "
              f"({total_term/total_q:.0%})")
    print()


def main():
    papers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PAPERS
    print(f"Evaluating {len(papers)} paper(s): {', '.join(papers)}")

    results = []
    for paper_id in papers:
        try:
            result = evaluate_paper(paper_id)
            results.append(result)
        except Exception as e:
            print(f"  [!] Evaluation failed for {paper_id}: {e}")
            results.append(EvalResult(paper_id=paper_id, title="(error)"))

    print_report(results)

    # Save detailed results to JSON
    out = []
    for r in results:
        out.append({
            "paper_id": r.paper_id,
            "title": r.title,
            "total_questions": r.total_questions,
            "retrieval_hits": r.retrieval_hits,
            "answer_term_matches": r.answer_term_matches,
            "details": r.details,
        })
    with open("eval_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("Detailed results saved to eval_results.json")


if __name__ == "__main__":
    main()

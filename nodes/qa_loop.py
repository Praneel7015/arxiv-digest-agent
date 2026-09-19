"""
Grounded QA over the selected paper's Chroma collection.
"""
from __future__ import annotations

from state import AgentState
import llm
import vectorstore

QA_SYSTEM = """You answer questions about ONE academic paper using ONLY the provided context chunks.
Rules:
- If the answer is not supported by the context, say exactly: "I couldn't find that in the paper."
- Do not use outside knowledge.
- Keep answers concise (3-8 sentences unless asked for detail).
- Cite chunk ids inline like [chunk_id] when you use evidence.
"""


def answer_question(state: AgentState, question: str, *, top_k: int = 4) -> AgentState:
    if not state.collection_name:
        state.warnings.append("qa_loop: no vector collection — run chunk_and_embed first.")
        answer = "I couldn't find that in the paper."
        state.qa_history.append((question, answer, []))
        return state

    hits = vectorstore.query_chunks(state.collection_name, question, top_k=top_k)
    if not hits:
        answer = "I couldn't find that in the paper."
        state.qa_history.append((question, answer, []))
        return state

    context_blocks = []
    chunk_ids = []
    for h in hits:
        chunk_ids.append(h["chunk_id"])
        context_blocks.append(
            f"[{h['chunk_id']}] (section={h['section']})\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    user = (
        f"Question: {question}\n\n"
        f"Context chunks from the paper:\n{context}\n\n"
        "Answer using only the context above."
    )

    try:
        answer = llm.chat(QA_SYSTEM, user, temperature=0.0).strip()
    except Exception as e:
        answer = f"I couldn't find that in the paper. (LLM error: {e})"
        state.warnings.append(f"qa_loop LLM failed: {e}")

    state.qa_history.append((question, answer, chunk_ids))
    return state

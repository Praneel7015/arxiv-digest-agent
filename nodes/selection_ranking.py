from state import AgentState
import embeddings


def selection_ranking(state: AgentState) -> AgentState:
    if not state.candidates:
        state.warnings.append("selection_ranking called with no candidates.")
        return state

    query_vec = embeddings.embed([state.query])[0]
    abstract_vecs = embeddings.embed([c.abstract for c in state.candidates])
    scores = embeddings.cosine_sim_matrix(query_vec, abstract_vecs)

    best_idx = int(scores.argmax())
    state.selected_paper = state.candidates[best_idx]
    ranked = sorted(
        zip(state.candidates, scores.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    preview = "; ".join(f"{p.arxiv_id}={s:.3f}" for p, s in ranked[:3])
    state.warnings.append(
        f"Ranked {len(state.candidates)} candidates by abstract similarity; "
        f"picked '{state.selected_paper.title}' (score={scores[best_idx]:.3f}). "
        f"Top scores: {preview}"
    )
    return state

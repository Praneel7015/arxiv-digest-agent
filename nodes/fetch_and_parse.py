from state import AgentState
from pdf_parser import fetch_and_parse_pdf


def fetch_and_parse(state: AgentState) -> AgentState:
    paper = state.selected_paper
    if paper is None:
        state.warnings.append("fetch_and_parse called with no selected_paper.")
        state.parse_status = "failed"
        return state

    sections, status, warnings = fetch_and_parse_pdf(paper.pdf_url)
    state.parsed_sections = sections
    state.parse_status = status
    state.warnings.extend(warnings)

    if status == "failed":
        # Graceful degradation (required failure case): summarize node still has
        # the abstract to work with, so the pipeline doesn't dead-end.
        state.parsed_sections = {"abstract": paper.abstract}

    return state

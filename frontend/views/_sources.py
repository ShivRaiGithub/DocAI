import streamlit as st


def score_bar(score: float) -> None:
    # similarity runs from -1 to 1; negative scores are noise, so show them as empty
    st.progress(min(max(score, 0.0), 1.0), text=f"relevance {score:.2f}")


def render_sources(sources: list[dict], cited: set[int]) -> None:
    """Show retrieved chunks; those the answer cited are marked."""
    for source in sources:
        mark = "  ·  **cited**" if source["number"] in cited else ""
        st.markdown(
            f"**[{source['number']}]** {source['filename']}  ·  page {source['page']}  ·  "
            f"score {source['score']:.2f}{mark}"
        )
        st.text(source["text"])  # plain text: a chunk is source material, not Markdown to render

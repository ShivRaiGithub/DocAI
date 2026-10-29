import streamlit as st

from frontend.docai_client import ApiError
from frontend.state import get_client
from frontend.views._sources import score_bar


def render() -> None:
    client = get_client()
    st.header("Search")
    st.caption("Finds passages by meaning, not by exact words. No language model is involved here.")

    with st.form("search"):
        query = st.text_input("What are you looking for?")
        left, right = st.columns(2)
        limit = left.slider("Results", 1, 20, 5)
        min_score = right.slider("Hide results scoring below", 0.0, 0.8, 0.0, 0.05)
        submitted = st.form_submit_button("Search", icon=":material/search:")

    if not submitted:
        return
    if not query.strip():
        st.warning("Type something to search for.")
        return
    try:
        with st.spinner("Searching..."):
            results = client.search(query, limit, min_score or None)
    except ApiError as e:
        st.error(str(e))
        return
    if not results:
        st.info("Nothing found. Is anything indexed yet? Try lowering the score filter.")
        return

    st.caption(f"{len(results)} result(s)")
    for hit in results:
        with st.container(border=True):
            st.markdown(f"**{hit['filename']}**  ·  page {hit['page']}  ·  version {hit['version']}")
            score_bar(hit["score"])
            st.text(hit["text"])  # plain text: show the passage exactly as stored

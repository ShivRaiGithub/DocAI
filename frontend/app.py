"""DocAI web interface. Start it with:  uv run streamlit run frontend/app.py

Streamlit re-runs this whole file from top to bottom on every click or keystroke.
That is its core idea: you write a plain script, and the UI is whatever the script
draws on that run. Values that must survive between runs live in `st.session_state`.
"""

import sys
from pathlib import Path

# Streamlit puts this folder on the import path, not the project root; add the root
# so `from frontend ...` works the same here and in the tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from frontend.docai_client import DEFAULT_URL, ApiError  # noqa: E402
from frontend.state import get_client  # noqa: E402
from frontend.views import ask, documents, search  # noqa: E402

st.set_page_config(page_title="DocAI", page_icon=":material/description:", layout="wide")

with st.sidebar:
    st.title("DocAI")
    st.text_input("API address", value=DEFAULT_URL, key="api_url")
    try:
        health = get_client().health()
        st.success(f"Connected to {health['app_name']} {health['version']}", icon=":material/check:")
    except ApiError as e:
        st.error(str(e), icon=":material/error:")

navigation = st.navigation(
    [
        st.Page(ask.render, title="Ask", icon=":material/chat:", url_path="ask", default=True),
        st.Page(search.render, title="Search", icon=":material/search:", url_path="search"),
        st.Page(documents.render, title="Documents", icon=":material/folder:", url_path="documents"),
    ]
)
navigation.run()

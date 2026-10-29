import streamlit as st

from frontend.docai_client import ApiError
from frontend.state import get_client
from frontend.views._sources import render_sources

GREETING = "Ask a question about your documents. Answers come with the passages they are based on."


def render() -> None:
    client = get_client()
    st.header("Ask your documents")

    # Chat history lives in session_state, otherwise it would vanish on every rerun.
    # It is for display only: each question is answered on its own, with no memory of the
    # earlier ones.
    messages = st.session_state.setdefault("messages", [])

    docs = _documents(client)
    options = settings_panel(docs)

    # The chat box is pinned to the bottom of the page wherever it is created, so it is
    # read first: that way the new question is already in `messages` when the history is drawn.
    question = st.chat_input("Ask something...")
    if question:
        messages.append({"role": "user", "content": question})

    if not messages:
        st.info(GREETING)
    for message in messages:
        show_message(message)

    if question:
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents and asking the model..."):
                    response = client.ask(question, **options)
            except ApiError as e:
                st.error(str(e))
                messages.pop()  # don't keep a question that never got an answer
                return
            message = {
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"],
                "cited": sorted({c["number"] for c in response["citations"]}),
                "steps": response.get("steps"),
                "model": response["model"],
            }
            messages.append(message)
            show_message_body(message)

    if st.button("Clear conversation", icon=":material/delete_sweep:", disabled=not messages):
        messages.clear()
        st.rerun()


def _documents(client) -> list[dict]:
    try:
        return client.list_documents()
    except ApiError as e:
        st.error(str(e))
        return []


def settings_panel(docs: list[dict]) -> dict:
    """Controls for how the question is answered. Returns keyword arguments for `ask`."""
    with st.expander("Settings"):
        agent = st.toggle(
            "Rewrite the query and retry if nothing relevant is found",
            value=True,
            help="Uses /ask/agent. Slower when the first search misses, but finds answers "
            "phrased differently from the documents.",
        )
        left, right = st.columns(2)
        limit = left.slider("Passages given to the model", 1, 10, 5)
        min_score = right.slider(
            "Minimum relevance", 0.0, 0.8, 0.25, 0.05,
            help="Passages scoring below this are ignored. Higher means fewer wrong answers "
            "but more 'I couldn't find anything'.",
        )
        names = {d["id"]: d["filename"] for d in docs}
        chosen = st.multiselect(
            "Only use these documents (empty means all)", list(names), format_func=names.get
        )
    return {"agent": agent, "limit": limit, "min_score": min_score, "document_ids": chosen or None}


def show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        show_message_body(message)


def show_message_body(message: dict) -> None:
    """The contents of one chat bubble. The caller provides the bubble (`st.chat_message`)."""
    st.markdown(message["content"])
    if message["role"] != "assistant":
        return
    sources = message.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            render_sources(sources, set(message.get("cited") or []))
    if message.get("steps"):
        with st.expander("What the agent did"):
            for step in message["steps"]:
                st.markdown(f"- {step}")
    if message.get("model"):
        st.caption(f"answered by {message['model']}")


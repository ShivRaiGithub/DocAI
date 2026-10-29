import streamlit as st

from frontend.docai_client import ApiError, DocAIClient
from frontend.state import get_client, human_size, short_date

ALLOWED_TYPES = ["pdf", "txt", "md"]


def render() -> None:
    client = get_client()
    st.header("Documents")
    upload_form(client)

    try:
        docs = client.list_documents()
    except ApiError as e:
        st.error(str(e))
        return
    if not docs:
        st.info("No documents yet. Upload a PDF, text or Markdown file above.")
        return

    st.subheader(f"{len(docs)} document{'s' if len(docs) != 1 else ''}")
    for doc in docs:
        document_card(client, doc)


def upload_form(client: DocAIClient) -> None:
    # A form submits once, on the button press. Without it, Streamlit would re-run the
    # script on every file selection, and a file still sitting in the uploader would be
    # uploaded again on every later rerun. `clear_on_submit` empties the uploader after.
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("Add documents", type=ALLOWED_TYPES, accept_multiple_files=True)
        index_now = st.checkbox("Index after upload (required before a document can be searched)", value=True)
        submitted = st.form_submit_button("Upload", icon=":material/upload:")

    if not (submitted and files):
        return
    with st.status(f"Uploading {len(files)} file(s)...", expanded=True) as status:
        failed = 0
        for file in files:
            try:
                doc = client.upload(file.name, file.getvalue(), file.type)
                st.write(f"Uploaded **{file.name}**")
                if index_now:
                    result = client.index(doc["id"])
                    st.write(f"Indexed {result['chunks_indexed']} chunk(s)")
            except ApiError as e:
                failed += 1
                st.write(f"**{file.name}** failed: {e}")
        status.update(
            label="Finished with errors" if failed else "Done",
            state="error" if failed else "complete",
            expanded=bool(failed),
        )


def document_card(client: DocAIClient, doc: dict) -> None:
    try:
        index = client.index_status(doc["id"])
    except ApiError:
        index = {"indexed": False, "chunk_count": 0}
    state = f"indexed ({index['chunk_count']} chunks)" if index["indexed"] else "not indexed"

    with st.expander(f"{doc['filename']}  ·  v{doc['version']}  ·  {state}"):
        st.caption(
            f"{human_size(doc['size_bytes'])}  ·  added {short_date(doc['created_at'])}  ·  "
            f"updated {short_date(doc['updated_at'])}  ·  id `{doc['id']}`"
        )
        key = doc["id"]
        index_col, delete_col, _ = st.columns([1, 1, 4])
        if index_col.button("Re-index" if index["indexed"] else "Index", key=f"index-{key}"):
            with st.spinner("Chunking and embedding..."):
                try:
                    result = client.index(doc["id"])
                    st.toast(f"Indexed {result['chunks_indexed']} chunk(s) of {doc['filename']}")
                    st.rerun()
                except ApiError as e:
                    st.error(str(e))
        if delete_col.button("Delete", key=f"delete-{key}", icon=":material/delete:"):
            st.session_state[f"confirm-{key}"] = True
        if st.session_state.get(f"confirm-{key}"):
            st.warning(f"Delete **{doc['filename']}**, all its versions and its search index?")
            yes, no, _ = st.columns([1, 1, 4])
            if yes.button("Yes, delete", key=f"yes-{key}", type="primary"):
                try:
                    client.delete(doc["id"])
                except ApiError as e:
                    st.error(str(e))
                else:
                    st.session_state.pop(f"confirm-{key}", None)
                    st.toast(f"Deleted {doc['filename']}")
                    st.rerun()
            if no.button("Cancel", key=f"no-{key}"):
                st.session_state.pop(f"confirm-{key}", None)
                st.rerun()

        new_version_form(client, doc)
        if st.toggle("Show extracted text", key=f"text-{key}"):
            show_text(client, doc)


def new_version_form(client: DocAIClient, doc: dict) -> None:
    with st.form(f"version-{doc['id']}", clear_on_submit=True):
        file = st.file_uploader("Upload a new version", type=ALLOWED_TYPES, key=f"vfile-{doc['id']}")
        submitted = st.form_submit_button("Upload new version")
    if submitted and file:
        try:
            updated = client.upload_version(doc["id"], file.name, file.getvalue(), file.type)
            client.index(doc["id"])  # keep the search index in step with the latest version
        except ApiError as e:
            st.error(str(e))
        else:
            st.toast(f"{updated['filename']} is now version {updated['version']} (indexed)")
            st.rerun()


def show_text(client: DocAIClient, doc: dict) -> None:
    try:
        text = client.text(doc["id"])
    except ApiError as e:
        st.error(str(e))
        return
    for page in text["pages"]:
        if text["page_count"] > 1:
            st.markdown(f"**Page {page['number']}**")
        st.text(page["text"])

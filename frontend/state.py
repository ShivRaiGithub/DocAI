"""Helpers shared by the pages: getting the API client and formatting values."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st

from frontend import docai_client


@st.cache_resource
def _client_for(url: str) -> docai_client.DocAIClient:
    # cached so every rerun of the script reuses one HTTP connection pool
    return docai_client.make_client(url)


def get_client() -> docai_client.DocAIClient:
    """The client for the API address typed in the sidebar."""
    return _client_for(st.session_state.get("api_url", docai_client.DEFAULT_URL))


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _browser_timezone() -> str | None:
    """The viewer's timezone name (e.g. "Asia/Kolkata"), or None when Streamlit can't say."""
    try:
        return st.context.timezone
    except Exception:  # no browser session, e.g. when called outside a running app
        return None


def short_date(iso_timestamp: str) -> str:
    """Format an API timestamp in the viewer's own timezone, not whatever the database uses."""
    moment = datetime.fromisoformat(iso_timestamp)
    zone = _browser_timezone()
    if zone and moment.tzinfo:
        try:
            moment = moment.astimezone(ZoneInfo(zone))
        except ZoneInfoNotFoundError:
            pass
    return moment.strftime("%d %b %Y, %H:%M")

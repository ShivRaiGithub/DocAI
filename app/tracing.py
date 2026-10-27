"""LangSmith tracing, switched on only by explicit configuration.

The `@traceable` decorators and LangChain's ChatOllama both look at LANGSMITH_*
environment variables. We set those from DocAI's own settings, so nothing is
sent anywhere unless DOCAI_LANGSMITH_TRACING=true *and* an API key is present.
"""

import os

from app.config import Settings


def configure_tracing(settings: Settings) -> bool:
    """Turn tracing on or off for this process. Call once at startup, before any LLM call."""
    enabled = settings.langsmith_tracing and settings.langsmith_api_key is not None
    # Always set the flag explicitly, so a stray LANGSMITH_TRACING in the shell can't
    # switch tracing on behind our back.
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        hide = "true" if settings.langsmith_hide_content else "false"
        os.environ["LANGSMITH_HIDE_INPUTS"] = hide
        os.environ["LANGSMITH_HIDE_OUTPUTS"] = hide
    return enabled

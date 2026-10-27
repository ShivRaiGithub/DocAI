from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Depends

from app.config import Settings, get_settings


class LLMError(Exception):
    """The language model could not be reached or gave no usable answer."""


class OllamaClient:
    """Minimal client for a local Ollama server (https://ollama.com)."""

    def __init__(self, base_url: str, model: str, timeout_s: float):
        self.model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s)

    def chat(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self._http.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    # Qwen3 models "think" before answering by default. That is slow and
                    # unnecessary when the answer must come from the supplied sources.
                    "think": False,
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama timed out (model '{self.model}' may be too slow)") from e
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama returned {e.response.status_code}: {e.response.text[:200]}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach Ollama at {self._http.base_url}") from e
        content = response.json().get("message", {}).get("content", "").strip()
        if not content:
            raise LLMError("The model returned an empty answer")
        return content


@lru_cache
def _client_for(url: str, model: str, timeout_s: float) -> OllamaClient:
    return OllamaClient(url, model, timeout_s)


def get_llm(settings: Annotated[Settings, Depends(get_settings)]) -> OllamaClient:
    return _client_for(settings.ollama_url, settings.llm_model, settings.llm_timeout_s)


LLMDep = Annotated[OllamaClient, Depends(get_llm)]

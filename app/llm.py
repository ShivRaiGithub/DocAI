from functools import lru_cache
from typing import Annotated, Any, TypeVar

import httpx
from fastapi import Depends
from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The language model could not be reached or gave no usable answer."""


class OllamaClient:
    """Our small interface to the LLM, built on LangChain's ChatOllama.

    The rest of the app only sees `chat()` and `structured()`, and `LLMError`.
    """

    def __init__(self, base_url: str, model: str, timeout_s: float, chat_model: Any = None):
        self.model = model
        self.base_url = base_url
        self._chat = chat_model or ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.1,
            # Qwen3-family models "think" before answering by default. That is slow and
            # unnecessary when the answer must come from the supplied sources.
            reasoning=False,
            client_kwargs={"timeout": timeout_s},
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Plain-text reply to a list of {"role", "content"} messages."""
        content = self._call(self._chat, messages).content
        if not isinstance(content, str) or not content.strip():
            raise LLMError("The model returned an empty answer")
        return content.strip()

    def structured(self, messages: list[dict[str, str]], schema: type[T]) -> T | None:
        """Reply parsed into `schema` (a Pydantic model), or None if it isn't valid."""
        try:
            return self._call(self._chat.with_structured_output(schema), messages)
        except (OutputParserException, ValidationError):
            return None

    def _call(self, runnable: Any, messages: list[dict[str, str]]) -> Any:
        try:
            return runnable.invoke(messages)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise LLMError(f"Could not reach Ollama at {self.base_url}") from e
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama timed out (model '{self.model}' may be too slow)") from e
        except ResponseError as e:
            raise LLMError(f"Ollama returned {e.status_code}: {e.error[:200]}") from e
        except (httpx.HTTPError, ConnectionError) as e:
            raise LLMError(f"Could not reach Ollama at {self.base_url}") from e


@lru_cache
def _client_for(url: str, model: str, timeout_s: float) -> OllamaClient:
    return OllamaClient(url, model, timeout_s)


def get_llm(settings: Annotated[Settings, Depends(get_settings)]) -> OllamaClient:
    return _client_for(settings.ollama_url, settings.llm_model, settings.llm_timeout_s)


LLMDep = Annotated[OllamaClient, Depends(get_llm)]

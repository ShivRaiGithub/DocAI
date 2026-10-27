from types import SimpleNamespace

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from ollama import ResponseError
from pydantic import BaseModel, ValidationError

from app.llm import LLMError, OllamaClient


class Q(BaseModel):
    query: str


class FakeChatModel:
    """Stands in for LangChain's ChatOllama: returns a value or raises, and records input."""

    def __init__(self, result=None, error=None):
        self.result, self.error, self.seen = result, error, []
        self.schema = None

    def invoke(self, messages):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return self.result

    def with_structured_output(self, schema):
        self.schema = schema
        return self


def client_with(chat_model):
    return OllamaClient("http://ollama.test", "m", 5, chat_model=chat_model)


MSGS = [{"role": "user", "content": "yo"}]


def test_chat_returns_trimmed_text_and_passes_messages_through():
    fake = FakeChatModel(SimpleNamespace(content=" Hi "))
    assert client_with(fake).chat(MSGS) == "Hi"
    assert fake.seen == [MSGS]


def test_structured_returns_parsed_object_or_none_when_invalid():
    assert client_with(FakeChatModel(Q(query="pasta"))).structured(MSGS, Q) == Q(query="pasta")

    with pytest.raises(ValidationError) as invalid:
        Q.model_validate({"query": None})
    for parse_error in (OutputParserException("not json"), invalid.value):
        assert client_with(FakeChatModel(error=parse_error)).structured(MSGS, Q) is None


@pytest.mark.parametrize(
    "error,message",
    [
        (httpx.ConnectError("down"), "Could not reach Ollama"),
        (httpx.ConnectTimeout("dropped"), "Could not reach Ollama"),
        (ConnectionError("refused"), "Could not reach Ollama"),
        (httpx.ReadTimeout("slow"), "timed out"),
        (ResponseError("model not found", 404), "404"),
    ],
)
def test_transport_errors_become_llmerror(error, message):
    client = client_with(FakeChatModel(error=error))
    with pytest.raises(LLMError, match=message):
        client.chat(MSGS)
    with pytest.raises(LLMError, match=message):
        client.structured(MSGS, Q)


def test_empty_answer_is_an_error():
    with pytest.raises(LLMError, match="empty"):
        client_with(FakeChatModel(SimpleNamespace(content="  "))).chat(MSGS)

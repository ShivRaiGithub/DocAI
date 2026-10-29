"""A small client for the DocAI REST API, used by the Streamlit pages.

It is kept free of any Streamlit code so it can be tested on its own, and so the
pages never deal with URLs, status codes or JSON error bodies.
"""

import os
from typing import Any

import httpx

DEFAULT_URL = os.environ.get("DOCAI_API_URL", "http://localhost:8000")


class ApiError(Exception):
    """A request failed. The message is safe to show to the user."""


class DocAIClient:
    def __init__(self, base_url: str = DEFAULT_URL, http: httpx.Client | None = None):
        # `http` can be injected: tests pass a TestClient bound to the real FastAPI app
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url, timeout=300.0)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise ApiError(f"Cannot reach the DocAI API at {self.base_url}. Is it running?") from e
        except httpx.TimeoutException as e:
            raise ApiError("The request timed out. The model may be slow or busy.") from e
        if response.status_code >= 400:
            raise ApiError(_error_message(response))
        return None if response.status_code == 204 else response.json()

    # documents
    def health(self) -> dict:
        return self._request("GET", "/health")

    def list_documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def upload(self, filename: str, data: bytes, content_type: str | None = None) -> dict:
        return self._request("POST", "/documents", files={"file": (filename, data, content_type)})

    def upload_version(
        self, doc_id: str, filename: str, data: bytes, content_type: str | None = None
    ) -> dict:
        files = {"file": (filename, data, content_type)}
        return self._request("POST", f"/documents/{doc_id}/versions", files=files)

    def delete(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{doc_id}")

    def text(self, doc_id: str) -> dict:
        return self._request("GET", f"/documents/{doc_id}/text")

    def index(self, doc_id: str) -> dict:
        return self._request("POST", f"/documents/{doc_id}/index")

    def index_status(self, doc_id: str) -> dict:
        return self._request("GET", f"/documents/{doc_id}/index")

    # questions
    def search(
        self, query: str, limit: int = 5, min_score: float | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        body = {"query": query, "limit": limit, "min_score": min_score, "document_ids": document_ids}
        return self._request("POST", "/search", json=body)["results"]

    def ask(
        self, question: str, agent: bool = False, limit: int | None = None,
        min_score: float | None = None, document_ids: list[str] | None = None,
    ) -> dict:
        body = {
            "question": question, "limit": limit, "min_score": min_score,
            "document_ids": document_ids,
        }
        return self._request("POST", "/ask/agent" if agent else "/ask", json=body)


def _error_message(response: httpx.Response) -> str:
    """FastAPI puts the reason in {"detail": ...}; for validation errors it is a list."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail or f"Request failed with status {response.status_code}")


def make_client(base_url: str) -> DocAIClient:
    """The one place pages get a client from (tests replace this)."""
    return DocAIClient(base_url)

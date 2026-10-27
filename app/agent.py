"""Agentic retrieval: if the first search finds nothing relevant, rewrite the query and retry.

Plain RAG (`/ask`) searches once with the user's wording and gives up if that fails.
This adds a bounded loop around it:

    search -> relevant chunks?  yes -> generate answer
                                no  -> rewrite the query with the LLM -> search again
                                       (at most `agent_max_rewrites` times, never
                                        repeating a query) -> give up honestly
"""

from langsmith import traceable

from app import services
from app.config import Settings
from app.embeddings import Embedder
from app.llm import OllamaClient
from app.rag import RewrittenQuery, build_rewrite_messages, clean_rewrite
from app.schemas import AgentAskResponse
from app.vectorstore import VectorStore


@traceable(
    name="agent",
    process_inputs=lambda i: {k: i.get(k) for k in ("question", "limit", "document_ids", "min_score")},
)
def run_agent(
    settings: Settings,
    embedder: Embedder,
    vectors: VectorStore,
    llm: OllamaClient,
    question: str,
    limit: int | None = None,
    document_ids: list[str] | None = None,
    min_score: float | None = None,
) -> AgentAskResponse:
    min_score = min_score if min_score is not None else settings.rag_min_score
    limit = limit or settings.rag_top_k

    queries = [question]  # every query tried, in order
    steps: list[str] = []
    hits = []

    while True:
        query = queries[-1]
        hits = services.semantic_search(embedder, vectors, query, limit, document_ids, min_score)
        steps.append(f"retrieve: '{query}' -> {len(hits)} relevant chunk(s)")
        if hits or len(queries) > settings.agent_max_rewrites:
            break

        reply = llm.structured(build_rewrite_messages(question, queries), RewrittenQuery)
        new_query = clean_rewrite(reply.query, queries) if reply else None
        if new_query is None:
            steps.append("rewrite: no new query, stopping")
            break
        steps.append(f"rewrite: '{new_query}'")
        queries.append(new_query)

    answer = services.generate_answer(llm, question, hits)  # skips the LLM if hits is empty
    steps.append(
        "generate: answered from retrieved chunks" if hits else "give_up: nothing relevant found"
    )
    return AgentAskResponse(**answer.model_dump(), search_queries=queries, steps=steps)

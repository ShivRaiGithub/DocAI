# DocAI

A document intelligence service. Upload documents, ask questions in plain language, and get answers grounded in your own files with citations. The same knowledge base is exposed to AI assistants such as Claude Code through the **Model Context Protocol (MCP)**.

It runs entirely on your machine: local embeddings, a local LLM through Ollama, and your own Postgres and Qdrant. Nothing is sent to a third party unless you opt in to LangSmith tracing.


## What it does

- **Document management:** upload PDF, TXT and Markdown files. Re-uploading creates a new version and keeps the old ones.
- **Ingestion pipeline:** extract text per page, split into overlapping chunks, embed, and store in a vector database.
- **Semantic search:** find passages by meaning, not keywords ("money back" finds the refund policy). Results carry filename, page and score.
- **Grounded answers (RAG):** a local LLM answers only from the retrieved passages, with `[1]`-style citations, and says it doesn't know when the documents don't contain the answer.
- **Agentic retrieval:** if the first search finds nothing relevant, the query is rewritten by the LLM and retried (bounded, never repeating a query).
- **Web interface:** a Streamlit app to upload and manage documents, chat with them (with sources and the agent's steps shown), and run searches.
- **MCP server:** `search_documents`, `ask_documents`, `list_documents` and `get_document_text` tools for any MCP client.
- **Tracing and evaluation:** optional LangSmith tracing, and a built-in evaluation suite that scores retrieval and answer quality.

## Architecture

```mermaid
flowchart LR
    subgraph Clients
        W[Streamlit web UI]
        U[HTTP clients]
        A[AI assistants<br/>Claude Code, etc.]
    end
    W --> API[FastAPI]
    U --> API
    A --> MCP[MCP server]
    API --> S[services.py<br/>shared logic]
    MCP --> S
    S --> PG[(PostgreSQL<br/>documents, versions)]
    S --> FS[(Local files<br/>original uploads)]
    S --> EMB[sentence-transformers<br/>all-MiniLM-L6-v2]
    S --> Q[(Qdrant<br/>chunk vectors)]
    S --> LLM[Ollama<br/>via LangChain ChatOllama]
```

The REST API and the MCP server are thin wrappers over the same functions in [`app/services.py`](app/services.py), so retrieval behaves identically through both.

Ingestion: `file -> text (pypdf) -> chunks (800 chars, 200 overlap) -> embeddings (384-d) -> Qdrant`.
Answering: `question -> embed -> nearest chunks above a score cutoff -> LLM with numbered sources -> answer + citations`.

## Tech stack

| Area | Technology |
|---|---|
| Language, packaging | Python 3.13, [uv](https://docs.astral.sh/uv/) |
| API | FastAPI, Pydantic, pydantic-settings |
| Metadata | PostgreSQL with SQLAlchemy 2 and psycopg 3 |
| Vector database | Qdrant (Docker) |
| Embeddings | sentence-transformers, `all-MiniLM-L6-v2` (local, uses the GPU if present) |
| LLM | Ollama through LangChain's `ChatOllama` (default `qwen3.5:9b`) |
| Extraction | pypdf |
| Web interface | Streamlit (a client of the REST API) |
| Assistant integration | Official MCP Python SDK (`mcp` 2.x) |
| Observability, evaluation | LangSmith (optional) |
| Tests | pytest |

## Quick start

**Prerequisites:** Python 3.13 with `uv`, Docker, a PostgreSQL server, and [Ollama](https://ollama.com) with a model pulled (`ollama pull qwen3.5:9b`).

```bash
git clone https://github.com/ShivRaiGithub/DocAI.git
cd DocAI
uv sync

# 1. vector database
docker compose up -d                       # Qdrant on :6333

# 2. database: create an empty one, e.g. `createdb docai`
cp .env.example .env                       # then set DOCAI_DATABASE_URL

# 3. run the API (tables are created on startup)
uv run uvicorn app.main:app --reload       # http://localhost:8000/docs
```

The first request that needs embeddings downloads the model (about 90 MB).

Try it:

```bash
# upload, index, ask
ID=$(curl -s -F file=@notes.pdf localhost:8000/documents | jq -r .id)
curl -X POST localhost:8000/documents/$ID/index
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What is the refund policy?"}'

# retries with a rewritten query when the first search finds nothing
curl -X POST localhost:8000/ask/agent -H 'content-type: application/json' \
     -d '{"question": "I bought the wrong size, what now?"}'
```

## Web interface

```bash
uv run streamlit run frontend/app.py       # http://localhost:8501 (the API must be running)
```

Three pages: **Ask** (chat with your documents; each answer has expandable sources and, in agent mode, the steps taken), **Search** (ranked passages with relevance bars, no LLM involved) and **Documents** (upload several files at once, index, upload new versions, view extracted text, delete). It talks to the REST API only, so it can run on a different machine: set the API address in the sidebar or with `DOCAI_API_URL`. The upload limit shown in the UI is set in `.streamlit/config.toml` (`maxUploadSize`, in MB) and should match `DOCAI_MAX_UPLOAD_MB`.

Chat history is for display only: each question is answered on its own.

## API

| Method and path | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /documents` | Upload a PDF, TXT or MD file |
| `GET /documents`, `GET /documents/{id}` | List documents, get one |
| `POST /documents/{id}/versions` | Upload a new version |
| `DELETE /documents/{id}` | Delete the document, its files and its vectors |
| `GET /documents/{id}/text` | Extracted text by page (`?version=N`) |
| `GET /documents/{id}/chunks` | Preview the chunking (`?chunk_size=`, `?chunk_overlap=`) |
| `POST /documents/{id}/index`, `GET .../index` | Embed and store the chunks, check index status |
| `POST /search` | Semantic search with optional document filter and `min_score` |
| `POST /ask` | Grounded answer with citations |
| `POST /ask/agent` | Same, with query rewriting and a step-by-step trace |
| `POST /embeddings`, `POST /embeddings/similarity` | Inspect embeddings and cosine similarity |

Interactive documentation is served at `/docs`.

## MCP server

```bash
uv run python -m app.mcp_server            # stdio, for local MCP clients
uv run python -m app.mcp_server --http     # streamable HTTP
```

Register it with Claude Code:

```bash
claude mcp add docai -- uv --directory /path/to/DocAI run python -m app.mcp_server
```

| Tool | Description |
|---|---|
| `search_documents` | Semantic search; passages with score, filename and page |
| `ask_documents` | Cited answer generated by the local LLM |
| `list_documents` | Stored documents |
| `get_document_text` | Read a document or a single page, with a size cap |

All tools are read-only. Note that a connected client can read every document in the knowledge base; there are no per-user permissions yet.

## Configuration

Settings are read from environment variables prefixed `DOCAI_` or from a `.env` file. The commonly changed ones:

| Variable | Default | Meaning |
|---|---|---|
| `DOCAI_DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/docai` | PostgreSQL (`postgres://` URLs also work) |
| `DOCAI_QDRANT_URL` | `http://localhost:6333` | Qdrant server |
| `DOCAI_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `DOCAI_LLM_MODEL` | `qwen3.5:9b` | Model used for answers and query rewriting |
| `DOCAI_CHUNK_SIZE`, `DOCAI_CHUNK_OVERLAP` | `800`, `200` | Characters per chunk and overlap |
| `DOCAI_RAG_MIN_SCORE` | `0.25` | Chunks scoring below this are not given to the LLM |
| `DOCAI_AGENT_MAX_REWRITES` | `2` | Query rewrites `/ask/agent` may try |
| `DOCAI_MAX_UPLOAD_MB` | `20` | Upload size limit |
| `DOCAI_LANGSMITH_TRACING` | `false` | Opt-in tracing (also needs `DOCAI_LANGSMITH_API_KEY`) |

LangSmith tracing uploads prompts and retrieved document text to LangSmith's servers, so it is off by default. `DOCAI_LANGSMITH_HIDE_CONTENT=true` keeps timings and scores but hides the text.

## Evaluation

A small labelled set (5 documents, 18 questions including 4 the documents cannot answer) lives in [`eval/`](eval/). It runs against an in-memory Qdrant and never touches your data.

```bash
uv run python -m app.evaluation                  # retrieval only, no LLM, seconds
uv run python -m app.evaluation --mode ask       # full RAG answers
uv run python -m app.evaluation --mode agent     # with query rewriting
uv run python -m app.evaluation --min-score 0.35 --chunk-size 400
```

Results with `qwen3.5:9b` and the default settings:

| | retrieval hit | facts present | citation correct | correct "I don't know" |
|---|---|---|---|---|
| `/ask` | 0.93 | 0.93 | 0.93 | 1.00 |
| `/ask/agent` | 1.00 | 1.00 | 1.00 | 1.00 |

The one miss for plain `/ask` is a question phrased unlike the document ("I bought the wrong size, what now?"); the agent rewrites it and succeeds. The corpus is small, so treat differences of one question as noise.

## Development

```bash
uv run pytest                    # 103 tests; no running services needed (the embedding model downloads on first run)
```

The tests use in-memory SQLite and in-memory Qdrant, a fake LLM, and the real embedding model.

```
app/
  main.py, config.py, schemas.py   FastAPI app, settings, Pydantic models
  api/                             routes: documents, search, ask, embeddings, health
  services.py                      logic shared by REST and MCP
  extraction.py, chunking.py       text extraction, chunk splitting
  embeddings.py, vectorstore.py    sentence-transformers, Qdrant
  llm.py, rag.py, agent.py         Ollama client, prompts and citations, retry loop
  mcp_server.py                    MCP tools
  tracing.py, evaluation.py        LangSmith switch, evaluation CLI
frontend/                          Streamlit app: app.py, API client, views/
eval/                              evaluation corpus and questions
.streamlit/config.toml             Streamlit settings (upload limit, no telemetry)
tests/
docker-compose.yml                 Qdrant
```

## Limitations

Not built yet: per-user permissions and authentication (the web UI has no login and shows every document), streamed answers, automatic indexing on upload, hybrid keyword and semantic search, conversation memory, OCR for scanned PDFs, and database migrations (tables are created on startup). The embedding model is English-only and reads at most 256 tokens per chunk. Answers come from a small local model and can be wrong, which is why every answer carries citations to check against.

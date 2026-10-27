"""Measure retrieval and answer quality against a small labelled question set.

    uv run python -m app.evaluation                 # retrieval only: fast, no LLM
    uv run python -m app.evaluation --mode ask      # full RAG answers (needs Ollama)
    uv run python -m app.evaluation --mode agent    # RAG with query rewriting
    uv run python -m app.evaluation --chunk-size 400 --min-score 0.3   # try other settings
    uv run python -m app.evaluation --mode ask --langsmith             # also log to LangSmith

It indexes `eval/corpus/` into a private in-memory Qdrant, so it never touches your
real documents or database, and scores each question in `eval/dataset.json`.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from app import services
from app.agent import run_agent
from app.chunking import chunk_pages
from app.config import Settings, get_settings
from app.embeddings import Embedder
from app.extraction import extract_pages
from app.llm import OllamaClient
from app.rag import NO_SOURCES_ANSWER
from app.tracing import configure_tracing
from app.vectorstore import VectorStore

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
MODES = ("search", "ask", "agent")
DONT_KNOW_PHRASES = ("don't know", "do not know", "couldn't find", "could not find", "no information")


@dataclass
class Example:
    id: str
    question: str
    expected_source: str | None  # None = the corpus does not contain the answer
    expected_facts: list[str] = field(default_factory=list)  # "a|b" means either wording


@dataclass
class Context:
    settings: Settings
    embedder: Embedder
    vectors: VectorStore
    llm: OllamaClient | None
    k: int = 3  # how many chunks retrieval returns (and the "k" in hit@k)


def load_dataset(path: Path = EVAL_DIR / "dataset.json") -> list[Example]:
    return [Example(**row) for row in json.loads(path.read_text())]


def index_corpus(corpus_dir: Path, ctx: Context) -> int:
    """Chunk, embed and store every file of the corpus. Returns the number of chunks."""
    total = 0
    for path in sorted(corpus_dir.iterdir()):
        pages = extract_pages(path.read_bytes(), path.suffix)
        chunks = chunk_pages(pages, ctx.settings.chunk_size, ctx.settings.chunk_overlap)
        vectors = ctx.embedder.embed([c.text for c in chunks])
        total += ctx.vectors.replace_document(path.name, 1, path.name, chunks, vectors)
    return total


# --- the system under test -------------------------------------------------------------


def answer_with(ctx: Context, mode: str, question: str) -> dict[str, Any]:
    """Run one question through the pipeline and describe what happened."""
    s = ctx.settings
    if mode == "search":
        # Same cutoff as the real pipeline: chunks scoring below min_score are dropped, so
        # a high cutoff costs recall here too (instead of looking free).
        hits = services.semantic_search(ctx.embedder, ctx.vectors, question, ctx.k, None, None)
        kept = [h for h in hits if h.score >= s.rag_min_score]
        return {
            "answer": None,
            "sources": [h.filename for h in kept],
            "cited": [],
            "abstained": not kept,
        }

    run = services.answer_question if mode == "ask" else run_agent
    response = run(s, ctx.embedder, ctx.vectors, ctx.llm, question, ctx.k)
    answer = response.answer
    return {
        "answer": answer,
        "sources": [c.filename for c in response.sources],
        "cited": [c.filename for c in response.citations],
        "abstained": answer == NO_SOURCES_ANSWER
        or any(p in answer.lower() for p in DONT_KNOW_PHRASES),
    }


# --- metrics (pure functions: no models, easy to test) ---------------------------------


def fact_present(fact: str, answer: str) -> bool:
    return any(option.strip().lower() in answer.lower() for option in fact.split("|"))


def score(outputs: dict[str, Any], reference: dict[str, Any], mode: str) -> dict[str, float]:
    """All metrics that apply to this question, each 0..1."""
    expected = reference["expected_source"]
    if expected is None:
        return {"correct_abstention": float(outputs["abstained"])}

    sources = outputs["sources"]
    scores = {
        "retrieval_hit": float(expected in sources),
        "reciprocal_rank": 1 / (sources.index(expected) + 1) if expected in sources else 0.0,
    }
    if mode != "search":
        answer = outputs["answer"] or ""
        facts = reference["expected_facts"]
        scores["facts_present"] = float(bool(facts) and all(fact_present(f, answer) for f in facts))
        scores["citation_correct"] = float(expected in outputs["cited"])
    return scores


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average of each metric over the questions it applies to."""
    keys = sorted({k for row in rows for k in row})
    return {k: sum(r[k] for r in rows if k in r) / sum(k in r for r in rows) for k in keys}


# --- running ---------------------------------------------------------------------------


def run_evaluation(ctx: Context, mode: str, examples: list[Example]):
    results = []
    for ex in examples:
        outputs = answer_with(ctx, mode, ex.question)
        reference = {"expected_source": ex.expected_source, "expected_facts": ex.expected_facts}
        results.append((ex, outputs, score(outputs, reference, mode)))
    return results


def format_report(results, mode: str, ctx: Context, chunk_count: int) -> str:
    s = ctx.settings
    lines = [
        f"mode={mode}  k={ctx.k}  chunk_size={s.chunk_size}  overlap={s.chunk_overlap}  "
        f"min_score={s.rag_min_score}  chunks={chunk_count}  model={s.llm_model if mode != 'search' else '-'}",
        "",
    ]
    for ex, outputs, scores in results:
        # reciprocal_rank is informational: a right answer from the 3rd-ranked chunk is a pass
        ok = all(v == 1.0 for k, v in scores.items() if k != "reciprocal_rank")
        detail = " ".join(f"{k}={v:.2f}" for k, v in scores.items())
        lines.append(f"{'PASS' if ok else 'FAIL'} {ex.id} {ex.question[:48]:<48} {detail}")
        if not ok:
            lines.append(f"       expected={ex.expected_source}  retrieved={outputs['sources']}")
            if outputs["answer"]:
                lines.append(f"       answer={outputs['answer'][:110]!r}")
    lines += ["", "SUMMARY (average over the questions each metric applies to)"]
    lines += [f"  {k:<18} {v:.2f}" for k, v in summarize([r[2] for r in results]).items()]
    return "\n".join(lines)


# --- LangSmith -------------------------------------------------------------------------


def langsmith_callables(ctx: Context, mode: str):
    """The (target, evaluator) pair in the shapes `langsmith.evaluate` expects."""

    def target(inputs: dict) -> dict:
        return answer_with(ctx, mode, inputs["question"])

    def all_metrics(outputs: dict, reference_outputs: dict) -> dict:
        scores = score(outputs, reference_outputs, mode)
        return {"results": [{"key": k, "score": v} for k, v in scores.items()]}

    return target, all_metrics


def run_in_langsmith(ctx: Context, mode: str, examples: list[Example], dataset_name: str):
    """Log the dataset and one experiment to LangSmith, so runs can be compared over time."""
    from langsmith import Client, evaluate

    client = Client()
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name, description="DocAI evaluation questions")
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"question": e.question} for e in examples],
            outputs=[
                {"expected_source": e.expected_source, "expected_facts": e.expected_facts}
                for e in examples
            ],
            metadata=[{"id": e.id} for e in examples],
        )

    target, all_metrics = langsmith_callables(ctx, mode)

    return evaluate(
        target,
        data=dataset_name,
        evaluators=[all_metrics],
        experiment_prefix=f"docai-{mode}",
        metadata={
            "chunk_size": ctx.settings.chunk_size, "chunk_overlap": ctx.settings.chunk_overlap,
            "min_score": ctx.settings.rag_min_score, "k": ctx.k, "llm": ctx.settings.llm_model,
        },
        client=client,
        max_concurrency=1,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocAI retrieval and answers.")
    parser.add_argument("--mode", choices=MODES, default="search")
    parser.add_argument("--k", type=int, default=3, help="chunks retrieved per question")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--langsmith", action="store_true", help="also log an experiment to LangSmith")
    args = parser.parse_args(argv)

    overrides = {
        name: value
        for name, value in {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "rag_min_score": args.min_score,
        }.items()
        if value is not None
    }
    settings = get_settings().model_copy(update=overrides)
    configure_tracing(settings)  # eval runs are traced too, if tracing is enabled

    ctx = Context(
        settings=settings,
        embedder=Embedder(settings.embedding_model),
        vectors=VectorStore(QdrantClient(":memory:"), "eval", settings.embedding_dimension),
        llm=None
        if args.mode == "search"
        else OllamaClient(settings.ollama_url, settings.llm_model, settings.llm_timeout_s),
        k=args.k,
    )
    chunk_count = index_corpus(EVAL_DIR / "corpus", ctx)
    examples = load_dataset()

    if args.langsmith:
        run_in_langsmith(ctx, args.mode, examples, "docai-eval")
        print("Experiment logged to LangSmith (see the 'Experiments' tab of the docai-eval dataset).")
    else:
        results = run_evaluation(ctx, args.mode, examples)
        print(format_report(results, args.mode, ctx, chunk_count))


if __name__ == "__main__":
    main()

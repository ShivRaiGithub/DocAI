import uuid

import pytest
from langsmith import evaluate
from langsmith.schemas import Example as LangSmithExample

from app import evaluation as ev
from app.config import Settings


@pytest.fixture
def ctx(embedder, vector_store, llm):
    context = ev.Context(Settings(), embedder, vector_store, llm, k=3)
    ev.index_corpus(ev.EVAL_DIR / "corpus", context)
    return context


def reference(source, facts=()):
    return {"expected_source": source, "expected_facts": list(facts)}


def test_dataset_is_consistent():
    examples = ev.load_dataset()
    corpus = {p.name for p in (ev.EVAL_DIR / "corpus").iterdir()}
    assert len({e.id for e in examples}) == len(examples)
    assert {e.expected_source for e in examples if e.expected_source} <= corpus
    assert all(e.expected_facts for e in examples if e.expected_source)
    assert any(e.expected_source is None for e in examples)  # includes unanswerable questions


def test_fact_present_accepts_alternatives_and_ignores_case():
    assert ev.fact_present("1 hour|one hour", "Report it within ONE HOUR.")
    assert not ev.fact_present("14", "Use a long password.")


def test_score_answerable_search_mode():
    out = {"sources": ["b.md", "a.md"], "abstained": False, "answer": None, "cited": []}
    assert ev.score(out, reference("a.md"), "search") == {"retrieval_hit": 1.0, "reciprocal_rank": 0.5}
    miss = ev.score({**out, "sources": ["b.md"]}, reference("a.md"), "search")
    assert miss == {"retrieval_hit": 0.0, "reciprocal_rank": 0.0}


def test_score_answer_modes_check_facts_and_citation():
    out = {"sources": ["a.md"], "cited": ["a.md"], "answer": "It takes 30 days [1].", "abstained": False}
    good = ev.score(out, reference("a.md", ["30 days"]), "ask")
    assert good["facts_present"] == good["citation_correct"] == 1.0

    wrong = ev.score({**out, "answer": "Soon.", "cited": []}, reference("a.md", ["30 days"]), "ask")
    assert wrong["facts_present"] == wrong["citation_correct"] == 0.0


def test_score_unanswerable_rewards_abstaining():
    assert ev.score({"abstained": True}, reference(None), "ask") == {"correct_abstention": 1.0}
    assert ev.score({"abstained": False}, reference(None), "ask") == {"correct_abstention": 0.0}


def test_summarize_averages_each_metric_over_its_own_questions():
    rows = [{"hit": 1.0, "abstain": 0.0}, {"hit": 0.0}, {"hit": 1.0}]
    assert ev.summarize(rows) == {"abstain": 0.0, "hit": pytest.approx(2 / 3)}


def test_search_mode_finds_every_answerable_question_without_a_cutoff(ctx):
    ctx.settings = ctx.settings.model_copy(update={"rag_min_score": -1.0})
    results = ev.run_evaluation(ctx, "search", ev.load_dataset())
    hits = [s["retrieval_hit"] for _, _, s in results if "retrieval_hit" in s]
    assert len(hits) == 14 and all(h == 1.0 for h in hits)


def test_a_stricter_cutoff_trades_recall_for_abstention(ctx):
    def run(min_score):
        ctx.settings = ctx.settings.model_copy(update={"rag_min_score": min_score})
        return ev.summarize([s for _, _, s in ev.run_evaluation(ctx, "search", ev.load_dataset())])

    loose, strict = run(0.0), run(0.35)
    assert strict["correct_abstention"] > loose["correct_abstention"]
    assert strict["retrieval_hit"] < loose["retrieval_hit"]


def test_ask_mode_uses_the_llm_answer_and_citations(ctx, llm):
    llm.reply = "You have 30 days [1]."
    example = ev.load_dataset()[0]
    outputs = ev.answer_with(ctx, "ask", example.question)
    assert outputs["answer"] == "You have 30 days [1]."
    assert outputs["cited"] == ["returns-policy.md"]
    scores = ev.score(outputs, reference(example.expected_source, example.expected_facts), "ask")
    assert scores["facts_present"] == scores["citation_correct"] == 1.0


def test_ask_mode_counts_dont_know_answers_as_abstaining(ctx, llm):
    llm.reply = "I don't know based on the provided documents."
    assert ev.answer_with(ctx, "ask", "Who is the CEO of Acme?")["abstained"] is True


def test_report_lists_failures_with_details(ctx):
    results = ev.run_evaluation(ctx, "search", ev.load_dataset()[:3])
    report = ev.format_report(results, "search", ctx, chunk_count=5)
    assert "mode=search" in report and "SUMMARY" in report and "retrieval_hit" in report
    assert "PASS q01" in report


def test_langsmith_engine_accepts_our_target_and_evaluator(ctx):
    """Runs LangSmith's own evaluate() locally (upload_results=False): no key or network."""
    target, metrics = ev.langsmith_callables(ctx, "search")
    examples = [
        LangSmithExample(
            id=uuid.uuid4(), dataset_id=uuid.uuid4(),
            inputs={"question": e.question}, outputs=reference(e.expected_source, e.expected_facts),
        )
        for e in ev.load_dataset()[:2]
    ]
    results = list(evaluate(target, data=examples, evaluators=[metrics],
                            upload_results=False, max_concurrency=1))
    keys = {r.key for res in results for r in res["evaluation_results"]["results"]}
    assert keys == {"retrieval_hit", "reciprocal_rank"}

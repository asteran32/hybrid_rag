import json

from enterprise_agent.evaluation.metrics import (
    ReferenceSpan,
    document_metrics,
    span_recall,
)
from enterprise_agent.evaluation.runner import evaluate, load_questions, write_report
from enterprise_agent.models import SearchHit


def test_document_metrics_multiple_gold_and_duplicate_chunks():
    assert document_metrics(["d", "d", "g", "g"], {"g", "h"}) == {
        "recall@1": 0,
        "recall@5": 0.5,
        "recall@10": 0.5,
        "mrr": 0.5,
    }
    assert document_metrics(["g"], set()) is None
    assert document_metrics([], {"g"})["mrr"] == 0
    assert document_metrics(["1", "2", "3", "4", "5", "6"], {"6"}) == {
        "recall@1": 0,
        "recall@5": 0,
        "recall@10": 1,
        "mrr": 1 / 6,
    }


def test_reference_passage_requires_full_coverage_in_the_right_document():
    ref = ReferenceSpan(document_id="dsid_right", start=5, end=20)
    chunks = [
        {"document_id": "dsid_right", "spans": [[5, 12]]},
        {"document_id": "dsid_wrong", "spans": [[0, 20]]},
        {"document_id": "dsid_right", "spans": [[12, 20]]},
    ]
    assert span_recall(chunks, [ref]) == {
        "span_recall@1": 0,
        "span_recall@5": 1,
        "span_recall@10": 1,
    }


def test_runner_passes_question_only_and_saves_trace(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": str(i),
                    "question": "search words",
                    "expected_doc_ids": gold,
                    "gold_answer": "secret",
                }
            )
            for i, gold in enumerate((["dsid_1"], []))
        )
    )
    questions = load_questions(path, 2)

    class Spy:
        def search(self, query, k):
            assert (query, k) == ("search words", 10)
            return [
                SearchHit(
                    document_id="dsid_1",
                    chunk_id=str(i),
                    position=i,
                    start_offset=i,
                    end_offset=i + 1,
                    content="a",
                    score=1,
                    rank=i + 1,
                    component_ranks={},
                    source_type="jira",
                    title=None,
                    metadata={},
                    source_path="jira/a",
                )
                for i in range(2)
            ]

    result = evaluate({"C0": Spy()}, questions, chunk_depth=10)
    summary = result["C0"]["summary"]
    assert summary["evaluated_questions"] == summary["no_gold_questions"] == 1
    assert summary["mean_retrieved_documents"] == summary["mrr"] == 1
    output = tmp_path / "results/run.json"
    write_report(output, result)
    assert (
        json.loads(output.read_text()) == result and "secret" not in output.read_text()
    )
    assert (
        evaluate({"C1": Spy()}, questions[1:], chunk_depth=10)["C1"]["summary"]["mrr"]
        is None
    )

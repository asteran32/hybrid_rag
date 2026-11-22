import json

import pytest

from enterprise_agent import cli
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import ChunkSettings
from enterprise_agent.retrieval.pageindex_selector import (
    LocalSelector,
    Selection,
    SelectorUnavailable,
)
from enterprise_agent.storage.pageindex import build_pageindex

pytestmark = pytest.mark.integration


def test_tree_only_evaluation_tracks_coverage_errors_and_candidate_recall(
    db, documents, monkeypatch, tmp_path
):
    pytest.importorskip("pageindex")
    ingest(db, documents, ChunkSettings(), limit=None, batch_size=2)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"question_id": str(i), "question": q, "expected_doc_ids": ids})
            for i, (q, ids) in enumerate(
                [("travel", ["dsid_one"]), ("absent", ["dsid_absent"]), ("outage", [])]
            )
        )
    )
    args = cli.parser().parse_args(
        [
            "evaluate",
            "--methods",
            "pageindex",
            "--dataset-version",
            "fixture",
            "--questions",
            str(path),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    monkeypatch.setattr(
        cli, "configured_embedder", lambda: pytest.fail("unexpected embedding")
    )
    with pytest.raises(ValueError, match="0/2"):
        cli.run_evaluation(db, args)
    build_pageindex(db)

    def select(self, question, stage, choices, limit, timeout):
        if question == "outage":
            raise SelectorUnavailable("offline")
        return Selection([1] if question == "travel" else [], {})

    monkeypatch.setattr(LocalSelector, "select", select)
    assert cli.run_evaluation(db, args) == 1
    report = json.loads(args.output.read_text())
    config, result = report["configuration"], report["results"]["PI"]
    assert config["snapshot"]["indexed_chunks"] is None
    assert config["snapshot"]["pageindex"]["documents"] == 2
    assert config["embedding"] is None
    assert result["summary"]["candidate_metrics"]["recall@1"] == 0.5
    assert result["summary"]["with_gold_in_corpus"]["hits_at_10"] == 1
    assert result["summary"]["navigation"]["errors"] == 1
    assert result["queries"][-1]["trace"]["stop_reason"] == "model_error"

import json
import sys
from contextlib import nullcontext

import pytest

from enterprise_agent import cli
from enterprise_agent.agent import runtime, tools
from enterprise_agent.agent.budget import Budget, RunStopped
from enterprise_agent.agent.contracts import AgentSettings
from enterprise_agent.agent.planning import Plan, Review
from enterprise_agent.generation.contracts import (
    AnswerRequest,
    Generation,
    GenerationInfo,
)
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import AnswerScope, ChunkSettings
from enterprise_agent.storage.embeddings import index_embeddings

pytestmark = pytest.mark.integration
pytest.importorskip("langgraph.graph")


class Decisions:
    """Fixed graph inputs. Tests do not invoke a planner, generator or classifier model."""

    def __init__(self):
        self.initial = Plan(complexity="simple", searches=[])
        self.reviews = []
        self.contexts = []
        self.generations = 0

    def plan(self, question):
        return self.initial

    def review(self, question, tasks, history, passages, can_retry):
        self.contexts.append(passages)
        decision = self.reviews.pop(0)
        return decision() if callable(decision) else decision

    def generate(self, question, passages):
        self.generations += 1
        label, quote = next(iter(passages.items()))
        return Generation(
            outcome={
                "coverage": "complete",
                "conflicts": [],
                "claims": [{"text": quote, "kind": "fact", "citations": [label]}],
            },
            info=GenerationInfo(
                provider="fixture", model="fixed", prompt_version="fixture"
            ),
        )


def decision(supported=(), followup=None):
    return Review(
        supported=[{"task": task, "passages": [label]} for task, label in supported],
        followup=followup,
    )


SCOPE = AnswerScope(source_types=("slack",), document_ids=("dsid_one",))
REQUEST = AnswerRequest(
    question="Business travel receipts",
    method="agentic",
    source_type="slack",
    metadata={"team": "finance"},
    top_k=3,
)


@pytest.fixture
def decisions(db, documents, embedder, monkeypatch, classify_fixture):
    ingest(db, documents, ChunkSettings(), limit=2, batch_size=2)
    index_embeddings(db, embedder, max_chunks=2, batch_size=2)
    fake = Decisions()
    monkeypatch.setattr(runtime, "OpenAIPlanner", lambda budget: fake)
    monkeypatch.setattr(runtime, "classify_local", classify_fixture)
    monkeypatch.setattr(tools, "configured_embedder", lambda: embedder)
    return fake


def test_simple_cli_search_and_http_answer_share_graph_and_verified_scope(
    db, decisions, documents, monkeypatch, capsys
):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from enterprise_agent import api

    decisions.reviews = [decision([(1, "E1.1")])] * 2
    monkeypatch.setenv("ANSWER_ACCESS_SCOPE", SCOPE.model_dump_json())
    monkeypatch.setattr(cli, "connect", lambda: nullcontext(db))
    monkeypatch.setattr(api, "connect", lambda: nullcontext(db))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "enterprise-agent",
            "search",
            REQUEST.question,
            "--method",
            "agentic",
            "--source-type",
            "slack",
            "--metadata",
            '{"team":"finance"}',
        ],
    )
    assert cli.main() == 0
    search = json.loads(capsys.readouterr().out)
    assert [h["document_id"] for h in search["hits"]] == ["dsid_one"]
    assert decisions.generations == 0 and decisions.contexts == [
        {"E1.1": documents[0].content}
    ]
    with TestClient(api.app) as client:
        response = client.post("/answer", json=REQUEST.model_dump())
        assert (
            client.post(
                "/answer", json={**REQUEST.model_dump(), "scope": {}}
            ).status_code
            == 422
        )
    assert response.status_code == 200
    answer = response.json()
    for trace in (search["trace"], answer["trace"]):
        assert trace["stop_reason"] == "sufficient_evidence"
        assert (
            trace["complexity"] == "simple"
            and trace["reviews"][0]["missing_tasks"] == []
        )
        assert (
            trace["reviews"][0]["sources_by_task"]["1"][0]["document_id"] == "dsid_one"
        )
        assert [(c["tool"], c["retry"]) for c in trace["tool_calls"]] == [
            ("hybrid", False)
        ]
    assert answer["status"] == "answered" and decisions.generations == 1
    cite = answer["claims"][0]["citations"][0]
    assert cite["document_id"] == "dsid_one"
    assert (
        documents[0].content[cite["start_offset"] : cite["end_offset"]] == cite["quote"]
    )


def test_complex_decomposition_recovers_missing_task_and_merges_evidence(db, decisions):
    decisions.initial = Plan(
        complexity="complex",
        searches=[
            {"query": "business receipts", "tool": "keyword"},
            {"query": "unindexedidentifier", "tool": "keyword"},
        ],
    )
    decisions.reviews = [
        decision([(1, "E1.1")], {"task": 2, "tool": "dense", "query": "tomato soup"}),
        decision([(1, "E1.1"), (2, "E2.1")]),
    ]
    request = AnswerRequest(
        question="What receipts are required and what meal is available?",
        method="agentic",
        top_k=2,
    )
    result = runtime.run_agent(
        db, request, AnswerScope(source_types=("slack", "gmail")), want_answer=False
    )
    assert [(c.task, c.tool, c.hits, c.retry) for c in result.trace.tool_calls] == [
        (1, "keyword", 1, False),
        (2, "keyword", 0, False),
        (2, "dense", 2, True),
    ]
    assert [r.missing_tasks for r in result.trace.reviews] == [[2], []]
    assert result.trace.stop_reason == "sufficient_evidence"
    assert {h.document_id for h in result.hits} == {"dsid_one", "dsid_two"}
    assert len(result.hits) == len({h.chunk_id for h in result.hits}) == 2


@pytest.mark.parametrize(
    ("mode", "reason", "count"),
    [
        ("duplicate", "no_new_query", 1),
        ("limited", "retry_limit", 2),
        ("bad_label", "invalid_model_output", 1),
        ("bad_task", "invalid_model_output", 1),
    ],
)
def test_retry_bound_validation_and_partial_answer(
    db, decisions, monkeypatch, mode, reason, count
):
    monkeypatch.setenv("RAG_AGENT_MAX_RETRIES", "1")
    next_query = {"task": 1, "query": "receipts", "tool": "keyword"}
    choices = {
        "duplicate": decision(
            followup={
                "task": 1,
                "query": " BUSINESS  travel RECEIPTS ",
                "tool": "hybrid",
            }
        ),
        "limited": decision(followup=next_query),
        "bad_label": decision([(1, "E99.1")]),
        "bad_task": decision(followup={**next_query, "task": 2}),
    }
    decisions.reviews = [choices[mode], decision(followup=next_query)]
    result = runtime.run_agent(db, REQUEST, SCOPE, want_answer=True)
    assert len(result.trace.tool_calls) == count and result.trace.stop_reason == reason
    if mode.startswith("bad"):
        assert result.answer.status == "abstained" and decisions.generations == 0
    else:
        assert (
            result.answer.status == "partial"
            and "incomplete_evidence" in result.answer.limitations
        )


def test_revoke_after_review_stops_before_using_derived_query(db, decisions):
    def revoke():
        db.execute("UPDATE documents SET metadata='{}' WHERE document_id='dsid_one'")
        return decision(
            followup={"task": 1, "query": "private derived name", "tool": "keyword"}
        )

    decisions.reviews = [revoke]
    result = runtime.run_agent(db, REQUEST, SCOPE, want_answer=True)
    assert (
        result.trace.stop_reason == "evidence_changed"
        and len(result.trace.tool_calls) == 1
    )
    assert (
        result.hits == []
        and result.answer.status == "abstained"
        and decisions.generations == 0
    )
    assert "private derived name" not in result.trace.model_dump_json()


def test_no_access_and_zero_budget_end_before_model_or_tool_dispatch(db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    denied = runtime.run_agent(
        db, REQUEST, AnswerScope(source_types=()), want_answer=True
    )
    monkeypatch.setenv("RAG_AGENT_MAX_COST_USD", "0")
    limited = runtime.run_agent(db, REQUEST, SCOPE, want_answer=True)
    assert (
        denied.trace.stop_reason == "no_access"
        and limited.trace.stop_reason == "cost_limit"
    )
    for result in (denied, limited):
        assert result.answer.status == "abstained"
        assert (
            not result.hits
            and not result.trace.tool_calls
            and not result.trace.model_calls
        )


def test_sql_deadline_cancels_statement_and_restores_connection(db):
    before = db.execute("SHOW statement_timeout").fetchone()
    store = tools.TimedSearchStore(db, Budget(AgentSettings(max_seconds=0.05)))
    with pytest.raises(RunStopped, match="time_limit"):
        store._execute("SELECT pg_sleep(2)", ())
    assert db.execute("SHOW statement_timeout").fetchone() == before


def test_retry_search_retains_source_document_and_metadata_filters(
    db, decisions, documents
):
    decisions.reviews = [
        decision(followup={"task": 1, "tool": "dense", "query": "lunch soup"}),
        decision(followup=None),
    ]
    result = runtime.run_agent(db, REQUEST, SCOPE, want_answer=False)
    assert [call.tool for call in result.trace.tool_calls] == ["hybrid", "dense"]
    assert decisions.contexts == [{"E1.1": documents[0].content}] * 2
    assert [hit.document_id for hit in result.hits] == ["dsid_one"]


def test_empty_initial_searches_and_no_retries_skip_review_and_generation(
    db, decisions, monkeypatch
):
    monkeypatch.setenv("RAG_AGENT_MAX_RETRIES", "0")
    decisions.initial = Plan(
        complexity="complex",
        searches=[
            {"query": "nonexistentfirst", "tool": "keyword"},
            {"query": "nonexistentsecond", "tool": "keyword"},
        ],
    )
    result = runtime.run_agent(db, REQUEST, SCOPE, want_answer=True)
    assert result.trace.stop_reason == "retry_limit" and not decisions.contexts
    assert decisions.generations == 0 and result.answer.status == "abstained"


def test_operational_error_logs_stage_without_private_content(
    db, decisions, monkeypatch, caplog
):
    from enterprise_agent.generation.chat import GenerationUnavailable

    def unavailable(*args):
        raise GenerationUnavailable("private upstream payload")

    monkeypatch.setattr(decisions, "review", unavailable)
    with (
        caplog.at_level("INFO", logger="enterprise_agent.agent.graph"),
        pytest.raises(GenerationUnavailable),
    ):
        runtime.run_agent(db, REQUEST, SCOPE, want_answer=False)
    entry = next(
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("agent_trace ")
    )
    trace = json.loads(entry.removeprefix("agent_trace "))
    assert trace["stop_reason"] == "service_error" and trace["last_stage"] == "review"
    assert (
        trace["error_type"] == "GenerationUnavailable" and len(trace["tool_calls"]) == 1
    )
    assert "private upstream payload" not in entry and "receipts" not in entry.lower()

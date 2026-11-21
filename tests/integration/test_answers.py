import json
import sys
from contextlib import nullcontext

import pytest

from enterprise_agent import cli
from enterprise_agent.generation import runtime
from enterprise_agent.generation.contracts import (
    AnswerDraft,
    AnswerRequest,
    Generation,
    GenerationInfo,
)
from enterprise_agent.generation.service import answer_question
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import AnswerScope, ChunkSettings
from enterprise_agent.retrieval.service import KeywordRetriever
from enterprise_agent.storage.search import SearchStore

pytestmark = pytest.mark.integration
SCOPE = AnswerScope(source_types=("slack",), document_ids=("dsid_one",))
REQUEST = AnswerRequest(
    question="Which business travel receipts are required?", method="keyword"
)


def receipts_draft():
    return Generation(
        outcome=AnswerDraft(
            coverage="complete",
            claims=[
                {
                    "kind": "fact",
                    "text": "Business travel receipts are required.",
                    "citations": ["E1.1"],
                }
            ],
            conflicts=[],
        ),
        info=GenerationInfo(provider="test", model="fixture", prompt_version="test"),
    )


def test_only_authorized_canonical_text_reaches_model_even_if_retriever_returns_other_hits(
    db, documents, classify_fixture
):
    ingest(db, documents, ChunkSettings(), limit=2, batch_size=2)
    store = SearchStore(db)
    allowed, denied = (
        KeywordRetriever(store).search(q, 1)[0] for q in ("receipts", "soup")
    )
    calls = []

    class UnreliableRetriever:
        def search(self, query, k, filters):
            calls.append((query, k, filters.scope))
            return [
                denied,
                allowed.model_copy(update={"content": "invented search result"}),
            ]

    def generate(question, passages):
        assert question == REQUEST.question
        assert passages == {"E1.1": documents[0].content}
        return receipts_draft()

    out = answer_question(
        REQUEST,
        scope=SCOPE,
        retriever=UnreliableRetriever(),
        store=store,
        generate=generate,
        classify=classify_fixture,
    )
    assert calls == [(REQUEST.question, REQUEST.top_k, SCOPE)]
    assert out.status == "answered"
    (cite,) = out.claims[0].citations
    assert documents[0].content[cite.start_offset : cite.end_offset] == cite.quote
    assert cite.document_id == allowed.document_id and cite.chunk_id == allowed.chunk_id


@pytest.mark.parametrize(
    "change", ["delete", "edit", "revoke_source", "revoke_metadata"]
)
def test_citations_checked_after_generation_against_current_document_and_policy(
    db, documents, change, classify_fixture
):
    ingest(db, documents, ChunkSettings(), limit=2, batch_size=2)
    store = SearchStore(db)
    query = REQUEST.model_copy(update={"metadata": {"team": "finance"}})

    def generate(_, passages):
        assert passages
        if change == "delete":
            db.execute("DELETE FROM documents WHERE document_id = 'dsid_one'")
        else:
            changes = {
                "edit": {"content": "No travel policy."},
                "revoke_source": {"source_type": "linear"},
                "revoke_metadata": {"metadata": {}},
            }
            ingest(
                db,
                [documents[0].model_copy(update=changes[change])],
                ChunkSettings(),
                limit=1,
                batch_size=1,
            )
        return receipts_draft()

    out = answer_question(
        query,
        scope=SCOPE,
        retriever=KeywordRetriever(store),
        store=store,
        generate=generate,
        classify=classify_fixture,
    )
    assert out.status == "abstained" and out.answer is None
    assert (
        "evidence_changed" in out.limitations
        and "receipts" not in out.model_dump_json()
    )


def test_no_hits_never_invokes_generator_or_discloses_reason_for_missing_document(
    db, documents
):
    ingest(db, documents, ChunkSettings(), limit=2, batch_size=2)
    store = SearchStore(db)
    both = []
    for query, scope in [
        (REQUEST, AnswerScope(source_types=())),
        (REQUEST.model_copy(update={"question": "missingterm"}), SCOPE),
    ]:
        out = answer_question(
            query,
            scope=scope,
            retriever=KeywordRetriever(store),
            store=store,
            generate=lambda *_: pytest.fail("unexpected model call"),
            classify=lambda *_: pytest.fail("unexpected verifier call"),
        )
        both.append(out.model_dump())
    assert both[0] == both[1] and both[0]["generation"] is None


@pytest.mark.parametrize("method", ["keyword", "dense", "hybrid"])
def test_answer_cli_all_retrievers_return_same_contract(
    db, documents, embedder, monkeypatch, capsys, method, classify_fixture
):
    from enterprise_agent.retrieval import service
    from enterprise_agent.storage.embeddings import index_embeddings

    ingest(db, documents, ChunkSettings(), limit=2, batch_size=2)
    index_embeddings(db, embedder, max_chunks=2, batch_size=2)
    monkeypatch.setenv("ANSWER_ACCESS_SCOPE", SCOPE.model_dump_json())
    monkeypatch.setattr(cli, "connect", lambda: nullcontext(db))
    monkeypatch.setattr(service, "configured_embedder", lambda: embedder)
    monkeypatch.setattr(runtime, "generate", lambda *_: receipts_draft())
    monkeypatch.setattr(runtime, "classify_local", classify_fixture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "enterprise-agent",
            "answer",
            REQUEST.question,
            "--method",
            method,
            "--top-k",
            "1",
        ],
    )
    assert cli.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert (
        out["status"] == "answered"
        and out["answer"] == "Business travel receipts are required."
    )
    assert out["claims"][0]["citations"][0]["document_id"] == "dsid_one"


def test_cli_reports_bad_trusted_policy_without_generating(db, monkeypatch, capsys):
    monkeypatch.setenv("ANSWER_ACCESS_SCOPE", '{"source_typess":["slack"]}')
    monkeypatch.setattr(cli, "connect", lambda: nullcontext(db))
    monkeypatch.setattr(sys, "argv", ["enterprise-agent", "answer", REQUEST.question])
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert "ANSWER_ACCESS_SCOPE" in captured.err and captured.out == ""

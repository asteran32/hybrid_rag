import pytest

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
                    "citations": [
                        {"passage": "E1", "quote": "travel receipts are required."}
                    ],
                }
            ],
            conflicts=[],
        ),
        info=GenerationInfo(provider="test", model="fixture", prompt_version="test"),
    )


def test_only_authorized_canonical_text_reaches_model_even_if_retriever_returns_other_hits(
    db, documents
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
        assert passages == {"E1": documents[0].content}
        return receipts_draft()

    out = answer_question(
        REQUEST,
        scope=SCOPE,
        retriever=UnreliableRetriever(),
        store=store,
        generate=generate,
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
    db, documents, change
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
        )
        both.append(out.model_dump())
    assert both[0] == both[1] and both[0]["generation"] is None

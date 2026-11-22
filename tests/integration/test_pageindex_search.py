import pytest

from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import AnswerScope, ChunkSettings, SearchFilter
from enterprise_agent.retrieval.pageindex import PageIndexRetriever
from enterprise_agent.retrieval.pageindex_selector import Selection, SelectorSettings
from enterprise_agent.storage.pageindex import PageIndexStore, build_pageindex

pytestmark = pytest.mark.integration


class PickFirst:
    def __init__(self):
        self.calls = []

    def select(self, question, stage, choices, limit, timeout):
        self.calls.append((stage, choices))
        return Selection([1], {"prompt_tokens": 10, "completion_tokens": 3})


def test_scope_is_applied_before_model_and_spans_cross_real_chunks(db, documents):
    pytest.importorskip("pageindex")
    doc = documents[0].model_copy(
        update={"content": "First.\nInvoice ledgers must balance by Friday."}
    )
    ingest(
        db,
        [doc, documents[1]],
        ChunkSettings(size=17, overlap=3),
        limit=None,
        batch_size=2,
    )
    build_pageindex(db)
    selector = PickFirst()
    retriever = PageIndexRetriever(PageIndexStore(db), SelectorSettings(), selector)
    filters = SearchFilter(
        scope=AnswerScope(source_types=["slack"]), metadata={"team": "finance"}
    )
    denied = SearchFilter(scope=AnswerScope(source_types=["slack"], document_ids=[]))
    assert retriever.search("invoice", 5, denied) == []
    assert not selector.calls
    with pytest.raises(ValueError, match="scope"):
        retriever.search("invoice")

    def pick_invoice(question, stage, choices, limit, timeout):
        selector.calls.append((stage, choices))
        assert all("tomato" not in c and "Café" not in c for c in choices)
        return Selection([2] if stage == "evidence" else [1], {})

    selector.select = pick_invoice
    hits, trace = retriever.search_with_trace("invoice", 10, filters)
    assert trace.candidates == [doc.document_id]
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert len(hits) > 1
    parts = [doc.content[a:b] for h in hits for a, b in h.matched_spans]
    assert "".join(parts) == "Invoice ledgers must balance by Friday."
    # An edited parent suppresses stale outlines before any model call.
    ingest(
        db,
        [doc.model_copy(update={"content": "replaced"})],
        ChunkSettings(),
        limit=1,
        batch_size=1,
    )
    calls = len(selector.calls)
    assert retriever.search("invoice", 10, filters) == []
    assert len(selector.calls) == calls

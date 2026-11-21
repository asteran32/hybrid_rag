import pytest

from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import AnswerScope, ChunkSettings, SearchFilter
from enterprise_agent.retrieval.service import (
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
)
from enterprise_agent.storage.embeddings import index_embeddings
from enterprise_agent.storage.search import SearchStore

pytestmark = pytest.mark.integration


def test_document_scope_applies_before_ranking_and_again_at_evidence_lookup(
    db, documents, embedder
):
    second = documents[1].model_copy(
        update={
            "source_type": "slack",
            "content": "Travel receipts include tomato soup.",
            "metadata": {"team": "finance"},
        }
    )
    ingest(db, [documents[0], second], ChunkSettings(), limit=2, batch_size=2)
    index_embeddings(db, embedder, max_chunks=2, batch_size=2)
    store = SearchStore(db)
    keyword, dense = KeywordRetriever(store), DenseRetriever(store, embedder)
    allowed = AnswerScope(source_types=("slack",), document_ids=(second.document_id,))
    scope = SearchFilter(scope=allowed, metadata={"team": "finance"})
    all_ids = [h.chunk_id for h in keyword.search("receipts", 2)]
    assert len(all_ids) == 2
    for retriever in (keyword, dense, HybridRetriever(keyword, dense, candidates=1)):
        (best,) = retriever.search("receipts", 1, scope)
        assert best.document_id == second.document_id
        for denied in (
            SearchFilter(scope=AnswerScope(source_types=())),
            SearchFilter(scope=AnswerScope(source_types=("slack",), document_ids=())),
            SearchFilter(scope=allowed, source_type="gmail"),
            SearchFilter(scope=allowed, metadata={"team": "sales"}),
        ):
            assert retriever.search("receipts", 1, denied) == []
            assert store.load_evidence(all_ids, denied) == {}
    evidence = store.load_evidence([*all_ids, "missing"], scope)
    assert list(evidence) == [best.chunk_id]
    assert evidence[best.chunk_id].content == second.content
    with pytest.raises(ValueError, match="scope"):
        store.load_evidence(all_ids, SearchFilter())
    db.execute(
        "UPDATE documents SET content = 'changed' WHERE document_id = %s",
        (second.document_id,),
    )
    assert store.load_evidence(all_ids, scope) == {}

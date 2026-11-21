import argparse
import json

import pytest

from enterprise_agent import cli
from enterprise_agent.embeddings import ModelSpec
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import ChunkSettings, SearchFilter
from enterprise_agent.retrieval.service import (
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
)
from enterprise_agent.storage.db import build_lexical_index
from enterprise_agent.storage.embeddings import index_embeddings
from enterprise_agent.storage.search import SearchStore

pytestmark = pytest.mark.integration


def test_search_filters_cache_and_model_isolation(db, documents, embedder):
    settings = ChunkSettings()
    ingest(db, documents, settings, limit=None, batch_size=2)
    store = SearchStore(db)
    with pytest.raises(ValueError, match="no indexed"):
        DenseRetriever(store, embedder)
    build_lexical_index(db)
    keyword = KeywordRetriever(store)
    (h,) = keyword.search("receipts", 1, SearchFilter(metadata={"team": "finance"}))
    assert (h.title, h.document_id, h.source_type, h.metadata) == (
        documents[0].title,
        documents[0].document_id,
        "slack",
        documents[0].metadata,
    )
    assert keyword.search("receipts", 1, SearchFilter(source_type="gmail")) == []
    assert keyword.search("expense", 1)[0].chunk_id == h.chunk_id  # title indexing
    assert index_embeddings(db, embedder, max_chunks=1, batch_size=5) == 1
    assert index_embeddings(db, embedder, max_chunks=10, batch_size=2) == 1
    assert index_embeddings(db, embedder, max_chunks=10, batch_size=2) == 0
    assert len(embedder.calls) == 2
    dense = DenseRetriever(store, embedder)
    assert dense.search("meal", 1)[0].document_id == "dsid_two"
    assert (
        dense.search("meal", 1, SearchFilter(source_type="slack"))[0].document_id
        == "dsid_one"
    )
    hybrid = HybridRetriever(keyword, dense, candidates=2)
    (best,) = hybrid.search("lunch", 1, SearchFilter(metadata={}))
    assert (
        best.component_ranks == {"keyword": 1, "dense": 1}
        and best.document_id == "dsid_two"
    )
    edited = documents[1].model_copy(update={"content": "Hot tomato soup"})
    ingest(db, [edited], settings, limit=1, batch_size=1)
    assert index_embeddings(db, embedder, max_chunks=10, batch_size=10) == 1
    assert embedder.calls[-1] == [edited.content]
    embedder.spec = ModelSpec(provider="test", model="other", dimensions=2)
    with pytest.raises(ValueError, match="no indexed"):
        DenseRetriever(store, embedder)


def test_eval_snapshot_coverage_and_cli_report(
    db, documents, embedder, monkeypatch, tmp_path
):
    ingest(db, documents, ChunkSettings(), limit=2, batch_size=1)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"question_id":"q","question":"lunch","expected_doc_ids":["dsid_two"]}\n'
    )
    monkeypatch.setattr(cli, "configured_embedder", lambda: embedder)
    args = argparse.Namespace(
        questions=path,
        output=tmp_path / "out.json",
        dataset_version="fixture-v1",
        limit=1,
        candidates=10,
        rrf_constant=60,
        chunk_depth=10,
    )
    index_embeddings(db, embedder, max_chunks=1, batch_size=2)
    with pytest.raises(ValueError, match="1/2"):
        cli.run_evaluation(db, args)
    index_embeddings(db, embedder, max_chunks=1, batch_size=2)
    cli.run_evaluation(db, args)
    report = json.loads(args.output.read_text())
    assert report["configuration"]["snapshot"]["chunks"] == 2
    assert report["configuration"]["questions_with_any_gold_in_corpus"] == 1
    assert set(report["results"]) == {"C0", "C1", "C2"}
    assert all(r["summary"]["recall@1"] == 1 for r in report["results"].values())
    original = path.read_bytes()
    args.output = path
    with pytest.raises(ValueError, match="separate file"):
        cli.run_evaluation(db, args)
    assert path.read_bytes() == original


def test_interrupted_embedding_batch_can_resume(db, documents, embedder):
    ingest(db, documents, ChunkSettings(), limit=2, batch_size=2)
    original = embedder.embed

    def interrupted(texts):
        if len(embedder.calls) == 1:
            return [[0.0]]
        return original(texts)

    embedder.embed = interrupted
    with pytest.raises(ValueError, match="dimensions"):
        index_embeddings(db, embedder, max_chunks=2, batch_size=1)
    embedder.embed = original
    assert index_embeddings(db, embedder, max_chunks=2, batch_size=1) == 1

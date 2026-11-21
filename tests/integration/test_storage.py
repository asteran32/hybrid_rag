import pytest

from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import ChunkSettings
from enterprise_agent.storage.db import build_lexical_index, initialize

pytestmark = pytest.mark.integration


def test_reingest_and_rechunk(db, documents):
    initialize(db)
    build_lexical_index(db)
    settings = ChunkSettings(size=14, overlap=3)
    args = {"limit": 1, "batch_size": 2}
    first = ingest(db, documents, settings, **args)
    assert first.seen == first.written == 1 and first.chunks_written > 1
    doc = documents[0]
    row = db.execute("SELECT * FROM documents").fetchone()
    assert row["metadata"] == doc.metadata and row["title"] == doc.title
    assert ingest(db, documents, settings, **args).unchanged == 1
    assert all(
        row["document_id"] == doc.document_id
        for row in db.execute("SELECT document_id FROM chunks")
    )
    assert (
        ingest(
            db,
            [doc.model_copy(update={"source_path": "gmail/alias"})],
            settings,
            **args,
        ).errors
        == 1
    )
    empty = doc.model_copy(update={"content": ""})
    assert ingest(db, [empty], settings, **args).written == 1
    assert db.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 0
    assert db.execute("SELECT content FROM documents").fetchone()["content"] == ""


def test_overlap_change_preserves_vectors_for_unchanged_windows(
    db, documents, embedder
):
    from enterprise_agent.storage.embeddings import index_embeddings

    doc = documents[0].model_copy(update={"content": "abcdefghijklmnop"})
    for overlap, new_vectors in [(0, 4), (2, 3), (0, 0)]:
        ingest(db, [doc], ChunkSettings(size=4, overlap=overlap), limit=1, batch_size=1)
        assert (
            index_embeddings(db, embedder, max_chunks=20, batch_size=2) == new_vectors
        )
        rows = db.execute(
            "SELECT position, content FROM chunks ORDER BY position"
        ).fetchall()
        assert [r["content"] for r in rows] == [
            doc.content[i : i + 4] for i in range(0, 13, 4 - overlap)
        ]
        assert [r["position"] for r in rows] == list(range(len(rows)))


def test_invalid_record_does_not_undo_valid_batch_members(db, documents):
    bad = documents[0].model_copy(update={"content": "invalid\x00content"})
    result = ingest(db, [bad, documents[1]], ChunkSettings(), limit=None, batch_size=8)
    assert (result.seen, result.errors, result.written) == (2, 1, 1)
    assert (
        db.execute("SELECT document_id FROM documents").fetchone()["document_id"]
        == "dsid_two"
    )

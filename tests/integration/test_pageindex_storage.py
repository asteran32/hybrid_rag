import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from psycopg import sql

from enterprise_agent.ingestion.normalize import normalize
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import ChunkSettings
from enterprise_agent.storage import pageindex
from enterprise_agent.storage.db import connect

pytestmark = pytest.mark.integration


def test_concurrent_builders_publish_one_tree(db, documents, monkeypatch):
    pytest.importorskip("pageindex")
    ingest(db, documents[:1], ChunkSettings(), limit=None, batch_size=1)
    schema = sql.Identifier(
        db.execute("SELECT current_schema() AS name").fetchone()["name"]
    )
    dsn = db.info.dsn
    barrier = Barrier(2)
    build = pageindex.build_tree

    def synchronized(*args):
        barrier.wait(timeout=10)
        return build(*args)

    def worker():
        with connect(dsn) as other:
            other.execute(sql.SQL("SET search_path TO {},public").format(schema))
            return pageindex.build_pageindex(other)

    monkeypatch.setattr(pageindex, "build_tree", synchronized)
    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [pool.submit(worker) for _ in range(2)]
        stats = [task.result() for task in workers]
    assert sorted((s["written"], s["unchanged"], s["errors"]) for s in stats) == [
        (0, 1, 0),
        (1, 0, 0),
    ]


def test_rebuild_skips_current_trees_and_recovers_field_spans(db, tmp_path):
    pytest.importorskip("pageindex")
    source = tmp_path / "generated_data/sources/gmail"
    source.mkdir(parents=True)
    raw = {
        "dataset_doc_uuid": "dsid_x",
        "content_field_names": ["body", "notes"],
        "body": "proposal",
        "notes": "approval",
    }
    doc = normalize(raw, "gmail", "gmail/one.json")
    empty = doc.model_copy(
        update={
            "document_id": "dsid_y",
            "source_path": "gmail/empty.txt",
            "content": "",
        }
    )
    ingest(db, [doc, empty], ChunkSettings(), limit=None, batch_size=2)
    assert pageindex.build_pageindex(db) == {
        "written": 2,
        "unchanged": 0,
        "empty": 1,
        "errors": 0,
    }
    assert pageindex.build_pageindex(db, tmp_path)["errors"] == 1
    (source / "one.json").write_text(json.dumps(raw))
    assert pageindex.build_pageindex(db, tmp_path)["written"] == 1
    assert pageindex.build_pageindex(db)["unchanged"] == 2
    tree = db.execute(
        "SELECT nodes FROM pageindex_documents WHERE document_id='dsid_x'"
    ).fetchone()["nodes"]
    assert [n["title"] for n in tree] == ["dsid_x", "body", "notes"]
    edited = doc.model_copy(update={"content": "updated"})
    ingest(db, [edited], ChunkSettings(), limit=None, batch_size=1)
    assert pageindex.build_pageindex(db, tmp_path)["errors"] == 1
    assert pageindex.build_pageindex(db)["written"] == 1
    db.execute("DELETE FROM documents WHERE document_id='dsid_x'")
    assert (
        db.execute("SELECT count(*) AS n FROM pageindex_documents").fetchone()["n"] == 1
    )


def test_stale_builder_cannot_publish_over_new_content(db, documents, monkeypatch):
    doc = documents[0]
    settings = ChunkSettings()
    ingest(db, [doc], settings, limit=None, batch_size=1)

    def racing_build(*args):
        ingest(
            db,
            [doc.model_copy(update={"content": "newer text"})],
            settings,
            limit=None,
            batch_size=1,
        )
        return []

    monkeypatch.setattr(pageindex, "build_tree", racing_build)
    assert pageindex.build_pageindex(db)["errors"] == 1
    assert (
        db.execute("SELECT count(*) AS n FROM pageindex_documents").fetchone()["n"] == 0
    )

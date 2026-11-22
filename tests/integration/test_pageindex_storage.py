import json

import pytest

from enterprise_agent.ingestion.normalize import normalize
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import ChunkSettings
from enterprise_agent.storage import pageindex

pytestmark = pytest.mark.integration


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

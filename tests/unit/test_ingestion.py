import json
from pathlib import Path

import pytest

from enterprise_agent.ingestion.chunking import chunk_document
from enterprise_agent.ingestion.loader import LoadError, load_documents
from enterprise_agent.ingestion.normalize import normalize
from enterprise_agent.models import ChunkSettings


def test_normalize_labeled_content_and_metadata():
    raw = {
        "dataset_doc_uuid": "dsid_ab",
        "title_field_name": "subject",
        "subject": "Tax",
        "content_field_names": ["body", "comments"],
        "body": "Receipt",
        "comments": [{"z": 1, "a": 2}],
        "timestamp": "bad-date",
        "count": "7",
    }
    doc = normalize(raw, "gmail", "gmail/thread.json")
    assert (doc.document_id, doc.source_type, doc.title) == ("dsid_ab", "gmail", "Tax")
    assert doc.content == 'body\nReceipt\n\ncomments\n{"a": 2, "z": 1}'
    assert doc.metadata == {"timestamp": "bad-date", "count": "7", "subject": "Tax"}
    assert normalize(dict(reversed(raw.items())), "gmail", doc.source_path) == doc


def test_missing_and_malformed_labels_do_not_invent_values():
    doc = normalize(
        {
            "dataset_doc_uuid": "dsid_x",
            "title_field_name": [],
            "content_field_names": 4,
            "messages": "existing",
            "participants": None,
        },
        "slack",
        "slack/x",
    )
    assert (
        doc.title is None
        and doc.content == "existing"
        and doc.metadata == {"participants": None}
    )
    with pytest.raises(ValueError):
        normalize({"body": "no ID"}, "slack", "slack/x")
    with pytest.raises(ValueError):
        normalize(
            {"dataset_doc_uuid": "dsid_x", "expected_doc_ids": []}, "slack", "slack/x"
        )


def test_loader_round_robin_and_errors(tmp_path: Path):
    for source in ("gmail", "slack"):
        d = tmp_path / "generated_data/sources" / source
        d.mkdir(parents=True)
        (d / "a.json").write_text(
            json.dumps({"dataset_doc_uuid": f"dsid_{source}", "body": source})
        )
        (d / "b.json").write_text("{")
    (tmp_path / "questions.jsonl").write_text('{"gold_answer":"excluded"}')
    results = list(load_documents(tmp_path))
    assert [r.source_type for r in results[:2]] == ["gmail", "slack"]
    assert all(isinstance(r, LoadError) for r in results[2:]) and len(results) == 4


def test_text_export_flat_slice(tmp_path):
    (tmp_path / "dsid_beef__memo.txt").write_text(
        "Quarterly memo\n\nQuarterly results."
    )
    (doc,) = load_documents(tmp_path, "confluence")
    assert (doc.document_id, doc.title, doc.content) == (
        "dsid_beef",
        "Quarterly memo",
        "Quarterly results.",
    )


def test_parquet_export(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "documents.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "doc_id": "dsid_test",
                    "source_type": "jira",
                    "content": "details",
                    "title": "ticket",
                    "timestamp": "raw",
                }
            ]
        ),
        path,
    )
    (doc,) = load_documents(path)
    assert doc.metadata == {"timestamp": "raw"} and doc.document_id == "dsid_test"


def test_chunks_are_stable_and_traceable(documents):
    doc = documents[0].model_copy(update={"content": "한글abcde"})
    settings = ChunkSettings(size=4, overlap=1)
    chunks = chunk_document(doc, settings)
    assert [c.content for c in chunks] == ["한글ab", "bcde"]
    assert [(c.start_offset, c.end_offset, c.position) for c in chunks] == [
        (0, 4, 0),
        (3, 7, 1),
    ]
    assert all(c.document_id == doc.document_id for c in chunks)
    assert (
        len({c.chunk_id for c in chunks}) == 2
        and chunk_document(doc, settings) == chunks
    )
    assert len(chunk_document(doc, ChunkSettings(size=8, overlap=1))) == 1
    assert chunk_document(doc.model_copy(update={"content": " \n"}), settings) == []
    with pytest.raises(ValueError):
        ChunkSettings(size=2, overlap=2)


def test_source_selector_never_relabels_another_source(tmp_path):
    (tmp_path / "slack").mkdir()
    with pytest.raises(ValueError, match="missing"):
        list(load_documents(tmp_path, "gmail"))
    with pytest.raises(ValueError, match="conflicts"):
        list(load_documents(tmp_path / "slack", "gmail"))


def test_meeting_summary_without_title_remains_body():
    doc = normalize(
        {"dataset_doc_uuid": "dsid_call", "summary": "Topics discussed"},
        "fireflies",
        "fireflies/call.json",
    )
    assert doc.title is None and doc.content == "Topics discussed"

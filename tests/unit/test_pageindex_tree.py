import pytest

from enterprise_agent.ingestion.normalize import normalize_with_sections
from enterprise_agent.ingestion.pageindex import BLOCK_CHARS, build_tree


def test_upstream_tree_preserves_native_sections_and_repeated_unicode_spans():
    pytest.importorskip("pageindex")
    body = (
        "## Scope\r\nsame 🌲\r\n~~~\r\n# literal\r\n~~~\r\n### Steps\nsame 🌲\n"
        + "長" * 2200
    )
    doc, sections = normalize_with_sections(
        {
            "dataset_doc_uuid": "dsid_tree",
            "content_field_names": ["body", "notes"],
            "body": body,
            "notes": "same 🌲",
        },
        "confluence",
        "confluence/tree.json",
    )
    nodes = build_tree(doc, sections)
    titles = {n["title"]: n for n in nodes}
    assert titles["Steps"]["parent"] == titles["Scope"]["id"]
    assert "literal" not in titles
    spans = [(n["start"], n["end"]) for n in nodes if n["start"] < n["end"]]
    assert all(b - a <= BLOCK_CHARS for a, b in spans)
    recovered = "".join(doc.content[a:b] for a, b in spans)
    assert "".join(recovered.split()) == "".join(doc.content.split())
    assert build_tree(doc, sections) == nodes


def test_short_and_empty_document(documents):
    pytest.importorskip("pageindex")
    doc = documents[0]
    tree = build_tree(doc)
    assert [n["title"] for n in tree] == [doc.title, "Content"]
    assert tree[1]["end"] == len(doc.content)
    assert build_tree(doc.model_copy(update={"content": "  "})) == []

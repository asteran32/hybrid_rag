import pytest

from enterprise_agent.models import EvidenceChunk, PageIndexHit, SearchHit
from enterprise_agent.validation.citations import (
    PassageRef,
    resolve_citation,
    select_passages,
)


def chunk(name="a", text="α🌿 승인. 승인. 내일.", offset=11):
    return EvidenceChunk(
        chunk_id=name,
        document_id=f"dsid_{name}",
        start_offset=offset,
        end_offset=offset + len(text),
        position=1,
        content=text,
        source_type="slack",
        source_path=f"slack/{name}.json",
        title=None,
    )


def hit(item):
    return SearchHit(
        **item.model_dump(), score=1, rank=1, component_ranks={}, metadata={}
    )


def test_only_canonical_ranked_distinct_evidence_fits_in_budget():
    a, b, denied = chunk(), chunk("b"), chunk("denied")
    forged = hit(a).model_copy(update={"content": "untrusted search result"})
    wrong_pair = hit(b).model_copy(update={"document_id": "dsid_other"})
    passages = select_passages(
        [hit(denied), wrong_pair, forged, forged, hit(b)], {"a": a, "b": b}, 11
    )
    assert [(p.label, p.chunk, p.text) for p in passages.values()] == [
        ("E1.1", a, a.content[:5]),
        ("E2.1", b, b.content[:6]),
    ]
    with pytest.raises(ValueError, match="budget"):
        select_passages([], {}, 0)


def test_citations_locate_repeated_text_in_unicode_document_coordinates():
    item = chunk(text="🌿 보고. 승인. 승인. 내일.")
    current = {"a": item}
    passages = select_passages([hit(item)], current, 100)
    ref = PassageRef("E1.2")
    citation = resolve_citation(ref, passages, current)
    assert citation.document_id == item.document_id and citation.chunk_id == "a"
    start = item.start_offset + len("🌿 보고. ")
    assert citation.start_offset == start and citation.end_offset == start + len(
        "승인."
    )
    assert citation.quote == "승인." and citation.source_path == item.source_path
    repeated = resolve_citation(PassageRef("E1.3"), passages, current)
    assert (
        repeated.quote == citation.quote
        and repeated.start_offset == citation.end_offset + 1
    )
    for label in ["dsid_a", "E9.1", " ", item.content]:
        assert resolve_citation(PassageRef(label), passages, current) is None
    assert resolve_citation(ref, passages, {}) is None
    assert resolve_citation(ref, passages, {"a": chunk(text="changed")}) is None
    truncated = select_passages([hit(item)], current, 2)
    assert resolve_citation(ref, truncated, current) is None


def test_many_short_lines_have_bounded_labels_and_preserve_source_slices():
    item = chunk(text="\n".join(f"한 줄 {i}." for i in range(100)) + "\n")
    current = {"a": item}
    passages = select_passages([hit(item)], current, 1000)
    assert 1 <= len(passages) <= 8
    for label, passage in passages.items():
        cite = resolve_citation(PassageRef(label), passages, current)
        assert (
            item.content[
                cite.start_offset - item.start_offset : cite.end_offset
                - item.start_offset
            ]
            == passage.text
        )


def test_pageindex_selected_tail_survives_budget_and_fresh_read():
    item = chunk(text="old. " * 40 + "🌿 Friday.")
    selected = PageIndexHit(
        **hit(item).model_dump(), matched_spans=[(item.end_offset - 9, item.end_offset)]
    )
    current = {"a": item}
    passages = select_passages([selected], current, 9)
    citation = resolve_citation(PassageRef("E1.1"), passages, current)
    assert (citation.quote, citation.start_offset) == (
        "🌿 Friday.",
        item.end_offset - 9,
    )
    changed = {"a": item.model_copy(update={"content": "changed"})}
    assert select_passages([selected], changed, 9) == {}
    assert resolve_citation(PassageRef("E1.1"), passages, changed) is None

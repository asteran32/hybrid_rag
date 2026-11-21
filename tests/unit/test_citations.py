import pytest

from enterprise_agent.models import EvidenceChunk, SearchHit
from enterprise_agent.validation.citations import (
    QuoteRef,
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
        ("E1", a, a.content[:5]),
        ("E2", b, b.content[:6]),
    ]
    with pytest.raises(ValueError, match="budget"):
        select_passages([], {}, 0)


def test_citation_matches_unique_visible_text_in_unicode_document_coordinates():
    item = chunk()
    current = {"a": item}
    passages = select_passages([hit(item)], current, 100)
    ref = QuoteRef(passage="E1", quote="🌿 승인.")
    citation = resolve_citation(ref, passages, current)
    assert citation.document_id == item.document_id and citation.chunk_id == "a"
    assert citation.start_offset == 12 and citation.end_offset == 12 + len(ref.quote)
    assert citation.quote == ref.quote and citation.source_path == item.source_path
    for label, quote in [
        ("dsid_a", ref.quote),
        ("E9", ref.quote),
        ("E1", "승인."),
        ("E1", " "),
    ]:
        assert (
            resolve_citation(QuoteRef(passage=label, quote=quote), passages, current)
            is None
        )
    assert resolve_citation(ref, passages, {}) is None
    assert resolve_citation(ref, passages, {"a": chunk(text="changed")}) is None
    truncated = select_passages([hit(item)], current, 2)
    assert resolve_citation(ref, truncated, current) is None

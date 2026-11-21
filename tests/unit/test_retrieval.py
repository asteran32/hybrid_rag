from types import SimpleNamespace

import pytest

from enterprise_agent.embeddings import ModelSpec, OpenAIEmbeddings, checked_vectors
from enterprise_agent.models import SearchHit
from enterprise_agent.retrieval.fusion import reciprocal_rank_fusion


def hit(cid: str, doc: str = "dsid_x") -> SearchHit:
    return SearchHit(
        chunk_id=cid,
        document_id=doc,
        start_offset=0,
        end_offset=4,
        position=0,
        title=None,
        metadata={},
        content="test",
        source_type="gmail",
        source_path="gmail/test",
        rank=1,
        score=1,
        component_ranks={},
    )


def test_rrf_independent_ranks_deduplicated_and_ties():
    a, b, c = hit("a"), hit("b"), hit("c")
    ranked = reciprocal_rank_fusion(
        {"keyword": [a, b, b], "dense": [c, b]}, k=3, constant=6
    )
    assert [h.chunk_id for h in ranked] == ["b", "a", "c"]
    assert ranked[0].score == pytest.approx(2 / 8)
    assert ranked[0].component_ranks == {"keyword": 2, "dense": 2}
    assert a.rank == 1 and a.component_ranks == {}
    assert all(h.document_id == "dsid_x" for h in ranked)
    assert reciprocal_rank_fusion({}, 1) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({}, 1, -1)


@pytest.mark.parametrize("invalid", [[], [[0.1]], [[0, 0]], [[float("inf"), 1]]])
def test_invalid_embeddings(embedder, invalid):
    embedder.embed = lambda _: invalid
    with pytest.raises(ValueError):
        checked_vectors(embedder, ["a"])


def test_openai_adapter_orders_by_input_index(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-value")
    spec = ModelSpec(provider="openai", model="text-embedding-3-small", dimensions=2)
    adapter = OpenAIEmbeddings(spec)

    def create(**kwargs):
        assert kwargs == {"model": spec.model, "dimensions": 2, "input": ["a", "b"]}
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[1, 0]),
                SimpleNamespace(index=0, embedding=[0, 1]),
            ]
        )

    monkeypatch.setattr(adapter.client.embeddings, "create", create)
    assert checked_vectors(adapter, ["a", "b"]) == [[0, 1], [1, 0]]
    with pytest.raises(ValueError):
        ModelSpec(provider="openai", model=spec.model, dimensions=2000)

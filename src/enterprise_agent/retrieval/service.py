from typing import Protocol

from enterprise_agent.embeddings import Embedder, checked_vectors, configured_embedder
from enterprise_agent.models import SearchFilter, SearchHit
from enterprise_agent.retrieval.fusion import reciprocal_rank_fusion
from enterprise_agent.storage.search import SearchStore


class Retriever(Protocol):
    def search(
        self, query: str, k: int = 10, filters: SearchFilter | None = None
    ) -> list[SearchHit]: ...


def _validate(query: str, k: int) -> None:
    if not query.strip() or k < 1:
        raise ValueError("search needs a nonempty query and positive top-k")


class KeywordRetriever:
    def __init__(self, store: SearchStore):
        self.store = store

    def search(
        self, query: str, k: int = 10, filters: SearchFilter | None = None
    ) -> list[SearchHit]:
        _validate(query, k)
        return self.store.keyword(query, k, filters)


class DenseRetriever:
    def __init__(self, store: SearchStore, provider: Embedder):
        store.require_embeddings(provider.spec.key)
        self.store = store
        self.provider = provider

    def search(
        self, query: str, k: int = 10, filters: SearchFilter | None = None
    ) -> list[SearchHit]:
        _validate(query, k)
        (vector,) = checked_vectors(self.provider, [query])
        return self.store.vector(vector, self.provider.spec.key, k, filters)


class HybridRetriever:
    def __init__(
        self,
        keyword: Retriever,
        dense: Retriever,
        *,
        candidates: int = 50,
        constant: int = 60,
    ):
        if candidates < 1 or constant < 0:
            raise ValueError("candidates must be positive and RRF constant nonnegative")
        self.keyword, self.dense = keyword, dense
        self.candidates, self.constant = candidates, constant

    def search(
        self, query: str, k: int = 10, filters: SearchFilter | None = None
    ) -> list[SearchHit]:
        _validate(query, k)
        depth = max(k, self.candidates)
        ranks = {
            "keyword": self.keyword.search(query, depth, filters),
            "dense": self.dense.search(query, depth, filters),
        }
        return reciprocal_rank_fusion(ranks, k, self.constant)


def build_retriever(
    store: SearchStore, method: str, *, candidates: int, constant: int
) -> Retriever:
    if method == "pageindex":
        from enterprise_agent.retrieval.pageindex import PageIndexRetriever
        from enterprise_agent.retrieval.pageindex_selector import SelectorSettings
        from enterprise_agent.storage.pageindex import PageIndexStore

        return PageIndexRetriever(
            PageIndexStore(store.db), SelectorSettings.configured()
        )
    if method == "keyword":
        return KeywordRetriever(store)
    if method not in {"dense", "hybrid"}:
        raise ValueError(f"unknown retrieval method: {method}")
    dense = DenseRetriever(store, configured_embedder())
    if method == "dense":
        return dense
    return HybridRetriever(
        KeywordRetriever(store), dense, candidates=candidates, constant=constant
    )

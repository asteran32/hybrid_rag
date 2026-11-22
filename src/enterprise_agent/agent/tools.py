import os
from decimal import Decimal, InvalidOperation
from functools import cached_property

import psycopg

from enterprise_agent.agent.budget import Budget, RunStopped
from enterprise_agent.agent.contracts import ModelCall, Tool
from enterprise_agent.embeddings import Embedder, OpenAIEmbeddings, configured_embedder
from enterprise_agent.models import SearchEvidence, SearchFilter, SearchHit
from enterprise_agent.retrieval.service import (
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
)
from enterprise_agent.storage.search import PROJECTION, SearchStore


class TimedSearchStore(SearchStore):
    """Apply the request deadline to each SQL statement, including evidence re-reads."""

    evidence_projection = PROJECTION
    evidence_model = SearchEvidence

    def __init__(self, db: psycopg.Connection, budget: Budget):
        super().__init__(db)
        self.budget = budget

    def _execute(self, query: str, params: list | tuple):
        self.budget.remaining()
        try:
            with self.db.transaction():
                previous = self.db.execute(
                    "SELECT setting::bigint AS ms FROM pg_settings WHERE name='statement_timeout'"
                ).fetchone()["ms"]
                ms = max(1, int(self.budget.remaining() * 1000))
                ms = min(ms, previous or 2147483647)
                self.db.execute(
                    "SELECT set_config('statement_timeout', %s, true)", (f"{ms}ms",)
                )
                cursor = self.db.execute(query, params)
                self.db.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (f"{previous}ms",),
                )
        except psycopg.errors.QueryCanceled:
            raise RunStopped("time_limit") from None
        self.budget.remaining()
        return cursor


class MeteredEmbeddings:
    def __init__(self, provider: OpenAIEmbeddings, budget: Budget):
        self.provider, self.spec, self.budget = provider, provider.spec, budget
        prices = {
            "text-embedding-3-small": "0.02",
            "text-embedding-3-large": "0.13",
            "text-embedding-ada-002": "0.10",
        }
        price = os.getenv("RAG_AGENT_EMBEDDING_USD_PER_MILLION") or prices.get(
            self.spec.model
        )
        if price is None:
            raise ValueError(
                "set RAG_AGENT_EMBEDDING_USD_PER_MILLION for this embedding model"
            )
        try:
            self.rate = Decimal(price)
        except InvalidOperation:
            raise ValueError("invalid RAG_AGENT_EMBEDDING_USD_PER_MILLION") from None
        if not self.rate.is_finite() or self.rate <= 0:
            raise ValueError(
                "RAG_AGENT_EMBEDDING_USD_PER_MILLION must be finite and positive"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        inputs = sum(len(t.encode("utf-8")) + 8 for t in texts)
        call = ModelCall(
            stage="embed",
            model=self.spec.model,
            input_tokens=inputs,
            output_tokens=0,
            cost_usd=0,
        )
        self.budget.reserve(call, self.rate, Decimal(0))
        vectors, tokens = self.budget.wait_for(
            lambda: self.provider.embed_with_usage(
                texts, timeout=self.budget.remaining()
            )
        )
        if tokens is not None:
            self.budget.settle(call, tokens, 0, self.rate, Decimal(0))
        self.budget.remaining()
        return vectors


class TimedEmbeddings:
    def __init__(self, provider: Embedder, budget: Budget):
        self.provider, self.spec, self.budget = provider, provider.spec, budget

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.budget.wait_for(lambda: self.provider.embed(texts))


class SearchTools:
    def __init__(
        self, store: SearchStore, budget: Budget, candidates: int, constant: int
    ):
        self.store, self.budget = store, budget
        self.candidates, self.constant = candidates, constant
        self.keyword = KeywordRetriever(store)

    @cached_property
    def dense(self) -> DenseRetriever:
        self.budget.remaining()
        provider = self.budget.wait_for(configured_embedder)
        if isinstance(provider, OpenAIEmbeddings):
            provider = MeteredEmbeddings(provider, self.budget)
        else:
            provider = TimedEmbeddings(provider, self.budget)
        return DenseRetriever(self.store, provider)

    def search(
        self, method: Tool, query: str, k: int, filters: SearchFilter
    ) -> list[SearchHit]:
        self.budget.remaining()
        if method == "keyword":
            return self.keyword.search(query, k, filters)
        if method == "dense":
            return self.dense.search(query, k, filters)
        return HybridRetriever(
            self.keyword, self.dense, candidates=self.candidates, constant=self.constant
        ).search(query, k, filters)

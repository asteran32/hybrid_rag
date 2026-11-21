import os
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from psycopg import sql

from enterprise_agent.models import Document
from enterprise_agent.storage.db import connect, initialize

load_dotenv()


@pytest.fixture
def db():
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set TEST_DATABASE_URL for PostgreSQL + pgvector integration tests")
    with connect(dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        name = sql.Identifier("rag_test_" + uuid4().hex)
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(name))
        try:
            conn.execute(sql.SQL("SET search_path TO {}, public").format(name))
            initialize(conn)
            yield conn
        finally:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(name))


@pytest.fixture
def documents():
    return [
        Document(
            document_id="dsid_one",
            title="Expense policy",
            source_type="slack",
            source_path="slack/one.json",
            content="Business travel receipts are required.",
            metadata={"team": "finance", "created_at": "2025-12-01"},
        ),
        Document(
            document_id="dsid_two",
            title="Café menu",
            source_type="gmail",
            source_path="gmail/two.json",
            content="Today's lunch includes fresh tomato soup.",
        ),
    ]


@pytest.fixture
def embedder():
    from enterprise_agent.embeddings import ModelSpec

    class FakeEmbeddings:
        spec = ModelSpec(provider="test", model="finance-food", dimensions=2)

        def __init__(self):
            self.calls = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [
                [0.1, 1.0]
                if any(w in t.lower() for w in ("soup", "lunch", "meal"))
                else [1.0, 0.1]
                for t in texts
            ]

    return FakeEmbeddings()

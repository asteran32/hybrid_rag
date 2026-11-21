import os
from importlib.resources import files

import psycopg
from psycopg.rows import dict_row


def connect(url: str | None = None) -> psycopg.Connection:
    dsn = url or os.getenv("DATABASE_URL")
    if not dsn:
        raise ValueError("configure DATABASE_URL in the environment or .env")
    return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)


def initialize(db: psycopg.Connection) -> None:
    with db.transaction():
        db.execute(files(__package__).joinpath("schema.sql").read_text())


def build_lexical_index(db: psycopg.Connection) -> None:
    with db.transaction():
        db.execute(
            "CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING gin(search_vector)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS documents_source_idx ON documents(source_type)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING gin(metadata jsonb_path_ops)"
        )

import json
from hashlib import sha256
from typing import Literal

from psycopg import Connection
from psycopg.types.json import Jsonb

from enterprise_agent.models import ChunkSettings, Document, DocumentChunk


def save_document(
    db: Connection,
    document: Document,
    chunks: list[DocumentChunk],
    settings: ChunkSettings,
) -> Literal["written", "unchanged", "conflict"]:
    """Called inside the batch transaction. Parent row locks serialize same-ID writers."""
    encoded = json.dumps(
        [document.model_dump(), settings.model_dump(), "char-v1"],
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    fingerprint = sha256(encoded.encode()).hexdigest()
    changed = db.execute(
        """
        INSERT INTO documents AS d
            (document_id, source_type, source_path, title, content, metadata, chunk_size, chunk_overlap, fingerprint)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (document_id) DO UPDATE SET
            source_type=excluded.source_type, title=excluded.title, content=excluded.content,
            metadata=excluded.metadata, chunk_size=excluded.chunk_size,
            chunk_overlap=excluded.chunk_overlap, fingerprint=excluded.fingerprint
        WHERE d.source_path = excluded.source_path AND d.fingerprint <> excluded.fingerprint
        RETURNING document_id
    """,
        (
            document.document_id,
            document.source_type,
            document.source_path,
            document.title,
            document.content,
            Jsonb(document.metadata),
            settings.size,
            settings.overlap,
            fingerprint,
        ),
    ).fetchone()
    if changed is None:
        existing = db.execute(
            "SELECT source_path FROM documents WHERE document_id=%s",
            (document.document_id,),
        ).fetchone()
        return (
            "unchanged"
            if existing["source_path"] == document.source_path
            else "conflict"
        )
    db.execute(
        "DELETE FROM chunks WHERE document_id=%s AND NOT (chunk_id=ANY(%s::text[]))",
        (document.document_id, [c.chunk_id for c in chunks]),
    )
    with db.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO chunks (chunk_id, document_id, position, start_offset, end_offset, content, search_vector)
            VALUES (%s,%s,%s,%s,%s,%s,
                    setweight(to_tsvector('english', %s), 'A') || setweight(to_tsvector('english', %s), 'D'))
            ON CONFLICT (chunk_id) DO UPDATE SET position=excluded.position, search_vector=excluded.search_vector
        """,
            [
                (
                    c.chunk_id,
                    c.document_id,
                    c.position,
                    c.start_offset,
                    c.end_offset,
                    c.content,
                    document.title or "",
                    c.content,
                )
                for c in chunks
            ],
        )
    return "written"

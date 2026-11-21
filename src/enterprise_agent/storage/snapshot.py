import hashlib

from psycopg import Connection


def corpus_snapshot(db: Connection, model_key: str) -> dict:
    """Caller holds a repeatable-read transaction for this snapshot and all searches."""
    digest = hashlib.sha256()
    with db.cursor(name="corpus_fingerprints") as cursor:
        cursor.execute(
            'SELECT fingerprint FROM documents ORDER BY document_id COLLATE "C"'
        )
        for row in cursor:
            digest.update(row["fingerprint"].encode("ascii"))
    chunks = db.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    indexed = db.execute(
        "SELECT count(*) AS n FROM chunk_embeddings WHERE model_key=%s", (model_key,)
    ).fetchone()["n"]
    if not chunks or chunks != indexed:
        raise ValueError(
            f"evaluation needs full embedding coverage: {indexed}/{chunks} chunks indexed"
        )
    return {
        "sha256": digest.hexdigest(),
        "chunks": chunks,
        "indexed_chunks": indexed,
        "by_source": {
            r["source_type"]: r["n"]
            for r in db.execute(
                "SELECT source_type, count(*) AS n FROM documents GROUP BY source_type ORDER BY source_type"
            )
        },
        "chunking": db.execute("""
            SELECT chunk_size, chunk_overlap, count(*) AS documents FROM documents
            GROUP BY chunk_size, chunk_overlap ORDER BY chunk_size, chunk_overlap
        """).fetchall(),
    }

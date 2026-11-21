from psycopg import Connection

from enterprise_agent.embeddings import Embedder, checked_vectors


def index_embeddings(
    db: Connection, provider: Embedder, *, max_chunks: int, batch_size: int
) -> int:
    if min(batch_size, max_chunks) < 1:
        raise ValueError("embedding limits must be positive")
    written = 0
    spec = provider.spec
    # Keyset pagination avoids repeatedly scanning the indexed prefix.
    after = ""
    while written < max_chunks:
        with db.transaction():
            # Same-model workers check the cache after the previous batch commits.
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (spec.key,)
            )
            rows = db.execute(
                """
                SELECT c.chunk_id, c.content FROM chunks c
                WHERE c.chunk_id > %s AND NOT EXISTS (
                    SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id=c.chunk_id AND e.model_key=%s)
                ORDER BY c.chunk_id LIMIT %s FOR UPDATE OF c SKIP LOCKED
            """,
                (after, spec.key, min(batch_size, max_chunks - written)),
            ).fetchall()
            if not rows:
                break
            vectors = checked_vectors(provider, [row["content"] for row in rows])
            with db.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO chunk_embeddings (model_key, chunk_id, dimensions, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT DO NOTHING
                """,
                    [
                        (spec.key, row["chunk_id"], spec.dimensions, str(vec))
                        for row, vec in zip(rows, vectors, strict=True)
                    ],
                )
                written += cursor.rowcount
            after = rows[-1]["chunk_id"]
    return written

import hashlib

from psycopg import Connection

from enterprise_agent.ingestion.pageindex import INDEX_VERSION


def corpus_snapshot(
    db: Connection, model_key: str | None, *, pageindex: bool = False
) -> dict:
    """Caller holds a repeatable-read transaction for this snapshot and all searches."""
    digest = hashlib.sha256()
    with db.cursor(name="corpus_fingerprints") as cursor:
        cursor.execute(
            'SELECT fingerprint FROM documents ORDER BY document_id COLLATE "C"'
        )
        for row in cursor:
            digest.update(row["fingerprint"].encode("ascii"))
    chunks = db.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    indexed = None
    if model_key is not None:
        indexed = db.execute(
            "SELECT count(*) AS n FROM chunk_embeddings WHERE model_key=%s",
            (model_key,),
        ).fetchone()["n"]
        if not chunks or chunks != indexed:
            raise ValueError(
                f"evaluation needs full embedding coverage: {indexed}/{chunks} chunks indexed"
            )
    snapshot = {
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
    if pageindex:
        trees = hashlib.sha256()
        indexed_docs = nodes = size = 0
        modes: dict[str, int] = {}
        with db.cursor(name="pageindex_fingerprints") as cursor:
            cursor.execute(
                """
                SELECT p.document_id, p.source_mode, jsonb_array_length(p.nodes) AS n,
                       pg_column_size(p.nodes) AS bytes, sha256(convert_to(p.nodes::text,'UTF8')) AS digest
                FROM pageindex_documents p JOIN documents d USING(document_id)
                WHERE p.fingerprint=d.fingerprint AND p.index_version=%s
                ORDER BY p.document_id COLLATE "C"
                """,
                (INDEX_VERSION,),
            )
            for row in cursor:
                trees.update(
                    row["document_id"].encode()
                    + b"\0"
                    + row["source_mode"].encode()
                    + b"\0"
                    + bytes(row["digest"])
                )
                indexed_docs += 1
                nodes += row["n"]
                size += row["bytes"]
                modes[row["source_mode"]] = modes.get(row["source_mode"], 0) + 1
        total = sum(snapshot["by_source"].values())
        if not total or indexed_docs != total:
            raise ValueError(
                f"evaluation needs full PageIndex coverage: {indexed_docs}/{total} documents indexed"
            )
        snapshot["pageindex"] = {
            "documents": total,
            "nodes": nodes,
            "stored_tree_bytes": size,
            "source_modes": modes,
            "version": INDEX_VERSION,
            "sha256": trees.hexdigest(),
        }
    return snapshot

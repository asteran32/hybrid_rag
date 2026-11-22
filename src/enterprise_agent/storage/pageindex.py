"""Versioned per-document PageIndex trees; documents and chunks remain authoritative."""

import json
import logging
from pathlib import Path

from psycopg import Connection
from psycopg.types.json import Jsonb

from enterprise_agent.ingestion.normalize import normalize_with_sections
from enterprise_agent.ingestion.pageindex import INDEX_VERSION, build_tree
from enterprise_agent.models import Document


def _source_file(root: Path, doc: Document) -> Path:
    relative = Path(doc.source_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[0] != doc.source_type
    ):
        raise ValueError("invalid benchmark source path")
    for base in (root / "generated_data/sources", root / "sources", root):
        if (base / doc.source_type).is_dir():
            path = (base / relative).resolve()
            break
    else:
        path = root.joinpath(*relative.parts[1:]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("source path is outside input directory")
    return path


def build_pageindex(
    db: Connection, source_input: Path | None = None, batch_size: int = 32
) -> dict:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if source_input is not None and not (
        source_input.is_dir()
        or (source_input.is_file() and source_input.suffix == ".parquet")
    ):
        raise ValueError("expected an extracted source directory or documents.parquet")
    stats = {"written": 0, "unchanged": 0, "empty": 0, "errors": 0}
    after = ""
    while rows := db.execute(
        """
        SELECT d.*, p.fingerprint AS built, p.index_version, p.source_mode
        FROM documents d LEFT JOIN pageindex_documents p USING(document_id)
        WHERE d.document_id > %s ORDER BY d.document_id LIMIT %s
        """,
        (after, batch_size),
    ).fetchall():
        for row in rows:
            doc = Document.model_validate(row)
            after = doc.document_id
            mode = (
                "fields"
                if source_input
                and source_input.is_dir()
                and doc.source_path.endswith(".json")
                else "content"
            )
            if not doc.content.strip():
                stats["empty"] += 1
            if (
                row["built"] == row["fingerprint"]
                and row["index_version"] == INDEX_VERSION
                and (mode == "content" or row["source_mode"] == mode)
            ):
                stats["unchanged"] += 1
                continue
            try:
                fields = None
                if mode == "fields" and source_input is not None:
                    raw = json.loads(
                        _source_file(source_input, doc).read_text(encoding="utf-8")
                    )
                    rebuilt, fields = normalize_with_sections(
                        raw, doc.source_type, doc.source_path
                    )
                    if (
                        rebuilt.document_id != doc.document_id
                        or rebuilt.content != doc.content
                    ):
                        raise ValueError(
                            "source does not match stored document; re-ingest first"
                        )
                nodes = build_tree(doc, fields)
                with db.transaction():
                    current = db.execute(
                        "SELECT fingerprint FROM documents WHERE document_id=%s FOR UPDATE",
                        (doc.document_id,),
                    ).fetchone()
                    if not current or current["fingerprint"] != row["fingerprint"]:
                        raise ValueError("document changed during indexing")
                    saved = db.execute(
                        """
                        INSERT INTO pageindex_documents AS p
                            (document_id, fingerprint, index_version, source_mode, nodes)
                        VALUES (%s,%s,%s,%s,%s)
                        ON CONFLICT (document_id) DO UPDATE SET fingerprint=excluded.fingerprint,
                            index_version=excluded.index_version, source_mode=excluded.source_mode,
                            nodes=excluded.nodes
                        WHERE (p.fingerprint, p.index_version) <>
                              (excluded.fingerprint, excluded.index_version)
                           OR (p.source_mode='content' AND excluded.source_mode='fields')
                        """,
                        (
                            doc.document_id,
                            row["fingerprint"],
                            INDEX_VERSION,
                            mode,
                            Jsonb(nodes),
                        ),
                    )
                stats["written" if saved.rowcount else "unchanged"] += 1
            except (OSError, ValueError) as exc:
                logging.getLogger(__name__).error("%s: %s", doc.document_id, exc)
                stats["errors"] += 1
    return stats

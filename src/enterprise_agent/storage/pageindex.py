"""Versioned per-document PageIndex trees; documents and chunks remain authoritative."""

import json
import logging
from pathlib import Path

from psycopg import Connection
from psycopg.types.json import Jsonb

from enterprise_agent.ingestion.normalize import normalize_with_sections
from enterprise_agent.ingestion.pageindex import INDEX_VERSION, build_tree
from enterprise_agent.models import Document, SearchFilter
from enterprise_agent.storage.search import PROJECTION, _filters


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


class PageIndexStore:
    def __init__(self, db: Connection):
        self.db = db

    @staticmethod
    def _scope(filters: SearchFilter) -> tuple[str, list]:
        if filters.scope is None:
            raise ValueError("PageIndex requires an explicit read scope")
        where, params = _filters(filters)
        return "p.index_version=%s AND p.fingerprint=d.fingerprint" + where, [
            INDEX_VERSION,
            *params,
        ]

    def catalog(self, filters: SearchFilter) -> list[dict]:
        where, params = self._scope(filters)
        return self.db.execute(
            f"""
            SELECT d.document_id, d.fingerprint, d.title, d.source_type, left(d.content,120) AS preview
            FROM documents d JOIN pageindex_documents p USING(document_id)
            WHERE {where} AND p.nodes <> '[]'::jsonb ORDER BY d.document_id
            """,
            params,
        ).fetchall()

    def outlines(self, docs: dict[str, str], filters: SearchFilter) -> dict[str, list]:
        where, params = self._scope(filters)
        rows = self.db.execute(
            f"""
            SELECT d.document_id, p.nodes FROM documents d JOIN pageindex_documents p USING(document_id)
            JOIN unnest(%s::text[], %s::text[]) v(id, fp)
              ON v.id=d.document_id AND v.fp=d.fingerprint
            WHERE {where}
            """,
            [list(docs), list(docs.values()), *params],
        )
        return {r["document_id"]: r["nodes"] for r in rows}

    def read_spans(
        self,
        spans: list[tuple[str, str, int, int]],
        filters: SearchFilter,
        *,
        chunks: bool = False,
    ) -> list[dict]:
        if not spans:
            return []
        where, params = self._scope(filters)
        projection = """v.id AS document_id, v.a, v.b,
                        substring(d.content FROM v.a+1 FOR v.b-v.a) AS content"""
        join = ""
        if chunks:
            projection = "DISTINCT " + PROJECTION
            join = """JOIN chunks c ON c.document_id=d.document_id
                      AND c.start_offset < v.b AND c.end_offset > v.a
                      AND c.content=substring(d.content FROM c.start_offset+1 FOR c.end_offset-c.start_offset)"""
        return self.db.execute(
            f"""
            SELECT {projection} FROM documents d JOIN pageindex_documents p USING(document_id)
            JOIN unnest(%s::text[],%s::text[],%s::int[],%s::int[]) v(id,fp,a,b)
              ON v.id=d.document_id AND v.fp=d.fingerprint
            {join} WHERE {where} AND v.a >= 0 AND v.b <= char_length(d.content) AND v.a < v.b
            """,
            [*[list(col) for col in zip(*spans, strict=True)], *params],
        ).fetchall()

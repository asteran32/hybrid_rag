import re

from psycopg import Connection
from psycopg.types.json import Jsonb

from enterprise_agent.models import EvidenceChunk, SearchFilter, SearchHit

EVIDENCE_FIELDS = """
    c.document_id, c.chunk_id, c.position, c.start_offset, c.end_offset, c.content,
    d.source_type, d.source_path, d.title
"""
PROJECTION = EVIDENCE_FIELDS + ", d.metadata"


def _filters(filters: SearchFilter | None) -> tuple[str, list]:
    conditions, values = [], []
    if filters and filters.source_type:
        conditions.append("d.source_type = %s")
        values.append(filters.source_type)
    if filters and filters.metadata:
        conditions.append("d.metadata @> %s")
        values.append(Jsonb(filters.metadata))
    if filters and filters.scope is not None:
        conditions.append("d.source_type = ANY(%s::text[])")
        values.append(list(filters.scope.source_types))
        if filters.scope.document_ids is not None:
            conditions.append("d.document_id = ANY(%s::text[])")
            values.append(list(filters.scope.document_ids))
    return "".join(f" AND {condition}" for condition in conditions), values


def _hits(rows: list[dict], method: str) -> list[SearchHit]:
    return [
        SearchHit(**row, rank=i, component_ranks={method: i})
        for i, row in enumerate(rows, 1)
    ]


class SearchStore:
    def __init__(self, db: Connection):
        self.db = db

    def load_evidence(
        self, chunk_ids: list[str], filters: SearchFilter
    ) -> dict[str, EvidenceChunk]:
        """Load only authorized chunks whose saved span still matches the parent text."""
        if filters.scope is None:
            raise ValueError("answer evidence requires an access scope")
        if not chunk_ids:
            return {}
        where, params = _filters(filters)
        rows = self.db.execute(
            f"""
            SELECT {EVIDENCE_FIELDS} FROM chunks c JOIN documents d USING(document_id)
            WHERE c.chunk_id = ANY(%s::text[]) {where}
              AND c.start_offset >= 0 AND c.end_offset <= char_length(d.content)
              AND c.content = substring(d.content FROM c.start_offset + 1
                                        FOR c.end_offset - c.start_offset)
            """,
            [chunk_ids, *params],
        )
        return {row["chunk_id"]: EvidenceChunk(**row) for row in rows}

    def keyword(
        self, query: str, k: int, filters: SearchFilter | None
    ) -> list[SearchHit]:
        # Natural-language questions: match any non-stopword term, weight title matches higher.
        terms = re.findall(r"\w+", query)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        where, params = _filters(filters)
        rows = self.db.execute(
            f"""
            SELECT {PROJECTION}, ts_rank_cd(c.search_vector, q) AS score
            FROM chunks c JOIN documents d USING(document_id),
                 websearch_to_tsquery('english', %s) q
            WHERE c.search_vector @@ q {where}
            ORDER BY score DESC, c.chunk_id LIMIT %s
        """,
            [fts_query, *params, k],
        ).fetchall()
        return _hits(rows, "keyword")

    def vector(
        self, vector: list[float], model_key: str, k: int, filters: SearchFilter | None
    ) -> list[SearchHit]:
        where, params = _filters(filters)
        rows = self.db.execute(
            f"""
            SELECT {PROJECTION}, 1 - (e.embedding <=> %s::vector) AS score
            FROM chunk_embeddings e JOIN chunks c USING(chunk_id) JOIN documents d USING(document_id)
            WHERE e.model_key=%s {where}
            ORDER BY score DESC, c.chunk_id LIMIT %s
        """,
            [str(vector), model_key, *params, k],
        ).fetchall()
        return _hits(rows, "dense")

    def require_embeddings(self, model_key: str) -> None:
        if not self.db.execute(
            "SELECT 1 FROM chunk_embeddings WHERE model_key=%s LIMIT 1", (model_key,)
        ).fetchone():
            raise ValueError(
                "no indexed chunks for this model configuration; run build-index first"
            )

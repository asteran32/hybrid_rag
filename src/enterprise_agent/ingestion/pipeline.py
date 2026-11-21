import logging
from collections.abc import Iterable
from itertools import islice

from psycopg import Connection, DataError
from pydantic import BaseModel

from enterprise_agent.ingestion.chunking import chunk_document
from enterprise_agent.ingestion.loader import LoadError
from enterprise_agent.models import ChunkSettings, Document
from enterprise_agent.storage.documents import save_document

log = logging.getLogger(__name__)


class IngestionStats(BaseModel):
    seen: int = 0
    written: int = 0
    unchanged: int = 0
    errors: int = 0
    chunks_written: int = 0


def ingest(
    db: Connection,
    records: Iterable[Document | LoadError],
    settings: ChunkSettings,
    *,
    limit: int | None,
    batch_size: int,
) -> IngestionStats:
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("limit and batch_size must be positive")
    stream = islice(records, limit)
    stats = IngestionStats()
    while batch := list(islice(stream, batch_size)):
        # Lock shared document IDs in the same order across concurrent ingestion runs.
        docs = []
        for record in batch:
            stats.seen += 1
            if isinstance(record, LoadError):
                stats.errors += 1
                log.error("%s: %s", record.path, record.message)
            else:
                docs.append(record)
        with db.transaction():
            for doc in sorted(docs, key=lambda d: d.document_id):
                try:
                    chunks = chunk_document(doc, settings)
                    with db.transaction():  # savepoint for invalid source records
                        outcome = save_document(db, doc, chunks, settings)
                except (ValueError, DataError) as exc:
                    log.error("%s (%s): %s", doc.source_path, doc.document_id, exc)
                    stats.errors += 1
                    continue
                if outcome == "conflict":
                    log.error(
                        "%s: ID %s already belongs to another path",
                        doc.source_path,
                        doc.document_id,
                    )
                    stats.errors += 1
                elif outcome == "unchanged":
                    stats.unchanged += 1
                else:
                    stats.written += 1
                    stats.chunks_written += len(chunks)
        log.info("committed batch: %s", stats.model_dump())
    return stats

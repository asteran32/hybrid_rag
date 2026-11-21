from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from enterprise_agent.models import DocumentID, EvidenceChunk, SearchHit


class QuoteRef(BaseModel):
    """The generator can point only to text it received in this request."""

    model_config = ConfigDict(extra="forbid")
    passage: str
    quote: str


class Citation(BaseModel):
    document_id: DocumentID
    chunk_id: str
    source_type: str
    source_path: str
    title: str | None
    start_offset: int
    end_offset: int
    quote: str


@dataclass(frozen=True)
class Passage:
    label: str
    chunk: EvidenceChunk
    text: str


def select_passages(
    hits: list[SearchHit], stored: dict[str, EvidenceChunk], char_budget: int
) -> dict[str, Passage]:
    if char_budget < 1:
        raise ValueError("context character budget must be positive")
    ordered = {}
    for hit in hits:
        chunk = stored.get(hit.chunk_id)
        if chunk is not None and chunk.document_id == hit.document_id:
            ordered.setdefault(chunk.chunk_id, chunk)

    passages = {}
    remaining = char_budget
    chunks = list(ordered.values())[:char_budget]
    for index, chunk in enumerate(chunks):
        # Reserve room for lower-ranked evidence, including potential disagreements.
        text = chunk.content[: remaining // (len(chunks) - index)]
        if text.strip():
            label = f"E{len(passages) + 1}"
            passages[label] = Passage(label, chunk, text)
            remaining -= len(text)
    return passages


def resolve_citation(
    ref: QuoteRef, passages: dict[str, Passage], current: dict[str, EvidenceChunk]
) -> Citation | None:
    passage = passages.get(ref.passage)
    if passage is None or not ref.quote.strip():
        return None
    chunk = current.get(passage.chunk.chunk_id)
    if (
        chunk is None
        or chunk.document_id != passage.chunk.document_id
        or chunk.start_offset != passage.chunk.start_offset
        or chunk.content != passage.chunk.content
    ):
        return None
    start = passage.text.find(ref.quote)
    if start < 0 or passage.text.find(ref.quote, start + 1) >= 0:
        return None
    offset = chunk.start_offset + start
    return Citation(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        source_type=chunk.source_type,
        source_path=chunk.source_path,
        title=chunk.title,
        start_offset=offset,
        end_offset=offset + len(ref.quote),
        quote=ref.quote,
    )

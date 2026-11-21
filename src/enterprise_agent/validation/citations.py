import re
from dataclasses import dataclass
from itertools import pairwise
from math import ceil

from pydantic import BaseModel, RootModel

from enterprise_agent.models import DocumentID, EvidenceChunk, SearchHit


class PassageRef(RootModel[str]):
    """Request-local label for a visible, server-located source passage."""


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
    chunk_offset: int


def _spans(text: str) -> list[tuple[int, int]]:
    """Group adjacent sentences or lines into at most eight citable Unicode spans."""
    boundaries = [
        0,
        *(m.end() for m in re.finditer(r"(?<=[.!?。！？])[^\S\n]+|\n+", text)),
        len(text),
    ]
    pieces = [(a, b) for a, b in pairwise(boundaries) if text[a:b].strip()]
    width = max(1, ceil(len(pieces) / 8))
    result = []
    for i in range(0, len(pieces), width):
        group = pieces[i : i + width]
        start, end = group[0][0], group[-1][1]
        while text[start].isspace():
            start += 1
        while text[end - 1].isspace():
            end -= 1
        result.append((start, end))
    return result


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
        for number, (start, end) in enumerate(_spans(text), 1):
            label = f"E{index + 1}.{number}"
            passages[label] = Passage(label, chunk, text[start:end], start)
        remaining -= len(text)
    return passages


def resolve_citation(
    ref: PassageRef, passages: dict[str, Passage], current: dict[str, EvidenceChunk]
) -> Citation | None:
    passage = passages.get(ref.root)
    if passage is None:
        return None
    chunk = current.get(passage.chunk.chunk_id)
    if (
        chunk is None
        or chunk.document_id != passage.chunk.document_id
        or chunk.start_offset != passage.chunk.start_offset
        or chunk.content != passage.chunk.content
    ):
        return None
    start = passage.chunk_offset
    end = start + len(passage.text)
    if start < 0 or not passage.text or chunk.content[start:end] != passage.text:
        return None
    offset = chunk.start_offset + start
    return Citation(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        source_type=chunk.source_type,
        source_path=chunk.source_path,
        title=chunk.title,
        start_offset=offset,
        end_offset=chunk.start_offset + end,
        quote=passage.text,
    )

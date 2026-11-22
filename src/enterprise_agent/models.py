from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveInt,
    model_validator,
)

DocumentID = Annotated[str, Field(pattern=r"^dsid_[A-Za-z0-9_-]+$")]


class Document(BaseModel):
    document_id: DocumentID
    source_type: str = Field(min_length=1)
    source_path: str
    title: str | None = None
    content: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ChunkSettings(BaseModel):
    size: PositiveInt = 1200
    overlap: int = Field(default=160, ge=0)

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than chunk size")
        return self


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: DocumentID
    position: int
    start_offset: int
    end_offset: int
    content: str


class AnswerScope(BaseModel):
    """Trusted operator policy. An empty source or document allowlist grants no access."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_types: tuple[str, ...]
    document_ids: tuple[DocumentID, ...] | None = None


class SearchFilter(BaseModel):
    source_type: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    scope: AnswerScope | None = None


class EvidenceChunk(DocumentChunk):
    source_type: str
    source_path: str
    title: str | None


class SearchHit(EvidenceChunk):
    metadata: dict[str, JsonValue]
    score: float
    rank: PositiveInt
    component_ranks: dict[str, int]


class PageIndexHit(SearchHit):
    matched_spans: list[tuple[int, int]] = Field(min_length=1)

    @model_validator(mode="after")
    def within_chunk(self) -> Self:
        if any(
            not self.start_offset <= a < b <= self.end_offset
            for a, b in self.matched_spans
        ):
            raise ValueError("matched spans must be inside the source chunk")
        return self

from typing import Annotated, Self

from pydantic import BaseModel, Field, JsonValue, PositiveInt, model_validator

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

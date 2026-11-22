from collections.abc import Iterable
from typing import Self

from pydantic import BaseModel, Field, model_validator

from enterprise_agent.models import DocumentID


class ReferenceSpan(BaseModel):
    document_id: DocumentID
    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if self.end <= self.start:
            raise ValueError("reference span must be nonempty")
        return self


def span_recall(
    chunks: list[dict], references: list[ReferenceSpan]
) -> dict[str, float]:
    """A reference must be fully covered, including across adjacent/overlapping hits."""
    if not references:
        return {}
    scores = {}
    for k in (1, 5, 10):
        found = 0
        for ref in references:
            pos = ref.start
            ranges = sorted(
                span
                for hit in chunks[:k]
                if hit["document_id"] == ref.document_id
                for span in hit["spans"]
            )
            for left, right in ranges:
                if left > pos:
                    break
                pos = max(pos, right)
            found += pos >= ref.end
        scores[f"span_recall@{k}"] = found / len(references)
    return scores


def document_metrics(
    retrieved: Iterable[str], expected: set[str]
) -> dict[str, float] | None:
    if not expected:
        return None
    documents = list(dict.fromkeys(retrieved))
    first = next(
        (rank for rank, doc in enumerate(documents, 1) if doc in expected), None
    )
    return {
        **{
            f"recall@{k}": len(expected.intersection(documents[:k])) / len(expected)
            for k in (1, 5, 10)
        },
        "mrr": 1 / first if first else 0.0,
    }

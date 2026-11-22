"""Interpret the benchmark's per-document field labels without inferring timestamps."""

import json
from typing import Any

from enterprise_agent.models import Document

LABELS = {"title_field_name", "content_field_names", "dataset_doc_uuid"}
GOLD_FIELDS = {
    "gold_answer",
    "answer_facts",
    "expected_doc_ids",
    "expected_spans",
    "question_id",
}
BODY_FIELDS = (
    "content",
    "body",
    "messages",
    "description",
    "summary",
    "transcript",
    "notes",
)


def render(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(render(item) for item in value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def normalize(raw: dict, source: str, path: str) -> Document:
    return normalize_with_sections(raw, source, path)[0]


def normalize_with_sections(
    raw: dict, source: str, path: str
) -> tuple[Document, list[tuple[str, int, int]]]:
    """Build the canonical text and the exact spans of its contributing fields."""
    if not isinstance(raw, dict) or GOLD_FIELDS.intersection(raw):
        raise ValueError("expected a source document, received a non-document record")
    title_key = raw.get("title_field_name")
    if not isinstance(title_key, str) or title_key not in raw:
        source_title = {"gmail": "subject", "jira": "summary"}.get(source)
        title_key = "title" if "title" in raw else source_title
    title = raw.get(title_key)
    fields = raw.get("content_field_names")
    if isinstance(fields, str):
        fields = [fields]
    fields = list(
        dict.fromkeys(
            k
            for k in (fields if isinstance(fields, list) else [])
            if isinstance(k, str) and k in raw and k not in LABELS
        )
    )
    if not fields:
        fields = [k for k in BODY_FIELDS if k in raw and k != title_key]
    parts = []
    sections = []
    offset = 0
    for key in fields:
        value = render(raw[key])
        if value.strip():
            part = value if len(fields) == 1 else f"{key}\n{value}"
            sections.append((key, offset, offset + len(part)))
            parts.append(part)
            offset += len(part) + 2
    metadata = {k: v for k, v in raw.items() if k not in LABELS and k not in fields}
    document = Document(
        document_id=raw.get("dataset_doc_uuid"),
        source_type=source,
        source_path=path,
        title=title if isinstance(title, str) else None,
        content="\n\n".join(parts),
        metadata=metadata,
    )
    return document, sections

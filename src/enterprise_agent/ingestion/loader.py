"""Stream source documents. Benchmark questions and repository metadata are outside this tree."""

import json
import os
import re
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from enterprise_agent.ingestion.normalize import GOLD_FIELDS, normalize
from enterprise_agent.models import Document

SOURCES = (
    "confluence",
    "fireflies",
    "github",
    "gmail",
    "google_drive",
    "hubspot",
    "jira",
    "linear",
    "slack",
)


@dataclass
class LoadError:
    path: str
    message: str


def _source_files(directory: Path) -> Iterator[Path]:
    def on_error(exc: OSError) -> None:
        raise exc

    for base, dirs, files in os.walk(directory, onerror=on_error):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):
                continue
            if name.endswith(".json") or (
                name.startswith("dsid_") and name.endswith(".txt")
            ):
                yield Path(base, name)


def _read_file(path: Path, source: str, relative: str) -> Document:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return normalize(json.loads(text), source, relative)
    match = re.fullmatch(r"(dsid_[A-Za-z0-9-]+)__(.+)", path.stem)
    if not match:
        raise ValueError("expected dsid_*__name.txt export filename")
    title, _, content = text.partition("\n")
    return Document(
        document_id=match[1],
        source_type=source,
        source_path=relative,
        title=title or None,
        content=content.removeprefix("\n"),
    )


def _parquet(path: Path) -> Iterator[Document | LoadError]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "install Parquet support with uv sync --extra parquet"
        ) from exc
    table = pq.ParquetFile(path)
    names = set(table.schema_arrow.names)
    if GOLD_FIELDS & names or not {"doc_id", "source_type", "content"} <= names:
        raise ValueError("expected a benchmark documents Parquet file")
    for batch in table.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            document_id = row.pop("doc_id")
            reference = f"{path.name}#{document_id}"
            try:
                yield Document(
                    document_id=document_id,
                    source_path=reference,
                    source_type=row.pop("source_type"),
                    content=row.pop("content") or "",
                    title=row.pop("title", None),
                    metadata=row,
                )
            except ValueError as exc:
                yield LoadError(reference, str(exc))


def load_documents(
    path: Path, source_type: str | None = None
) -> Iterator[Document | LoadError]:
    if path.is_file() and path.suffix == ".parquet":
        if source_type:
            raise ValueError("Parquet documents already supply source_type")
        yield from _parquet(path)
        return
    if not path.is_dir():
        raise ValueError(
            f"expected an extracted dataset directory or documents.parquet: {path}"
        )
    for subdir in ("generated_data/sources", "sources"):
        if (path / subdir).is_dir():
            path = path / subdir
            break
    if source_type is not None and source_type not in SOURCES:
        raise ValueError(f"unknown source type: {source_type}")
    available = [s for s in SOURCES if (path / s).is_dir()]
    if source_type and available and source_type not in available:
        raise ValueError(f"source directory {source_type!r} is missing under {path}")
    if source_type and path.name in SOURCES and source_type != path.name:
        raise ValueError(f"source directory {path.name!r} conflicts with --source-type")
    if source_type or path.name in SOURCES:
        source = source_type or path.name
        directory = path / source if (path / source).is_dir() else path
        entries = [(source, directory)]
    else:
        entries = [(s, path / s) for s in available]
    if not entries:
        raise ValueError(
            "no benchmark source directories; supply --source-type for an extracted flat slice"
        )
    # Round robin across sources so a small prefix is repeatable and covers multiple formats.
    streams = deque((s, d, _source_files(d)) for s, d in entries)
    while streams:
        source, directory, files = streams.popleft()
        file = next(files, None)
        if file is None:
            continue
        relative = f"{source}/{file.relative_to(directory).as_posix()}"
        try:
            yield _read_file(file, source, relative)
        except (ValueError, OSError) as exc:
            yield LoadError(relative, str(exc))
        streams.append((source, directory, files))

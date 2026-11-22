"""Project normalized text into Markdown, then attach PageIndex nodes to source spans."""

import asyncio
import re
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

from enterprise_agent.models import Document

UPSTREAM_REVISION = "add6b17cebff528612748ede3fed565b5f726c6b"
INDEX_VERSION = f"markdown-v1/{UPSTREAM_REVISION}"
BLOCK_CHARS = 1000
HEADING = re.compile(r" {0,3}(#{1,6})[ \t]+(.+?)\s*$")
TIMESTAMP = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")


def excerpt(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def _sections(text: str, title: str, start: int, end: int):
    """Preserve native section hierarchy, ignoring headings inside fenced code."""
    headings = []
    offset = start
    fence = ""
    for line in text[start:end].splitlines(keepends=True):
        marker = re.match(r" {0,3}(\x60{3,}|~{3,})", line)
        if marker:
            if not fence:
                fence = marker[1]
            elif marker[1].startswith(fence):
                fence = ""
        elif not fence and (match := HEADING.fullmatch(line)):
            headings.append((offset, len(match[1]), match[2]))
        offset += len(line)
    first = headings[0][0] if headings else end
    yield 2, title, start, first
    levels = []
    for i, (left, level, label) in enumerate(headings):
        while levels and levels[-1] >= level:
            levels.pop()
        levels.append(level)
        right = headings[i + 1][0] if i + 1 < len(headings) else end
        yield min(5, 2 + len(levels)), label, left, right


def _blocks(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    """Prefer full turns/paragraphs, then sentences, then a bounded Unicode slice."""
    while end - start > BLOCK_CHARS:
        stop = start + BLOCK_CHARS
        midpoint = start + BLOCK_CHARS // 2
        split = text.rfind("\n", midpoint, stop)
        if split < 0:
            split = text.rfind(". ", midpoint, stop)
        if split < 0:
            split = text.rfind(" ", midpoint, stop)
        right = split + 1 if split >= 0 else stop
        yield start, right
        start = right
    if start < end:
        yield start, end


def build_tree(
    document: Document, fields: list[tuple[str, int, int]] | None = None
) -> list[dict]:
    """Return a flat preorder tree. Offsets always address Document.content."""
    text = document.content
    if not text.strip():
        return []
    try:
        from pageindex.page_index_md import md_to_tree
    except ImportError as exc:
        raise ValueError("install PageIndex with uv sync --extra pageindex") from exc

    lines: list[str] = []
    locations: dict[int, tuple[int, int]] = {}

    def heading(level: int, title: str, start: int, end: int) -> None:
        lines.append("#" * level + " " + (excerpt(title, 160) or "Content"))
        locations[len(lines)] = start, end
        # Only our structural headings are interpreted by the Markdown parser.
        lines.extend("> " + line for line in text[start:end].splitlines())

    heading(1, document.title or document.document_id, 0, 0)
    for name, start, end in fields or [("Content", 0, len(text))]:
        for level, title, left, right in _sections(text, name, start, end):
            if right - left <= BLOCK_CHARS:
                heading(level, title, left, right)
                continue
            heading(level, title, left, left)
            for a, b in _blocks(text, left, right):
                part = text[a:b]
                if part.strip():
                    times = TIMESTAMP.findall(part)
                    label = excerpt(part, 80)
                    if len(times) > 1:
                        label = f"{times[0]}–{times[-1]} {excerpt(TIMESTAMP.sub('', part), 80)}"
                    heading(level + 1, label, a, b)

    with TemporaryDirectory(prefix="enterprise-pageindex-") as tmp:
        path = Path(tmp) / "document.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        result = asyncio.run(md_to_tree(str(path), if_add_node_id="yes"))

    nodes = []

    def visit(items: list[dict], parent: str | None) -> None:
        for item in items:
            start, end = locations[item["line_num"]]
            nodes.append(
                {
                    "id": item["node_id"],
                    "parent": parent,
                    "title": item["title"],
                    "start": start,
                    "end": end,
                    "preview": excerpt(text[start:end], 180),
                }
            )
            visit(item.get("nodes", []), item["node_id"])

    visit(result["structure"], None)
    return nodes

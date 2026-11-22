"""Select authorized documents, navigate their PageIndex trees and read original evidence."""

import re
from collections import defaultdict
from itertools import zip_longest

from enterprise_agent.ingestion.pageindex import excerpt
from enterprise_agent.models import PageIndexHit, SearchFilter
from enterprise_agent.retrieval.pageindex_selector import (
    LocalSelector,
    Navigation,
    Selector,
    SelectorSettings,
    Trace,
)
from enterprise_agent.storage.pageindex import PageIndexStore


def source_passages(text: str, offset: int) -> list[tuple[int, int]]:
    """Keep speaker turns together; subdivide oversized lines into complete sentences."""
    spans = []
    sentences = re.compile(r".+?(?:[.!?。！？]+(?=\s|$)|$)")
    for line in re.finditer(r"[^\r\n]+", text):
        parts = (
            sentences.finditer(text, line.start(), line.end())
            if len(line[0]) > 450
            else [line]
        )
        for part in parts:
            clean = part[0].strip()
            if clean:
                start = offset + part.start() + len(part[0]) - len(part[0].lstrip())
                spans.append((start, start + len(clean)))
    return spans


def evidence_hits(
    ranked: list[tuple[str, str, int, int]], rows: list[dict], k: int
) -> list[PageIndexHit]:
    chunks = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["document_id"], r["position"])):
        chunks[row["document_id"]].append(row)
    by_document: dict[str, dict] = {}
    for doc, _, left, end in ranked:
        pieces = []
        while left < end:
            covering = [
                r for r in chunks[doc] if r["start_offset"] <= left < r["end_offset"]
            ]
            if not covering:
                break
            row = max(covering, key=lambda r: min(r["end_offset"], end))
            right = min(row["end_offset"], end)
            pieces.append((row, left, right))
            left = right
        if left != end:
            continue
        hits = by_document.setdefault(doc, {})
        for row, a, b in pieces:
            _, spans = hits.setdefault(row["chunk_id"], (row, []))
            if (a, b) not in spans:
                spans.append((a, b))
    results = []
    for group in zip_longest(*(hits.values() for hits in by_document.values())):
        for item in group:
            if item is None:
                continue
            row, spans = item
            rank = len(results) + 1
            results.append(
                PageIndexHit(
                    **row,
                    matched_spans=spans,
                    rank=rank,
                    score=1 / rank,
                    component_ranks={"pageindex": rank},
                )
            )
            if rank == k:
                return results
    return results


class PageIndexRetriever:
    def __init__(
        self,
        store: PageIndexStore,
        settings: SelectorSettings,
        selector: Selector | None = None,
    ):
        self.store = store
        self.settings = settings
        self.selector = selector or LocalSelector(settings)

    def search(
        self, query: str, k: int = 10, filters: SearchFilter | None = None
    ) -> list[PageIndexHit]:
        return self.search_with_trace(query, k, filters)[0]

    def search_with_trace(
        self, query: str, k: int = 10, filters: SearchFilter | None = None
    ) -> tuple[list[PageIndexHit], Trace]:
        if not query.strip() or k < 1:
            raise ValueError("search needs a nonempty query and positive top-k")
        if filters is None or filters.scope is None:
            raise ValueError("PageIndex requires an explicit read scope")
        nav = Navigation(self.selector, self.settings, query)
        catalog = {r["document_id"]: r for r in self.store.catalog(filters)}
        options = [
            (
                doc,
                f"{excerpt(r['title'] or r['source_type'], 110)} | {r['source_type']} {excerpt(r['preview'], 64)}",
            )
            for doc, r in catalog.items()
        ]
        selected = nav.choose("document", options, self.settings.candidates)
        nav.trace.candidates = selected
        versions = {doc: catalog[doc]["fingerprint"] for doc in selected}
        outlines = self.store.outlines(versions, filters) if versions else {}
        children = defaultdict(list)
        records = {}
        frontier = []
        for doc in selected:
            for n in outlines.get(doc, []):
                key = f"{doc}/{n['id']}"
                records[key] = doc, n
                if n["parent"] is None:
                    frontier.append(key)
                else:
                    children[f"{doc}/{n['parent']}"].append(key)
        to_read = []
        while frontier:
            auto = []
            options = []
            for key in frontier:
                doc, node = records[key]
                nav.trace.nodes.append(key)
                if node["end"] > node["start"]:
                    to_read.append((doc, versions[doc], node["start"], node["end"]))
                nested = children[key]
                if len(nested) == 1:
                    auto.extend(nested)
                else:
                    for child in nested:
                        n = records[child][1]
                        label = f"{n['title']} | {catalog[doc]['title'] or catalog[doc]['source_type']} / {node['title']} / {n['preview']}"
                        options.append((child, label))
            frontier = auto + (
                nav.choose("node", options, self.settings.candidates) if options else []
            )

        text_rows = self.store.read_spans(to_read, filters)
        bodies = {(r["document_id"], r["a"], r["b"]): r["content"] for r in text_rows}
        passages = []
        options = []
        for doc, fp, a, b in to_read:
            text = bodies.get((doc, a, b), "")
            for left, right in source_passages(text, a):
                options.append(
                    (
                        str(len(passages)),
                        f"{text[left - a : right - a]} | {catalog[doc]['title'] or catalog[doc]['source_type']}",
                    )
                )
                passages.append((doc, fp, left, right))
        choices = nav.choose("evidence", options, min(k * 2, 40)) if options else []
        document_order = {doc: rank for rank, doc in enumerate(selected)}
        ranked = sorted(
            (passages[int(idx)] for idx in choices), key=lambda s: document_order[s[0]]
        )
        rows = self.store.read_spans(ranked, filters, chunks=True)
        return evidence_hits(ranked, rows, k), nav.trace

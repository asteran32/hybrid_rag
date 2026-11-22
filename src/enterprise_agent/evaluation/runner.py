import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict
from itertools import islice
from math import ceil
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from time import perf_counter

from pydantic import BaseModel, Field

from enterprise_agent.evaluation.metrics import (
    ReferenceSpan,
    document_metrics,
    span_recall,
)
from enterprise_agent.models import DocumentID, PageIndexHit, SearchFilter
from enterprise_agent.retrieval.pageindex import PageIndexRetriever
from enterprise_agent.retrieval.pageindex_selector import SelectorUnavailable, Trace
from enterprise_agent.retrieval.service import Retriever


class Question(BaseModel):
    # Pydantic ignores unused benchmark fields, including reference answers.
    question_id: str
    question: str = Field(min_length=1)
    expected_doc_ids: set[DocumentID] = Field(default_factory=set)
    expected_spans: list[ReferenceSpan] = Field(default_factory=list)


def load_questions(path: Path, limit: int | None) -> list[Question]:
    if limit is not None and limit < 1:
        raise ValueError("question limit must be positive")
    with path.open(encoding="utf-8") as lines:
        questions = []
        for n, line in enumerate(
            islice((line for line in lines if line.strip()), limit), 1
        ):
            try:
                questions.append(Question.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"{path}, record {n}: invalid benchmark question"
                ) from exc
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("question set must be nonempty with unique question IDs")
    return questions


def _averages(metrics: list[dict]) -> dict:
    return {
        key: fmean(m[key] for m in metrics) if metrics else None
        for key in ("recall@1", "recall@5", "recall@10", "mrr")
    }


def evaluate(
    methods: dict[str, Retriever],
    questions: list[Question],
    *,
    chunk_depth: int,
    present: set[str] | None = None,
    filters: SearchFilter | None = None,
) -> dict:
    if chunk_depth < 10 or not questions:
        raise ValueError(
            "evaluation requires questions and at least 10 candidate chunks"
        )
    results = {}
    for name, retriever in methods.items():
        rows = []
        for count, q in enumerate(questions, 1):
            started = perf_counter()
            trace = None
            error = None
            if isinstance(retriever, PageIndexRetriever):
                try:
                    hits, trace = retriever.search_with_trace(
                        q.question, chunk_depth, filters
                    )
                except SelectorUnavailable as exc:
                    hits, trace, error = (
                        [],
                        exc.trace or Trace(stop_reason="model_error"),
                        str(exc),
                    )
            else:
                hits = retriever.search(q.question, chunk_depth)
            latency = (perf_counter() - started) * 1000
            doc_ids = list(dict.fromkeys(hit.document_id for hit in hits))
            rows.append(
                {
                    "question_id": q.question_id,
                    "latency_ms": latency,
                    "retrieved_documents": len(doc_ids),
                    "document_ids": doc_ids,
                    "metrics": document_metrics(doc_ids, q.expected_doc_ids),
                    "hit@10": bool(q.expected_doc_ids.intersection(doc_ids[:10])),
                    "gold_in_corpus": bool(q.expected_doc_ids & present)
                    if present is not None
                    else None,
                    "chunks": [
                        {
                            "chunk_id": h.chunk_id,
                            "document_id": h.document_id,
                            "rank": h.rank,
                            "score": h.score,
                            "component_ranks": h.component_ranks,
                            "spans": [list(span) for span in h.matched_spans]
                            if isinstance(h, PageIndexHit)
                            else [[h.start_offset, h.end_offset]],
                        }
                        for h in hits
                    ],
                }
            )
            if q.expected_spans:
                rows[-1]["span_metrics"] = span_recall(
                    rows[-1]["chunks"], q.expected_spans
                )
            if trace is not None:
                rows[-1].update(
                    trace=asdict(trace),
                    error=error,
                    candidate_metrics=document_metrics(
                        trace.candidates, q.expected_doc_ids
                    ),
                )
            if count % 25 == 0:
                logging.getLogger(__name__).info(
                    "%s completed %d/%d questions", name, count, len(questions)
                )
        eligible = [row["metrics"] for row in rows if row["metrics"] is not None]
        summary = _averages(eligible)
        spans = [row["span_metrics"] for row in rows if "span_metrics" in row]
        if spans:
            summary["annotated_evidence"] = {
                key: fmean(m[key] for m in spans) for key in spans[0]
            }
            summary["annotated_evidence"]["questions"] = len(spans)
        times = sorted(r["latency_ms"] for r in rows)
        summary.update(
            evaluated_questions=len(eligible),
            no_gold_questions=len(rows) - len(eligible),
            hits_at_10=sum(r["hit@10"] for r in rows),
            p50_latency_ms=times[ceil(len(times) * 0.5) - 1],
            p95_latency_ms=times[ceil(len(times) * 0.95) - 1],
            mean_latency_ms=fmean(r["latency_ms"] for r in rows),
            mean_retrieved_documents=fmean(r["retrieved_documents"] for r in rows),
        )
        if present is not None:
            subset = [r for r in rows if r["gold_in_corpus"]]
            summary["with_gold_in_corpus"] = {
                **_averages([r["metrics"] for r in subset]),
                "questions": len(subset),
                "hits_at_10": sum(r["hit@10"] for r in subset),
            }
        if isinstance(retriever, PageIndexRetriever):
            traces = [r["trace"] for r in rows]
            summary["candidate_metrics"] = _averages(
                [r["candidate_metrics"] for r in rows if r["candidate_metrics"]]
            )
            summary["navigation"] = {
                key: sum(t[key] for t in traces)
                for key in (
                    "calls",
                    "retries",
                    "input_tokens",
                    "output_tokens",
                    "missing_usage",
                )
            }
            summary["navigation"].update(
                stop_reasons=dict(Counter(t["stop_reason"] for t in traces)),
                errors=sum(r["error"] is not None for r in rows),
            )
        results[name] = {"summary": summary, "queries": rows}
    return results


def question_file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", dir=path.parent, encoding="utf-8", delete=False
    ) as output:
        tmp = Path(output.name)
        try:
            json.dump(
                report,
                output,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            output.write("\n")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)

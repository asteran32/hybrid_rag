import hashlib
import json
from itertools import islice
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from time import perf_counter

from pydantic import BaseModel, Field

from enterprise_agent.evaluation.metrics import document_metrics
from enterprise_agent.models import DocumentID
from enterprise_agent.retrieval.service import Retriever


class Question(BaseModel):
    # Pydantic ignores unused benchmark fields, including reference answers.
    question_id: str
    question: str = Field(min_length=1)
    expected_doc_ids: set[DocumentID] = Field(default_factory=set)


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


def evaluate(
    methods: dict[str, Retriever], questions: list[Question], *, chunk_depth: int
) -> dict:
    if chunk_depth < 10 or not questions:
        raise ValueError(
            "evaluation requires questions and at least 10 candidate chunks"
        )
    results = {}
    for name, retriever in methods.items():
        rows = []
        for q in questions:
            started = perf_counter()
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
                    "chunks": [
                        {
                            "chunk_id": h.chunk_id,
                            "document_id": h.document_id,
                            "rank": h.rank,
                            "score": h.score,
                            "component_ranks": h.component_ranks,
                        }
                        for h in hits
                    ],
                }
            )
        eligible = [row["metrics"] for row in rows if row["metrics"] is not None]
        summary = {
            key: fmean(m[key] for m in eligible) if eligible else None
            for key in ("recall@1", "recall@5", "recall@10", "mrr")
        }
        summary.update(
            evaluated_questions=len(eligible),
            no_gold_questions=len(rows) - len(eligible),
            mean_latency_ms=fmean(r["latency_ms"] for r in rows),
            mean_retrieved_documents=fmean(r["retrieved_documents"] for r in rows),
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

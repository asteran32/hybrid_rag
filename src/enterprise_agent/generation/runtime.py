import os

from psycopg import Connection
from pydantic import ValidationError

from enterprise_agent.generation.chat import generate
from enterprise_agent.generation.contracts import AnswerRequest, GroundedAnswer
from enterprise_agent.generation.service import answer_question
from enterprise_agent.models import AnswerScope
from enterprise_agent.retrieval.service import build_retriever
from enterprise_agent.storage.search import SearchStore
from enterprise_agent.validation.entailment import classify_local


def configured_scope() -> AnswerScope:
    raw_scope = os.getenv("ANSWER_ACCESS_SCOPE") or '{"source_types":[]}'
    try:
        return AnswerScope.model_validate_json(raw_scope)
    except ValidationError:
        raise ValueError(
            "ANSWER_ACCESS_SCOPE must specify a valid source and optional document allowlist"
        ) from None


def run_answer(db: Connection, request: AnswerRequest) -> GroundedAnswer:
    scope = configured_scope()
    if (
        not scope.source_types
        or scope.document_ids == ()
        or (
            request.source_type is not None
            and request.source_type not in scope.source_types
        )
    ):
        return GroundedAnswer(status="abstained", limitations=["insufficient_evidence"])
    store = SearchStore(db)
    retriever = build_retriever(
        store,
        request.method,
        candidates=request.candidates,
        constant=request.rrf_constant,
    )
    return answer_question(
        request,
        scope=scope,
        retriever=retriever,
        store=store,
        generate=generate,
        classify=classify_local,
    )

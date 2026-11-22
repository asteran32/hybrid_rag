from collections.abc import Callable
from typing import Protocol

from enterprise_agent.generation.contracts import (
    AnswerRequest,
    Claim,
    Conflict,
    Generation,
    GroundedAnswer,
    Limitation,
)
from enterprise_agent.models import AnswerScope, EvidenceChunk, SearchFilter
from enterprise_agent.retrieval.service import Retriever
from enterprise_agent.validation.citations import (
    Citation,
    Passage,
    resolve_citation,
    select_passages,
)
from enterprise_agent.validation.entailment import Classifier
from enterprise_agent.validation.support import review_support

Generate = Callable[[str, dict[str, str]], Generation]
CONTEXT_CHARS = 6000


class EvidenceStore(Protocol):
    def load_evidence(
        self, chunk_ids: list[str], filters: SearchFilter
    ) -> dict[str, EvidenceChunk]: ...


def _overlap(first: Citation, second: Citation) -> bool:
    return (
        first.document_id == second.document_id
        and first.start_offset < second.end_offset
        and second.start_offset < first.end_offset
    )


def finalize(
    generated: Generation,
    passages: dict[str, Passage],
    current: dict[str, EvidenceChunk],
    review_issues: tuple[Limitation, ...] = (),
) -> GroundedAnswer:
    draft = generated.outcome
    if isinstance(draft, str):
        return GroundedAnswer(
            status="abstained", limitations=[draft], generation=generated.info
        )
    # Even an uncited passage may have influenced the prose. Fail closed on any change.
    if any(current.get(p.chunk.chunk_id) != p.chunk for p in passages.values()):
        return GroundedAnswer(
            status="abstained",
            limitations=["evidence_changed"],
            generation=generated.info,
        )

    conflicts: list[Conflict] = []
    reasons: list[Limitation] = list(review_issues)
    invalid_conflict = False
    for pair in draft.conflicts:
        left = resolve_citation(pair.left, passages, current)
        right = resolve_citation(pair.right, passages, current)
        if (
            left is None
            or right is None
            or left.quote == right.quote
            or _overlap(left, right)
        ):
            invalid_conflict = True
        else:
            conflicts.append(Conflict(left=left, right=right))
    if conflicts:
        reasons.append("conflicting_evidence")

    claims: list[Claim] = []
    invalid_citation = invalid_conflict
    # An unverifiable side of a dispute cannot justify picking the remaining side.
    if draft.coverage != "insufficient" and not invalid_conflict:
        disputed = [side for pair in conflicts for side in (pair.left, pair.right)]
        for item in draft.claims:
            refs = [resolve_citation(ref, passages, current) for ref in item.citations]
            if not item.text.strip() or not refs or None in refs:
                invalid_citation = True
                continue
            citations = {
                (ref.document_id, ref.start_offset, ref.end_offset): ref
                for ref in refs
                if ref is not None
            }
            if any(
                _overlap(ref, side) for ref in citations.values() for side in disputed
            ):
                continue
            claims.append(
                Claim(
                    kind=item.kind,
                    text=item.text.strip(),
                    citations=list(citations.values()),
                )
            )
    if invalid_citation:
        reasons.append("invalid_citation")
    if not claims:
        reasons.append("insufficient_evidence")
    elif draft.coverage == "partial":
        reasons.append("incomplete_evidence")
    return GroundedAnswer(
        status=("abstained" if not claims else "partial" if reasons else "answered"),
        claims=claims,
        conflicts=conflicts,
        limitations=reasons,
        generation=generated.info,
    )


def answer_question(
    request: AnswerRequest,
    *,
    scope: AnswerScope,
    retriever: Retriever,
    store: EvidenceStore,
    generate: Generate,
    classify: Classifier,
) -> GroundedAnswer:
    filters = SearchFilter(
        source_type=request.source_type, metadata=request.metadata, scope=scope
    )
    hits = retriever.search(request.question, request.top_k, filters)[: request.top_k]
    stored = store.load_evidence([hit.chunk_id for hit in hits], filters)
    passages = select_passages(hits, stored, CONTEXT_CHARS)
    return answer_from_passages(
        request.question, passages, filters, store, generate, classify
    )


def answer_from_passages(
    question: str,
    passages: dict[str, Passage],
    filters: SearchFilter,
    store: EvidenceStore,
    generate: Generate,
    classify: Classifier,
) -> GroundedAnswer:
    if not passages:
        return GroundedAnswer(status="abstained", limitations=["insufficient_evidence"])
    texts = {label: p.text for label, p in passages.items()}
    generated = generate(question, texts)
    issues = []
    if not isinstance(generated.outcome, str):
        draft, issues = review_support(generated.outcome, passages, classify)
        generated = generated.model_copy(update={"outcome": draft})
    # Use a fresh DB read after the model call, which may outlive an edit or deletion.
    current = (
        store.load_evidence(
            list(dict.fromkeys(p.chunk.chunk_id for p in passages.values())), filters
        )
        if not isinstance(generated.outcome, str)
        else {}
    )
    return finalize(generated, passages, current, tuple(issues))

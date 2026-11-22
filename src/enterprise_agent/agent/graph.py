import logging
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from enterprise_agent.agent.budget import Budget, RunStopped
from enterprise_agent.agent.contracts import (
    AgentTrace,
    EvidenceRef,
    EvidenceReview,
    StopReason,
    ToolCall,
)
from enterprise_agent.agent.planning import Followup, OpenAIPlanner, Review
from enterprise_agent.agent.tools import SearchTools
from enterprise_agent.generation.contracts import GroundedAnswer, SearchRequest
from enterprise_agent.generation.service import (
    CONTEXT_CHARS,
    EvidenceStore,
    answer_from_passages,
)
from enterprise_agent.models import EvidenceChunk, SearchFilter, SearchHit
from enterprise_agent.retrieval.fusion import reciprocal_rank_fusion
from enterprise_agent.validation.citations import Passage, select_passages
from enterprise_agent.validation.entailment import Classifier, Relation


class State(TypedDict, total=False):
    pending: list[Followup]
    rankings: dict[str, list[SearchHit]]
    hits: list[SearchHit]
    passages: dict[str, Passage]
    disclosed: dict[str, EvidenceChunk]
    missing: list[int]
    reason: StopReason
    answer: GroundedAnswer


@dataclass
class AgentResult:
    hits: list[SearchHit]
    answer: GroundedAnswer | None
    trace: AgentTrace


def _checked_review(
    review: Review, tasks: dict[int, str], passages: dict[str, Passage]
) -> EvidenceReview:
    sources: dict[int, list[EvidenceRef]] = {}
    for item in review.supported:
        if (
            item.task not in tasks
            or item.task in sources
            or not set(item.passages) <= passages.keys()
        ):
            raise RunStopped("invalid_model_output")
        sources[item.task] = []
        for label in dict.fromkeys(item.passages):
            passage = passages[label]
            start = passage.chunk.start_offset + passage.chunk_offset
            sources[item.task].append(
                EvidenceRef(
                    document_id=passage.chunk.document_id,
                    chunk_id=passage.chunk.chunk_id,
                    start_offset=start,
                    end_offset=start + len(passage.text),
                )
            )
    return EvidenceReview(
        missing_tasks=sorted(tasks.keys() - sources.keys()), sources_by_task=sources
    )


def run_graph(
    request: SearchRequest,
    filters: SearchFilter,
    *,
    want_answer: bool,
    budget: Budget,
    planner: OpenAIPlanner,
    tools: SearchTools,
    store: EvidenceStore,
    classify: Classifier,
) -> AgentResult:
    trace = budget.trace

    def plan(state: State) -> dict:
        trace.last_stage = "plan"
        budget.remaining()
        if trace.settings.max_tokens == 0:
            raise RunStopped("token_limit")
        if trace.settings.max_cost_usd == 0:
            raise RunStopped("cost_limit")
        result = planner.plan(request.question)
        trace.complexity = result.complexity
        if result.complexity == "simple":
            steps = [Followup(task=1, tool="hybrid", query=request.question)]
        else:
            if (
                len({step.key[0] for step in result.searches}) != len(result.searches)
                or len(result.searches) < 2
            ):
                raise RunStopped("invalid_model_output")
            steps = [
                Followup(task=i, **step.model_dump())
                for i, step in enumerate(result.searches, 1)
            ]
        trace.tasks = [step.query for step in steps]
        return {"pending": steps}

    def search(state: State) -> dict:
        trace.last_stage = "search"
        rankings = dict(state["rankings"])
        retry = bool(rankings)
        for step in state["pending"]:
            budget.remaining()
            call = ToolCall(
                task=step.task, query=step.query, tool=step.tool, retry=retry
            )
            trace.tool_calls.append(call)
            started = budget.clock()
            try:
                group = tools.search(step.tool, step.query, request.top_k, filters)[
                    : request.top_k
                ]
                call.hits = len(group)
                rankings[f"search_{len(trace.tool_calls)}"] = group
            finally:
                call.elapsed_ms = round((budget.clock() - started) * 1000, 3)
        budget.remaining()
        hits = (
            next(iter(rankings.values()))
            if len(rankings) == 1
            else reciprocal_rank_fusion(rankings, request.top_k, request.rrf_constant)
        )
        return {"rankings": rankings, "hits": hits, "pending": []}

    def current(state: State) -> tuple[list[SearchHit], dict[str, EvidenceChunk]]:
        budget.remaining()
        # Include evidence from earlier rounds even if fusion has since displaced it.
        seen = state["disclosed"]
        ids = list(dict.fromkeys([*seen, *(hit.chunk_id for hit in state["hits"])]))
        stored = store.load_evidence(ids, filters)
        if any(stored.get(cid) != old for cid, old in seen.items()):
            raise RunStopped("evidence_changed")
        hits = []
        for hit in state["hits"]:
            chunk = stored.get(hit.chunk_id)
            if chunk is not None and chunk.document_id == hit.document_id:
                hits.append(
                    hit.model_copy(update={**chunk.model_dump(), "rank": len(hits) + 1})
                )
        budget.remaining()
        return hits, stored

    def review(state: State) -> dict:
        trace.last_stage = "review"
        hits, stored = current(state)
        passages = select_passages(hits[:20], stored, CONTEXT_CHARS)
        tasks = dict(enumerate(trace.tasks, 1))
        history = [
            call.model_dump(include={"task", "query", "tool"})
            for call in trace.tool_calls
        ]
        can_retry = (
            sum(call.retry for call in trace.tool_calls) < trace.settings.max_retries
        )
        decision = (
            planner.review(
                request.question,
                tasks,
                history,
                {key: p.text for key, p in passages.items()},
                can_retry,
            )
            if passages or can_retry
            else Review(supported=[], followup=None)
        )
        checked = _checked_review(decision, tasks, passages)
        trace.reviews.append(checked)
        missing = checked.missing_tasks
        changes = {
            "hits": hits,
            "passages": passages,
            "missing": missing,
            "disclosed": {
                **state["disclosed"],
                **{p.chunk.chunk_id: p.chunk for p in passages.values()},
            },
        }
        if not missing:
            return {**changes, "reason": "sufficient_evidence"}
        if not can_retry:
            return {**changes, "reason": "retry_limit"}
        followup = decision.followup
        if followup is None:
            return {**changes, "reason": "no_new_query"}
        if followup.task not in missing:
            raise RunStopped("invalid_model_output")
        previous = {
            (" ".join(call["query"].casefold().split()), call["tool"])
            for call in history
        }
        if followup.key in previous:
            return {**changes, "reason": "no_new_query"}
        # Recheck all passages before using a query derived from them in a new tool.
        current({**state, **changes})
        return {**changes, "pending": [followup]}

    def finish(state: State) -> dict:
        trace.last_stage = "finish"
        hits, _ = current(state)
        if not want_answer:
            return {"hits": hits}
        if state["passages"]:
            trace.last_stage = "generate"
        answer = answer_from_passages(
            request.question,
            state["passages"],
            filters,
            store,
            planner.generate,
            classify_before_deadline,
        )
        if "evidence_changed" in answer.limitations:
            raise RunStopped("evidence_changed")
        if (
            state["missing"]
            and answer.claims
            and "incomplete_evidence" not in answer.limitations
        ):
            answer = answer.model_copy(
                update={
                    "status": "partial",
                    "limitations": [*answer.limitations, "incomplete_evidence"],
                }
            )
        hits, _ = current(state)
        reason = next(
            (
                issue
                for issue in answer.limitations
                if issue in ("model_refusal", "invalid_model_output", "output_limit")
            ),
            state["reason"],
        )
        return {"hits": hits, "answer": answer, "reason": reason}

    def classify_before_deadline(pairs: list[tuple[str, str]]) -> list[Relation]:
        return budget.wait_for(lambda: classify(pairs))

    def build():
        graph = StateGraph(State)
        graph.add_node("plan", plan)
        graph.add_node("search", search)
        graph.add_node("review", review)
        graph.add_node("finish", finish)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "review")
        graph.add_conditional_edges(
            "review",
            lambda s: "search" if s["pending"] else "finish",
            ["search", "finish"],
        )
        graph.add_edge("finish", END)
        return graph.compile()

    try:
        if filters.scope is None or not filters.scope.allows_source(
            filters.source_type
        ):
            raise RunStopped("no_access")
        # Keep enterprise text within the configured services even if a global
        # LangSmith tracing environment is present. Only the application trace is emitted.
        with tracing_context(enabled=False):
            final = build().invoke(
                {"rankings": {}, "disclosed": {}},
                {"recursion_limit": 8 + 2 * trace.settings.max_retries},
            )
        budget.remaining()
        return AgentResult(
            final["hits"], final.get("answer"), budget.finish(final["reason"])
        )
    except RunStopped as exc:
        for review_result in trace.reviews:
            review_result.sources_by_task.clear()
        for call in trace.tool_calls:
            if call.retry:
                call.query = None
        limitation = {
            "evidence_changed": "evidence_changed",
            "model_refusal": "model_refusal",
            "invalid_model_output": "invalid_model_output",
            "output_limit": "output_limit",
            "time_limit": "agent_budget_exhausted",
            "token_limit": "agent_budget_exhausted",
            "cost_limit": "agent_budget_exhausted",
        }.get(exc.reason, "insufficient_evidence")
        answer = (
            GroundedAnswer(status="abstained", limitations=[limitation])
            if want_answer
            else None
        )
        return AgentResult([], answer, budget.finish(exc.reason))
    except Exception as exc:
        trace.error_type = type(exc).__name__
        budget.finish("service_error")
        raise
    finally:
        # Operator logs retain timing, routing, usage and failures without source-derived queries.
        level = (
            logging.WARNING if trace.stop_reason == "service_error" else logging.INFO
        )
        logging.getLogger(__name__).log(
            level,
            "agent_trace %s",
            trace.model_dump_json(
                exclude={
                    "tasks": True,
                    "tool_calls": {"__all__": {"query"}},
                    "reviews": {"__all__": {"sources_by_task"}},
                }
            ),
        )

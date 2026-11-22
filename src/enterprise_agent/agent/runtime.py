from typing import TYPE_CHECKING

from psycopg import Connection

from enterprise_agent.agent.budget import Budget
from enterprise_agent.agent.contracts import AgentSettings
from enterprise_agent.agent.planning import OpenAIPlanner
from enterprise_agent.agent.tools import SearchTools, TimedSearchStore
from enterprise_agent.generation.chat import GenerationUnavailable
from enterprise_agent.generation.contracts import SearchRequest
from enterprise_agent.models import AnswerScope, SearchFilter
from enterprise_agent.validation.entailment import classify_local

if TYPE_CHECKING:
    from enterprise_agent.agent.graph import AgentResult


def run_agent(
    db: Connection, request: SearchRequest, scope: AnswerScope, *, want_answer: bool
) -> "AgentResult":
    try:
        from enterprise_agent.agent.graph import run_graph
    except ModuleNotFoundError as exc:
        if exc.name != "langgraph":
            raise
        raise GenerationUnavailable(
            "install LangGraph with uv sync --extra answer"
        ) from None
    budget = Budget(AgentSettings.from_env())
    store = TimedSearchStore(db, budget)
    return run_graph(
        request,
        SearchFilter(
            source_type=request.source_type, metadata=request.metadata, scope=scope
        ),
        want_answer=want_answer,
        budget=budget,
        planner=OpenAIPlanner(budget),
        tools=SearchTools(store, budget, request.candidates, request.rrf_constant),
        store=store,
        classify=classify_local,
    )

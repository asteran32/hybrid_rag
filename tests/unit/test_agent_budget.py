from decimal import Decimal
from threading import Event

import pytest
from pydantic import ValidationError

from enterprise_agent.agent.budget import Budget, RunStopped
from enterprise_agent.agent.contracts import DEFAULT_MODEL, AgentSettings, ModelCall
from enterprise_agent.agent.tools import MeteredEmbeddings
from enterprise_agent.embeddings import ModelSpec


def reservation(inputs=100, outputs=50):
    return ModelCall(
        stage="review",
        model=DEFAULT_MODEL,
        input_tokens=inputs,
        output_tokens=outputs,
        cost_usd=0,
    )


def test_reserve_before_dispatch_and_release_only_on_reported_usage():
    budget = Budget(AgentSettings(max_tokens=200, max_cost_usd="0.000210"))
    rates = Decimal(1), Decimal(2)
    first = reservation()
    budget.reserve(first, *rates)
    assert budget.trace.estimated_cost_usd == Decimal("0.000200")
    with pytest.raises(RunStopped, match="token_limit"):
        budget.reserve(reservation(), *rates)
    with pytest.raises(RunStopped, match="cost_limit"):
        budget.reserve(reservation(0, 10), *rates)
    assert budget.trace.model_calls == [first] and not first.usage_reported
    budget.settle(first, 8, 3, *rates)
    budget.reserve(reservation(100, 40), *rates)
    assert budget.trace.accounted_tokens == 151
    assert budget.trace.estimated_cost_usd == Decimal("0.000194")
    assert [call.usage_reported for call in budget.trace.model_calls] == [True, False]


def test_deadline_stops_blocking_work_and_forbids_another_call():
    budget = Budget(AgentSettings(max_seconds=0.05))
    release = Event()
    try:
        with pytest.raises(RunStopped, match="time_limit"):
            budget.wait_for(lambda: release.wait(timeout=3))
        assert not release.is_set()
        with pytest.raises(RunStopped, match="time_limit"):
            budget.wait_for(lambda: pytest.fail("work dispatched after deadline"))
    finally:
        release.set()


def test_model_switch_requires_matching_rates_and_limits_are_finite(monkeypatch):
    monkeypatch.setenv("RAG_AGENT_MODEL", "another-model")
    with pytest.raises(ValidationError, match="RAG_AGENT_INPUT_USD_PER_MILLION"):
        AgentSettings.from_env()
    monkeypatch.setenv("RAG_AGENT_INPUT_USD_PER_MILLION", "1.5")
    monkeypatch.setenv("RAG_AGENT_OUTPUT_USD_PER_MILLION", "3")
    assert AgentSettings.from_env().input_usd_per_million == Decimal("1.5")
    with pytest.raises(ValidationError):
        AgentSettings(max_seconds=float("inf"))


def test_paid_embedding_uses_shared_ledger_and_remaining_deadline():
    class Embeddings:
        spec = ModelSpec(
            provider="openai", model="text-embedding-3-small", dimensions=2
        )

        def embed_with_usage(self, texts, *, timeout):
            assert texts == ["meal"] and 0 < timeout <= 60
            return [[1.0, 0.0]], 1

    budget = Budget(AgentSettings())
    assert MeteredEmbeddings(Embeddings(), budget).embed(["meal"]) == [[1.0, 0.0]]
    assert budget.trace.accounted_tokens == 1
    assert budget.trace.estimated_cost_usd == Decimal("0.00000002")
    assert budget.trace.model_calls[0].stage == "embed"

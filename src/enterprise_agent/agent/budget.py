from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from time import monotonic
from typing import TypeVar

from enterprise_agent.agent.contracts import (
    AgentSettings,
    AgentTrace,
    ModelCall,
    StopReason,
)

T = TypeVar("T")
_provider_workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-provider")


class RunStopped(Exception):
    def __init__(self, reason: StopReason):
        self.reason = reason
        super().__init__(reason)


class Budget:
    """Reserve before dispatch. An interrupted or unmetered call keeps its reservation."""

    def __init__(self, settings: AgentSettings, clock: Callable[[], float] = monotonic):
        self.trace = AgentTrace(settings=settings)
        self.clock = clock
        self.start = clock()
        self.deadline = self.start + settings.max_seconds

    def remaining(self) -> float:
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise RunStopped("time_limit")
        return remaining

    def wait_for(self, operation: Callable[[], T]) -> T:
        """Bound a blocking provider/CPU call. Workers never touch the request's DB or ledger."""
        self.remaining()

        def dispatch() -> T:
            self.remaining()  # The operation may have waited for a worker.
            return operation()

        future = _provider_workers.submit(dispatch)
        try:
            value = future.result(timeout=self.remaining())
        except TimeoutError:
            future.cancel()
            raise RunStopped("time_limit") from None
        self.remaining()
        return value

    def reserve(
        self, call: ModelCall, input_rate: Decimal, output_rate: Decimal
    ) -> None:
        self.remaining()
        trace = self.trace
        tokens = call.input_tokens + call.output_tokens
        cost = (
            input_rate * call.input_tokens + output_rate * call.output_tokens
        ) / 1_000_000
        if trace.accounted_tokens + tokens > trace.settings.max_tokens:
            raise RunStopped("token_limit")
        if trace.estimated_cost_usd + cost > trace.settings.max_cost_usd:
            raise RunStopped("cost_limit")
        call.cost_usd = cost
        trace.model_calls.append(call)
        trace.accounted_tokens += tokens
        trace.estimated_cost_usd += cost

    def settle(
        self,
        call: ModelCall,
        inputs: int,
        outputs: int,
        input_rate: Decimal,
        output_rate: Decimal,
    ) -> None:
        trace = self.trace
        cost = (input_rate * inputs + output_rate * outputs) / 1_000_000
        trace.estimated_cost_usd += cost - call.cost_usd
        trace.accounted_tokens += (
            inputs + outputs - call.input_tokens - call.output_tokens
        )
        call.input_tokens, call.output_tokens = inputs, outputs
        call.cost_usd, call.usage_reported = cost, True
        if trace.accounted_tokens > trace.settings.max_tokens:
            raise RunStopped("token_limit")
        if trace.estimated_cost_usd > trace.settings.max_cost_usd:
            raise RunStopped("cost_limit")

    def finish(self, reason: StopReason) -> AgentTrace:
        self.trace.stop_reason = reason
        self.trace.elapsed_ms = round((self.clock() - self.start) * 1000, 3)
        return self.trace

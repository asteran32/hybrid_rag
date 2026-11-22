import os
from decimal import Decimal
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from enterprise_agent.models import DocumentID

DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
PROMPT_VERSION = "agentic-retrieval-v1"
Tool = Literal["keyword", "dense", "hybrid"]
Complexity = Literal["simple", "complex"]
StopReason = Literal[
    "sufficient_evidence",
    "retry_limit",
    "no_new_query",
    "no_access",
    "time_limit",
    "cost_limit",
    "token_limit",
    "invalid_model_output",
    "model_refusal",
    "output_limit",
    "evidence_changed",
    "service_error",
]


class AgentSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_seconds: float = Field(default=60, gt=0)
    max_tokens: int = Field(default=64000, ge=0)
    max_cost_usd: Decimal = Field(default=Decimal("0.05"), ge=0)
    input_usd_per_million: Decimal = Field(default=Decimal("0.75"), gt=0)
    output_usd_per_million: Decimal = Field(default=Decimal("4.50"), gt=0)

    @model_validator(mode="after")
    def require_model_prices(self) -> Self:
        if (
            self.model != DEFAULT_MODEL
            and not {"input_usd_per_million", "output_usd_per_million"}
            <= self.model_fields_set
        ):
            raise ValueError(
                "set both RAG_AGENT_INPUT_USD_PER_MILLION and RAG_AGENT_OUTPUT_USD_PER_MILLION for a different RAG_AGENT_MODEL"
            )
        return self

    @classmethod
    def from_env(cls) -> Self:
        return cls.model_validate(
            {
                field: value
                for field in cls.model_fields
                if (value := os.getenv("RAG_AGENT_" + field.upper())) is not None
            }
        )


class SearchStep(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str
    tool: Tool

    @field_validator("query")
    @classmethod
    def bounded_query(cls, value: str) -> str:
        if not 1 <= len(value) <= 4000:
            raise ValueError("search query must contain 1 to 4000 characters")
        return value

    @property
    def key(self) -> tuple[str, Tool]:
        return " ".join(self.query.casefold().split()), self.tool


class ToolCall(BaseModel):
    task: int
    tool: Tool
    query: str | None
    retry: bool
    hits: int = 0
    elapsed_ms: float = 0


class ModelCall(BaseModel):
    stage: Literal["plan", "review", "embed", "generate"]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    usage_reported: bool = False


class EvidenceRef(BaseModel):
    document_id: DocumentID
    chunk_id: str
    start_offset: int
    end_offset: int


class EvidenceReview(BaseModel):
    missing_tasks: list[int]
    sources_by_task: dict[int, list[EvidenceRef]]


class AgentTrace(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    settings: AgentSettings
    prompt_version: str = PROMPT_VERSION
    complexity: Complexity | None = None
    tasks: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model_calls: list[ModelCall] = Field(default_factory=list)
    reviews: list[EvidenceReview] = Field(default_factory=list)
    last_stage: Literal["scope", "plan", "search", "review", "finish", "generate"] = (
        "scope"
    )
    stop_reason: StopReason | None = None
    error_type: str | None = None
    elapsed_ms: float = 0
    estimated_cost_usd: Decimal = Decimal(0)
    accounted_tokens: int = 0

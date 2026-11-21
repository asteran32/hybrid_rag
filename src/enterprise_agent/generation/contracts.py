from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    computed_field,
)

from enterprise_agent.validation.citations import Citation, PassageRef

ClaimKind = Literal["fact", "inference"]
GenerationIssue = Literal["model_refusal", "output_limit", "invalid_model_output"]
Limitation = (
    Literal[
        "insufficient_evidence",
        "incomplete_evidence",
        "conflicting_evidence",
        "invalid_citation",
        "evidence_changed",
        "unsupported_claim",
        "unverified_conflict",
    ]
    | GenerationIssue
)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    method: Literal["keyword", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    source_type: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    candidates: int = Field(default=50, ge=1, le=500)
    rrf_constant: int = Field(default=60, ge=0)


class DraftClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: ClaimKind
    text: str
    citations: list[PassageRef] = Field(max_length=8)


class DraftConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: PassageRef
    right: PassageRef


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coverage: Literal["complete", "partial", "insufficient"]
    claims: list[DraftClaim] = Field(max_length=8)
    conflicts: list[DraftConflict] = Field(max_length=4)


class GenerationInfo(BaseModel):
    provider: str
    model: str
    prompt_version: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Generation(BaseModel):
    outcome: AnswerDraft | GenerationIssue
    info: GenerationInfo


class Claim(BaseModel):
    kind: ClaimKind
    text: str
    citations: list[Citation] = Field(min_length=1)


class Conflict(BaseModel):
    left: Citation
    right: Citation


class GroundedAnswer(BaseModel):
    status: Literal["answered", "partial", "abstained"]
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    generation: GenerationInfo | None = None

    @computed_field
    @property
    def answer(self) -> str | None:
        lines = [
            ("[inference] " if claim.kind == "inference" else "") + claim.text
            for claim in self.claims
        ]
        return "\n".join(lines) or None

"""State is checkpointed per conversation; model responses are strictly validated."""
from typing import Literal, TypedDict
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Understanding(StrictModel):
    new_topic: bool
    question: str = Field(min_length=1, max_length=1500)
    slots: dict[str, str]
    missing: list[str]
    clarification: str


class Citation(StrictModel):
    evidence_id: str
    quote: str | None = Field(default=None, min_length=4, max_length=700)
    # Optional for historical unique quotes; new model calls select an offered ID.
    span_id: str | None = Field(default=None, min_length=1, max_length=160)
    table_row_id: str | None = Field(default=None, min_length=1, max_length=180)

    @model_validator(mode='after')
    def one_reference_kind(self):
        if self.table_row_id is not None:
            if self.quote is not None or self.span_id is not None:
                raise ValueError('Table row references cannot supply a quote or span')
        elif self.quote is None:
            raise ValueError('A text quote or a table row reference is required')
        return self


class EvidenceCheck(StrictModel):
    requirement: str = Field(min_length=1, max_length=500)
    required_for_answer: bool = True
    status: Literal["supported", "knowledge_gap", "user_gap", "unavailable", "not_applicable"]
    evidence_ids: list[str] = Field(default_factory=list)
    detail: str = Field(min_length=1, max_length=700)


class Assessment(StrictModel):
    decision: Literal["answer", "clarify", "retrieve_more", "insufficient"]
    evidence_ids: list[str]
    reason: str = Field(min_length=1, max_length=1200)
    answerable_parts: list[str] = Field(default_factory=list, max_length=10)
    missing_parts: list[str] = Field(default_factory=list, max_length=10)
    retrieval_queries: list[str] = Field(default_factory=list, max_length=2)
    missing: list[str] = Field(default_factory=list)
    clarification: str = ""
    checks: list[EvidenceCheck] = Field(default_factory=list, max_length=10)
    critical_conditions: list[Citation] = Field(default_factory=list, max_length=4)
    boundary: Literal["none", "unknown_fault", "unknown_compatibility", "future",
                      "service", "device_history", "missing_evidence"] = "none"


class Claim(StrictModel):
    kind: Literal["fact", "inference", "advice"] = "fact"
    text: str = Field(min_length=1, max_length=1500)
    citations: list[Citation] = Field(min_length=1, max_length=5)


class Answer(StrictModel):
    claims: list[Claim] = Field(min_length=1, max_length=12)


class AnswerRevision(StrictModel):
    issues: list[str] = Field(max_length=10)
    answer: Answer


class RAGState(TypedDict, total=False):
    aspect_plan: dict
    aspect_coverage: list[dict]
    aspect_queries: list[dict]
    output_audits: list[dict]
    aspect_revision: dict
    output_fallback: dict
    message: str
    force_new_topic: bool
    original_question: str
    user_utterances: list[str]
    question: str
    slots: dict
    history: list[dict]
    pending: bool
    clarification_count: int
    missing: list[str]
    clarification: str
    query: str
    evidence: list[dict]
    initial_evidence: list[dict]
    retrieval_rounds: int
    retrieval_queries: list[str]
    retrieval_log: list[dict]
    context_expansions: list[str]
    answerable_parts: list[str]
    missing_parts: list[str]
    assessment_decision: str
    boundary: str
    partial_answer: bool
    generation_repair: dict
    generation_skipped: dict
    budget_fallback: dict
    evidence_checks: list[dict]
    critical_conditions: list[dict]
    condition_evidence_added: list[str]
    assessment_format_repair: dict
    assessment_route_repair: dict
    generation_format_repair: dict
    condition_locator_repairs: list[dict]
    verification_requests: list[dict]
    decision_override: dict
    draft_answer: dict
    generation_review: dict
    generation_base_clarification_count: int
    selected_ids: list[str]
    sufficient: bool
    reason: str
    response: str
    citations: list[dict]
    trace: list[dict]
    error: str
    status: Literal["running", "clarify", "answered", "partial", "insufficient", "error"]

"""C-only contracts: a stable question plan, evidence coverage and output audit."""
from typing import Literal
from pydantic import Field
from .state import StrictModel, Citation


class Aspect(StrictModel):
    aspect_id: str = Field(min_length=1, max_length=80)
    requirement: str = Field(min_length=1, max_length=500)
    category: Literal['operation', 'version', 'model', 'numeric', 'condition', 'conclusion', 'user_info', 'boundary']
    required: bool
    reason: str = Field(min_length=1, max_length=500)


class AspectPlan(StrictModel):
    aspects: list[Aspect] = Field(min_length=1, max_length=10)
    boundary: Literal['none', 'unknown_fault', 'unknown_compatibility', 'future', 'service', 'device_history', 'missing_evidence']


class AspectCoverage(StrictModel):
    aspect_id: str
    status: Literal['supported', 'missing', 'conflict', 'user_gap', 'unavailable']
    citations: list[Citation] = Field(max_length=5)
    detail: str = Field(min_length=1, max_length=700)
    query: str = Field(max_length=1500)
    question_to_user: str = Field(max_length=500)


class CoverageReport(StrictModel):
    coverage: list[AspectCoverage] = Field(min_length=1, max_length=10)


class AspectAnswerCheck(StrictModel):
    aspect_id: str
    status: Literal['addressed', 'missing', 'appropriately_limited']
    claim_indices: list[int] = Field(max_length=12)
    detail: str = Field(min_length=1, max_length=500)


class ClaimCheck(StrictModel):
    claim_index: int = Field(ge=0)
    status: Literal['supported', 'unsupported', 'contradicted', 'irrelevant']
    detail: str = Field(min_length=1, max_length=500)


class OutputAudit(StrictModel):
    aspects: list[AspectAnswerCheck] = Field(min_length=1, max_length=10)
    claims: list[ClaimCheck] = Field(min_length=1, max_length=12)

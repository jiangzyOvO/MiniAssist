"""Allowlisted diagnostics; never serialize arbitrary exception/provider bodies."""
from pydantic import ValidationError


class CitationError(ValueError):
    def __init__(self, code, *, claim_index=None, citation_index=None, answer=None):
        super().__init__(code)
        self.diagnostic = {"code": code, "claim_index": claim_index,
                           "citation_index": citation_index}
        if answer is not None:
            # Only schema-validated generated claims, never an HTTP response body.
            self.diagnostic["generated_answer"] = answer


class DecisionError(ValueError):
    def __init__(self, code, assessment, **details):
        super().__init__(code)
        self.diagnostic = {"code": code, "assessment": assessment, **details}


class AttributionError(ValueError):
    def __init__(self, claim_index, answer):
        super().__init__("user_document_identity_unverified")
        self.diagnostic = {"code": "user_document_identity_unverified", "claim_index": claim_index,
                           "generated_answer": answer}


def schema_issues(exc):
    # Locations may include user-provided dictionary keys: do not log those keys.
    allowed = {"claims", "text", "citations", "evidence_id", "quote", "span_id", "question",
               "new_topic", "slots", "missing", "clarification", "sufficient",
               "evidence_ids", "reason", "decision", "answerable_parts", "missing_parts",
               "retrieval_queries", "boundary", "kind", "checks", "critical_conditions",
               "requirement", "status", "detail", "issues", "answer", "required_for_answer", "table_row_id"}
    return [{"type": e["type"], "location": [
        x if isinstance(x, int) or x in allowed else "<field>" for x in e["loc"]]}
        for e in exc.errors(include_input=False, include_context=False, include_url=False)]


def diagnostic_for(exc):
    if isinstance(exc, (CitationError, DecisionError, AttributionError)):
        return exc.diagnostic
    if isinstance(exc, ValidationError):
        return {"code": "schema_invalid", "issues": schema_issues(exc)}
    # ModelError deliberately exposes only locally constructed diagnostics.
    from .llm import ModelError
    if isinstance(exc, ModelError):
        return exc.diagnostic
    return {"code": "unexpected_exception", "exception_type": type(exc).__name__}

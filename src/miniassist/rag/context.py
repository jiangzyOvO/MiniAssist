"""Separate literal user statements, model-derived context and source evidence.

Catalog metadata stays in state for attribution, never becomes case information.
This is an input boundary, not a semantic guarantee about a model's conclusions.
"""
import re
from .citations import model_citation_spans
from .table_evidence import model_table_rows
from .errors import AttributionError


def user_context(state):
    utterances = list(state.get("user_utterances", []))
    if not utterances:
        # Historical replay states predate user_utterances. Never reconstruct user
        # facts from resolved questions, slots, assistant messages or source metadata.
        original = state.get("original_question", "")
        message = state.get("message", "")
        utterances = [original] if original else []
        if message and message != original:
            utterances.append(message)
    return {"utterances": utterances, "origin": "literal_user_messages"}


def model_context(state):
    return {"resolved_question": state.get("question", ""), "slots": dict(state.get("slots", {})),
            "origin": "model_derived_not_independent_evidence"}


def evidence_payload(document):
    # Deliberate allowlist: version/date/title annotations may describe a retrieved
    # document, but cannot establish the identity of a document the user owns.
    # Original text is not edited: genuine in-body versions/dates remain citable.
    return {**{key: document[key] for key in ("chunk_id", "text", "source_id", "pages")},
            "citable_spans": model_citation_spans(document),
            "table_rows": model_table_rows(document)}


_DOCUMENT = r"(?:手册|说明书|文档|资料)"
_PERSONAL_DOCUMENT = re.compile(r"(?:你|您|用户)[^。；\n]{0,20}" + _DOCUMENT)
_VERSION_DATE = r"(?:[vV]\d+(?:\.\d+)+|(?:19|20|21)\d{2}(?:[.年/-]\d{1,2})?(?:[.月/-]\d{1,2})?)"
_IDENTITY_VALUE = re.compile(_DOCUMENT + r"[^，。；\n]{0,16}?(" + _VERSION_DATE + r")")
_CHRONOLOGY = re.compile(r"早于|晚于|发布于|出版于|发行于|印刷于|早期|较早|未更新|尚未更新|内容落后|版本落后|因此没有介绍|所以没有介绍")
_QUALIFIED = re.compile(r"无法(?:确认|确定|判断)|不能(?:确认|确定|判断|认定|断言|推断)|尚未(?:确认|核实)|如果|假如|若")
_USER_DOC_DATE = re.compile(_DOCUMENT + r"[^。；？！\n]{0,20}(?:19|20|21)\d{2}|"
                            r"(?:19|20|21)\d{2}[^。；？！\n]{0,20}" + _DOCUMENT)


def validate_case_attribution(state, answer):
    """Narrow Chinese guard for affirmative claims about the user's own document.

    This is deliberately not a universal entailment checker. Retrieved source dates
    cannot fill missing personal-document dates. Explicit caveats/hypotheses remain
    allowed; genuine user-supplied dates still require ordinary semantic review.
    """
    user_text = "\n".join(user_context(state)["utterances"])
    for index, claim in enumerate(answer.claims):
        text = claim.text
        if not _PERSONAL_DOCUMENT.search(text):
            continue
        for match in _IDENTITY_VALUE.finditer(text):
            if match.group(1).lower() not in user_text.lower() and not _QUALIFIED.search(text[max(0, match.start()-20):match.start()]):
                raise AttributionError(index, answer.model_dump())
        if not _USER_DOC_DATE.search(user_text):
            for match in _CHRONOLOGY.finditer(text):
                if not _QUALIFIED.search(text[max(0, match.start()-20):match.start()]):
                    raise AttributionError(index, answer.model_dump())

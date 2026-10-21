"""Bounded graph nodes with explicit failure and insufficient-evidence exits."""
import time
from .state import Understanding, Assessment, Answer, AnswerRevision, Claim
from .errors import CitationError, DecisionError, AttributionError, diagnostic_for
from .citations import resolve_quote, resolve_citation, resolve_reference
from .table_evidence import model_table_rows, validate_category_models
from .boundaries import BOUNDARIES
from .context import user_context, model_context, evidence_payload, validate_case_attribution
from .verification import verification_queries

ALLOWED_SLOTS = {"aircraft", "controller", "accessory", "firmware", "symptom"}


class Nodes:
    def __init__(self, llm, retriever, settings):
        self.llm, self.retriever, self.settings = llm, retriever, settings

    def guarded(self, name):
        action = getattr(self, name)

        def run(state):
            started = time.monotonic()
            input_keys = {
                "plan_aspects": ("original_question", "user_utterances", "question"),
                "audit_answer": ("aspect_plan", "aspect_coverage", "draft_answer", "selected_ids", "critical_conditions"),
                "understand": ("message", "original_question", "question", "pending", "slots", "history", "user_utterances"),
                "clarify": ("missing", "clarification", "clarification_count"),
                "retrieve": ("query",),
                "retrieve_more": ("retrieval_queries", "retrieval_rounds", "evidence"),
                "assess": ("question", "original_question", "slots", "evidence", "retrieval_rounds",
                           "missing", "clarification", "user_utterances"),
                "generate": ("question", "slots", "evidence", "selected_ids",
                             "original_question", "answerable_parts", "missing_parts", "reason", "boundary",
                             "critical_conditions", "missing", "clarification", "user_utterances"),
                "insufficient": ("reason", "boundary", "missing_parts"),
                "review_answer": ("question", "original_question", "draft_answer", "critical_conditions",
                                  "evidence", "selected_ids"),
                "finish": ("status",),
            }[name]
            node_input = {k: state.get(k) for k in input_keys}
            try:
                update = action(state)
                entry = {"node": name, "input": node_input, "result": dict(update)}
            except Exception as exc:
                # Provider payloads and exceptions may contain secrets: retain type only.
                update = {"status": "error", "error": f"{name}:{type(exc).__name__}",
                          "response": "本轮处理失败，请稍后重试；尚未形成可核验的答案。",
                          "pending": False, "citations": []}
                entry = {"node": name, "input": node_input, "error": update["error"],
                         "diagnostic": diagnostic_for(exc)}
            entry["duration_ms"] = round((time.monotonic() - started) * 1000)
            update["trace"] = state.get("trace", []) + [entry]
            return update
        return run

    def understand(self, state):
        data = Understanding.model_validate(self.llm.call("understand", {
            "user_context": user_context(state),
            "message": state["message"], "pending": state.get("pending", False),
            "original_question": state["original_question"],
            "previous_question": state.get("question", ""),
            "slots": state.get("slots", {}), "history": state.get("history", []),
        })).model_dump(exclude_none=True)
        if set(data["slots"]) - ALLOWED_SLOTS:
            raise ValueError("Unknown slot")
        if data["missing"] and not data["clarification"].strip():
            raise ValueError("Missing clarification")
        if not data["missing"] and data["clarification"]:
            raise ValueError("Unexpected clarification")
        # New-topic classification explicitly clears obsolete conversation slots.
        slots = {} if data["new_topic"] else dict(state.get("slots", {}))
        slots.update(data["slots"])
        count = 0 if data["new_topic"] else state.get("clarification_count", 0)
        # Slots remain available for applicability checks. Repeating them in the
        # full-text query distorts retrieval scores, especially product names.
        query = data["question"]
        return {"question": data["question"],
                "user_utterances": [state["message"]] if data["new_topic"] else user_context(state)["utterances"],
                "original_question": state["message"] if data["new_topic"] else state["original_question"],
                "slots": slots, "missing": data["missing"],
                "clarification": data["clarification"], "query": query.strip(),
                "clarification_count": count}

    def clarify(self, state):
        if state.get("clarification_count", 0) >= self.settings.max_clarifications:
            return {"status": "insufficient", "pending": False,
                    "response": "关键设备或使用条件仍未确认，暂时无法给出可靠结论。请整理型号和具体现象后重新提问。"}
        return {"status": "clarify", "pending": True,
                "clarification_count": state.get("clarification_count", 0) + 1,
                "response": "\n\n".join(filter(None, (
                    BOUNDARIES.get(state.get("boundary", "none"), ""), state["clarification"])))}

    def retrieve(self, state):
        docs = self.retriever.search(state["query"])
        related = getattr(self.retriever, "related_release_context", lambda docs, limit: [])(docs, 2)
        return {"evidence": docs + related, "initial_evidence": docs, "pending": False,
                "context_expansions": [d["chunk_id"] for d in related],
                "retrieval_log": [{"query": state["query"], "round": 0,
                                   "chunk_ids": [d["chunk_id"] for d in docs]}]}

    def retrieve_more(self, state):
        rounds = state.get("retrieval_rounds", 0)
        if rounds >= self.settings.max_retrieval_rounds:
            raise ValueError("Retrieval budget exceeded")
        docs = {d["chunk_id"]: d for d in state["evidence"]}
        log = list(state.get("retrieval_log", []))
        queries = state["retrieval_queries"]
        if not 1 <= len(queries) <= 2:
            raise ValueError("Invalid follow-up query budget")
        for query in queries:
            if not query.strip() or len(query) > 1500:
                raise ValueError("Invalid follow-up query")
            found = self.retriever.search(query)
            log.append({"query": query, "round": rounds + 1,
                        "chunk_ids": [d["chunk_id"] for d in found]})
            for doc in found:
                docs.setdefault(doc["chunk_id"], doc)
        expanded = list(state.get("context_expansions", []))
        remaining = 2 - len(expanded)
        if remaining > 0:
            related = getattr(self.retriever, "related_release_context",
                              lambda docs, limit: [])(list(docs.values()), remaining)
            for doc in related:
                docs.setdefault(doc["chunk_id"], doc)
                expanded.append(doc["chunk_id"])
        return {"evidence": list(docs.values()), "retrieval_rounds": rounds + 1,
                "retrieval_log": log, "context_expansions": expanded}

    def assess(self, state):
        try:
            return self._assess_once(state)
        except DecisionError as first:
            if first.diagnostic['code'] not in {'missing_clarification_fields', 'invalid_service_boundary_route',
                                                'condition_not_in_source', 'unsupported_check', 'invalid_not_applicable_check'}:
                raise
            previous_format = getattr(self.llm, 'last_call_diagnostic', {})
            if previous_format.get('attempts'):
                first.diagnostic['format_repair'] = previous_format
                raise
            # Rejudge the original user question and evidence; do not force an
            # insufficient exit just because the first model chose a boundary.
            try:
                result = self._assess_once(state, {'code': first.diagnostic['code'],
                    'failure_location': {k: first.diagnostic[k] for k in ('condition_index', 'check_index', 'citation_error') if k in first.diagnostic},
                    'previous_assessment': first.diagnostic['assessment'],
                    'instruction': '在原证据与原问题范围内修复指定字段。表格事实选择当前evidence.table_rows内对应的table_row_id，引用时省略quote和span_id；普通引文逐字复制citable_spans，不改字或拼接标签。不得删除必需条件来过校验。不适用检查项省略或标not_applicable且required_for_answer=false、evidence_ids=[]；确有事实支持的项仍须证据，不得伪填ID。保留真实knowledge_gap与补查意图。missing只放用户可补充的信息并配clarification；服务不可提供的事项用unavailable，future/service/device_history只用于实际服务边界，不得为修格式将可答问题改成拒答。'})
            except Exception as second:
                first.diagnostic['repair_failure'] = diagnostic_for(second)
                raise first from None
            result['assessment_route_repair'] = {'attempts': 1, 'status': 'validated',
                'first_failure': first.diagnostic}
            return result

    def _assess_once(self, state, repair=None):
        if not state["evidence"]:
            return {"sufficient": False, "assessment_decision": "insufficient",
                    "boundary": "missing_evidence", "selected_ids": [],
                    "reason": "本次检索没有返回资料。"}
        evidence = [evidence_payload(c) for c in state["evidence"]]
        payload = {
            "user_context": user_context(state), "model_context": model_context(state), "evidence": evidence,
            "original_question": state.get("original_question", state["question"]),
            "proposed_missing": state.get("missing", []),
            "proposed_clarification": state.get("clarification", ""),
            "retrieval_rounds_remaining": max(0, self.settings.max_retrieval_rounds -
                                               state.get("retrieval_rounds", 0)),
        }
        if repair:
            payload['routing_repair'] = repair
        data = Assessment.model_validate(self.llm.call('assess', payload)).model_dump(exclude_none=True)
        known = {c["chunk_id"] for c in state["evidence"]}
        if set(data["evidence_ids"]) - known or len(set(data["evidence_ids"])) != len(data["evidence_ids"]):
            raise DecisionError("invalid_evidence_ids", data)
        decision = data["decision"]
        original_decision = decision
        selected = {c["chunk_id"]: c for c in state["evidence"] if c["chunk_id"] in data["evidence_ids"]}
        retrieved = {c["chunk_id"]: c for c in state["evidence"]}
        condition_evidence_added = []
        locator_repairs = []
        for check_index, check in enumerate(data["checks"]):
            if set(check["evidence_ids"]) - known:
                raise DecisionError("invalid_check_evidence_ids", data)
            if check["status"] == "supported" and not check["evidence_ids"]:
                raise DecisionError("unsupported_check", data, check_index=check_index)
            if check['status'] == 'not_applicable' and (check['required_for_answer'] or check['evidence_ids']):
                raise DecisionError('invalid_not_applicable_check', data, check_index=check_index)
        for condition_index, condition in enumerate(data["critical_conditions"]):
            if condition["evidence_id"] not in retrieved:
                raise DecisionError("condition_not_retrieved", data)
            try:
                resolved = resolve_reference(retrieved[condition["evidence_id"]], condition)
            except CitationError as exc:
                raise DecisionError("condition_not_in_source", data, condition_index=condition_index,
                                    citation_error=exc.diagnostic['code']) from None
            if resolved.get('locator_repair'):
                locator_repairs.append({'evidence_id': condition['evidence_id'], **resolved['locator_repair']})
                condition['span_id'] = resolved['span_id']
            # A verified explicit condition citation is itself a selection. Merge
            # only retrieved, source-mapped evidence, not uncited checklist claims.
            if condition["evidence_id"] not in selected:
                selected[condition["evidence_id"]] = retrieved[condition["evidence_id"]]
                data["evidence_ids"].append(condition["evidence_id"])
                condition_evidence_added.append(condition["evidence_id"])
        knowledge_gaps = [c["requirement"] for c in data["checks"]
                          if c["status"] == "knowledge_gap" and c["required_for_answer"]]
        user_gaps = [c["requirement"] for c in data["checks"]
                     if c["status"] == "user_gap" and c["required_for_answer"]]
        optional_requirements = {c["requirement"] for c in data["checks"] if not c["required_for_answer"]}
        data["missing_parts"] = [p for p in data["missing_parts"] if p not in optional_requirements]
        for requirement in knowledge_gaps + user_gaps:
            if requirement not in data["missing_parts"]:
                data["missing_parts"].append(requirement)
        if user_gaps and not data["missing"]:
            data["missing"] = user_gaps
        verifications = []
        if (self.settings.surface_verification
                and state.get("retrieval_rounds", 0) < self.settings.max_retrieval_rounds
                and data["boundary"] not in {"future", "service", "device_history"}
                and not user_gaps):
            verifications = verification_queries(state)
            if verifications:
                # Do not rely on the model to notice/checklist these implicit
                # subquestions. Reserve bounded neutral queries, not answers.
                queries = [p['query'] for p in verifications] + data['retrieval_queries']
                data['retrieval_queries'] = list(dict.fromkeys(queries))[:2]
                for plan in verifications:
                    if plan['purpose'] not in data['missing_parts']:
                        data['missing_parts'].append(plan['purpose'])
                decision = 'retrieve_more'
        if bool(data["missing"]) != bool(data["clarification"].strip()):
            raise DecisionError("missing_clarification_fields", data)
        # A claimed answer cannot override its own explicit knowledge gaps.
        if knowledge_gaps and decision == "answer" and state.get("retrieval_rounds", 0) < self.settings.max_retrieval_rounds:
            if data["retrieval_queries"]:
                decision = "retrieve_more"
            # If the model declares a gap but supplies no usable subquery, keep
            # supported parts as partial. Never invent a query or a complete answer.
        if data["boundary"] in {"future", "service", "device_history"} and decision != "insufficient":
            raise DecisionError("invalid_service_boundary_route", data)
        partial = False
        budget_fallback = {}
        if decision == "retrieve_more" and state.get("retrieval_rounds", 0) >= self.settings.max_retrieval_rounds:
            # No executable query is required when execution is already forbidden.
            # Evidence IDs, conditions and clarification fields were still validated.
            decision = "answer" if data["evidence_ids"] and data["answerable_parts"] else "insufficient"
            partial = decision == "answer"
            budget_fallback = {"reason": "retrieval_budget_exhausted", "decision": decision,
                               "unexecuted_queries": list(data["retrieval_queries"])}
            data["retrieval_queries"] = []
        if decision == "answer" and not data["evidence_ids"]:
            raise DecisionError("empty_supporting_evidence", data)
        if decision == "clarify" and (not data["missing"] or not data["clarification"].strip()):
            raise DecisionError("missing_clarification_fields", data)
        if decision == "retrieve_more" and (not data["retrieval_queries"] or not data["missing_parts"]):
            raise DecisionError("missing_retrieval_gap_fields", data)
        if decision == "answer" and data["missing_parts"]:
            # Never silently turn missing required content into a complete answer.
            partial = True
        if decision == "insufficient" and data["boundary"] == "none":
            data["boundary"] = "missing_evidence"
        return {"sufficient": decision == "answer", "assessment_decision": decision,
                "selected_ids": data["evidence_ids"], "reason": data["reason"],
                "answerable_parts": data["answerable_parts"], "missing_parts": data["missing_parts"],
                "retrieval_queries": data["retrieval_queries"], "boundary": data["boundary"],
                "missing": data["missing"], "clarification": data["clarification"],
                "partial_answer": partial,
                "budget_fallback": budget_fallback,
                "assessment_format_repair": getattr(self.llm, "last_call_diagnostic", {}),
                "assessment_route_repair": {},
                "verification_requests": verifications,
                "condition_evidence_added": condition_evidence_added,
                "condition_locator_repairs": locator_repairs,
                "evidence_checks": data["checks"], "critical_conditions": data["critical_conditions"],
                "decision_override": {"from": original_decision, "to": decision}
                                     if decision != original_decision else {}}

    def insufficient(self, state):
        # Limits describe evidence/service capabilities, not unverified product facts.
        return {"status": "insufficient", "pending": False,
                "response": BOUNDARIES.get(state.get("boundary")) or BOUNDARIES["missing_evidence"]}

    def generate(self, state):
        known = {d['chunk_id'] for d in state['evidence']}
        selected = set(state.get('selected_ids', []))
        if selected - known:
            raise CitationError('unknown_evidence_id')
        if any(c['evidence_id'] not in selected for c in state.get('critical_conditions', [])):
            raise CitationError('unknown_evidence_id')
        if not selected:
            # Empty evidence is an explicit non-factual exit, not an invitation
            # to fabricate citation IDs. Preserve already requested user questions.
            follow_up = bool(state.get('missing')) and bool(state.get('clarification', '').strip())
            can_clarify = follow_up and state.get('boundary') not in {'future', 'service', 'device_history'}
            result = self.clarify(state) if can_clarify else self.insufficient(state)
            return {**result, 'citations': [], 'draft_answer': {}, 'sufficient': False,
                    'partial_answer': False, 'generation_repair': {},
                    'generation_skipped': {'reason': 'no_selected_evidence'},
                    'assessment_decision': 'clarify' if result['status'] == 'clarify' else 'insufficient'}
        try:
            return self._generate_once(state)
        except (CitationError, AttributionError) as first:
            previous_format = getattr(self.llm, 'last_call_diagnostic', {})
            if previous_format.get('attempts'):
                first.diagnostic['format_repair'] = previous_format
                raise
            # One shared citation/attribution repair; the same validators run again.
            # No answer-quality retry and no retrieval/gold information is added.
            repair = {"diagnostic": first.diagnostic}
            try:
                result = self._generate_once(state, repair)
            except Exception as second:
                first.diagnostic["repair_failure"] = diagnostic_for(second)
                raise first from None
            result["generation_repair"] = {"attempts": 1, "first_failure": first.diagnostic,
                                           "status": "validated"}
            return result

    def _answer_payload(self, state, repair=None):
        selected = {c["chunk_id"]: c for c in state["evidence"] if c["chunk_id"] in state["selected_ids"]}
        return {
            "user_context": user_context(state), "model_context": model_context(state),
            "original_question": state.get("original_question", state["question"]),
            "missing_parts": state.get("missing_parts", []),
            "boundary": state.get("boundary", "none"),
            "critical_conditions": state.get("critical_conditions", []),
            "follow_up": state.get("clarification", ""),
            "citation_repair": repair,
            "evidence": [evidence_payload(c) for c in selected.values()],
        }

    def _generate_once(self, state, repair=None):
        answer = Answer.model_validate(self.llm.call("generate", self._answer_payload(state, repair)))
        return {**self._render_answer(state, answer), "draft_answer": answer.model_dump(exclude_none=True),
                "generation_format_repair": getattr(self.llm, 'last_call_diagnostic', {}),
                "generation_base_clarification_count": state.get("clarification_count", 0)}

    def review_answer(self, state):
        payload = self._answer_payload(state)
        # Review the actual supporting excerpts, rather than inheriting the
        # assessor's prose or the chunk's potentially over-broad section grouping.
        # The original full evidence remains in state and is used for validation.
        quotes = {}
        documents = {d['chunk_id']: d for d in state['evidence']}
        for cite in state["citations"]:
            quotes.setdefault(cite["chunk_id"], [])
            if cite.get('table_row_id'):
                span = {'table_row_id': cite['table_row_id']}
                if span not in quotes[cite['chunk_id']]:
                    quotes[cite['chunk_id']].append(span)
                continue
            # Legacy stored citations have no locator; recover it only through
            # the same unique-original-coordinate validation, never first-match.
            span_id = cite.get('span_id') or resolve_quote(
                documents[cite['chunk_id']], cite['quote'])['span_id']
            span = {"text": cite["quote"], "span_id": span_id}
            if span not in quotes[cite["chunk_id"]]:
                quotes[cite["chunk_id"]].append(span)
        payload["evidence"] = [
            {"chunk_id": doc["chunk_id"], "source_id": doc["source_id"], "pages": doc["pages"],
             "text": "\n\n".join(s["text"] for s in quotes[doc["chunk_id"]] if 'text' in s),
             "citable_spans": [s for s in quotes[doc["chunk_id"]] if 'text' in s],
             "table_rows": [r for r in model_table_rows(doc) if any(s.get('table_row_id') == r['table_row_id'] for s in quotes[doc['chunk_id']])]}
            for doc in state["evidence"] if doc["chunk_id"] in quotes]
        revision = AnswerRevision.model_validate(self.llm.call("review_answer", {
            **payload, "draft": state["draft_answer"]}))
        # No new evidence, no approval label, no recursive model repair. A malformed
        # revised citation fails closed under the identical original-coordinate check.
        render_state = {**state, "clarification_count": state["generation_base_clarification_count"]}
        return {**self._render_answer(render_state, revision.answer),
                "generation_review": {"issues": revision.issues, "answer": revision.answer.model_dump(exclude_none=True),
                                      "changed": revision.answer.model_dump(exclude_none=True) != state["draft_answer"]}}

    def _render_answer(self, state, answer):
        validate_case_attribution(state, answer)
        selected = {c["chunk_id"]: c for c in state["evidence"] if c["chunk_id"] in state["selected_ids"]}
        citations = []
        boundary = BOUNDARIES.get(state.get("boundary", "none"), "")
        lines = [boundary] if boundary else []
        claims = [c.model_dump(exclude_none=True) for c in answer.claims]
        # Critical conditions are validated source excerpts, not another paraphrase.
        # Rendering them independently prevents the generator silently dropping a qualifier.
        for condition in state.get("critical_conditions", []):
            doc = selected.get(condition["evidence_id"])
            if doc is None:
                raise CitationError("unknown_evidence_id", answer=answer.model_dump(exclude_none=True))
            resolved = resolve_reference(doc, condition)
            label = '适用条件（表格单元格组合，分隔符为展示）：“' if condition.get('table_row_id') else '适用条件（原文）：“'
            claims.append({"kind": "fact", "text": label + resolved["quote"] + "”",
                           "citations": [condition]})
        for claim_index, item_claim in enumerate(claims):
            claim = Claim.model_validate(item_claim)
            labels = []
            claim_references = []
            for citation_index, cite in enumerate(claim.citations):
                def fail(code):
                    raise CitationError(code, claim_index=claim_index,
                                        citation_index=citation_index, answer=answer.model_dump(exclude_none=True))
                doc = selected.get(cite.evidence_id)
                if doc is None:
                    fail("unknown_evidence_id")
                try:
                    resolved = resolve_reference(doc, cite.model_dump(exclude_none=True))
                except CitationError as exc:
                    fail(exc.diagnostic["code"])
                claim_references.append(resolved)
                item = {"number": len(citations) + 1, "chunk_id": cite.evidence_id,
                        **resolved,
                        "source_id": doc["source_id"], "source_version": doc["source_version"],
                        "source_sha256": doc["source_sha256"], "url": doc["url"],
                        "pages": sorted({m["page"] for m in resolved["text_map"] if m["page"] is not None}
                                        | {s['page'] for a in resolved.get('source_context', []) for s in a['source_spans'] if s['page'] is not None})}
                citations.append(item)
                labels.append(f"[{item['number']}]")
            try:
                validate_category_models(claim.text, list(selected.values()), claim_references)
            except CitationError as exc:
                raise CitationError(exc.diagnostic['code'], claim_index=claim_index,
                                    answer=answer.model_dump(exclude_none=True)) from exc
            prefix = {"fact": "", "inference": "据此判断：", "advice": "操作建议："}[claim.kind]
            lines.append(prefix + claim.text + " " + "".join(labels))
        if state.get("partial_answer"):
            lines.append("以上仅回答已获得证据支持的部分。其余问题尚未核实，不能据此作确定判断。")
        follow_up = state.get("clarification", "").strip() if state.get("missing") else ""
        pending = bool(follow_up) and state.get("clarification_count", 0) < self.settings.max_clarifications
        if pending:
            lines.append(follow_up)
        elif follow_up:
            lines.append("仍缺少关键条件，本轮已达到追问上限，暂不能完成个体判断。")
        for c in citations:
            location = f"PDF物理页 {','.join(map(str, c['pages']))}" if c["pages"] else "网页快照"
            lines.append(f"[{c['number']}] {c['source_id']}，{location}，{c['source_version']}；{c['url']}")
        return {"status": "partial" if state.get("partial_answer") or follow_up else "answered",
                "pending": pending,
                "clarification_count": state.get("clarification_count", 0) + int(pending),
                "response": "\n\n".join(lines),
                "citations": citations}

    def finish(self, state):
        history = state.get("history", []) + [
            {"role": "user", "content": state["message"]},
            {"role": "assistant", "content": state["response"]},
        ]
        return {"history": history[-self.settings.max_history:]}

"""Opt-in C nodes. Gold labels and question IDs never enter this module."""
from .nodes import Nodes
from .state import Answer
from .aspect_schema import AspectPlan, CoverageReport, OutputAudit
from .context import user_context, evidence_payload
from .citations import resolve_reference
from .boundaries import BOUNDARIES
from .errors import DecisionError


def unique_inventory(items, expected, key, code):
    actual = [r[key] for r in items]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise DecisionError(code, {'expected': list(expected), 'actual': actual})


class AspectNodes(Nodes):
    def plan_aspects(self, state):
        plan = AspectPlan.model_validate(self.llm.call('c_plan', {
            'user_context': user_context(state),
            'original_question': state.get('original_question', state['question']),
        })).model_dump(exclude_none=True)
        ids = [a['aspect_id'] for a in plan['aspects']]
        if len(ids) != len(set(ids)) or not any(a['required'] for a in plan['aspects']):
            raise DecisionError('invalid_aspect_plan', plan)
        return {'aspect_plan': plan, 'aspect_coverage': [], 'aspect_queries': [],
                'output_audits': [], 'aspect_revision': {}, 'output_fallback': {}}

    def assess(self, state):
        plan = AspectPlan.model_validate(state['aspect_plan']).model_dump()
        aspects = {a['aspect_id']: a for a in plan['aspects']}
        remaining = state.get('retrieval_rounds', 0) < self.settings.max_retrieval_rounds
        report = CoverageReport.model_validate(self.llm.call('c_cover', {
            'user_context': user_context(state), 'aspect_plan': plan,
            'evidence': [evidence_payload(d) for d in state['evidence']],
            'previous_coverage': state.get('aspect_coverage', []),
            'remaining_queries': 2 if remaining else 0,
        })).model_dump(exclude_none=True)
        unique_inventory(report['coverage'], aspects, 'aspect_id', 'changed_aspect_inventory')
        docs = {d['chunk_id']: d for d in state['evidence']}
        selected, requests, gaps, user_gaps, questions, supported, checks = [], [], [], [], [], [], []
        for item in report['coverage']:
            aspect = aspects[item['aspect_id']]
            status = item['status']
            if status == 'supported' and not item['citations']:
                raise DecisionError('supported_aspect_without_citation', item)
            if status == 'unavailable' and aspect['category'] not in {'boundary', 'user_info'}:
                raise DecisionError('fact_gap_misclassified_as_unavailable', item)
            for cite in item['citations']:
                if cite['evidence_id'] not in docs:
                    raise DecisionError('aspect_unknown_evidence', item)
                resolved = resolve_reference(docs[cite['evidence_id']], cite)
                if resolved.get('span_id'):
                    cite['span_id'] = resolved['span_id']
                if status == 'supported' and cite['evidence_id'] not in selected:
                    selected.append(cite['evidence_id'])
            if item['query'] and status not in {'missing', 'conflict'}:
                raise DecisionError('query_for_nonmissing_aspect', item)
            if aspect['required']:
                if status in {'missing', 'conflict'}:
                    gaps.append(aspect['requirement'])
                    if remaining:
                        if not item['query'].strip():
                            raise DecisionError('missing_aspect_query', item)
                        requests.append({'aspect_id': item['aspect_id'], 'query': item['query'].strip()})
                elif status == 'user_gap':
                    if not item['question_to_user'].strip():
                        raise DecisionError('missing_aspect_clarification', item)
                    user_gaps.append(aspect['requirement'])
                    questions.append(item['question_to_user'].strip())
                elif status == 'supported':
                    supported.append(aspect['requirement'])
            checks.append({'requirement': aspect['requirement'], 'required_for_answer': aspect['required'],
                           'status': {'supported': 'supported', 'missing': 'knowledge_gap', 'conflict': 'knowledge_gap',
                                      'user_gap': 'user_gap', 'unavailable': 'unavailable'}[status],
                           'evidence_ids': list(dict.fromkeys(c['evidence_id'] for c in item['citations'])),
                           'detail': item['detail']})
        queries = list(dict.fromkeys(r['query'] for r in requests))[:2]
        decision = ('retrieve_more' if queries else 'answer' if supported and selected else
                    'clarify' if user_gaps else 'insufficient')
        boundary = plan['boundary']
        if decision == 'insufficient' and boundary == 'none':
            boundary = 'missing_evidence'
        return {'aspect_coverage': report['coverage'],
                'aspect_queries': [{'query': q, 'aspect_ids': [r['aspect_id'] for r in requests if r['query'] == q]} for q in queries],
                'assessment_decision': decision, 'sufficient': decision == 'answer',
                'selected_ids': selected, 'retrieval_queries': queries,
                'missing_parts': gaps + user_gaps, 'answerable_parts': supported,
                'missing': user_gaps, 'clarification': '\n'.join(dict.fromkeys(questions)),
                'boundary': boundary, 'reason': '按冻结方面逐项检查后由程序路由。',
                'partial_answer': bool(gaps or user_gaps), 'evidence_checks': checks,
                'critical_conditions': [], 'decision_override': {}, 'verification_requests': [],
                'budget_fallback': {'reason': 'retrieval_budget_exhausted', 'missing': gaps} if gaps and not remaining else {}}

    def _answer_payload(self, state, repair=None):
        return {**super()._answer_payload(state, repair),
                'required_aspects': state['aspect_plan']['aspects'],
                'aspect_coverage': state['aspect_coverage'],
                'output_audit': state.get('revision_audit'),
                'previous_draft': state.get('revision_draft')}

    def _generate_once(self, state, repair=None):
        answer = Answer.model_validate(self.llm.call('c_generate', self._answer_payload(state, repair)))
        return {**self._render_answer(state, answer), 'draft_answer': answer.model_dump(exclude_none=True),
                'generation_format_repair': getattr(self.llm, 'last_call_diagnostic', {}),
                'generation_base_clarification_count': state.get('clarification_count', 0)}

    def _render_answer(self, state, answer):
        # C represents every published factual condition as a normal claim, so
        # no unaudited fact can be appended after the semantic audit.
        if state.get('critical_conditions'):
            raise DecisionError('c_untracked_automatic_conditions', {})
        result = super()._render_answer(state, answer)
        if state.get('partial_answer') and state.get('missing_parts'):
            result['response'] += '\n\n尚未核实的方面：' + '；'.join(state['missing_parts'])
        return result

    def _audit(self, state, answer):
        payload = self._answer_payload(state)
        audit = OutputAudit.model_validate(self.llm.call('c_audit', {
            **payload, 'aspect_plan': state['aspect_plan'], 'answer': answer,
            'boundary_text': BOUNDARIES.get(state.get('boundary'), ''),
            'missing_notice': state.get('missing_parts', []),
        })).model_dump()
        aspects = {a['aspect_id']: a for a in state['aspect_plan']['aspects']}
        coverage = {a['aspect_id']: a for a in state['aspect_coverage']}
        unique_inventory(audit['aspects'], aspects, 'aspect_id', 'incomplete_output_aspect_audit')
        unique_inventory(audit['claims'], range(len(answer['claims'])), 'claim_index', 'incomplete_claim_audit')
        for a in audit['aspects']:
            if any(type(i) is not int or i < 0 or i >= len(answer['claims']) for i in a['claim_indices']):
                raise DecisionError('invalid_claim_index', a)
            if coverage[a['aspect_id']]['status'] == 'supported':
                # A checker cannot waive an answerable requirement by calling it
                # appropriately limited, nor approve a factual answer without a claim.
                if a['status'] == 'appropriately_limited' or (a['status'] == 'addressed' and not a['claim_indices']):
                    a['status'] = 'missing'
                    a['detail'] += '；程序校验：有支持的方面必须对应正文claim。'
        return audit

    @staticmethod
    def _problems(state, audit):
        required = {a['aspect_id'] for a in state['aspect_plan']['aspects'] if a['required']}
        bad_claims = {c['claim_index'] for c in audit['claims'] if c['status'] != 'supported'}
        missing = [a['aspect_id'] for a in audit['aspects'] if a['aspect_id'] in required and
                   (a['status'] == 'missing' or bool(set(a['claim_indices']) & bad_claims))]
        return bad_claims, missing

    def audit_answer(self, state):
        answer = Answer.model_validate(state['draft_answer']).model_dump(exclude_none=True)
        audit = self._audit(state, answer)
        audits = [audit]
        bad, missing = self._problems(state, audit)
        revision = {}
        render_state = {**state, 'clarification_count': state.get('generation_base_clarification_count', 0)}
        if bad or missing:
            # One semantic revision, then a fresh audit. No additional retrieval.
            revised = self.generate({**render_state, 'revision_audit': audit, 'revision_draft': answer})
            answer = revised['draft_answer']
            revision = {'attempts': 1, 'draft': answer, 'generation_repair': revised.get('generation_repair', {})}
            audit = self._audit(state, answer)
            audits.append(audit)
            bad, missing = self._problems(state, audit)
        fallback = {}
        if bad or missing:
            fallback = {'reason': 'postcheck_unresolved', 'removed_claim_indices': sorted(bad),
                        'missing_aspect_ids': missing}
            answer = {'claims': [c for i, c in enumerate(answer['claims']) if i not in bad]}
            if not answer['claims']:
                return {**self.insufficient({**state, 'boundary': 'missing_evidence'}),
                        'status': 'insufficient', 'citations': [], 'draft_answer': {},
                        'output_audits': audits, 'aspect_revision': revision, 'output_fallback': fallback}
            requirements = {a['aspect_id']: a['requirement'] for a in state['aspect_plan']['aspects']}
            render_state.update(partial_answer=True,
                                missing_parts=list(dict.fromkeys(state.get('missing_parts', []) + [requirements[i] for i in missing])))
        return {**self._render_answer(render_state, Answer.model_validate(answer)),
                'draft_answer': answer, 'output_audits': audits, 'aspect_revision': revision,
                'output_fallback': fallback, 'partial_answer': render_state.get('partial_answer', False),
                'missing_parts': render_state.get('missing_parts', [])}

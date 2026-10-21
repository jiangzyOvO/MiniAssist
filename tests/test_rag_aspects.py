"""C routing/guard tests with injected judgments, NOT model quality evidence."""
import copy
import json
import unittest
from miniassist.rag.aspects import AspectNodes
from miniassist.rag.graph import RAGService
from miniassist.rag.settings import Settings
from miniassist.rag.errors import DecisionError, CitationError
from miniassist.rag.llm import DeepSeek, ModelError
from test_rag import Model, Retriever, DOC, TEXT, ANSWER, understanding


def plan(ids=('normal_effect',), category='operation'):
    return {'aspects': [{'aspect_id': i, 'requirement': '按钮单击的正常作用' + i,
                        'category': category, 'required': True, 'reason': '解释用户动作'} for i in ids], 'boundary': 'none'}


def covered(ident='normal_effect', status='supported', query='', question=''):
    return {'aspect_id': ident, 'status': status,
            'citations': [{'evidence_id': 'c1', 'quote': TEXT}] if status == 'supported' else [],
            'detail': '引用仅描述该设备的单击效果', 'query': query, 'question_to_user': question}


def audit(aspect_status='addressed', claim_status='supported', claim_count=1):
    return {'aspects': [{'aspect_id': 'normal_effect', 'status': aspect_status,
                         'claim_indices': [0] if aspect_status == 'addressed' else [], 'detail': '逐项对照正文'}],
            'claims': [{'claim_index': i, 'status': claim_status, 'detail': '对照实际引用及适用对象'} for i in range(claim_count)]}


def seed():
    return {'question': '设备甲单击按钮没开机，怎么办？', 'original_question': '设备甲单击按钮没开机，怎么办？',
            'message': '设备甲单击按钮没开机，怎么办？', 'aspect_plan': plan(),
            'aspect_coverage': [covered()], 'evidence': [copy.deepcopy(DOC)],
            'selected_ids': ['c1'], 'retrieval_rounds': 0, 'critical_conditions': [],
            'clarification_count': 0, 'missing_parts': [], 'missing': [], 'boundary': 'none',
            'draft_answer': copy.deepcopy(ANSWER), 'trace': []}


class AspectTests(unittest.TestCase):
    def nodes(self, outputs, **settings):
        return AspectNodes(Model(outputs), Retriever(), Settings(policy='C', **settings))

    def test_plan_sees_user_only_and_not_seed_evidence(self):
        nodes = self.nodes([plan()])
        result = nodes.plan_aspects(seed())
        self.assertEqual(result['aspect_plan'], plan())
        payload = nodes.llm.calls[0][1]
        self.assertEqual(set(payload), {'user_context', 'original_question'})
        self.assertNotIn(TEXT, json.dumps(payload, ensure_ascii=False))

    def test_supported_aspect_does_not_trigger_surface_search(self):
        result = self.nodes([{'coverage': [covered()]}]).assess(seed())
        self.assertEqual(result['assessment_decision'], 'answer')
        self.assertEqual(result['retrieval_queries'], [])
        self.assertEqual(result['critical_conditions'], [])

    def test_missing_query_binds_only_missing_aspects(self):
        s = seed(); s['aspect_plan'] = plan(('normal_effect', 'correct_operation'))
        result = self.nodes([{'coverage': [covered(), covered('correct_operation', 'missing', '设备甲 开机操作')]}]).assess(s)
        self.assertEqual(result['retrieval_queries'], ['设备甲 开机操作'])
        self.assertEqual(result['aspect_queries'][0]['aspect_ids'], ['correct_operation'])

    def test_invalid_queries_and_inventory_are_rejected(self):
        for rows in [[covered(query='不该查询')], [covered('another')], [covered(), covered()],
                     [covered(status='missing')], [covered(status='unavailable')]]:
            with self.subTest(rows=rows), self.assertRaises(DecisionError):
                self.nodes([{'coverage': rows}]).assess(seed())

    def test_missing_query_budget_cap_and_no_replanning(self):
        s = seed(); s['aspect_plan'] = plan(('a','b','c'))
        rows = [covered(i, 'missing', '设备甲 '+i) for i in ('a','b','c')]
        result = self.nodes([{'coverage': rows}]).assess(s)
        self.assertEqual(len(result['retrieval_queries']), 2)
        self.assertEqual(len(result['missing_parts']), 3)
        self.assertEqual(len(result['aspect_coverage']), 3)

    def test_budget_exhaustion_retains_gap_and_partial(self):
        s = seed(); s['aspect_plan'] = plan(('normal_effect', 'correct_operation')); s['retrieval_rounds'] = 1
        result = self.nodes([{'coverage': [covered(), covered('correct_operation','missing')]}]).assess(s)
        self.assertEqual(result['assessment_decision'], 'answer')
        self.assertTrue(result['partial_answer'])
        self.assertEqual(result['retrieval_queries'], [])

    def test_user_gap_and_boundary_do_not_search(self):
        s = seed(); s['aspect_plan'] = plan(category='user_info')
        result = self.nodes([{'coverage': [covered(status='user_gap', question='设备完整型号是什么？')]}]).assess(s)
        self.assertEqual(result['assessment_decision'], 'clarify')
        self.assertEqual(result['retrieval_queries'], [])
        s['aspect_plan'] = plan(category='boundary'); s['aspect_plan']['boundary'] = 'device_history'
        result = self.nodes([{'coverage': [covered(status='unavailable')]}]).assess(s)
        self.assertEqual(result['assessment_decision'], 'insufficient')

    def test_quote_and_evidence_ids_are_validated(self):
        bad = covered(); bad['citations'][0]['quote'] = '原文没有提供这个结论'
        with self.assertRaises(CitationError):
            self.nodes([{'coverage': [bad]}]).assess(seed())
        bad['citations'][0]['evidence_id'] = 'absent'
        with self.assertRaises(DecisionError):
            self.nodes([{'coverage': [bad]}]).assess(seed())

    def test_selected_supported_but_omitted_aspect_is_revised_without_search(self):
        nodes = self.nodes([audit('missing'), ANSWER, audit()])
        result = nodes.audit_answer(seed())
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['aspect_revision']['attempts'], 1)
        self.assertEqual([s for s,_ in nodes.llm.calls], ['c_audit','c_generate','c_audit'])
        self.assertFalse(result['output_fallback'])

    def test_repeated_irrelevant_claim_is_removed_without_approving_empty_answer(self):
        nodes = self.nodes([audit(claim_status='irrelevant'), ANSWER, audit(claim_status='irrelevant')])
        result = nodes.audit_answer(seed())
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['citations'], [])
        self.assertEqual(result['output_fallback']['removed_claim_indices'], [0])

    def test_checker_cannot_waive_supported_aspect(self):
        nodes = self.nodes([audit('appropriately_limited'), ANSWER, audit('appropriately_limited')])
        result = nodes.audit_answer(seed())
        self.assertEqual(result['status'], 'partial')
        self.assertIn('normal_effect', result['output_fallback']['missing_aspect_ids'])
        self.assertIn('尚未核实的方面', result['response'])

    def test_incomplete_claim_audit_fails_closed(self):
        bad = audit(); bad['claims'][0]['claim_index'] = 2
        result = self.nodes([bad]).guarded('audit_answer')(seed())
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['citations'], [])
        self.assertNotIn('单击可查看', result['response'])

    def test_no_unaudited_condition_can_be_appended(self):
        s = seed(); s['critical_conditions'] = [{'evidence_id':'c1', 'quote':TEXT}]
        result = self.nodes([ANSWER]).guarded('generate')(s)
        self.assertEqual(result['status'], 'error')

    def test_graph_integration_and_turn_reset(self):
        outputs = [understanding(), plan(), {'coverage':[covered()]}, ANSWER, audit()]
        model = Model(outputs * 2)
        service = RAGService(model, Retriever(), Settings(policy='C'))
        for text in ['设备甲单击有何作用', '第二个新问题']:
            result = service.ask('c', text)
            self.assertEqual(result['status'], 'answered')
            self.assertEqual(len(result['output_audits']), 1)
            self.assertEqual([t['node'] for t in result['trace']],
                             ['understand','retrieve','plan_aspects','assess','generate','audit_answer','finish'])

    def test_no_postcheck_ablation_and_empty_retrieval(self):
        model = Model([understanding(), plan(), {'coverage':[covered()]}, ANSWER])
        result = RAGService(model, Retriever(), Settings(policy='C_no_postcheck')).ask('c','问题')
        self.assertEqual(result['status'], 'answered')
        self.assertFalse(result['output_audits'])
        s = seed(); s['evidence'] = []; s['retrieval_rounds'] = 1
        result = self.nodes([{'coverage':[covered(status='missing')]}]).assess(s)
        self.assertEqual(result['assessment_decision'], 'insufficient')

    def test_provider_request_ceiling_is_enforced_and_resettable(self):
        class Session:
            def post(self, *args, **kwargs):
                raise AssertionError('Budget check must happen before network')
        llm = DeepSeek(Settings(max_model_requests=1), api_key='test', session=Session())
        llm.requests_used = 1
        with self.assertRaises(ModelError) as exc:
            llm.call('c_plan', {})
        self.assertEqual(exc.exception.diagnostic['code'], 'model_request_budget_exhausted')
        llm.reset_budget()
        self.assertEqual(llm.requests_used, 0)


if __name__ == '__main__':
    unittest.main()

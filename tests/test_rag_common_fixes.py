"""Shared infrastructure regressions; injected outputs are not quality scores."""
import json
import unittest
from unittest.mock import patch

from miniassist.rag.citations import citation_spans, resolve_quote
from miniassist.rag.context import evidence_payload
from miniassist.rag.errors import CitationError, DecisionError
from miniassist.rag.graph import RAGService
from miniassist.rag.llm import DeepSeek
from miniassist.rag.nodes import Nodes
from miniassist.rag.settings import Settings
from test_rag import DOC, ANSWER, ASSESS, Model, Retriever, understanding
from test_rag_citations import document
from test_rag_format_repair import Session
from test_rag_routing import gap


class LocatorTests(unittest.TestCase):
    def test_repeated_lines_require_explicit_locator_and_keep_distinct_source_offsets(self):
        doc = document('垂直：\n±0.1 米（视觉定位正常工作时）\n水平：\n±0.1 米（视觉定位正常工作时）')
        quote = '±0.1 米（视觉定位正常工作时）'
        spans = [s for s in citation_spans(doc) if s['text'] == quote]
        self.assertEqual(len(spans), 2)
        resolved = [resolve_quote(doc, quote, s['span_id']) for s in spans]
        self.assertNotEqual(resolved[0]['text_map'], resolved[1]['text_map'])
        for span, result in zip(spans, resolved):
            self.assertEqual(result['text_map'][0]['source_start'], 10 + span['start'])
            self.assertEqual(result['match_mode'], 'exact')
        for locator in (None, citation_spans(doc)[0]['span_id']):
            with self.assertRaises(CitationError) as caught:
                resolve_quote(doc, quote, locator)
            self.assertEqual(caught.exception.diagnostic['code'], 'ambiguous_quote')

    def test_locator_never_falls_back_to_another_window_or_chunk(self):
        doc = document('电池需要逐块检查。\n遥控器需要另行检查。')
        line = next(s for s in citation_spans(doc) if s['text'] == '电池需要逐块检查。')
        for quote, locator, code in [
            ('遥控器需要另行检查。', line['span_id'], 'quote_not_in_span'),
            ('电池需要逐块检查。', line['span_id'].replace('c1:', 'c2:'), 'unknown_span_id'),
            ('电池无需逐块检查。', line['span_id'], 'quote_not_in_span')]:
            with self.subTest(code=code), self.assertRaises(CitationError) as caught:
                resolve_quote(doc, quote, locator)
            self.assertEqual(caught.exception.diagnostic['code'], code)

    def test_whole_window_preserves_multiline_whitespace_recovery(self):
        doc = document('温度恢复正常，\n电池将自动恢复充电。')
        result = resolve_quote(doc, '温度恢复正常，电池将自动恢复充电。', citation_spans(doc)[0]['span_id'])
        self.assertEqual(result['match_mode'], 'whitespace_only')
        self.assertEqual(result['quote'], doc['text'])

    def test_critical_condition_uses_same_locator_validation(self):
        doc = document('自动恢复充电。\n自动恢复充电。')
        line = citation_spans(doc)[-1]
        decision = {**ASSESS, 'critical_conditions': [{'evidence_id': 'c1',
                    'quote': line['text'], 'span_id': line['span_id']}]}
        node = Nodes(Model([decision]), None, Settings(surface_verification=False))
        result = node.assess({'question': '要求？', 'evidence': [doc]})
        self.assertEqual(result['critical_conditions'][0]['span_id'], line['span_id'])

    def test_provider_schema_exposes_locators_without_losing_assessment_constraints(self):
        session = Session([{**ASSESS, 'checks': [{'requirement': '查看电量',
            'required_for_answer': True, 'status': 'supported',
            'evidence_ids': ['c1'], 'detail': '原文支持'}]}])
        DeepSeek(Settings(), api_key='test', session=session).call('assess', {'evidence': [evidence_payload(DOC)]})
        content = session.requests[0]['messages'][0]['content']
        schema = json.loads(content.split('不增加其他字段：\n', 1)[1])
        citation = schema['$defs']['Citation']
        self.assertNotIn('span_id', citation['required'])
        self.assertEqual(citation['properties']['span_id']['anyOf'][0]['enum'],
                         [s['span_id'] for s in evidence_payload(DOC)['citable_spans']])
        self.assertIn('checks', schema['required'])
        self.assertEqual(schema['properties']['checks']['minItems'], 1)


class BudgetTests(unittest.TestCase):
    def assess(self, decision, remaining=0):
        return Nodes(Model([decision]), None, Settings(max_retrieval_rounds=remaining,
                     surface_verification=False)).assess({'question': '两项要求？', 'evidence': [DOC]})

    def test_no_executable_query_required_when_budget_exhausted(self):
        result = self.assess({**gap(), 'retrieval_queries': []})
        self.assertEqual(result['assessment_decision'], 'answer')
        self.assertTrue(result['partial_answer'])
        self.assertEqual(result['budget_fallback']['reason'], 'retrieval_budget_exhausted')
        self.assertEqual(result['retrieval_queries'], [])

    def test_no_supported_part_means_insufficient(self):
        result = self.assess({**gap(), 'retrieval_queries': [], 'answerable_parts': []})
        self.assertEqual(result['assessment_decision'], 'insufficient')
        self.assertFalse(result['partial_answer'])

    def test_invalid_ids_conditions_and_routes_still_fail_before_budget_fallback(self):
        for changes, code in [
            ({'evidence_ids': ['unknown']}, 'invalid_evidence_ids'),
            ({'critical_conditions': [{'evidence_id': 'c1', 'quote': '完全虚构的条件'}]}, 'condition_not_in_source'),
            ({'boundary': 'future'}, 'invalid_service_boundary_route')]:
            with self.subTest(code=code), self.assertRaises(DecisionError) as caught:
                self.assess({**gap(), 'retrieval_queries': [], **changes})
            self.assertEqual(caught.exception.diagnostic['code'], code)

    def test_available_budget_still_requires_query(self):
        with self.assertRaises(DecisionError) as caught:
            self.assess({**gap(), 'retrieval_queries': []}, remaining=1)
        self.assertEqual(caught.exception.diagnostic['code'], 'missing_retrieval_gap_fields')

    def test_graph_does_not_search_again_and_resets_budget_diagnostic_next_turn(self):
        model = Model([understanding(), {**gap(), 'retrieval_queries': []}, ANSWER,
                       understanding(), ASSESS, ANSWER])
        retriever = Retriever()
        service = RAGService(model, retriever, Settings(max_retrieval_rounds=0))
        first = service.ask('budget', '两个要求？')
        self.assertEqual(first['status'], 'partial')
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(first['retrieval_rounds'], 0)
        second = service.ask('budget', '另一个问题')
        self.assertEqual(second['budget_fallback'], {})


class EmptySelectionTests(unittest.TestCase):
    def test_empty_selection_never_calls_model_and_clears_stale_answer(self):
        model = Model([])
        result = Nodes(model, None, Settings()).generate({
            'evidence': [DOC], 'selected_ids': [], 'boundary': 'none', 'draft_answer': ANSWER})
        self.assertEqual(result['status'], 'insufficient')
        self.assertTrue(result['response'])
        self.assertEqual(result['draft_answer'], {})
        self.assertEqual(result['citations'], [])
        self.assertEqual(model.calls, [])

    def test_existing_clarification_and_service_boundary_are_preserved(self):
        node = Nodes(Model([]), None, Settings())
        state = {'evidence': [], 'selected_ids': [], 'missing': ['controller'],
                 'clarification': '遥控器是什么型号？'}
        self.assertEqual(node.generate(state)['status'], 'clarify')
        result = node.generate({**state, 'boundary': 'future'})
        self.assertEqual(result['status'], 'insufficient')
        self.assertTrue(result['response'])

    def test_malformed_selection_and_unselected_conditions_do_not_become_normal_fallback(self):
        for selection, conditions in [(['unknown'], []),
                ([], [{'evidence_id': 'c1', 'quote': '单击可查看剩余电量'}])]:
            with self.assertRaises(CitationError):
                Nodes(Model([]), None, Settings()).generate({'evidence': [DOC],
                    'selected_ids': selection, 'critical_conditions': conditions})

    def test_graph_skips_opt_in_reviewer_after_empty_evidence_fallback(self):
        # Simulate the deletion intervention between assessment and generation.
        original = Nodes.generate
        def delete_then_generate(node, state):
            return original(node, {**state, 'selected_ids': [], 'critical_conditions': []})
        model = Model([understanding(), ASSESS])
        with patch.object(Nodes, 'generate', delete_then_generate):
            result = RAGService(model, Retriever(), Settings(review_answer=True)).ask('empty', '问题')
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual([stage for stage, _ in model.calls], ['understand', 'assess'])
        self.assertNotIn('review_answer', [item['node'] for item in result['trace']])


if __name__ == '__main__':
    unittest.main()

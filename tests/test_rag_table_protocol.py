"""Source integrity, typed row references and bounded protocol recovery."""
import copy
import json
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from miniassist.rag.citations import resolve_reference, resolve_quote
from miniassist.rag.context import evidence_payload
from miniassist.rag.errors import CitationError
from miniassist.rag.nodes import Nodes
from miniassist.rag.llm import DeepSeek
from miniassist.rag.settings import Settings
from miniassist.rag.state import Citation
from miniassist.rag.table_evidence import build_rows, document_fingerprint, table_rows
from test_rag import DOC, ASSESS, Model
from test_rag_format_repair import Session


def fixture():
    # A genuine two-column table with one retained row and another not retrieved.
    block = {'block_id': 'table', 'page': 1,
             'metadata': {'source_id': 'S1', 'source_sha256': 'test'},
             'table': {'row_count': 3, 'column_count': 2, 'header_rows': [0], 'cells': []}}
    doc = {**copy.deepcopy(DOC), 'chunk_id': 'table-chunk', 'text': '', 'text_map': [], 'unit_ids': ['table:row:1']}
    for index, (row, col, text) in enumerate([(0, 0, '档位'), (0, 1, '功能'), (1, 0, 'A'), (1, 1, '开'), (2, 0, 'B'), (2, 1, '关')]):
        origin = 100 + 10*index
        cell = {'cell_id': f'cell-{index}', 'role': 'cell', 'row': row, 'column': col, 'text': text,
                'text_map': [{'text_start': 0, 'text_end': len(text), 'source_start': origin, 'source_end': origin+len(text)}]}
        block['table']['cells'].append(cell)
        if row < 2:
            doc['text'] += f'第{col+1}列：'
            start = len(doc['text'])
            doc['text'] += text + '\n'
            doc['text_map'].append({'text_start': start, 'text_end': start+len(text), 'source_start': origin,
                                    'source_end': origin+len(text), 'source_id': 'S1', 'page': 1, 'block_id': 'table'})
    return doc, {'table': block}


class TableProtocolTests(unittest.TestCase):
    def setUp(self):
        self.doc, self.blocks = fixture()
        self.rows = build_rows(self.doc, self.blocks)
        self.row_id = self.rows[0]['table_row_id']
        self.registry = {self.doc['chunk_id']: {'document_fingerprint': document_fingerprint(self.doc), 'rows': self.rows}}
        self.mock = patch('miniassist.rag.table_evidence._registry', return_value=self.registry)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def cite(self):
        return {'evidence_id': self.doc['chunk_id'], 'table_row_id': self.row_id}

    def state(self):
        return {'question': '档位A的功能？', 'evidence': [self.doc], 'selected_ids': [self.doc['chunk_id']],
                'slots': {}, 'trace': []}

    def assessment(self):
        return {**copy.deepcopy(ASSESS), 'evidence_ids': [self.doc['chunk_id']],
                'critical_conditions': [self.cite()]}

    def test_short_cells_allowed_only_as_source_bound_row(self):
        resolved = resolve_reference(self.doc, self.cite())
        self.assertEqual(resolved['match_mode'], 'table_row_exact')
        self.assertFalse(resolved['quote_is_contiguous'])
        self.assertEqual([p['text'] for p in resolved['quote_parts']], ['档位', '功能', 'A', '开'])
        self.assertEqual({m['source_start'] for m in resolved['text_map']}, {100, 110, 120, 130})
        self.assertNotIn('第1列', resolved['quote'])
        for short in ('A', '开'):
            with self.assertRaises(ValidationError):
                Citation(evidence_id=self.doc['chunk_id'], quote=short)
            with self.assertRaises(CitationError):
                resolve_quote(self.doc, short)

    def test_reference_types_cannot_mix_or_be_empty(self):
        for fields in ({}, {'table_row_id': self.row_id, 'quote': '虚构合法长度'},
                       {'table_row_id': self.row_id, 'span_id': 'invented'}):
            with self.assertRaises(ValidationError):
                Citation(evidence_id=self.doc['chunk_id'], **fields)

    def test_cross_row_cross_chunk_and_unretrieved_rows_rejected(self):
        for row_id in (self.row_id.replace(':row:1', ':row:2'), self.row_id.replace('table-chunk', 'elsewhere'), 'made-up'):
            with self.assertRaises(CitationError):
                resolve_reference(self.doc, {'evidence_id': self.doc['chunk_id'], 'table_row_id': row_id})
        with self.assertRaises(CitationError):
            resolve_reference(DOC, self.cite())

    def test_synthetic_labels_and_modified_chunk_still_fail(self):
        with self.assertRaises(CitationError):
            resolve_quote(self.doc, '第1列：档位')
        changed = copy.deepcopy(self.doc)
        changed['text'] = changed['text'].replace('开', '关')
        with self.assertRaises(CitationError):
            resolve_reference(changed, self.cite())
        changed = copy.deepcopy(self.doc)
        changed['text_map'][-1]['source_start'] += 1
        with self.assertRaises(CitationError):
            table_rows(changed)

    def test_partial_cell_or_missing_header_cannot_create_row(self):
        for index in (0, 2, 3):
            doc = copy.deepcopy(self.doc)
            doc['text_map'].pop(index)
            self.assertEqual(build_rows(doc, self.blocks), [])
        blocks = copy.deepcopy(self.blocks)
        blocks['table']['table']['cells'][3]['text'] += '未映射文字'
        self.assertEqual(build_rows(self.doc, blocks), [])

    def test_source_identity_and_cell_text_tampering_rejected(self):
        doc = copy.deepcopy(self.doc)
        doc['source_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            build_rows(doc, self.blocks)
        doc = copy.deepcopy(self.doc)
        doc['text'] = doc['text'].replace('开', '关')
        with self.assertRaises(ValueError):
            build_rows(doc, self.blocks)

    def test_assess_generate_and_conditions_expand_row_parts(self):
        node = Nodes(Model([self.assessment()]), None, Settings(surface_verification=False))
        update = node.assess(self.state())
        answer = {'claims': [{'text': 'A对应开。', 'citations': [self.cite()]}]}
        node = Nodes(Model([answer]), None, Settings())
        result = node.generate({**self.state(), **update})
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(result['citations']), 2)
        self.assertTrue(all(c['match_mode'] == 'table_row_exact' for c in result['citations']))
        self.assertIn('表格单元格组合', result['response'])
        self.assertNotIn('quote_start', result['citations'][0])

    def test_review_gets_only_cited_rows_not_other_rows(self):
        answer = {'claims': [{'text': 'A对应开。', 'citations': [self.cite()]}]}
        model = Model([answer, {'issues': [], 'answer': answer}])
        node = Nodes(model, None, Settings())
        generated = node.generate(self.state())
        node.review_answer({**self.state(), **generated})
        payload = model.calls[1][1]['evidence'][0]
        self.assertEqual([r['table_row_id'] for r in payload['table_rows']], [self.row_id])
        self.assertEqual(payload['citable_spans'], [])
        self.assertEqual(payload['text'], '')

    def test_provider_schema_offers_only_available_rows_and_exclusive_kind(self):
        answer = {'claims': [{'text': 'A对应开。', 'citations': [self.cite()]}]}
        session = Session([answer])
        DeepSeek(Settings(), api_key='test', session=session).call('generate', {'evidence': [evidence_payload(self.doc)]})
        schema = json.loads(session.requests[0]['messages'][0]['content'].split('不增加其他字段：\n', 1)[1])
        self.assertEqual(schema['$defs']['Citation']['properties']['table_row_id']['anyOf'][0]['enum'], [self.row_id])
        self.assertEqual(len(schema['$defs']['Citation']['oneOf']), 2)

    def test_one_assessment_repair_with_specific_condition_diagnostic(self):
        bad = self.assessment()
        bad['critical_conditions'] = [{'evidence_id': self.doc['chunk_id'], 'quote': '第1列：档位'}]
        model = Model([bad, self.assessment()])
        result = Nodes(model, None, Settings(surface_verification=False)).assess(self.state())
        self.assertEqual(result['assessment_decision'], 'answer')
        repair = model.calls[1][1]['routing_repair']
        self.assertEqual(repair['failure_location'], {'condition_index': 0, 'citation_error': 'unmapped_synthetic_text'})
        self.assertEqual(len(model.calls), 2)

    def test_repair_keeps_failure_and_budget_when_invalid_again(self):
        bad = self.assessment()
        bad['critical_conditions'] = [{'evidence_id': self.doc['chunk_id'], 'quote': '虚构不存在的字'}]
        model = Model([bad, bad, self.assessment()])
        result = Nodes(model, None, Settings()).guarded('assess')(self.state())
        self.assertEqual(result['status'], 'error')
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result['trace'][-1]['diagnostic']['repair_failure']['citation_error'], 'quote_not_found')

    def test_schema_repair_and_node_repair_share_budget(self):
        bad = self.assessment()
        bad['critical_conditions'] = [{'evidence_id': self.doc['chunk_id'], 'quote': '虚构不存在的字'}]
        model = Model([bad, self.assessment()])
        model.last_call_diagnostic = {'attempts': 1, 'status': 'validated'}
        result = Nodes(model, None, Settings()).guarded('assess')(self.state())
        self.assertEqual(result['status'], 'error')
        self.assertEqual(len(model.calls), 1)

    def test_nonapplicable_is_not_evidence_and_does_not_block_real_query(self):
        item = {'requirement': '不适用的动作项', 'detail': '用户没描述该动作', 'required_for_answer': False,
                'status': 'supported', 'evidence_ids': []}
        bad = {**self.assessment(), 'decision': 'retrieve_more', 'missing_parts': ['真实缺口'],
               'retrieval_queries': ['查询真实缺口'], 'checks': [item]}
        good = copy.deepcopy(bad)
        good['checks'][0]['status'] = 'not_applicable'
        model = Model([bad, good])
        result = Nodes(model, None, Settings(surface_verification=False)).assess(self.state())
        self.assertEqual(result['assessment_decision'], 'retrieve_more')
        self.assertEqual(result['retrieval_queries'], ['查询真实缺口'])
        for field, value in [('required_for_answer', True), ('evidence_ids', [self.doc['chunk_id']])]:
            invalid = copy.deepcopy(good)
            invalid['checks'][0][field] = value
            result = Nodes(Model([invalid, invalid]), None, Settings()).guarded('assess')(self.state())
            self.assertEqual(result['trace'][-1]['diagnostic']['code'], 'invalid_not_applicable_check')


if __name__ == '__main__':
    unittest.main()

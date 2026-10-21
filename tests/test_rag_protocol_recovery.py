"""Protocol recovery must not fabricate citations, change questions or stack retries."""
import copy
import json
from pathlib import Path
import unittest

from miniassist.rag.citations import resolve_quote, resolve_citation, model_citation_spans
from miniassist.rag.errors import CitationError
from miniassist.rag.llm import DeepSeek, ModelError
from miniassist.rag.nodes import Nodes
from miniassist.rag.settings import Settings, ROOT
from test_rag import ASSESS, ANSWER, DOC, Model
from test_rag_citations import document
from test_rag_format_repair import Session


class RawSession(Session):
    def post(self, url, **kwargs):
        self.requests.append(kwargs['json'])
        value = next(self.outputs)
        class Response:
            status_code = 200
            def json(self):
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': value}}]}
        return Response()


def assessed():
    return {**ASSESS, 'checks': [{'requirement': '查看电量', 'status': 'supported',
            'evidence_ids': ['c1'], 'required_for_answer': True, 'detail': '原文支持'}]}


class RecoveryTests(unittest.TestCase):
    def test_same_passage_nonoffered_range_recovers_only_unique_quote_and_audits(self):
        doc = document('标题\n电池需要逐块升级。\n后续说明')
        start = doc['text'].index('需要')
        quote = '需要逐块升级'
        ident = f'c1:{start}:{start+len(quote)}'
        with self.assertRaises(CitationError):
            resolve_quote(doc, quote, ident)
        result = resolve_citation(doc, quote, ident)
        self.assertEqual(result['quote'], quote)
        self.assertEqual(result['locator_repair']['original_span_id'], ident)
        self.assertEqual(result['text_map'][0]['source_start'], 10+start)

    def test_crossline_quote_expands_only_the_contained_line(self):
        doc = document('温度恢复正常，\n电池自动恢复充电。\n其他注意事项。')
        line_end = doc['text'].index('\n')
        result = resolve_citation(doc, '温度恢复正常，电池自动恢复充电。', f'c1:0:{line_end}')
        self.assertEqual(result['locator_repair']['reason'], 'quote_not_in_span')
        self.assertEqual(result['match_mode'], 'whitespace_only')
        with self.assertRaises(CitationError):
            resolve_citation(doc, '其他注意事项。', f'c1:0:{line_end}')

    def test_cross_chunk_ambiguous_changed_and_synthetic_quotes_still_rejected(self):
        doc = document('电池需要逐块升级。\n电池需要逐块升级。')
        for quote, span in [('电池需要逐块升级。', 'c1:1:22'),
                            ('电池需要逐块升级。', 'other:0:9'),
                            ('电池不必逐块升级。', 'c1:0:10')]:
            with self.subTest(quote=quote, span=span), self.assertRaises(CitationError):
                resolve_citation(doc, quote, span)
        synthetic = document('原文前缀。合成标签原文后缀。')
        synthetic['text_map'] = [
            {**DOC['text_map'][0], 'text_end':5,'source_end':15},
            {**DOC['text_map'][0], 'text_start':9,'text_end':14,'source_start':30,'source_end':35}]
        with self.assertRaises(CitationError):
            resolve_citation(synthetic, synthetic['text'], 'c1:0:14')

    def test_unique_passages_do_not_duplicate_every_line_in_model_payload(self):
        doc = document('第一行有效原文。\n第二行有效原文。\n第三行有效原文。')
        self.assertEqual(len(model_citation_spans(doc)), 1)
        duplicate = document('垂直：\n精度为零点一米。\n水平：\n精度为零点一米。')
        spans = [s for s in model_citation_spans(duplicate) if s['text']=='精度为零点一米。']
        self.assertEqual(len(spans),2)
        with self.assertRaises(CitationError):
            resolve_citation(duplicate,'精度为零点一米。')
        values = [resolve_citation(duplicate,s['text'],s['span_id'])['quote_start'] for s in spans]
        self.assertEqual(len(set(values)),2)

    def test_short_quote_one_repair_records_lengths_without_raw_invalid_value(self):
        bad = copy.deepcopy(ANSWER)
        bad['claims'][0]['citations'][0]['quote'] = '密值'
        session = Session([bad, ANSWER])
        model = DeepSeek(Settings(), api_key='test', session=session)
        result = Nodes(model,None,Settings()).generate({'question':'查看电量？','evidence':[DOC],'selected_ids':['c1']})
        self.assertEqual(result['status'],'answered')
        self.assertEqual(len(session.requests),2)
        diag = result['generation_format_repair']['first_failure']
        self.assertEqual(diag['quote_lengths'][0]['raw_chars'],2)
        self.assertNotIn('密值',json.dumps(diag,ensure_ascii=False))
        self.assertNotIn('密值',json.dumps(session.requests[1],ensure_ascii=False))

    def test_invalid_json_repair_is_bounded_and_has_safe_position_metadata(self):
        session = RawSession(['{"claims": SECRET', json.dumps(ANSWER)])
        model = DeepSeek(Settings(),api_key='test',session=session)
        model.call('generate',{'evidence':[]})
        self.assertEqual(len(session.requests),2)
        diag = model.last_call_diagnostic['first_failure']
        self.assertIn('json_position',diag)
        self.assertNotIn('SECRET',json.dumps(diag))
        self.assertNotIn('SECRET',json.dumps(session.requests[1]))

    def test_second_invalid_json_stays_error(self):
        session = RawSession(['{','{'])
        with self.assertRaises(ModelError) as error:
            DeepSeek(Settings(),api_key='test',session=session).call('generate',{})
        self.assertEqual(error.exception.diagnostic['code'],'schema_repair_failed')
        self.assertEqual(len(session.requests),2)

    def test_format_then_forged_citation_does_not_get_third_request(self):
        short = copy.deepcopy(ANSWER); short['claims'][0]['citations'][0]['quote']='短'
        forged = copy.deepcopy(ANSWER); forged['claims'][0]['citations'][0]['quote']='完全虚构的原文'
        session=Session([short,forged])
        model=DeepSeek(Settings(),api_key='test',session=session)
        result=Nodes(model,None,Settings()).guarded('generate')({'question':'问题','evidence':[DOC],'selected_ids':['c1']})
        self.assertEqual(result['status'],'error')
        self.assertEqual(len(session.requests),2)
        self.assertEqual(result['trace'][-1]['diagnostic']['code'],'quote_not_found')

    def test_citation_then_format_error_does_not_get_third_request(self):
        forged=copy.deepcopy(ANSWER);forged['claims'][0]['citations'][0]['evidence_id']='bad-id'
        session=RawSession([json.dumps(forged),'{'])
        result=Nodes(DeepSeek(Settings(),api_key='test',session=session),None,Settings()).guarded('generate')(
            {'question':'问题','evidence':[DOC],'selected_ids':['c1']})
        self.assertEqual(result['status'],'error')
        self.assertEqual(len(session.requests),2)

    def test_boundary_conflict_reasks_original_question_and_can_restore_real_answer(self):
        bad={**assessed(),'boundary':'future','decision':'answer'}
        model=Model([bad,assessed()])
        node=Nodes(model,None,Settings(surface_verification=False))
        result=node.assess({'question':'如何查看现在的电量？','original_question':'如何查看现在的电量？','evidence':[DOC]})
        self.assertEqual(result['assessment_decision'],'answer')
        self.assertEqual(result['boundary'],'none')
        self.assertEqual(result['assessment_route_repair']['attempts'],1)
        self.assertEqual(model.calls[0][1]['user_context'],model.calls[1][1]['user_context'])
        self.assertEqual(model.calls[0][1]['evidence'],model.calls[1][1]['evidence'])
        self.assertIn('routing_repair',model.calls[1][1])

    def test_real_user_gap_and_valid_boundary_need_no_repair(self):
        for output in [
            {'decision':'clarify','evidence_ids':[],'reason':'需型号','missing':['controller'],'clarification':'什么遥控器？'},
            {'decision':'insufficient','boundary':'service','evidence_ids':[],'reason':'不能报价'}]:
            model=Model([output])
            result=Nodes(model,None,Settings(surface_verification=False)).assess({'question':'问题','evidence':[DOC]})
            self.assertEqual(len(model.calls),1)
            self.assertEqual(result['assessment_decision'],output['decision'])

    def test_format_then_boundary_conflict_does_not_get_third_request(self):
        bad={**assessed(),'boundary':'future'}
        session=RawSession(['{',json.dumps(bad)])
        result=Nodes(DeepSeek(Settings(),api_key='test',session=session),None,Settings()).guarded('assess')(
            {'question':'现在如何看电量？','evidence':[DOC]})
        self.assertEqual(result['status'],'error')
        self.assertEqual(len(session.requests),2)

    def test_route_then_format_error_does_not_get_third_request(self):
        bad={**assessed(),'boundary':'future'}
        session=RawSession([json.dumps(bad),'{'])
        result=Nodes(DeepSeek(Settings(),api_key='test',session=session),None,Settings()).guarded('assess')(
            {'question':'现在如何看电量？','evidence':[DOC]})
        self.assertEqual(result['status'],'error')
        self.assertEqual(len(session.requests),2)



if __name__=='__main__':
    unittest.main()

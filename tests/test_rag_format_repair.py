import copy
import json
import unittest
from miniassist.rag.llm import DeepSeek, ModelError
from miniassist.rag.settings import Settings
from test_rag import ASSESS


class Session:
    def __init__(self, outputs):
        self.outputs, self.requests = iter(outputs), []

    def post(self, url, **kwargs):
        self.requests.append(kwargs['json'])
        value = next(self.outputs)
        class Response:
            status_code = value if isinstance(value, int) else 200

            def json(self):
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        return Response()


class FormatRepairTests(unittest.TestCase):
    def valid(self):
        return {**copy.deepcopy(ASSESS), 'checks': [{'requirement': '问题是否可核实',
                'status': 'unavailable', 'required_for_answer': True,
                'evidence_ids': [], 'detail': '此文档服务无法核实。'}]}

    def test_invalid_enum_rejudged_once_without_echoing_raw_value(self):
        good = self.valid()
        bad = copy.deepcopy(good)
        bad['checks'][0]['status'] = 'sensitive-invalid-value'
        session = Session([bad, good])
        model = DeepSeek(Settings(), api_key='test', session=session)
        result = model.call('assess', {'evidence': [], 'user_context': {'utterances': ['未来计划？']}})
        self.assertEqual(result['checks'][0]['status'], 'unavailable')
        self.assertEqual(len(session.requests), 2)
        retry = json.loads(session.requests[1]['messages'][1]['content'])
        self.assertIn('schema_repair', retry)
        self.assertEqual(retry['user_context']['utterances'], ['未来计划？'])
        self.assertNotIn('sensitive-invalid-value', json.dumps(retry))
        self.assertEqual(model.last_call_diagnostic['status'], 'validated')

    def test_second_invalid_response_stays_error(self):
        bad = self.valid()
        bad['checks'][0]['status'] = 'insufficient'
        session = Session([bad, bad])
        model = DeepSeek(Settings(), api_key='test', session=session)
        with self.assertRaises(ModelError) as caught:
            model.call('assess', {'evidence': []})
        self.assertEqual(caught.exception.diagnostic['code'], 'schema_repair_failed')
        self.assertEqual(len(session.requests), 2)

    def test_no_retry_for_other_invalid_fields_or_http_failure(self):
        bad = self.valid()
        bad['decision'] = 'not-a-route'
        for output in [bad, 429]:
            session = Session([output])
            model = DeepSeek(Settings(), api_key='test', session=session)
            with self.assertRaises(ModelError):
                model.call('assess', {'evidence': []})
            self.assertEqual(len(session.requests), 1)

    def test_valid_call_resets_previous_diagnostic_and_uses_one_request(self):
        session = Session([self.valid()])
        model = DeepSeek(Settings(), api_key='test', session=session)
        model.last_call_diagnostic = {'old': 'repair'}
        model.call('assess', {'evidence': []})
        self.assertEqual(model.last_call_diagnostic, {})
        self.assertEqual(len(session.requests), 1)


if __name__ == '__main__':
    unittest.main()

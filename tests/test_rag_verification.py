import unittest
import copy
from miniassist.rag.verification import verification_queries
from miniassist.rag.nodes import Nodes
from miniassist.rag.settings import Settings
from test_rag import ASSESS, DOC, Model, Retriever


class VerificationTests(unittest.TestCase):
    def test_action_queries_are_neutral_bounded_and_do_not_import_guessed_model(self):
        plans = verification_queries({'user_utterances': ['设备甲长按按钮没有启动。'],
                                      'slots': {'aircraft': '用户没有说的型号'}})
        self.assertEqual(len(plans), 2)
        self.assertTrue(all('用户没有说的型号' not in p['query'] for p in plans))
        self.assertIn('长按按钮', plans[0]['query'])
        self.assertIn('启动', plans[1]['query'])
        self.assertTrue(all('电量' not in p['query'] for p in plans))

    def test_explicit_controller_is_not_silently_treated_as_aircraft(self):
        plans = verification_queries({'user_utterances': ['设备甲的遥控器单击按钮不能启动。'],
                                      'slots': {'aircraft': '设备甲'}})
        self.assertTrue(all(p['query'].startswith('遥控器') for p in plans))

    def test_only_literal_document_version_triggers_cover_lookup(self):
        plans = verification_queries({'user_utterances': ['v7.2手册写了这个，对吗？'], 'slots': {}})
        self.assertEqual(len(plans), 1)
        self.assertIn('v7.2', plans[0]['query'])
        for text in ['我手里的旧手册没写。', 'App v7.2，手册里有说明吗？',
                     'App v7.2 手册里没有提到这个版本。']:
            with self.subTest(text=text):
                self.assertEqual(verification_queries({'user_utterances': [text]}), [])

    def test_policy_overrides_false_sufficiency_but_never_exceeds_one_round(self):
        state = {'question': '设备甲长按按钮没有启动。', 'original_question': '设备甲长按按钮没有启动。',
                 'user_utterances': ['设备甲长按按钮没有启动。'], 'slots': {'aircraft': '设备甲'},
                 'evidence': [DOC], 'retrieval_rounds': 0}
        nodes = Nodes(Model([ASSESS, ASSESS]), Retriever(), Settings())
        first = nodes.assess(state)
        self.assertEqual(first['assessment_decision'], 'retrieve_more')
        self.assertEqual(len(first['retrieval_queries']), 2)
        self.assertEqual(first['decision_override'], {'from': 'answer', 'to': 'retrieve_more'})
        final = nodes.assess({**state, 'retrieval_rounds': 1})
        self.assertEqual(final['assessment_decision'], 'answer')
        self.assertEqual(final['verification_requests'], [])

    def test_future_boundary_does_not_spend_verification_queries(self):
        assessment = {**ASSESS, 'decision': 'insufficient', 'boundary': 'future'}
        state = {'question': 'v7.2手册下一版何时发布？', 'slots': {}, 'evidence': [DOC],
                 'retrieval_rounds': 0}
        result = Nodes(Model([assessment]), Retriever(), Settings()).assess(state)
        self.assertEqual(result['assessment_decision'], 'insufficient')
        self.assertEqual(result['verification_requests'], [])

    def test_b0_disables_surface_override_but_keeps_model_retrieval(self):
        state = {'question': '设备甲长按按钮没有启动。',
                 'original_question': '设备甲长按按钮没有启动。',
                 'evidence': [DOC], 'retrieval_rounds': 0, 'slots': {}}
        settings = Settings(surface_verification=False)
        result = Nodes(Model([copy.deepcopy(ASSESS)]), Retriever(), settings).assess(state)
        self.assertEqual(result['assessment_decision'], 'answer')
        self.assertEqual(result['verification_requests'], [])
        self.assertEqual(result['decision_override'], {})
        autonomous = {**copy.deepcopy(ASSESS), 'decision': 'retrieve_more',
                      'retrieval_queries': ['正常启动的操作步骤'], 'missing_parts': ['操作步骤']}
        result = Nodes(Model([autonomous]), Retriever(), settings).assess(state)
        self.assertEqual(result['assessment_decision'], 'retrieve_more')
        self.assertEqual(result['retrieval_queries'], ['正常启动的操作步骤'])
        self.assertEqual(result['verification_requests'], [])


if __name__ == '__main__':
    unittest.main()

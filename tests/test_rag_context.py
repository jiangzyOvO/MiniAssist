"""Information ownership boundaries, using synthetic data rather than QA answers."""
import copy
import json
import unittest

from miniassist.rag.context import evidence_payload, user_context
from miniassist.rag.context import validate_case_attribution
from miniassist.rag.errors import AttributionError
from miniassist.rag.state import Answer
from miniassist.rag.graph import RAGService
from test_rag import ASSESS, ANSWER, DOC, Model, Retriever, understanding


class ContextTests(unittest.TestCase):
    def test_unknown_user_document_identity_and_chronology_are_rejected(self):
        for text in ("你手里的旧手册是v9.9，所以没介绍。", "您持有的手册早于新增支持条目。",
                     "你手里的旧手册没有介绍，只是手册内容早于该条目。",
                     "用户提供的说明书出版于2099.12，因此未记载该功能。"):
            with self.subTest(text=text):
                answer = copy.deepcopy(ANSWER)
                answer["claims"][0]["text"] = text
                with self.assertRaises(AttributionError):
                    validate_case_attribution({"user_utterances": ["我的旧手册没有介绍，支持吗？"]},
                                              Answer.model_validate(answer))

    def test_declared_identity_and_explicit_unknown_or_hypothetical_are_allowed(self):
        cases = [("你的手册是v7.2。", "我用的是v7.2手册。"),
                 ("你手里的手册早于该条目。", "我的手册出版于2020年。"),
                 ("不能认定你手里的旧手册早于该条目。", "旧手册没写。"),
                 ("如果你的手册早于该条目，需再查适用说明。", "旧手册没写。"),
                 ("你手里的旧手册未介绍，不代表不支持。", "旧手册没写。"),
                 ("检索到的手册正文版本为v7.2。", "旧手册没写。")]
        for text, message in cases:
            with self.subTest(text=text):
                answer = copy.deepcopy(ANSWER)
                answer["claims"][0]["text"] = text
                validate_case_attribution({"user_utterances": [message]}, Answer.model_validate(answer))

    def test_attribution_repair_uses_single_shared_budget_and_does_not_publish_bad_claim(self):
        bad = copy.deepcopy(ANSWER)
        bad["claims"][0]["text"] = "你手里的旧手册是v9.9。"
        for second, expected in ((ANSWER, "answered"), (bad, "error")):
            with self.subTest(expected=expected):
                model = Model([understanding(), ASSESS, bad, second])
                state = RAGService(model, Retriever()).ask("scope", "旧手册没写，怎么操作？")
                self.assertEqual(state["status"], expected)
                self.assertEqual(len(model.calls), 4)
                self.assertNotIn("v9.9", state["response"])
                self.assertEqual(model.calls[-1][1]["citation_repair"]["diagnostic"]["code"],
                                 "user_document_identity_unverified")

    def test_explicit_verified_condition_adds_only_already_retrieved_document(self):
        doc2 = {**copy.deepcopy(DOC), "chunk_id": "c2"}
        assessment = {**ASSESS, "critical_conditions": [
            {"evidence_id": "c2", "quote": "单击可查看剩余电量"}]}
        model = Model([understanding(), assessment, ANSWER])
        state = RAGService(model, Retriever([DOC, doc2])).ask("scope", "操作条件？")
        self.assertEqual(state["condition_evidence_added"], ["c2"])
        self.assertEqual(state["selected_ids"], ["c1", "c2"])
        self.assertEqual(state["citations"][-1]["chunk_id"], "c2")
        self.assertEqual(len(state["evidence"]), 2)

    def test_document_metadata_is_not_sent_as_case_or_fact_evidence(self):
        doc = {**copy.deepcopy(DOC), "source_version": "v9.9 封面2099.12",
               "snapshot_date": "2100-01-01", "title": "版本v9.9的设备甲手册"}
        model = Model([understanding(), ASSESS, ANSWER])
        state = RAGService(model, Retriever([doc])).ask("scope", "我手里的旧手册没写。")
        for stage, payload in model.calls:
            with self.subTest(stage=stage):
                serialized = json.dumps(payload, ensure_ascii=False)
                self.assertNotIn("2099.12", serialized)
                self.assertNotIn("2100-01-01", serialized)
                self.assertNotIn("v9.9", serialized)
        self.assertEqual(state["citations"][0]["source_version"], doc["source_version"])
        self.assertIn("2099.12", state["response"])  # attribution is still rendered
        self.assertEqual(state["evidence"][0]["source_version"], doc["source_version"])

    def test_old_assessment_explanation_cannot_be_laundered_into_generation(self):
        poison = "用户拥有的版本就是v9.9，因为2099年才上市"
        model = Model([understanding(), {**ASSESS, "reason": poison, "answerable_parts": [poison]}, ANSWER])
        state = RAGService(model, Retriever()).ask("scope", "旧手册没写，支持吗？")
        self.assertEqual(state["reason"], poison)  # audit trail remains intact
        self.assertNotIn(poison, json.dumps(model.calls[-1][1], ensure_ascii=False))

    def test_user_explicit_version_is_not_removed_with_metadata(self):
        message = "我看的是v7.2手册，封面2031.08，有其他办法吗？"
        model = Model([understanding(message), ASSESS, ASSESS, ANSWER])
        state = RAGService(model, Retriever()).ask("scope", message)
        self.assertEqual(state["status"], "answered")
        for _, payload in model.calls:
            self.assertEqual(payload["user_context"]["utterances"], [message])

    def test_in_body_version_text_remains_unchanged_and_citable(self):
        doc = copy.deepcopy(DOC)
        doc["text"] = "本功能需要软件v7.2及以上版本。"
        doc["text_map"] = [{**DOC["text_map"][0], "text_end": len(doc["text"]),
                            "source_end": 10 + len(doc["text"])}]
        payload = evidence_payload(doc)
        self.assertEqual(payload["text"], doc["text"])
        self.assertIn("v7.2及以上", payload["citable_spans"][0]["text"])
        self.assertNotIn("source_version", payload)

    def test_followup_ledger_contains_only_user_statements_and_resets_on_new_topic(self):
        ask = {"decision": "clarify", "evidence_ids": [], "reason": "需用户补型号",
               "missing": ["controller"], "clarification": "请补型号，助手猜测v9.9。"}
        model = Model([understanding("旧遥控器问题"), ask,
                       understanding("型号乙不能连接"), ASSESS, ANSWER,
                       understanding("新问题"), ASSESS, ANSWER])
        service = RAGService(model, Retriever())
        service.ask("scope", "我的旧遥控器连不上")
        second = service.ask("scope", "型号乙")
        self.assertEqual(second["user_utterances"], ["我的旧遥控器连不上", "型号乙"])
        self.assertNotIn("v9.9", str(model.calls[3][1]["user_context"]))
        third = service.ask("scope", "新问题", new_topic=True)
        self.assertEqual(third["user_utterances"], ["新问题"])

    def test_model_detected_new_topic_clears_old_user_ledger(self):
        ask = {"decision": "clarify", "evidence_ids": [], "reason": "需型号",
               "missing": ["controller"], "clarification": "型号？"}
        model = Model([understanding(), ask, understanding("设备丙", new_topic=True), ASSESS, ANSWER])
        service = RAGService(model, Retriever())
        service.ask("scope", "原设备甲问题")
        result = service.ask("scope", "换个话题，问设备丙")
        self.assertEqual(result["user_utterances"], ["换个话题，问设备丙"])

    def test_legacy_replay_fallback_ignores_resolved_question_and_assistant_history(self):
        context = user_context({"original_question": "旧手册没写", "message": "版本未知",
                                "question": "你的手册v9.9", "slots": {"firmware": "v9.9"},
                                "history": [{"role": "assistant", "content": "猜测v9.9"}]})
        self.assertEqual(context["utterances"], ["旧手册没写", "版本未知"])


if __name__ == "__main__":
    unittest.main()

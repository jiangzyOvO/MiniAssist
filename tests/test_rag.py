"""Behavioral workflow tests; injected model output is not model-quality evaluation."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import requests
from miniassist.rag.graph import RAGService
from miniassist.rag.llm import DeepSeek, ModelError
from miniassist.rag.retriever import Catalog, HybridRetriever, weighted_rrf
from miniassist.rag.settings import Settings, ROOT, load_config


def unreviewed_service(llm, retriever, settings=None):
    """Existing routing/citation tests isolate the pre-review pipeline.

    The experimental opt-in review node is covered by test_rag_answer_review.py.
    """
    from dataclasses import replace
    return RAGService(llm, retriever, replace(settings or Settings(), review_answer=False))

TEXT = "设备甲的按钮位于右侧，单击可查看剩余电量。"
DOC = {"chunk_id": "c1", "text": TEXT, "source_id": "S1", "source_version": "测试版本",
       "source_sha256": "test", "pages": [1], "url": "https://example.test/manual",
       "text_map": [{"text_start": 0, "text_end": len(TEXT), "source_start": 10,
                     "source_end": 10 + len(TEXT), "page": 1, "source_id": "S1", "block_id": "b1"}]}


def understanding(question="设备甲怎样查看剩余电量？", missing=None, new_topic=False, slots=None):
    return {"new_topic": new_topic, "question": question, "slots": slots or {"aircraft": "设备甲"},
            "missing": missing or [], "clarification": "请补充遥控器型号。" if missing else ""}


ASSESS = {"decision": "answer", "evidence_ids": ["c1"], "reason": "测试证据覆盖"}
CLARIFY = {"decision": "clarify", "evidence_ids": [], "reason": "型号阻塞兼容判断",
           "missing": ["controller"], "clarification": "请补充遥控器型号。"}
ANSWER = {"claims": [{"text": "单击按钮可查看剩余电量。",
                     "citations": [{"evidence_id": "c1", "quote": "单击可查看剩余电量"}]}]}


class Model:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def call(self, stage, payload):
        self.calls.append((stage, copy.deepcopy(payload)))
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class Retriever:
    def __init__(self, docs=None, error=None):
        self.docs, self.error, self.calls = [DOC] if docs is None else docs, error, []

    def search(self, query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return copy.deepcopy(self.docs)


class GraphTests(unittest.TestCase):
    def test_slot_metadata_stays_in_state_not_retrieval_but_reaches_answer_nodes(self):
        question = "设备甲怎样查看剩余电量？"
        slots = {"aircraft": "设备甲", "controller": "遥控器乙", "symptom": "查看剩余电量"}
        model = Model([understanding(question, slots=slots), ASSESS, ANSWER])
        retriever = Retriever()
        state = unreviewed_service(model, retriever).ask("slot-regression", question)
        self.assertEqual(state["status"], "answered")
        self.assertEqual(retriever.calls, [question])
        self.assertEqual(state["slots"], slots)
        for stage, payload in model.calls[1:]:
            self.assertIn(stage, ("assess", "generate"))
            self.assertEqual(payload["model_context"]["slots"], slots)

    def test_direct_answer_real_graph_trace_and_source_mapping(self):
        model, retriever = Model([understanding(), ASSESS, ANSWER]), Retriever()
        state = unreviewed_service(model, retriever).ask("a", "设备甲怎样查看电量？")
        self.assertEqual(state["status"], "answered")
        self.assertEqual([x["node"] for x in state["trace"]],
                         ["understand", "retrieve", "assess", "generate", "finish"])
        citation = state["citations"][0]
        self.assertEqual(citation["text_map"][0]["source_start"], 10 + TEXT.index(citation["quote"]))
        self.assertEqual(citation["text_map"][0]["source_end"] - citation["text_map"][0]["source_start"],
                         len(citation["quote"]))
        self.assertIn("[1]", state["response"])
        json.dumps(state)  # Regression: traces must not contain self-references.

    def test_clarification_resumes_original_question_and_merges_slots(self):
        model = Model([understanding("设备甲为什么不能连接旧遥控器？", ["controller"]),
                       CLARIFY,
                       understanding("设备甲与遥控器乙为什么连接不上？", slots={"controller": "遥控器乙"}),
                       ASSESS, ANSWER])
        retriever, service = Retriever(), None
        service = unreviewed_service(model, retriever)
        first = service.ask("a", "设备甲为什么不能连接旧遥控器？")
        self.assertEqual(first["status"], "clarify")
        self.assertEqual(retriever.calls, ["设备甲为什么不能连接旧遥控器？"])
        second = service.ask("a", "遥控器乙")
        self.assertEqual(second["status"], "answered")
        self.assertEqual(second["original_question"], "设备甲为什么不能连接旧遥控器？")
        self.assertTrue(model.calls[2][1]["pending"])
        self.assertEqual(model.calls[2][1]["previous_question"], first["question"])
        self.assertEqual(second["slots"], {"aircraft": "设备甲", "controller": "遥控器乙"})
        self.assertEqual(retriever.calls, ["设备甲为什么不能连接旧遥控器？", "设备甲与遥控器乙为什么连接不上？"])
        self.assertIn("连接不上", retriever.calls[-1])
        self.assertEqual(len(second["history"]), 4)

    def test_threads_are_isolated(self):
        model = Model([understanding(missing=["controller"]), CLARIFY, understanding(), ASSESS, ANSWER])
        service = unreviewed_service(model, Retriever())
        service.ask("a", "我的旧遥控器不工作")
        service.ask("b", "设备甲如何看电量")
        self.assertFalse(model.calls[2][1]["pending"])
        self.assertEqual(model.calls[2][1]["previous_question"], "")
        self.assertEqual(model.calls[2][1]["history"], [])

    def test_clarification_is_bounded(self):
        model = Model([understanding(missing=["controller"]), CLARIFY] * 3)
        retriever = Retriever()
        service = unreviewed_service(model, retriever)
        for _ in range(3):
            state = service.ask("a", "我不清楚")
        self.assertEqual(state["status"], "insufficient")
        self.assertFalse(state["pending"])
        self.assertEqual(len(retriever.calls), 3)

    def test_explicit_new_topic_clears_history_and_slots(self):
        model = Model([understanding(missing=["controller"]), CLARIFY, understanding(), ASSESS, ANSWER])
        service = unreviewed_service(model, Retriever())
        service.ask("a", "连接不上")
        service.ask("a", "新问题", new_topic=True)
        self.assertFalse(model.calls[2][1]["pending"])
        self.assertEqual(model.calls[2][1]["slots"], {})
        self.assertEqual(model.calls[2][1]["history"], [])

    def test_model_new_topic_clears_pending_slots(self):
        model = Model([understanding(missing=["controller"]), CLARIFY,
                       understanding(new_topic=True, slots={"aircraft": "设备丙"}), ASSESS, ANSWER])
        service = unreviewed_service(model, Retriever())
        service.ask("a", "连接不上")
        state = service.ask("a", "设备丙如何查看电量")
        self.assertEqual(state["slots"], {"aircraft": "设备丙"})
        self.assertEqual(state["clarification_count"], 0)

    def test_no_results_does_not_generate_or_claim_unsupported(self):
        model = Model([understanding()])
        state = unreviewed_service(model, Retriever([])).ask("a", "问题")
        self.assertEqual(state["status"], "insufficient")
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn("官方不支持", state["response"])

    def test_insufficient_evidence_stops_before_generation(self):
        model = Model([understanding(), {"decision": "insufficient", "evidence_ids": [],
                                         "reason": "型号不同，不能使用"}])
        state = unreviewed_service(model, Retriever()).ask("a", "问题")
        self.assertEqual(state["status"], "insufficient")
        self.assertEqual(len(model.calls), 2)

    def test_invalid_understanding_schema_fails_closed(self):
        state = unreviewed_service(Model([{"question": "invalid"}]), Retriever()).ask("a", "问题")
        self.assertEqual(state["status"], "error")

    def test_timeout_has_no_provider_secret_in_trace(self):
        state = unreviewed_service(Model([requests.Timeout("SECRET_PROVIDER_BODY")]), Retriever()).ask("a", "问题")
        self.assertEqual(state["status"], "error")
        self.assertNotIn("SECRET", json.dumps(state))

    def test_retrieval_failure_not_empty_result(self):
        state = unreviewed_service(Model([understanding()]), Retriever(error=RuntimeError("bad"))).ask("a", "问题")
        self.assertEqual(state["status"], "error")
        self.assertIn("retrieve:", state["error"])

    def test_unknown_evidence_selection_fails(self):
        bad = {**ASSESS, "evidence_ids": ["invented"]}
        state = unreviewed_service(Model([understanding(), bad]), Retriever()).ask("a", "问题")
        self.assertEqual(state["status"], "error")

    def test_fabricated_quote_or_id_fails(self):
        for cite in [{"evidence_id": "c1", "quote": "原文没有写的内容"},
                     {"evidence_id": "invented", "quote": TEXT}]:
            with self.subTest(cite=cite):
                bad = {"claims": [{"text": "答案", "citations": [cite]}]}
                state = unreviewed_service(Model([understanding(), ASSESS, bad]), Retriever()).ask("a", "问题")
                self.assertEqual(state["status"], "error")
                self.assertEqual(state["citations"], [])

    def test_synthetic_unmapped_text_cannot_be_cited(self):
        doc = copy.deepcopy(DOC)
        doc["text"] += "人工合成标题"
        answer = {"claims": [{"text": "答案", "citations": [
            {"evidence_id": "c1", "quote": "人工合成标题"}]}]}
        state = unreviewed_service(Model([understanding(), ASSESS, answer]), Retriever([doc])).ask("a", "问题")
        self.assertEqual(state["status"], "error")

    def test_new_turn_resets_stale_evidence_and_error(self):
        model = Model([understanding(), ASSESS, ANSWER, ValueError("bad")])
        service = unreviewed_service(model, Retriever())
        service.ask("a", "问题")
        state = service.ask("a", "第二个问题")
        self.assertEqual(state["evidence"], [])
        self.assertEqual(state["citations"], [])
        self.assertEqual(state["status"], "error")

    def test_history_bounded(self):
        model = Model([understanding(), ASSESS, ANSWER] * 8)
        service = unreviewed_service(model, Retriever(), Settings(max_history=4))
        for _ in range(8):
            state = service.ask("a", "问题")
        self.assertEqual(len(state["history"]), 4)

    def test_input_size_checked(self):
        service = unreviewed_service(Model([]), Retriever())
        for text in ("", "a"*1501):
            with self.assertRaises(ValueError):
                service.ask("a", text)


class AdapterTests(unittest.TestCase):
    def test_missing_key_explicit(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ModelError):
                DeepSeek(Settings())

    def test_json_mode_http_contract_and_schema(self):
        class Response:
            status_code = 200
            def json(self):
                return {"choices": [{"finish_reason": "stop",
                                     "message": {"content": json.dumps(understanding())}}]}
        class Session:
            def post(self, url, **kwargs):
                self.url, self.kwargs = url, kwargs
                return Response()
        session = Session()
        model = DeepSeek(Settings(), api_key="test", session=session)
        self.assertEqual(model.call("understand", {"message": "问题"}), understanding())
        self.assertEqual(session.url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(session.kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertFalse(session.kwargs["allow_redirects"])

    def test_http_and_empty_or_truncated_output_rejected(self):
        for status, content, reason in [(429, "secret", "stop"), (200, "", "stop"),
                                        (200, "{}", "length"), (200, "{}", "stop")]:
            class Response:
                status_code = status
                def json(self):
                    return {"choices": [{"finish_reason": reason, "message": {"content": content}}]}
            class Session:
                def post(self, *args, **kwargs):
                    return Response()
            with self.subTest(status=status, content=content, reason=reason):
                with self.assertRaises(ModelError):
                    DeepSeek(Settings(), api_key="test", session=Session()).call("understand", {})

    def test_weighted_rrf_scores_ties_and_duplicates(self):
        bm25 = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        dense = [{"chunk_id": "b"}, {"chunk_id": "c"}]
        self.assertEqual(weighted_rrf(bm25, dense, alpha=1, k=0, limit=2),
                         [{"chunk_id": "a", "score": 1.0}, {"chunk_id": "b", "score": 0.5}])
        self.assertEqual(weighted_rrf(bm25, dense, alpha=0, k=0, limit=2),
                         [{"chunk_id": "b", "score": 1.0}, {"chunk_id": "c", "score": 0.5}])
        tied = weighted_rrf([{"chunk_id": "z"}], [{"chunk_id": "a"}], alpha=.5, k=0)
        self.assertEqual([r["chunk_id"] for r in tied], ["a", "z"])
        with self.assertRaises(ValueError):
            weighted_rrf(bm25 + bm25, dense)

    def test_frozen_catalog_valid_and_corruption_rejected(self):
        config = load_config()
        self.assertEqual(len(Catalog(config).rows), 503)
        bad = {**config, "chunks_sha256": "bad"}
        with self.assertRaises(ValueError):
            Catalog(bad)

    def test_es_incomplete_response_rejected(self):
        for response in ({"timed_out": True, "_shards": {"failed": 0}},
                         {"timed_out": False, "_shards": {"failed": 1}}):
            with self.assertRaises(RuntimeError):
                HybridRetriever._complete(response)


if __name__ == "__main__":
    unittest.main()

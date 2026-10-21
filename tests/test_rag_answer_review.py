"""Opt-in review path: provenance, bounded calls and pending state."""
import copy
import unittest

from miniassist.rag.graph import RAGService as Service
from miniassist.rag.settings import Settings
from test_rag import Model, Retriever, ASSESS, ANSWER, understanding


def RAGService(model, retriever):
    return Service(model, retriever, Settings(review_answer=True))


class AnswerReviewTests(unittest.TestCase):
    def test_opt_in_review_replaces_draft_and_preserves_both_in_trace(self):
        revised = copy.deepcopy(ANSWER)
        revised["claims"][0]["text"] = "可以单击按钮查看剩余电量。"
        model = Model([understanding(), ASSESS, ANSWER, {"issues": ["直接回答用户"], "answer": revised}])
        result = RAGService(model, Retriever()).ask("review", "如何看电量？")
        self.assertEqual(result["status"], "answered")
        self.assertTrue(result["generation_review"]["changed"])
        self.assertEqual([stage for stage, _ in model.calls], ["understand", "assess", "generate", "review_answer"])
        self.assertEqual(model.calls[-1][1]["draft"], result["draft_answer"])
        self.assertIn("可以单击", result["response"])
        self.assertEqual(result["trace"][-2]["node"], "review_answer")

    def test_review_cannot_publish_forged_quote_or_retry_forever(self):
        revised = copy.deepcopy(ANSWER)
        revised["claims"][0]["citations"][0]["quote"] = "虚构的一段话"
        model = Model([understanding(), ASSESS, ANSWER, {"issues": [], "answer": revised}])
        result = RAGService(model, Retriever()).ask("review", "问题")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["citations"], [])
        self.assertNotIn("单击按钮", result["response"])
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(result["trace"][-2]["diagnostic"]["code"], "quote_not_found")

    def test_review_failure_does_not_publish_unreviewed_draft(self):
        model = Model([understanding(), ASSESS, ANSWER, ValueError("provider text")])
        result = RAGService(model, Retriever()).ask("review", "问题")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["citations"], [])
        self.assertNotIn("provider text", str(result["trace"]))

    def test_review_does_not_count_partial_followup_twice(self):
        assess = {**ASSESS, "missing_parts": ["异常阶段"], "missing": ["stage"],
                  "clarification": "发生在哪个阶段？"}
        model = Model([understanding(), assess, ANSWER, {"issues": [], "answer": ANSWER}])
        result = RAGService(model, Retriever()).ask("review", "问题")
        self.assertTrue(result["pending"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["clarification_count"], 1)
        self.assertEqual(result["response"].count("发生在哪个阶段？"), 1)

    def test_clarification_never_calls_answer_reviewer(self):
        assess = {"decision": "clarify", "evidence_ids": [], "reason": "型号未知",
                  "missing": ["controller"], "clarification": "具体什么型号？"}
        model = Model([understanding(), assess])
        result = RAGService(model, Retriever()).ask("review", "问题")
        self.assertTrue(result["pending"])
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()

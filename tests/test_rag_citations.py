"""Citation integrity: original coordinates survive whitespace recovery."""
import copy
import json
import unittest
from miniassist.rag.citations import citable_spans, resolve_quote
from miniassist.rag.errors import CitationError
from miniassist.rag.nodes import Nodes
from miniassist.rag.settings import Settings
from test_rag import DOC, Model


def document(text):
    doc = copy.deepcopy(DOC)
    doc["text"] = text
    doc["text_map"][0].update(text_end=len(text), source_end=10 + len(text))
    return doc


class CitationTests(unittest.TestCase):
    def test_whitespace_recovery_keeps_original_offsets_and_text(self):
        doc = document("适用于设备甲：温度恢复正常，电池将自动恢复充电，\n无需重新拔插充电器。")
        quote = "温度恢复正常，电池将自动恢复充电，无需重新拔插充电器。"
        result = resolve_quote(doc, quote)
        self.assertEqual(result["match_mode"], "whitespace_only")
        self.assertEqual(result["model_quote"], quote)
        self.assertIn("\n", result["quote"])
        lo, hi = result["quote_start"], result["quote_end"]
        self.assertEqual(doc["text"][lo:hi], result["quote"])
        self.assertEqual(result["text_map"][0]["source_start"], 10 + lo)
        self.assertEqual(result["text_map"][0]["source_end"], 10 + hi)

    def test_changed_numbers_punctuation_or_splicing_are_rejected(self):
        doc = document("温度为5–40℃。开机时不支持充电。请等待。")
        for quote in ("温度为5–50℃。", "温度为5-40℃。", "温度为5–40℃。请等待。"):
            with self.subTest(quote=quote), self.assertRaises(CitationError):
                resolve_quote(doc, quote)

    def test_synthetic_table_labels_split_citable_spans(self):
        doc = document("LED4 每秒闪3次\n第2列：充电温度过高")
        text = doc["text"]
        end = text.index("\n")
        second = text.index("充电")
        doc["text_map"] = [
            {**DOC["text_map"][0], "text_end": end, "source_end": 10 + end},
            {**DOC["text_map"][0], "text_start": second, "text_end": len(text),
             "source_start": 100, "source_end": 100 + len(text) - second}]
        self.assertEqual([s["text"] for s in citable_spans(doc)],
                         ["LED4 每秒闪3次", "充电温度过高"])
        for quote in ("第2列：充电温度过高", "LED4每秒闪3次充电温度过高"):
            with self.assertRaises(CitationError):
                resolve_quote(doc, quote)
        self.assertEqual(resolve_quote(doc, "充电温度过高")["text_map"][0]["source_start"], 100)

    def test_duplicate_text_with_different_source_coordinates_is_ambiguous(self):
        doc = document("自动恢复充电。\n自动恢复充电。")
        with self.assertRaises(CitationError) as caught:
            resolve_quote(doc, "自动恢复充电。")
        self.assertEqual(caught.exception.diagnostic["code"], "ambiguous_quote")

    def test_nonlinear_mapping_fails_closed(self):
        doc = copy.deepcopy(DOC)
        doc["text_map"][0]["source_end"] += 1
        with self.assertRaises(CitationError):
            resolve_quote(doc, "单击可查看剩余电量")

    def test_whitespace_only_quote_rejected(self):
        with self.assertRaises(CitationError):
            resolve_quote(DOC, "    ")

    def test_failure_diagnostic_retains_only_validated_answer_and_location(self):
        answer = {"claims": [{"kind": "fact", "text": "答案", "citations": [
            {"evidence_id": "c1", "quote": "不存在的原文"}]}]}
        result = Nodes(Model([answer]), None, Settings()).guarded("generate")({
            "question": "问题", "slots": {}, "evidence": [DOC], "selected_ids": ["c1"]})
        diagnostic = result["trace"][0]["diagnostic"]
        self.assertEqual(diagnostic["code"], "quote_not_found")
        self.assertEqual(diagnostic["claim_index"], 0)
        self.assertEqual(diagnostic["generated_answer"], answer)
        self.assertEqual(result["citations"], [])

    def test_schema_error_does_not_log_invalid_extra_value(self):
        model = Model([{"claims": [], "SECRET_FIELD": "SECRET_PROVIDER_BODY"}])
        result = Nodes(model, None, Settings()).guarded("generate")({
            "question": "问题", "slots": {}, "evidence": [DOC], "selected_ids": ["c1"]})
        self.assertEqual(result["trace"][0]["diagnostic"]["code"], "schema_invalid")
        self.assertNotIn("SECRET", json.dumps(result))


if __name__ == "__main__":
    unittest.main()

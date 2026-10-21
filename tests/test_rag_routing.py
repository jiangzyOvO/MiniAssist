"""Decision boundaries and bounded follow-up retrieval, independent of DJI gold."""
import copy
import unittest
from test_rag import Model, Retriever, DOC, ASSESS, ANSWER, understanding
from test_rag import unreviewed_service
from miniassist.rag.nodes import Nodes
from miniassist.rag.settings import Settings
from miniassist.rag.retriever import Catalog


def gap():
    return {"decision": "retrieve_more", "evidence_ids": ["c1"], "reason": "第二个子问题缺证据",
            "answerable_parts": ["已支持第一项"], "missing_parts": ["第二项的要求"],
            "retrieval_queries": ["设备甲第二项的要求"]}


class RoutingTests(unittest.TestCase):
    def test_optional_knowledge_gap_does_not_block_answer(self):
        decision = {**ASSESS, "missing_parts": ["个体原因"], "checks": [
            {"requirement": "个体原因", "required_for_answer": False, "status": "knowledge_gap",
             "evidence_ids": [], "detail": "原问只问是否存在其他方法，无须诊断原因"}]}
        result = unreviewed_service(Model([understanding(), decision, ANSWER]), Retriever()).ask("r", "有其他方法吗？")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["retrieval_rounds"], 0)
        self.assertEqual(result["missing_parts"], [])

    def test_required_gap_without_query_is_partial_not_error_or_complete(self):
        decision = {**ASSESS, "checks": [
            {"requirement": "另一项必要条件", "required_for_answer": True, "status": "knowledge_gap",
             "evidence_ids": [], "detail": "未找到该条件且未提供可用子查询"}]}
        result = unreviewed_service(Model([understanding(), decision, ANSWER]), Retriever()).ask("r", "两项条件？")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["retrieval_rounds"], 0)
        self.assertFalse(result["pending"])

    def test_declared_knowledge_gap_overrides_answer_and_uses_bounded_retrieval(self):
        decision = {**gap(), "decision": "answer", "missing_parts": [], "checks": [
            {"requirement": "完整开始条件", "status": "knowledge_gap", "evidence_ids": [],
             "detail": "只有失败原因不能确认全部要求"}]}
        result = unreviewed_service(Model([understanding(), decision, ASSESS, ANSWER]), Retriever()).ask("r", "能开始吗？")
        self.assertEqual(result["retrieval_rounds"], 1)
        assessment = next(t["result"] for t in result["trace"] if t["node"] == "assess")
        self.assertEqual(assessment["decision_override"], {"from": "answer", "to": "retrieve_more"})
        self.assertIn("完整开始条件", assessment["missing_parts"])

    def test_critical_condition_is_preserved_even_if_generated_paraphrase_omits_it(self):
        text = "设备甲的功能需使用工具V2.0及以上版本。单击按钮查看电量。"
        doc = {**copy.deepcopy(DOC), "text": text, "text_map": [
            {**DOC["text_map"][0], "text_end": len(text), "source_end": 10 + len(text)}]}
        condition = {"evidence_id": "c1", "quote": "需使用工具V2.0及以上版本"}
        decision = {**ASSESS, "critical_conditions": [condition]}
        answer = {"claims": [{"text": "单击按钮查看电量。", "citations": [
            {"evidence_id": "c1", "quote": "单击按钮查看电量"}]}]}
        model = Model([understanding(), decision, answer])
        state = unreviewed_service(model, Retriever([doc])).ask("r", "工具要求是什么？")
        self.assertIn("适用条件（原文）：“需使用工具V2.0及以上版本”", state["response"])
        citation = state["citations"][-1]
        self.assertEqual(citation["quote"], condition["quote"])
        self.assertEqual(citation["text_map"][0]["source_start"], 10 + text.index(condition["quote"]))

    def test_fabricated_or_unretrieved_critical_conditions_never_publish(self):
        for condition in ({"evidence_id": "c1", "quote": "不存在的原文条件"},
                          {"evidence_id": "another", "quote": "单击可查看剩余电量"}):
            with self.subTest(condition=condition):
                state = unreviewed_service(Model([understanding(), {**ASSESS, "critical_conditions": [condition]}]),
                                   Retriever()).ask("r", "条件？")
                self.assertEqual(state["status"], "error")
                self.assertEqual(state["citations"], [])

    def test_partial_answer_with_followup_resumes_original_problem(self):
        decision = {**ASSESS, "answerable_parts": ["按钮说明"],
                    "missing": ["stage"], "clarification": "具体在哪个阶段异常？",
                    "checks": [{"requirement": "异常阶段", "status": "user_gap",
                                "evidence_ids": [], "detail": "需用户补充"}]}
        model = Model([understanding("设备甲异常"), decision, ANSWER,
                       understanding("设备甲启动阶段异常"), ASSESS, ANSWER])
        service = unreviewed_service(model, Retriever())
        first = service.ask("r", "我的设备甲异常了")
        self.assertEqual(first["status"], "partial")
        self.assertTrue(first["pending"])
        self.assertIn("具体在哪个阶段异常？", first["response"])
        second = service.ask("r", "启动阶段")
        self.assertTrue(model.calls[3][1]["pending"])
        self.assertEqual(second["original_question"], "我的设备甲异常了")
        self.assertFalse(second["pending"])
        self.assertEqual(second["critical_conditions"], [])

    def test_partial_followup_obeys_same_two_round_limit(self):
        decision = {**ASSESS, "missing_parts": ["发生阶段"],
                    "missing": ["stage"], "clarification": "发生在哪个阶段？"}
        service = unreviewed_service(Model([understanding(), decision, ANSWER] * 3), Retriever())
        for _ in range(3):
            result = service.ask("r", "没看清")
        self.assertFalse(result["pending"])
        self.assertEqual(result["clarification_count"], 2)
        self.assertIn("追问上限", result["response"])

    def test_user_gap_without_actual_question_is_rejected(self):
        decision = {**ASSESS, "checks": [{"requirement": "现象", "status": "user_gap",
                                          "evidence_ids": [], "detail": "需要用户描述"}]}
        result = unreviewed_service(Model([understanding(), decision]), Retriever()).ask("r", "问题")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["trace"][-2]["diagnostic"]["code"], "missing_clarification_fields")

    def test_one_citation_repair_is_audited_and_revalidated(self):
        bad = copy.deepcopy(ANSWER)
        bad["claims"][0]["citations"][0]["evidence_id"] = "S1"
        model = Model([understanding(), ASSESS, bad, ANSWER])
        result = unreviewed_service(model, Retriever()).ask("r", "问题")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["generation_repair"]["attempts"], 1)
        self.assertEqual(result["generation_repair"]["first_failure"]["code"], "unknown_evidence_id")
        self.assertEqual(result["citations"][0]["chunk_id"], "c1")
        self.assertEqual(len(model.calls), 4)
        self.assertIn("diagnostic", model.calls[-1][1]["citation_repair"])

    def test_repeated_bad_citation_stops_after_one_repair_without_publishing(self):
        bad = copy.deepcopy(ANSWER)
        bad["claims"][0]["citations"][0]["quote"] = "伪造的引用内容"
        model = Model([understanding(), ASSESS, bad, bad])
        result = unreviewed_service(model, Retriever()).ask("r", "问题")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["citations"], [])
        self.assertNotIn("单击按钮", result["response"])
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(result["trace"][-2]["diagnostic"]["repair_failure"]["code"], "quote_not_found")

    def test_proposed_missing_slots_do_not_block_available_conditional_answer(self):
        model = Model([understanding(missing=["controller"]), ASSESS, ANSWER])
        search = Retriever()
        state = unreviewed_service(model, search).ask("r", "设备甲是否完全没有其他方法？")
        self.assertEqual(state["status"], "answered")
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(model.calls[1][1]["proposed_missing"], ["controller"])

    def test_one_additional_round_merges_deduplicates_and_preserves_initial_context(self):
        extra = {**copy.deepcopy(DOC), "chunk_id": "c2"}
        class Search(Retriever):
            def search(self, query):
                self.calls.append(query)
                return [copy.deepcopy(DOC)] if len(self.calls) == 1 else [copy.deepcopy(DOC), extra]
        model = Model([understanding(), gap(), ASSESS, ANSWER])
        search = Search()
        result = unreviewed_service(model, search).ask("r", "设备甲问题")
        self.assertEqual(result["status"], "answered")
        self.assertEqual([d["chunk_id"] for d in result["initial_evidence"]], ["c1"])
        self.assertEqual([d["chunk_id"] for d in result["evidence"]], ["c1", "c2"])
        self.assertEqual(result["retrieval_rounds"], 1)
        self.assertEqual(search.calls, ["设备甲怎样查看剩余电量？", "设备甲第二项的要求"])
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(model.calls[2][1]["retrieval_rounds_remaining"], 0)

    def test_exhausted_retrieval_does_not_loop_or_claim_complete_answer(self):
        model = Model([understanding(), gap(), gap(), ANSWER])
        search = Retriever()
        result = unreviewed_service(model, search).ask("r", "设备甲问题")
        self.assertEqual(result["status"], "partial")
        self.assertIn("其余问题尚未核实", result["response"])
        self.assertEqual(len(search.calls), 2)
        self.assertEqual(len(model.calls), 4)

    def test_unresolved_empty_evidence_fails_to_insufficient(self):
        missing = {**gap(), "evidence_ids": [], "answerable_parts": []}
        model = Model([understanding(), missing, missing])
        result = unreviewed_service(model, Retriever()).ask("r", "设备甲问题")
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(len(model.calls), 3)

    def test_assessment_clarification_preserves_pending_and_original_question(self):
        ask = {"decision": "clarify", "evidence_ids": [], "reason": "症状不够具体",
               "missing": ["stage"], "clarification": "异常发生在什么阶段？",
               "boundary": "unknown_fault"}
        model = Model([understanding("设备甲出现异常"), ask,
                       understanding("设备甲在启动阶段异常"), ASSESS, ANSWER])
        service = unreviewed_service(model, Retriever())
        first = service.ask("r", "设备甲出现异常")
        self.assertEqual(first["status"], "clarify")
        self.assertTrue(first["pending"])
        self.assertIn("不能判断", first["response"])
        second = service.ask("r", "启动阶段")
        self.assertEqual(second["original_question"], "设备甲出现异常")
        self.assertTrue(model.calls[2][1]["pending"])
        self.assertEqual(second["status"], "answered")

    def test_specific_information_boundaries_do_not_generate_unsupported_facts(self):
        for boundary, expected in (("future", "公告"), ("service", "官方检测"),
                                   ("device_history", "没有接入设备历史查询服务")):
            with self.subTest(boundary=boundary):
                decision = {"decision": "insufficient", "evidence_ids": [],
                            "reason": "产品资料不能证明个案", "boundary": boundary}
                model = Model([understanding(), decision])
                result = unreviewed_service(model, Retriever()).ask("r", "设备甲问题")
                self.assertIn(expected, result["response"])
                self.assertNotIn("请补充相关型号", result["response"])
                self.assertFalse(result["pending"])
                self.assertEqual(result["citations"], [])
                self.assertEqual(len(model.calls), 2)

    def test_invalid_route_and_unbounded_query_list_rejected(self):
        for decision in ({**gap(), "retrieval_queries": ["a", "b", "c"]},
                         {**gap(), "retrieval_queries": []},
                         {"decision": "clarify", "evidence_ids": [], "reason": "缺失"},
                         {**ASSESS, "boundary": "device_history"}):
            with self.subTest(decision=decision):
                state = unreviewed_service(Model([understanding(), decision]), Retriever()).ask("r", "问题")
                self.assertEqual(state["status"], "error")

    def test_boundary_and_inference_are_not_misrepresented_as_official_diagnosis(self):
        decision = {**ASSESS, "boundary": "unknown_fault", "answerable_parts": ["按钮操作"]}
        answer = copy.deepcopy(ANSWER)
        answer["claims"][0]["kind"] = "advice"
        model = Model([understanding(), decision, answer])
        state = unreviewed_service(model, Retriever()).ask("r", "是不是坏了？")
        self.assertTrue(state["response"].startswith("仅凭当前描述"))
        self.assertIn("操作建议：", state["response"])
        self.assertEqual(len(state["citations"]), 1)
        self.assertNotIn("answerable_parts", model.calls[-1][1])
        self.assertNotIn("assessment_reason", model.calls[-1][1])

    def test_new_turn_resets_retry_budget_and_boundary(self):
        model = Model([understanding(), gap(), ASSESS, ANSWER,
                       understanding(), ASSESS, ANSWER])
        service = unreviewed_service(model, Retriever())
        service.ask("r", "第一问")
        result = service.ask("r", "第二问")
        self.assertEqual(result["retrieval_rounds"], 0)
        self.assertEqual(len(result["retrieval_log"]), 1)
        self.assertEqual(result["boundary"], "none")
        self.assertFalse(result["partial_answer"])

    def test_release_context_cannot_cross_source_or_entry_date(self):
        catalog = Catalog.__new__(Catalog)
        catalog.rows = [
            {"chunk_id": "a", "source_id": "s", "release_dates": ["2030-01-01"], "text": "更新内容"},
            {"chunk_id": "b", "source_id": "s", "release_dates": ["2030-01-01"], "text": "注意：使用条件"},
            {"chunk_id": "c", "source_id": "s", "release_dates": ["2030-02-01"], "text": "注意：其他日期"},
            {"chunk_id": "d", "source_id": "other", "release_dates": ["2030-01-01"], "text": "注意：其他文档"}]
        catalog.by_id = {r["chunk_id"]: r for r in catalog.rows}
        catalog.document = lambda ident: catalog.by_id[ident]
        related = catalog.related_release_context([catalog.rows[0]], 2)
        self.assertEqual([d["chunk_id"] for d in related], ["b"])
        self.assertEqual(catalog.related_release_context([catalog.rows[0]], 0), [])
        self.assertEqual(catalog.related_release_context([DOC], 2), [])


if __name__ == "__main__":
    unittest.main()

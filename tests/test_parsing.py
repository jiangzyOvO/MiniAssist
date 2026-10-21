"""Small regression tests for provenance, DOM boundaries, and section inheritance."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from miniassist.parsing.provenance import PageMapper
from miniassist.parsing.pdf_parser import SectionState, is_toc, baseline_lines, column_order, classify_heading
from miniassist.parsing.html_parser import parse_html
from miniassist.parsing.tables import grid_boundaries, grid_span


class ProvenanceTests(unittest.TestCase):
    def test_repeated_table_header_can_follow_a_standalone_heading(self):
        mapper = PageMapper("S3", 4, "功能适配性说明\n飞行器 功能适配性说明\n")
        heading = mapper.map("功能适配性说明")
        cell = mapper.map("功能适配性说明")
        self.assertEqual(cell["matched_chars"], cell["total_chars"])
        self.assertLess(heading["source_spans"][0]["start_char"], cell["source_spans"][0]["start_char"])

    def test_figure_numeral_does_not_match_charger_voltage(self):
        mapped = PageMapper("S1", 1, "充电限制电压最大12 V。").map("1")
        self.assertEqual(mapped["matched_chars"], 0)

    def test_heading_prefers_standalone_line_over_body_prefix(self):
        raw = "焦点跟随包含三种功能。\n焦点跟随\n"
        mapped = PageMapper("S1", 1, raw).map("焦点跟随")
        self.assertEqual(mapped["source_spans"][0]["start_char"], raw.rfind("焦点跟随"))

    def test_whitespace_change_preserves_original_offsets(self):
        raw = "前文\n充电温度 5℃至\n40℃。\n后文"
        out = "充电温度5℃至40℃。"
        mapped = PageMapper("S1", 1, raw).map(out)
        self.assertEqual(mapped["matched_chars"], len(out))
        for m in mapped["text_map"]:
            self.assertEqual(out[m["text_start"]:m["text_end"]],
                             raw[m["source_start"]:m["source_end"]])

    def test_repeated_lines_choose_distinct_occurrences(self):
        mapper = PageMapper("S1", 1, "同样的提示\n同样的提示")
        first = mapper.map("同样的提示")
        second = mapper.map("同样的提示")
        self.assertLess(first["source_spans"][0]["start_char"],
                        second["source_spans"][0]["start_char"])

    def test_partial_quote_cannot_claim_whole_paragraph(self):
        raw = "开机状态下不支持充电。其他条件仍需核对。"
        mapped = PageMapper("S1", 1, raw).map("开机状态下不支持充电。")
        self.assertEqual(mapped["source_spans"][0]["end_char"], raw.index("其他"))

    def test_changed_fact_is_not_given_false_provenance(self):
        raw = "允许充电温度范围为 5℃至 40℃。"
        mapped = PageMapper("S1", 1, raw).map("允许充电温度范围为 5℃至 80℃。")
        self.assertLess(mapped["matched_chars"], mapped["total_chars"])
        self.assertTrue(all("8" not in s["exact_text"] for s in mapped["source_spans"]))

    def test_reordered_cells_map_to_disjoint_source_ranges(self):
        mapper = PageMapper("S3", 2, "型号甲：版本一。\n型号乙：版本二。")
        second = mapper.map("型号乙：版本二。")
        first = mapper.map("型号甲：版本一。")
        self.assertEqual(second["matched_chars"], second["total_chars"])
        self.assertEqual(first["matched_chars"], first["total_chars"])
        self.assertFalse(mapper.unclaimed())


class StructureTests(unittest.TestCase):
    def test_merged_cells_use_grid_boundaries_not_enlarged_row_boxes(self):
        cells = [(0,0,100,10), (0,10,40,30), (40,10,100,20), (40,20,100,30)]
        xs, ys = grid_boundaries(cells, 0), grid_boundaries(cells, 1)
        self.assertEqual(grid_span(0,100,xs), 2)
        self.assertEqual(grid_span(0,40,xs), 1)
        self.assertEqual(grid_span(10,30,ys), 2)
        self.assertEqual(grid_span(10,20,ys), 1)

    def test_long_body_line_is_not_a_font_heading(self):
        text = "热情况。DJI Mini 4 Pro在待飞状态下长时间静置，可能会导致机身温度持续上升，但"
        line = {"text":text, "chars":[{"text":c, "size":8} for c in text]}
        self.assertIsNone(classify_heading(line, 103, [], 7, "manual"))

    def test_diagram_power_label_is_not_a_heading(self):
        line = {"text":"PD 30W", "chars":[{"text":c, "size":9} for c in "PD 30W"]}
        self.assertIsNone(classify_heading(line, 62, [], 8, "manual"))

    def test_font_fallback_respects_bookmark_parent_depth(self):
        state = SectionState("S3")
        state.heading("对频", 2, "bookmark", path=["适配说明", "对频"], font_size=14)
        state.heading("通过按键对频", 3, "layout_font", level=2, font_size=11)
        self.assertEqual(state.path, ["适配说明", "对频", "通过按键对频"])

    def test_two_columns_are_not_interleaved(self):
        lines = []
        for row in range(3):
            chars = []
            for start, text in [(10, f"左栏第{row}行"), (120, f"右栏第{row}行")]:
                chars.extend({"text": c, "x0":start+i*6, "x1":start+(i+1)*6,
                              "top":row*12, "bottom":row*12+8}
                             for i,c in enumerate(text))
            lines.append({"text":"", "chars":chars, "top":row*12, "bottom":row*12+8, "x0":10, "x1":150})
        result = sorted(column_order(lines, 200), key=lambda l:l["reading_order"])
        self.assertEqual([x["text"] for x in result],
                         ["左栏第0行", "左栏第1行", "左栏第2行", "右栏第0行", "右栏第1行", "右栏第2行"])

    def test_mixed_font_boxes_use_common_baseline(self):
        class FakePage:
            height = 800
            chars = [
                {"text":"日期：", "matrix":(10,0,0,10,20,700), "x0":20, "x1":50, "top":100, "bottom":110},
                {"text":"2025", "matrix":(10,0,0,10,65,699.8), "x0":65, "x1":90, "top":92, "bottom":102},
                {"text":"下一行", "matrix":(10,0,0,10,20,680), "x0":20, "x1":50, "top":120, "bottom":130},
            ]
        lines = baseline_lines(FakePage())
        self.assertEqual([x["text"] for x in lines], ["日期： 2025", "下一行"])

    def test_cross_page_section_keeps_parent(self):
        state = SectionState("S1")
        state.heading("使用充电管家", 63, "bookmark", path=["智能飞行电池", "充电", "使用充电管家"])
        state.heading("充电步骤", 64, "layout_font", level=5)
        self.assertEqual(state.path[-2:], ["使用充电管家", "充电步骤"])

    def test_next_release_does_not_inherit_previous_release(self):
        state = SectionState("S2")
        state.heading("2025-11-05", 1, "release_date", level=1)
        state.heading("更新", 1, "release_label", level=2)
        state.heading("2025-05-13", 2, "release_date", level=1)
        self.assertEqual(state.path, ["2025-05-13"])

    def test_toc_detection_not_every_numbered_list(self):
        self.assertTrue(is_toc([{"text": f"章节{i} {i}"} for i in range(1, 8)]))
        self.assertFalse(is_toc([{"text": f"{i}. 请按步骤操作。"} for i in range(1, 8)]))

    def test_html_keeps_hidden_answer_and_footnotes(self):
        markup = """<html><body><nav>不应解析</nav>
        <div><div class="style__question___x">如何使用？</div>
        <div style="display:none" class="style__answer___x">先充电。<br>再开机。<sup>注意温度。</sup></div></div>
        <div class="style__specs-group-item___x"><h4 class="style__specs-group-title___x">电池</h4>
        <div class="style__specs-detail___x"><strong class="style__specs-detail-key___x">温度</strong>
        <span class="style__specs-detail-value___x">5℃至40℃</span></div></div></body></html>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.html"
            path.write_text(markup)
            doc = parse_html(path, {"source_id":"S5", "sha256":"test", "version":"test"},
                             "如何使用？\n先充电。\n再开机。注意温度。\n温度\n5℃至40℃")
        self.assertEqual(doc["dom_counts"], {"faq":1, "specification":1})
        self.assertIn("注意温度", doc["blocks"][0]["text"])
        self.assertNotIn("不应解析", str(doc["blocks"]))


if __name__ == "__main__":
    unittest.main()

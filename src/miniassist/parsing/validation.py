"""Independent structural checks and frozen-evidence preservation checks.

Evidence is used only here, never in the parsing pipeline.
"""
from collections import Counter
import json
from pathlib import Path
import re

from .models import sha256
from .pipeline import read_canonical
from .provenance import compact


def validate(root, parsed):
    root, parsed = Path(root), Path(parsed)
    manifest = json.loads((parsed/"manifest.json").read_text())
    errors, results, documents, coverage = [], [], {}, {}
    for entry in manifest["documents"]:
        sid = entry["source_id"]
        path = parsed/entry["path"]
        if sha256(path.read_bytes()) != entry["sha256"]:
            errors.append(f"{sid}: parsed artifact hash mismatch")
        doc = json.loads(path.read_text())
        documents[sid] = doc
        canonical = read_canonical(root, sid)
        source = doc["source"]
        if sha256((root/"data/sources/raw"/source["file"]).read_bytes()) != source["sha256"]:
            errors.append(f"{sid}: source hash mismatch")
        if sha256((root/doc["canonical_extraction"]["path"]).read_bytes()) != doc["canonical_extraction"]["sha256"]:
            errors.append(f"{sid}: canonical extraction hash mismatch")
        active = {p: set() for p in canonical}
        excluded = {p: set() for p in canonical}
        sections = {s["section_id"] for s in doc["sections"]}
        duplicates = Counter()
        partial, unmatched_output = 0, 0
        for b in doc["blocks"]:
            if any(x not in sections for x in b["section_ids"]):
                errors.append(f"{sid}: broken section reference {b['block_id']}")
            text = canonical[b["page"]]
            for m in b["text_map"]:
                emitted = b["text"][m["text_start"]:m["text_end"]]
                original = text[m["source_start"]:m["source_end"]]
                if not emitted or emitted != original:
                    errors.append(f"{sid}: invalid text map {b['block_id']}")
            mapped_chars = sum(m["text_end"]-m["text_start"] for m in b["text_map"])
            nonspace = len(compact(b["text"]))
            unmatched_output += nonspace-mapped_chars
            partial += nonspace != mapped_chars
            for span in b["source_spans"]:
                p, a, z = span["page"], span["start_char"], span["end_char"]
                original = canonical[p]
                if not (0 <= a < z <= len(original)) or original[a:z] != span["exact_text"] or sha256(original.encode()) != span["page_text_sha256"]:
                    errors.append(f"{sid}: invalid source span {b['block_id']}")
                for i in range(a, z):
                    if not original[i].isspace():
                        active[p].add(i)
                        duplicates[(p, i)] += 1
            if b["table"]:
                t = b["table"]
                coordinates = {(c["row"], c["column"]) for c in t["cells"]}
                if len(coordinates) != t["row_count"]*t["column_count"]:
                    errors.append(f"{sid}: incomplete table matrix")
                for c in t["cells"]:
                    if c["role"] == "covered_by_merged_cell" and tuple(c["covered_by"]) not in coordinates:
                        errors.append(f"{sid}: broken merged-cell relation")
                    if c["role"] == "cell" and not (
                        1 <= c["row_span"] <= t["row_count"]-c["row"]
                        and 1 <= c["col_span"] <= t["column_count"]-c["column"]
                    ):
                        errors.append(f"{sid}: invalid table span {b['block_id']}")
                    if c["role"] == "covered_by_merged_cell":
                        parent = next(p for p in t["cells"] if [p["row"], p["column"]] == c["covered_by"])
                        if not (parent["row"] <= c["row"] < parent["row"]+parent["row_span"]
                                and parent["column"] <= c["column"] < parent["column"]+parent["col_span"]):
                            errors.append(f"{sid}: merged slot outside parent")
        for ex in doc["excluded"]:
            for span in ex["source_spans"]:
                p, a, z = span["page"], span["start_char"], span["end_char"]
                if canonical[p][a:z] != span["exact_text"]:
                    errors.append(f"{sid}: invalid excluded span")
                excluded[p].update(i for i in range(a, z) if not canonical[p][i].isspace())
        total = sum(len(compact(t)) for t in canonical.values())
        kept = sum(map(len, active.values()))
        removed = sum(len(excluded[p]-active[p]) for p in canonical)
        unresolved = total-kept-removed
        if any(v > 1 for v in duplicates.values()):
            errors.append(f"{sid}: duplicate source character ownership")
        coverage[sid] = active
        results.append({
            "source_id": sid, "pages": len(doc["pages"]), "blocks": len(doc["blocks"]),
            "block_counts": dict(Counter(b["block_type"] for b in doc["blocks"])),
            "canonical_nonspace_chars": total, "mapped_body_chars": kept,
            "explicitly_excluded_chars": removed,
            "unresolved_or_out_of_scope_chars": unresolved,
            "body_char_coverage_after_explicit_exclusions": round(kept/max(1,total-removed), 6),
            "duplicate_mapped_chars": sum(v-1 for v in duplicates.values() if v>1),
            "partially_aligned_blocks": partial, "unmapped_output_chars": unmatched_output,
            "warning_counts": dict(Counter(w for b in doc["blocks"] for w in b["warnings"])),
            "dom_counts": doc.get("dom_counts"),
        })
    # Gold diagnostic covers source preservation only, NOT retrieval recall.
    evidence_path = root/"data/review/evidence_annotations_v1.0.json"
    evidence = json.loads(evidence_path.read_text())
    gold = []
    for record in evidence["records"]:
        for group in record["groups"]:
            for alt in group["alternatives"]:
                for kind in ("all_of_spans", "context_anchors"):
                    for a in alt[kind]:
                        sid, page = a["source_id"], a["pdf_page"]
                        if sid not in documents:
                            continue
                        canonical = read_canonical(root, sid)[page]
                        if sha256(canonical.encode()) != a["page_text_sha256"]:
                            errors.append(f"gold coordinate hash mismatch: {group['evidence_id']}")
                        required = {i for i in range(a["start_char"], a["end_char"]) if not canonical[i].isspace()}
                        found = required & coverage.get(sid, {}).get(page, set())
                        gold.append({
                            "evidence_id": group["evidence_id"], "role": group["role"], "kind": kind,
                            "source_id": sid, "page": page, "required_chars": len(required),
                            "preserved_chars": len(found), "complete": required == found,
                        })
                        if required != found:
                            errors.append(f"evidence not fully preserved: {group['evidence_id']} {kind} {sid} p{page}")
    checks = []
    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(name)
    if "S3" in documents:
        tables = [b for b in documents["S3"]["blocks"] if b["table"]]
        functions = [b for b in tables if b["page"] == 4 and b["table"]["column_count"] == 2]
        correct = False
        for b in functions:
            cells = b["table"]["cells"]
            for c in cells:
                if c["column"] == 0 and compact(c["text"]) == "DJIMini4Pro":
                    neighbor = next((x for x in cells if x["row"] == c["row"] and x["column"] == 1), None)
                    correct = neighbor and all(w in neighbor["text"] for w in ("竖拍", "智能跟随", "航点"))
        check("S3 功能表：Mini 4 Pro 与对应限制同一行", correct, "检查原表行列，而非仅全文出现关键词")
        versions = [b for b in tables if b["page"] == 2]
        correct = any("搭配DJIAir3：v01.00.0700" in compact(c["text"])
                      and "搭配DJIMini4Pro：v01.01.0300" in compact(c["text"])
                      for b in versions for c in b["table"]["cells"])
        check("S3 版本表：保留 RC-N2 两机型的不同版本", correct, "要求两条带型号的版本在同一个适配设备单元格")
    if "S1" in documents:
        order = [b for b in documents["S1"]["blocks"]
                 if b["page"] == 64 and "从高至低" in b["text"]]
        check("S1 跨页章节：充电步骤继承使用充电管家", bool(order) and all(
            any("使用充电管家" in x for x in b["section_path"]) for b in order),
            "第 63 页章节应延续到第 64 页步骤")
        steps = [int(m[1]) for b in documents["S1"]["blocks"] if b["page"] == 18
                 and b["block_type"] == "list_item"
                 if (m := re.match(r"^(\d+)\.", b["text"]))]
        check("S1 双栏阅读顺序：遥控器部件 1—15 顺序完整", steps == list(range(1,16)),
              "先读左栏再读右栏，不按相同高度交错拼接")
        charging = [b for b in documents["S1"]["blocks"]
                    if b["page"] == 62 and "开机状态下不支持充电" in b["text"]]
        check("S1 图示标签：PD 30W 不成为充电正文的父标题",
              bool(charging) and all("使用充电器" in b["section_path"]
                                     and all(compact(s) != "PD30W" for s in b["section_path"])
                                     for b in charging),
              "正文继承操作章节，不继承示意图功率标签")
        if manifest["parser_version"] != "0.1.0":
            blocks = documents["S1"]["blocks"]
            by_id = {b["block_id"]: b for b in blocks}
            specs = {b["page"]: b for b in blocks if b["table"]
                     and b["table"]["method"] == "shaded_key_value_layout"}
            check("S1 规格表：六页恢复为键值表且字符全部对齐",
                  set(specs) == set(range(97, 103)) and all(
                      not b["warnings"] for b in specs.values()), "原页 97—102；不把首个参数当作表头")
            def spec_rows(page):
                b = specs.get(page)
                if not b:
                    return []
                return [(r, [c["text"] for c in b["table"]["cells"]
                             if c["row"] == r["row"] and c["role"] == "cell"])
                        for r in b["table"]["rows"] if r["kind"] == "data"]
            check("S1 速度及悬停精度：条件与数值同一行",
                  any("海平面附近无风" in c[0] and "16 m/s" in c[1] for _, c in spec_rows(97))
                  and any("无风或微风环境" in c[0] and "±0.1" in c[1] for _, c in spec_rows(97)),
                  "不以全文关键词出现代替行关系检查")
            conditions = [("无干扰、无遮挡", "20 km", "[7]"),
                          ("有干扰、无遮挡", "1.5-4 km", "[8]"),
                          ("有干扰、有遮挡", "0-0.5 km", "[9]")]
            check("S1 图传距离：三类环境条件、数值与脚注对应",
                  all(any(condition in c[0] and value in c[1]
                          and any(by_id[x]["text"].startswith(marker) for x in r["footnote_block_ids"])
                          for r, c in spec_rows(100)) for condition, value, marker in conditions),
                  "脚注链接到第 103 页对应编号")
            check("S1 续航脚注：保留后续的差异说明段落",
                  any("最长飞行时间" in c[0] and "不同的外部环境" in "".join(
                      by_id[x]["text"] for x in r["footnote_block_ids"]) for r, c in spec_rows(97)),
                  "脚注 [4] 包含全部延续段落")
            check("S1 跨页规格：遥控器归属和页底参数完整",
                  any(c[0] == "协议" and "802.11" in c[1] and "RC 2" in str(r["category_path"])
                      for r, c in spec_rows(102))
                  and any(c[0] == "工作频段[6]" and "RC-N2" in str(r["category_path"])
                          for r, c in spec_rows(102))
                  and any(c[0] == "稳定系统" for _, c in spec_rows(98)),
                  "RC 2 与 RC-N2 不串设备；第 98 页最后一行不遗漏")
    if "S2" in documents:
        dates = {b["metadata"].get("release_date") for b in documents["S2"]["blocks"]}
        check("S2 发布记录：按日期分隔且无未归属正文", len(dates) == 10 and None not in dates,
              f"日期数 {len(dates)}")
    if "S5" in documents:
        blocks = documents["S5"]["blocks"]
        checks_to_run = [
            ("哪些情况会导致固件升级失败？", ["20%", "电源关闭", "网络", "其他应用"]),
            ("DJI Mini 4 Pro 可以使用 DJI Mini 3 Pro 的哪些配件？", ["249", "注意", "智能飞行电池"]),
        ]
        for title, words in checks_to_run:
            found = [b for b in blocks if b["metadata"].get("question") == title]
            check("S5 FAQ 完整配对："+title, len(found)==1 and all(w in found[0]["text"] for w in words),
                  "隐藏折叠答案和脚注均应保留")
    report = {
        "parser_version": manifest["parser_version"], "sources": results,
        "errors": errors, "structural_checks": checks,
        "evidence_spans": gold, "evidence_spans_complete": sum(x["complete"] for x in gold),
        "evidence_spans_total": len(gold),
        "status": "checks_passed_with_review_items" if not errors else "failed",
        "metric_warning": "These are parsing preservation checks, not retrieval recall or semantic accuracy.",
        "remaining_review": [
            "Ordinary tables keep candidate headers in parsed output; frozen corpus uses separately reviewed headers. Specification tables are key/value layouts without column headers.",
            "Images, vector-only indicators and multilingual diagrams remain visual dependencies.",
            "Unresolved text and partial alignments are reported separately; archive retention does not imply usable extraction.",
            "Section fallback and reading order require representative visual sampling, not just character coverage.",
        ],
    }
    return report


def render_report(report):
    out = [
        "# MiniAssist 文档解析验证报告\n\n",
        f"解析器版本：{report['parser_version']}。状态：{report['status']}。\n\n",
        "**本报告没有运行切分或检索，因此下面的数字不是召回率。**\n\n",
        "| 来源 | 页数 | 结构块数 | 正文映射字符 | 明确排除字符 | 未映射／范围外字符 | 重复映射字符 |\n",
        "| --- | --- | --- | --- | --- | --- | --- |\n",
    ]
    for s in report["sources"]:
        out.append(f"| {s['source_id']} | {s['pages'] or '网页'} | {s['blocks']} | {s['mapped_body_chars']} | {s['explicitly_excluded_chars']} | {s['unresolved_or_out_of_scope_chars']} | {s['duplicate_mapped_chars']} |\n")
    out.append("\n网页只解析 FAQ 和技术参数；网页范围外字符不能与 PDF 未解析字符直接比较。"
               "正文覆盖率检查的是冻结提取文本的保留情况，不保证 PDF 原始图像中所有信息都被提取。\n\n")
    out.append(f"## 已审核证据的原文保留\n\n{report['evidence_spans_complete']} / {report['evidence_spans_total']} 段完整保留"
               "（包含核心、补充证据及上下文定位）。这只检验原文范围，没有用问答来指导解析。\n\n")
    missing = [e for e in report["evidence_spans"] if not e["complete"]]
    for e in missing:
        out.append(f"- {e['evidence_id']}，{e['source_id']} 第 {e['page']} 页：{e['preserved_chars']}/{e['required_chars']} 字符。\n")
    out.append("\n## 针对结构的检查\n\n")
    for c in report["structural_checks"]:
        out.append(f"- {'通过' if c['passed'] else '失败'}：{c['name']}。{c['detail']}。\n")
    out.append("\n## 需要继续关注的解析边界\n\n")
    for s in report["sources"]:
        out.append(f"- {s['source_id']}：{s['partially_aligned_blocks']} 个块存在部分对齐，"
                   f"{s['unmapped_output_chars']} 个输出字符未获得原文映射；标记 {s['warning_counts']}。\n")
    out.append("\n工作版普通表格的表头仍为候选，规格键值表没有列标题。冻结语料的表头另按复核策略设置。"
               "复杂表头、图形灯语与图示依赖内容不能宣称全部自动理解。"
               "跨页章节会继承；跨页表格暂保留为独立表格，不自动合并。局部原文匹配只建立完全一致的字符映射，"
               "不能匹配的部分留待检查，不冒充证据。\n\n")
    if report["errors"]:
        out.append("## 失败项\n\n" + "\n".join("- "+e for e in report["errors"])+"\n")
    return "".join(out)

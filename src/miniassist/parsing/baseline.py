"""Create a versioned experiment input from parsed blocks and audited policy.

This is a corpus eligibility gate, not a chunker. QA is read only after selection
to report evidence preservation; it never changes inclusion decisions.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from importlib.metadata import version

from .models import sha256
from .pipeline import read_canonical
from .provenance import compact
from .validation import validate, render_report


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def select_blocks(documents, policy):
    selected, excluded = [], []
    decisions = {x["block_id"]: x for x in policy["table_reviews"]}
    seen_decisions = set()
    for sid, doc in documents.items():
        for original in doc["blocks"]:
            b = deepcopy(original)
            reasons = []
            if sid in policy["excluded_sources"]:
                reasons.append(policy["excluded_sources"][sid])
            if b["page"] in policy["excluded_pages"].get(sid, []):
                reasons.append("reviewed_incomplete_table_page")
            for region in policy["excluded_regions"]:
                if sid == region["source_id"] and b["page"] == region["page"] and b["bbox"]:
                    if b["bbox"][1] < region["bottom"] and b["bbox"][3] > region["top"]:
                        reasons.append(region["reason"])
            if b["warnings"]:
                reasons.extend(b["warnings"])
            if sum(m["text_end"]-m["text_start"] for m in b["text_map"]) != len(compact(b["text"])):
                reasons.append("incomplete_character_map")
            if b["block_type"] != "heading" and not b["table"] and re.fullmatch(
                r"[\d\sA-Za-z.%+°℃/(),:：×−—\-]+", b["text"]
            ):
                reasons.append("isolated_label_or_numeric_fragment")
            if b["table"]:
                decision = decisions.get(b["block_id"])
                if not decision:
                    reasons.append("table_semantics_not_reviewed")
                else:
                    if sha256(b["text"].encode()) != decision["text_sha256"]:
                        raise ValueError(f"Stale table review: {b['block_id']}")
                    seen_decisions.add(b["block_id"])
                    b["table"]["header_rows"] = decision["header_rows"]
                    b["table"]["header_status"] = decision["header_status"]
                    b["table"]["review"] = decision
                if any(c["role"] == "unresolved_slot" for c in b["table"]["cells"]):
                    reasons.append("unresolved_table_slot")
                if any(r.get("unresolved_footnote_markers") for r in b["table"].get("rows", [])):
                    reasons.append("unresolved_footnote")
                b["metadata"]["required_context_block_ids"] = sorted(set(
                    b["table"]["footnote_block_ids"] + (decision.get("context_block_ids", []) if decision else [])))
            if reasons:
                excluded.append({"source_id": sid, "block_id": b["block_id"],
                                 "page": b["page"], "block_type": b["block_type"],
                                 "reasons": sorted(set(reasons))})
            else:
                b["metadata"]["baseline_policy_version"] = policy["version"]
                selected.append(b)
    if seen_decisions != set(decisions):
        raise ValueError("Review refers to missing table blocks")
    # Never admit a table whose required note was quarantined.
    while True:
        ids = {b["block_id"] for b in selected}
        missing = [b for b in selected if any(x not in ids for x in
                   b["metadata"].get("required_context_block_ids", []))]
        if not missing:
            break
        for b in missing:
            selected.remove(b)
            excluded.append({"source_id": b["metadata"]["source_id"], "block_id": b["block_id"],
                             "page": b["page"], "block_type": b["block_type"],
                             "reasons": ["required_note_not_eligible"]})
    return selected, excluded


def evidence_retention(root, blocks):
    coverage = {}
    for b in blocks:
        for s in b["source_spans"]:
            coverage.setdefault((s["source_id"], s["page"]), set()).update(
                i for i in range(s["start_char"], s["end_char"])
                if not s["exact_text"][i-s["start_char"]].isspace())
    gold = json.loads((root/"data/review/evidence_annotations_v1.0.json").read_text())
    canonical = {}
    results = []
    for record in gold["records"]:
        for group in record["groups"]:
            for alt in group["alternatives"]:
                for kind in ("all_of_spans", "context_anchors"):
                    for s in alt[kind]:
                        sid, page = s["source_id"], s["pdf_page"]
                        if sid not in canonical:
                            canonical[sid] = read_canonical(root, sid)
                        text = canonical[sid][page]
                        required = {i for i in range(s["start_char"], s["end_char"]) if not text[i].isspace()}
                        found = required & coverage.get((sid, page), set())
                        results.append({"evidence_id": group["evidence_id"], "kind": kind,
                                        "source_id": sid, "page": page,
                                        "complete": found == required,
                                        "required_chars": len(required), "retained_chars": len(found)})
    return results


def build(root, parsed, policy_path, output):
    root, parsed, policy_path, output = map(Path, (root, parsed, policy_path, output))
    if output.exists():
        raise FileExistsError("Baseline already exists; choose a new version directory")
    policy = json.loads(policy_path.read_text())
    if sha256((parsed/"manifest.json").read_bytes()) != policy["parsed_manifest_sha256"]:
        raise ValueError("Policy was reviewed against a different parser run")
    gold_freeze = json.loads((root/"data/review/evidence_freeze_v1.0.json").read_text())
    for file, expected in gold_freeze["artifacts"].items():
        if sha256((root/"data/review"/file).read_bytes()) != expected:
            raise ValueError(f"Human-approved evidence changed: {file}")
    if not (parsed/"收尾复核记录.md").is_file():
        raise ValueError("Missing closeout review record")
    report = validate(root, parsed)
    if report["errors"]:
        raise ValueError(report["errors"])
    documents = {e["source_id"]: json.loads((parsed/e["path"]).read_text())
                 for e in json.loads((parsed/"manifest.json").read_text())["documents"]}
    blocks, excluded = select_blocks(documents, policy)
    gold = evidence_retention(root, blocks)
    output.mkdir(parents=True)
    (output/"corpus.jsonl").write_text("".join(json.dumps(b, ensure_ascii=False)+"\n" for b in blocks))
    dump(output/"exclusions.json", excluded)
    dump(output/"policy.json", policy)
    dump(output/"validation.json", report)
    (output/"解析验证报告.md").write_text(render_report(report))
    dump(output/"evidence_retention.json", gold)
    dump(output/"runtime.json", {"python": sys.version, "dependencies": {
        name: version(name) for name in ("pdfplumber", "pdfminer.six", "pypdf", "lxml")}})
    shutil.copytree(parsed, output/"parsed")
    # Keep executable code, review policy and dependency declaration with the data.
    for folder in ("src", "scripts", "tests", "config"):
        shutil.copytree(root/folder, output/"code"/folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    shutil.copy2(root/"pyproject.toml", output/"code/pyproject.toml")
    for file in ("收尾复核记录.md",):
        shutil.copy2(parsed/file, output/file)
    shutil.copy2(root/"README.md", output/"code/README.md")
    refs = {"data/sources/manifest.json", "data/review/evidence_annotations_v1.0.json",
            "data/review/evidence_freeze_v1.0.json"}
    refs.update("data/review/"+file for file in gold_freeze["artifacts"])
    for doc in documents.values():
        refs.add("data/sources/raw/"+doc["source"]["file"])
        refs.add(doc["canonical_extraction"]["path"])
    stats = []
    for sid in documents:
        stats.append({"source_id": sid, "parsed_blocks": len(documents[sid]["blocks"]),
                      "eligible_blocks": sum(b["metadata"]["source_id"] == sid for b in blocks),
                      "excluded_blocks": sum(b["source_id"] == sid for b in excluded)})
    total, kept = len(gold), sum(g["complete"] for g in gold)
    summary = [
        "# 解析基线 v1.0\n\n",
        "本版本冻结的是结构块与实验纳入范围，不是 chunk，也未运行检索。\n\n",
        "三种切分器必须读取同一份 `corpus.jsonl`；不得各自从原始 PDF 或整个 parsed 目录重新取数据。\n\n",
        "| 来源 | 解析块 | 纳入块 | 排除块 |\n| --- | --- | --- | --- |\n",
    ]
    for s in stats:
        summary.append(f"| {s['source_id']} | {s['parsed_blocks']} | {s['eligible_blocks']} | {s['excluded_blocks']} |\n")
    summary.extend([
        f"\n纳入后证据与上下文原文保留：**{kept}/{total}**。完整解析结果为 "
        f"{report['evidence_spans_complete']}/{report['evidence_spans_total']}。不是 Recall@K。\n\n",
        "纳入规则：完整字符映射、无未解决解析警告；表格必须经过记录在案的结构复核。"
        "S4 多语言图示整份暂不纳入；S1 图示区域、未解释灯语、局部未对齐文本、"
        "第 74/83 页的不完整挡位表页面等按 policy.json 排除。"
        "每块排除原因见 exclusions.json。已解析的内容没有删除，保存在 parsed/。\n\n",
        "表头复核是代理依据原 PDF 的视觉检查，不冒充用户人工签字。"
        "其余正文只经过自动校验和抽样检查，仍不保证全库语义正确。\n\n",
        "切分契约：保留 section_path、source/version/page 和逐字 text_map；"
        "表格需读取实际 header_rows、合并单元格、行 category_path；"
        "required_context_block_ids 是必须携带的脚注上下文。"
        "键值表没有列标题，不能把第一条参数作为表头。"
        "这些只是切分器输入约定，本轮尚未实现切分。\n\n",
        "冻结方式：代码副本、解析结果、策略和证据范围以 SHA-256 清单固定。"
        "原始资料仍保存在项目 data/sources，清单逐项校验；修改后校验会失败。"
        "目录不是操作系统只读或数字签名；工具拒绝覆盖已有版本。"
        "修复应创建下一版并让全部切分器重新评测。\n\n",
        "验证命令（项目根目录）：`python scripts/freeze_parsing.py --verify data/baselines/parsing_v1.0`\n",
    ])
    (output/"基线说明.md").write_text("".join(summary))
    manifest = {
        "baseline_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "parser_version": report["parser_version"], "scope": stats,
        "evidence_spans_complete": kept, "evidence_spans_total": total,
        "artifacts": {str(p.relative_to(output)): sha256(p.read_bytes())
                      for p in sorted(output.rglob("*")) if p.is_file()},
        "external_inputs": {p: sha256((root/p).read_bytes()) for p in sorted(refs)},
    }
    dump(output/"freeze_manifest.json", manifest)
    verify(root, output)
    return {k: manifest[k] for k in ("baseline_version", "scope", "evidence_spans_complete", "evidence_spans_total")}


def verify(root, output):
    root, output = Path(root), Path(output)
    manifest = json.loads((output/"freeze_manifest.json").read_text())
    errors = []
    for base, records in ((output, manifest["artifacts"]), (root, manifest["external_inputs"])):
        for relative, expected in records.items():
            path = base/relative
            if not path.is_file() or sha256(path.read_bytes()) != expected:
                errors.append(str(path))
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
              and p.name != "freeze_manifest.json" and "__pycache__" not in p.parts}
    if actual != set(manifest["artifacts"]):
        errors.append("Unexpected or missing frozen artifacts")
    if errors:
        raise ValueError("Frozen baseline changed: "+", ".join(errors))
    return {"status": "verified", "artifacts": len(manifest["artifacts"]),
            "external_inputs": len(manifest["external_inputs"])}

"""Reproducible parsing entry point; no access to QA annotations."""
import json
from pathlib import Path
from collections import Counter

from . import PARSER_VERSION
from .models import sha256
from .pdf_parser import parse_pdf
from .html_parser import parse_html

PROFILES = {
    "S1": ("manual", "manual_current_pages.json"),
    "S2": ("release_notes", "mini_4_pro_release_notes_cn.json"),
    "S3": ("compatibility", "goggles_3_compatibility_chs.json"),
    "S4": ("quick_start", "quick_start_pages.json"),
    "S5": ("support_html", "support.txt"),
}


def read_canonical(root, sid):
    file = root / "data/sources/extracted" / PROFILES[sid][1]
    if file.suffix == ".txt":
        return {None: file.read_text(encoding="utf-8")}
    return {r["pdf_page"]: r["text"] for r in json.loads(file.read_text())}


def preview(document):
    sid = document["source"]["source_id"]
    out = [f"# {sid}｜{document['source']['title']}\n\n",
           f"来源版本：{document['source']['version']}\n\n",
           "解析预览，不是切分结果。表格保留单元格内容及关系；候选表头、图示依赖等须结合验证报告阅读。\n\n"]
    last_path = None
    for b in document["blocks"]:
        if b["section_path"] != last_path:
            out.append("\n## " + " / ".join(b["section_path"] or ["未识别章节"]) + "\n\n")
            last_path = b["section_path"]
        out.append(f"<!-- block={b['block_id']} page={b['page']} type={b['block_type']} -->\n")
        if b["block_type"] == "table":
            t = b["table"]
            out.append(f"\n表格（{t['row_count']} 行 × {t['column_count']} 列；{t['method']}）\n\n")
            for ri in range(t["row_count"]):
                out.append(f"**第 {ri+1} 行：**\n\n")
                for c in t["cells"]:
                    if c["row"] != ri:
                        continue
                    if c["role"] == "covered_by_merged_cell":
                        value = f"由实际合并单元格 {c['covered_by']} 覆盖"
                    elif c["role"] == "unresolved_slot":
                        value = "空位：未推断"
                    else:
                        value = c["text"] or "（无文本，可能含图形）"
                    out.append(f"- 第 {c['column']+1} 列：{value.replace(chr(10), ' / ')}\n")
                out.append("\n")
        elif b["block_type"] != "heading":
            out.append(b["text"] + "\n\n")
        if b["warnings"]:
            out.append("解析标记：" + "、".join(b["warnings"]) + "\n\n")
    return "".join(out)


def run(root, output, source_ids=None):
    root, output = Path(root), Path(output)
    if any((p/"freeze_manifest.json").exists() for p in [output, *output.parents]):
        raise ValueError("Cannot overwrite a frozen baseline; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    sources = {s["source_id"]: s for s in json.loads((root/"data/sources/manifest.json").read_text())}
    selected = source_ids or list(PROFILES)
    manifest = {"parser_version": PARSER_VERSION, "documents": [],
                "excluded_sources": {"S0": "旧版手册归档，不混入当前正文", "S6": "下载目录用于发现资料，不作正文语料"}}
    existing = output/"manifest.json"
    if source_ids and existing.exists():
        previous = json.loads(existing.read_text())
        if previous["parser_version"] != PARSER_VERSION:
            raise ValueError("Do not mix parser versions in one output directory")
        manifest["documents"] = [d for d in previous["documents"] if d["source_id"] not in selected]
    for sid in selected:
        profile, canonical_name = PROFILES[sid]
        source = dict(sources[sid])
        raw = root / "data/sources/raw" / source["file"]
        if sha256(raw.read_bytes()) != source["sha256"]:
            raise ValueError(f"Source changed: {sid}")
        canonical = read_canonical(root, sid)
        print(f"Parsing {sid}: {profile}", flush=True)
        doc = (parse_html(raw, source, canonical[None]) if profile == "support_html"
               else parse_pdf(raw, source, canonical, profile))
        doc["parser_version"] = PARSER_VERSION
        doc["canonical_extraction"] = {
            "path": f"data/sources/extracted/{canonical_name}",
            "sha256": sha256((root/"data/sources/extracted"/canonical_name).read_bytes()),
        }
        path = output / f"{sid}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        (output/f"{sid}_预览.md").write_text(preview(doc), encoding="utf-8")
        manifest["documents"].append({
            "source_id": sid, "path": path.name, "sha256": sha256(path.read_bytes()),
            "block_counts": dict(Counter(b["block_type"] for b in doc["blocks"])),
            "pages": len(doc["pages"]), "warnings": len(doc["warnings"]),
        })
        print(f"Finished {sid}: {len(doc['blocks'])} blocks", flush=True)
    manifest["documents"].sort(key=lambda d: d["source_id"])
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest

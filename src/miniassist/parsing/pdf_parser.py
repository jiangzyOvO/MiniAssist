"""Parse PDF layout into headings, paragraphs, lists, and tables.

The frozen text is an independent provenance coordinate system, not the source
of section or table decisions. Outline + layout drive those decisions.
"""
from collections import Counter
from statistics import median
import re

import pdfplumber
from pypdf import PdfReader

from .models import Block, sha256, stable_id
from .provenance import PageMapper, compact
from .tables import contains, extract_tables, extract_shaded_key_value_table

LIST_START = re.compile(r"^(?:[•●▪*]|[0-9]{1,2}[.、]|[（(][0-9]+[）)])\s*")
DATE = re.compile(r"发布日期[：:]\s*(\d{4})[./-](\d{1,2})[./-](\d{1,2})")


def baseline_lines(page):
    """Mixed CJK/Latin fonts may have different top boxes on the same baseline."""
    groups = []
    for char in sorted(page.chars, key=lambda c: (page.height-c["matrix"][5], c["x0"])):
        baseline = page.height-char["matrix"][5]
        if groups and abs(baseline-groups[-1][0]) <= 2:
            groups[-1][1].append(char)
        else:
            groups.append((baseline, [char]))
    lines = []
    for _, chars in groups:
        chars.sort(key=lambda c: c["x0"])
        parts, previous = [], None
        for c in chars:
            if previous and c["x0"]-previous["x1"] > 2 and not (c["text"].isspace() or previous["text"].isspace()):
                parts.append(" ")
            parts.append(c["text"])
            previous = c
        text = "".join(parts).strip()
        if text:
            lines.append({"text": text, "chars": chars,
                          "x0": min(c["x0"] for c in chars), "x1": max(c["x1"] for c in chars),
                          "top": min(c["top"] for c in chars), "bottom": max(c["bottom"] for c in chars)})
    return lines


def column_order(lines, width):
    """Detect sustained two-column regions, without crossing full-width text.

    Tables have already been removed. A repeated central gutter is required;
    isolated wide gaps in figure labels don't establish a prose column region.
    """
    midpoint = width/2
    ordered = sorted(lines, key=lambda x: (x["top"], x["x0"]))
    runs, run = [], []
    for line in ordered:
        left = [c for c in line["chars"] if (c["x0"]+c["x1"])/2 < midpoint and not c["text"].isspace()]
        right = [c for c in line["chars"] if (c["x0"]+c["x1"])/2 >= midpoint and not c["text"].isspace()]
        gutter = min(c["x0"] for c in right)-max(c["x1"] for c in left) if left and right else None
        full_width = left and right and gutter < 10
        if full_width:
            if run:
                runs.append(run)
                run = []
        else:
            run.append((line, len(left)>=4 and len(right)>=4 and gutter is not None and gutter>=12))
    if run:
        runs.append(run)
    regions = []
    for run in runs:
        candidates = [l for l, yes in run if yes]
        if len(candidates) >= 3:
            regions.append((candidates[0]["top"], run[-1][0]["bottom"]))
    out = []
    for line in ordered:
        region = next((r for r in regions if r[0] <= line["top"] <= r[1]), None)
        if not region:
            out.append(dict(line, reading_order=(line["top"], 0, line["x0"]), column_id=None))
            continue
        for col in (0, 1):
            chars = [c for c in line["chars"] if int((c["x0"]+c["x1"])/2 >= midpoint) == col]
            if not any(not c["text"].isspace() for c in chars):
                continue
            chars.sort(key=lambda c: c["x0"])
            parts = []
            for i, c in enumerate(chars):
                if i and c["x0"]-chars[i-1]["x1"] > 2:
                    parts.append(" ")
                parts.append(c["text"])
            out.append(dict(line, text="".join(parts).strip(), chars=chars,
                            x0=min(c["x0"] for c in chars), x1=max(c["x1"] for c in chars),
                            top=min(c["top"] for c in chars), bottom=max(c["bottom"] for c in chars),
                            reading_order=(region[0], col+1, line["top"]), column_id=f"{region[0]}:{col}"))
    return out


def outlines(path):
    reader = PdfReader(path)
    result = []

    def visit(items, parents):
        previous = None
        for item in items:
            if isinstance(item, list):
                visit(item, parents + ([previous] if previous else []))
            else:
                previous = str(item.get("/Title", "")).strip()
                try:
                    page = reader.get_destination_page_number(item) + 1
                except (TypeError, ValueError, AttributeError):
                    continue
                result.append({"title": previous, "path": parents+[previous], "page": page})
    visit(reader.outline, [])
    return result


def is_toc(lines):
    texts = [l["text"].strip() for l in lines if l["text"].strip()]
    numbered = sum(bool(re.search(r"\s+\d{1,3}$", t)) for t in texts)
    return numbered >= 4 and numbered / max(1, len(texts)) > .55


def furniture(text, line, height):
    t = compact(text)
    if "版权所有" in t or "Copyright©" in t or "大疆创新的商标" in t:
        return "copyright"
    if line["top"] < 32 and ("用户手册" in t or "发布记录" in t):
        return "running_header"
    if re.fullmatch(r"\d{1,3}", t) and (line["top"] > height*.90 or line["top"] < height*.045):
        return "page_number"
    if line["top"] > height*.92 and re.match(r"https?://www\.dji\.com", t):
        return "footer_url"
    return None


class SectionState:
    def __init__(self, sid):
        self.sid = sid
        self.path, self.ids, self.sections = [], [], []
        self.levels = []
        self.fonts = []

    def heading(self, title, page, method, path=None, level=None, font_size=None):
        if path:
            # Bookmark ancestors are explicit even if their title is on a divider.
            common = 0
            while common < min(len(self.path), len(path)-1) and compact(self.path[common]) == compact(path[common]):
                common += 1
            self.path, self.ids, self.levels = self.path[:common], self.ids[:common], self.levels[:common]
            self.fonts = self.fonts[:common]
            for ancestor in path[common:-1]:
                self._add(ancestor, page, "bookmark_ancestor", len(self.levels)+1)
        else:
            if method == "layout_font" and font_size is not None and self.levels:
                # Bookmark depth and absolute font-size buckets are different
                # coordinate systems. Compare with observed parent sizes.
                depth = len(self.levels)
                while depth and self.fonts[depth-1] is not None and self.fonts[depth-1] <= font_size+.1:
                    depth -= 1
                level = self.levels[depth-1]+1 if depth else 1
        if not path and level is not None:
            while self.levels and self.levels[-1] >= level:
                self.path.pop()
                self.ids.pop()
                self.levels.pop()
                self.fonts.pop()
        self._add(title, page, method, level or len(self.path)+1, font_size)

    def _add(self, title, page, method, level, font_size=None):
        ident = stable_id(self.sid, "section", len(self.sections), page, title)
        self.sections.append({
            "section_id": ident, "title": title, "parent_id": self.ids[-1] if self.ids else None,
            "start_page": page, "method": method,
            "font_size": font_size,
            "confidence": "high" if method.startswith("bookmark") or method == "release_date" else "heuristic",
        })
        self.path.append(title)
        self.ids.append(ident)
        self.levels.append(level)
        self.fonts.append(font_size)


def classify_heading(line, page_num, bookmarks, body_size, profile):
    t = line["text"].strip()
    sizes = [c["size"] for c in line["chars"] if not c["text"].isspace()]
    size = round(median(sizes), 1) if sizes else body_size
    matching = [b for b in bookmarks if b["page"] == page_num and compact(b["title"]) == compact(t)]
    if matching:
        return {"method": "bookmark", "path": matching[0]["path"], "font_size": size}
    if profile == "release_notes" and DATE.search(t):
        return {"method": "release_date", "level": 1}
    if profile == "quick_start":
        return None
    if re.fullmatch(r"[A-Za-z0-9 /+%().°℃-]+", t) and not re.match(r"DJI\s+RC", t):
        # Power labels, dimensions and diagram numbers are not section titles.
        # Explicit PDF bookmarks above remain authoritative.
        return None
    if LIST_START.match(t) or len(t) > 30 or re.search(r"[。；，：:！？]$", t):
        return None
    # Small bold section labels are common in the Chinese manual.
    if size >= body_size + .45:
        if size >= 14:
            level = 1
        elif size >= 10.8:
            level = 2
        elif size >= 9.8:
            level = 3
        elif size >= 8.8:
            level = 4
        else:
            level = 5
        return {"method": "layout_font", "level": level, "font_size": size}
    if profile == "release_notes" and t in ("本次更新", "注意：", "注意:"):
        return {"method": "release_label", "level": 2}
    return None


def parse_pdf(path, source, canonical_pages, profile):
    sid = source["source_id"]
    bookmarks = outlines(path)
    spec = next((b for b in bookmarks if b["title"] == "规格参数"), None)
    spec_end = min((b["page"] for b in bookmarks if spec and b["page"] > spec["page"]
                    and len(b["path"]) <= len(spec["path"])), default=10**9)
    state = SectionState(sid)
    blocks, exclusions, page_info, warnings = [], [], [], []
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != len(canonical_pages):
            raise ValueError(f"{sid}: frozen page count mismatch")
        for page in pdf.pages:
            n = page.page_number
            mapper = PageMapper(sid, n, canonical_pages[n])
            raw_lines = baseline_lines(page) if profile == "release_notes" else page.extract_text_lines(return_chars=True)
            tables, rejected = extract_tables(page) if profile != "quick_start" else ([], [])
            if profile == "manual" and spec and spec["page"] <= n < spec_end:
                key_value = extract_shaded_key_value_table(page)
                if key_value:
                    tables = [key_value]
            toc = is_toc(raw_lines)
            if toc:
                tables = []
            sizes = Counter(round(c["size"], 1) for c in page.chars if not c["text"].isspace())
            body_size = sizes.most_common(1)[0][0] if sizes else 8
            events = []
            for line in raw_lines:
                reason = furniture(line["text"], line, page.height)
                if toc and not reason:
                    reason = "table_of_contents"
                if reason:
                    mapped = mapper.map(line["text"])
                    exclusions.append({"page": n, "reason": reason, "text": line["text"],
                                       "source_spans": mapped["source_spans"]})
                    continue
                remaining = [c for c in line["chars"] if not any(contains(t["bbox"], c) for t in tables)]
                if len(remaining) != len(line["chars"]):
                    if not remaining:
                        continue
                    # Characters outside tables are retained, with explicit whitespace.
                    line = dict(line, text="".join(c["text"] for c in remaining), chars=remaining,
                                x0=min(c["x0"] for c in remaining), x1=max(c["x1"] for c in remaining),
                                top=min(c["top"] for c in remaining), bottom=max(c["bottom"] for c in remaining))
                events.append(("line", line["top"], line))
            adjusted = column_order([e[2] for e in events], page.width) if profile != "release_notes" else [e[2] for e in events]
            events = [("line", l["top"], l) for l in adjusted]
            events.extend(("table", t["bbox"][1], t) for t in tables)
            events.sort(key=lambda e: e[2].get("reading_order", (e[1], 0, e[2].get("x0", 0))))
            pending = []
            pending_kind = "paragraph"
            date_value = None
            if profile == "release_notes":
                match = DATE.search(canonical_pages[n])
                if match:
                    date_value = "-".join([match[1], match[2].zfill(2), match[3].zfill(2)])

            def emit(text, kind, bbox, mapped, metadata=None, table=None):
                ident = stable_id(sid, source["sha256"], n, kind, len(blocks), text)
                item = Block(
                    ident, kind, n, text, list(state.path), list(state.ids), list(bbox) if bbox else None,
                    mapped["source_spans"], mapped["text_map"],
                    metadata={"source_id": sid, "source_version": source["version"],
                              "source_sha256": source["sha256"], "release_date": date_value,
                              **(metadata or {})},
                    table=table,
                )
                if mapped["matched_chars"] < mapped["total_chars"]:
                    item.warnings.append("partial_source_alignment")
                if profile == "quick_start":
                    item.warnings.append("multilingual_diagram_dependent")
                blocks.append(item.to_dict())

            def flush():
                nonlocal pending
                if not pending:
                    return
                text = "\n".join(l["text"] for l in pending)
                box = [min(l["x0"] for l in pending), pending[0]["top"],
                       max(l["x1"] for l in pending), pending[-1]["bottom"]]
                emit(text, pending_kind, box, mapper.map(text),
                     {"column_id": pending[0].get("column_id")})
                pending = []

            for kind, _, obj in events:
                if kind == "table":
                    flush()
                    cells = obj["cells"]
                    text_parts, combined_spans, combined_maps = [], [], []
                    cursor = 0
                    for c in cells:
                        if c["role"] != "cell":
                            continue
                        mapped = mapper.map(c["text"])
                        c.update(mapped)
                        c["cell_id"] = stable_id(sid, n, obj["bbox"], c["row"], c["column"])
                        if text_parts:
                            text_parts.append("\n")
                            cursor += 1
                        c["text_start"] = cursor
                        c["text_end"] = cursor + len(c["text"])
                        text_parts.append(c["text"])
                        combined_spans.extend(mapped["source_spans"])
                        combined_maps.extend({**m, "text_start": m["text_start"]+cursor,
                                              "text_end": m["text_end"]+cursor} for m in mapped["text_map"])
                        cursor += len(c["text"])
                    text = "".join(text_parts)
                    mapped = {"source_spans": combined_spans, "text_map": combined_maps,
                              "matched_chars": sum(c.get("matched_chars", 0) for c in cells),
                              "total_chars": sum(c.get("total_chars", 0) for c in cells)}
                    emit(text, "table", obj["bbox"], mapped, table=obj)
                    if obj["visual_dependency"]:
                        blocks[-1]["warnings"].append("table_contains_nontext_or_empty_cells")
                    continue
                heading = classify_heading(obj, n, bookmarks, body_size, profile)
                if heading:
                    flush()
                    state.heading(obj["text"], n, **heading)
                    emit(obj["text"], "heading", [obj[k] for k in ("x0", "top", "x1", "bottom")],
                         mapper.map(obj["text"]), {"heading_method": heading["method"]})
                    continue
                new_list = bool(LIST_START.match(obj["text"]))
                gap = obj["top"] - pending[-1]["bottom"] if pending else 0
                column_changed = pending and obj.get("column_id") != pending[-1].get("column_id")
                # Don't merge a new numbered/bullet step into its predecessor.
                paragraph_end = pending and re.search(r"[。！？]$", pending[-1]["text"].strip())
                if pending and (new_list or gap > body_size*1.3 or paragraph_end or column_changed):
                    flush()
                if not pending:
                    pending_kind = "list_item" if new_list else "paragraph"
                pending.append(obj)
            flush()
            # Preserve unmatched frozen text separately: archive coverage isn't usable coverage.
            unclaimed = mapper.unclaimed()
            unresolved = [{
                "start_char": a, "end_char": b, "exact_text": canonical_pages[n][a:b],
            } for a, b in unclaimed]
            total = len(mapper.positions)
            page_info.append({
                "page": n, "width": page.width, "height": page.height,
                "canonical_text_sha256": mapper.hash, "canonical_nonspace_chars": total,
                "is_toc": toc, "detected_tables": len(tables), "rejected_grids": rejected,
                "image_count": len(page.images), "vector_edges": len(page.edges),
                "unresolved_source_ranges": unresolved,
                "unresolved_nonspace_chars": sum(len(x["exact_text"]) for x in unresolved),
            })
            if unresolved:
                warnings.append({"page": n, "kind": "unresolved_source_characters",
                                 "count": sum(len(x["exact_text"]) for x in unresolved)})
            page.close()
    # Associate note candidates by geometry; never claim all nearby text applies.
    for i, b in enumerate(blocks):
        if b["block_type"] != "table":
            continue
        for following in blocks[i+1:]:
            if following["page"] != b["page"] or following["block_type"] in ("heading", "table"):
                break
            if following["bbox"][1] - b["bbox"][3] > 40:
                break
            if re.match(r"^[•*注]", following["text"]):
                b["table"]["footnote_block_ids"].append(following["block_id"])
    # Numbered notes are linked only within the same specification section.
    # Page-local asterisks remain inside their value cells.
    category, owner = None, None
    notes = {}
    active_note, note_page = None, None
    for b in blocks:
        if "规格参数" in b["section_path"] and b["block_type"] != "table":
            match = re.match(r"^\[(\d+)\]\s", b["text"])
            if match:
                if match[1] in notes:
                    raise ValueError(f"Ambiguous specification footnote: {match[1]}")
                active_note, note_page = match[1], b["page"]
                notes[active_note] = [b["block_id"]]
            elif active_note and b["page"] == note_page and b["block_type"] == "paragraph":
                notes[active_note].append(b["block_id"])
            else:
                active_note = None
        else:
            active_note = None
    for b in blocks:
        t = b["table"]
        if not t or t["method"] != "shaded_key_value_layout":
            continue
        for row in t["rows"]:
            values = [c for c in t["cells"] if c["row"] == row["row"] and c["role"] == "cell"]
            if row["kind"] == "category":
                category = values[0]["text"]
                if re.match(r"DJI\s+RC", category):
                    owner = category
            row["category"] = category
            row["category_path"] = [owner, category] if owner and category != owner else [category]
            markers = set(re.findall(r"\[(\d+)\]", " ".join(c["text"] for c in values)))
            row["footnote_block_ids"] = [ident for m in sorted(markers) for ident in notes.get(m, [])]
            row["unresolved_footnote_markers"] = [m for m in sorted(markers) if m not in notes]
        t["footnote_block_ids"] = sorted({x for r in t["rows"] for x in r["footnote_block_ids"]})
        t["footnote_status"] = "numbered_markers_linked" if not any(
            r["unresolved_footnote_markers"] for r in t["rows"]) else "unresolved_numbered_markers"
    return {
        "source": source, "profile": profile, "outline": bookmarks,
        "sections": state.sections, "blocks": blocks, "excluded": exclusions,
        "pages": page_info, "warnings": warnings,
        "limitations": [
            "Heading fallback and leading-row table headers are heuristics; low-confidence structures remain visible.",
            "Images/vector diagrams are not converted to factual text; quick-start guide requires visual interpretation.",
            "Cross-page sections are retained. Cross-page tables are kept as separate tables, not silently merged.",
        ],
    }

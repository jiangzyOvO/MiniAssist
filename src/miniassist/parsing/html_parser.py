"""DOM-based extraction for the archived DJI support page."""
import re
from lxml import html

from .models import Block, stable_id
from .provenance import PageMapper


def elements(node, class_prefix):
    return node.xpath(f'.//*[contains(@class,"{class_prefix}")]')


def dom_text(node):
    """Preserve <br>, list boundaries and all footnotes, including collapsed UI."""
    parts = []
    def visit(el):
        if el.tag in ("script", "style"):
            return
        if el.text:
            parts.append(el.text)
        for child in el:
            if child.tag == "br":
                parts.append("\n")
            else:
                visit(child)
                if child.tag in ("p", "li", "div"):
                    parts.append("\n")
            if child.tail:
                parts.append(child.tail)
    visit(node)
    return re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()


def parse_html(path, source, canonical_text):
    root = html.fromstring(path.read_text(encoding="utf-8"))
    tree = root.getroottree()
    mapper = PageMapper(source["source_id"], None, canonical_text)
    blocks, sections, warnings = [], [], []
    selectors = {"faq": 0, "specification": 0}

    def emit(kind, parent, title, body, node, metadata):
        text = title + "\n" + body
        mapped = mapper.map(text)
        section_id = stable_id(source["source_id"], parent)
        if not any(s["section_id"] == section_id for s in sections):
            sections.append({"section_id": section_id, "title": parent, "parent_id": None,
                             "method": "dom", "confidence": "high", "start_page": None})
        ident = stable_id(source["sha256"], tree.getpath(node))
        b = Block(ident, kind, None, text, [parent, title], [section_id], None,
                  mapped["source_spans"], mapped["text_map"],
                  metadata={"source_id": source["source_id"], "source_version": source["version"],
                            "source_sha256": source["sha256"], "dom_xpath": tree.getpath(node),
                            **metadata})
        if mapped["matched_chars"] < mapped["total_chars"]:
            b.warnings.append("partial_source_alignment")
        blocks.append(b.to_dict())

    # Site-specific adapter intentionally errors if the known DOM disappears.
    for q in elements(root, "style__question___"):
        item = q.getparent()
        answers = elements(item, "style__answer___")
        if len(answers) != 1:
            raise ValueError("FAQ question has no unique answer in this snapshot")
        title, answer = dom_text(q), dom_text(answers[0])
        emit("faq", "常见问题", title, answer, item,
             {"question": title, "answer": answer,
              "links": [{"text": dom_text(a), "url": a.get("href")} for a in answers[0].xpath(".//a[@href]")]})
        selectors["faq"] += 1
    for group in elements(root, "style__specs-group-item___"):
        titles = elements(group, "style__specs-group-title___")
        if len(titles) != 1:
            raise ValueError("Specification group has no unique title")
        group_title = dom_text(titles[0])
        for item in elements(group, "style__specs-detail___"):
            keys = elements(item, "style__specs-detail-key___")
            values = elements(item, "style__specs-detail-value___")
            if len(keys) != 1 or len(values) != 1:
                raise ValueError("Specification item has no unique key/value")
            emit("specification", "技术参数 / "+group_title, dom_text(keys[0]), dom_text(values[0]),
                 item, {"group": group_title, "key": dom_text(keys[0]), "value": dom_text(values[0])})
            selectors["specification"] += 1
    if not selectors["faq"] or not selectors["specification"]:
        raise ValueError("DJI support adapter matched an empty document")
    # Only the declared content regions are in scope, not global navigation / contact UI.
    return {
        "source": source, "profile": "support_html",
        "sections": sections, "blocks": blocks, "excluded": [], "pages": [],
        "warnings": warnings, "dom_counts": selectors,
        "limitations": ["Only FAQ and specification DOM regions are parsed; navigation, service links and video descriptions are out of scope.",
                        "Dynamic page has no inferred publication date; version is the archived snapshot."],
    }

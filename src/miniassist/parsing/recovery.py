"""Conservative recovery operators for visually audited PDF layout regions.

No QA records are inputs. Layout decisions live in a source-hashed policy.
These operators recover text, never interpret missing icons or invent glyphs.
"""
from copy import deepcopy
import re

from .models import sha256, stable_id
from .provenance import compact


def exact_map(text, canonical, sid, page):
    """Require one exact whitespace-insensitive occurrence; never fuzzy-match."""
    output_positions = [i for i, c in enumerate(text) if not c.isspace()]
    source_positions = [i for i, c in enumerate(canonical) if not c.isspace()]
    needle, haystack = compact(text), compact(canonical)
    if not needle:
        return {"text": text, "text_map": [], "source_spans": []}
    hits = [m.start() for m in re.finditer("(?=" + re.escape(needle) + ")", haystack)]
    if len(hits) != 1:
        raise ValueError(f"Expected unique exact source match, got {len(hits)}: {text!r}")
    start = hits[0]
    mappings = []
    for out, src in zip(output_positions, source_positions[start:start + len(needle)]):
        if mappings and mappings[-1]["text_end"] == out and mappings[-1]["source_end"] == src:
            mappings[-1]["text_end"] += 1
            mappings[-1]["source_end"] += 1
        else:
            mappings.append(dict(text_start=out, text_end=out + 1, source_start=src, source_end=src + 1))
    a, z = source_positions[start], source_positions[start + len(needle) - 1] + 1
    return {"text": text, "text_map": mappings, "source_spans": [{
        "source_id": sid, "page": page, "start_char": a, "end_char": z,
        "exact_text": canonical[a:z], "page_text_sha256": sha256(canonical.encode()),
    }]}


def concatenate(parts):
    result = {"text": "", "text_map": [], "source_spans": []}
    for index, part in enumerate(parts):
        if index:
            result["text"] += "\n"
        offset = len(result["text"])
        result["text"] += part["text"]
        result["text_map"].extend(dict(m, text_start=m["text_start"] + offset,
                                      text_end=m["text_end"] + offset) for m in part["text_map"])
        result["source_spans"].extend(deepcopy(part["source_spans"]))
    return result


def slice_value(value, start, end, canonical, sid, page):
    maps = []
    for m in value["text_map"]:
        a, z = max(start, m["text_start"]), min(end, m["text_end"])
        if a < z:
            maps.append(dict(text_start=a-start, text_end=z-start,
                             source_start=m["source_start"]+a-m["text_start"],
                             source_end=m["source_start"]+z-m["text_start"]))
    spans = []
    if maps:
        s, e = min(m["source_start"] for m in maps), max(m["source_end"] for m in maps)
        spans.append(dict(source_id=sid, page=page, start_char=s, end_char=e,
                          exact_text=canonical[s:e], page_text_sha256=sha256(canonical.encode())))
    return dict(text=value["text"][start:end], text_map=maps, source_spans=spans)


def validate_block(block, canonical):
    mapped = set()
    for m in block["text_map"]:
        a, z, s, e = (m[k] for k in ("text_start", "text_end", "source_start", "source_end"))
        if not (0 <= a < z <= len(block["text"]) and 0 <= s < e <= len(canonical)):
            raise ValueError("Invalid source coordinate")
        if block["text"][a:z] != canonical[s:e]:
            raise ValueError("Source text mismatch")
        mapped.update(range(a, z))
    expected = {i for i, c in enumerate(block["text"]) if not c.isspace()}
    if mapped != expected:
        raise ValueError("Incomplete emitted-character provenance")
    for span in block["source_spans"]:
        if (span["exact_text"] != canonical[span["start_char"]:span["end_char"]]
                or span["page_text_sha256"] != sha256(canonical.encode())):
            raise ValueError("Invalid source span")


def finalize(seed, value, rule, table=None, bbox=None):
    block = deepcopy(seed)
    block.update(value)
    block["table"] = table
    block["block_type"] = "table" if table else seed["block_type"]
    block["block_id"] = stable_id(seed["block_id"], rule["rule_id"], value["text"])
    block["warnings"] = []
    if bbox:
        block["bbox"] = bbox
    block["metadata"].update({
        "baseline_policy_version": "1.2",
        "recovery_rule_id": rule["rule_id"],
        "derived_from_block_ids": rule.get("derived_from_block_ids", [seed["block_id"]]),
        "recovery_review": "agent_visual_review",
        "human_signoff": False,
        "recovery_limit": rule["scope"],
        "required_context_block_ids": rule.get("context_block_ids", []),
    })
    return block


def build_table(seed, values, rule, header_rows, bbox):
    """Serialize a rectangular text projection with real, mapped header cells."""
    width = len(values[0])
    if not width or any(len(row) != width for row in values):
        raise ValueError("Nonrectangular text projection")
    cells, flattened, offset = [], [], 0
    for row, values_row in enumerate(values):
        for col, value in enumerate(values_row):
            if flattened:
                offset += 1
            cells.append(dict(deepcopy(value), row=row, column=col, role="cell",
                              row_span=1, col_span=1, cell_id=stable_id(rule["rule_id"], row, col),
                              text_start=offset, text_end=offset + len(value["text"]),
                              matched_chars=len(compact(value["text"])), total_chars=len(compact(value["text"]))))
            flattened.append(value)
            offset += len(value["text"])
    review = {"reviewer": "agent_visual_review", "human_signoff": False,
              "context_block_ids": rule.get("context_block_ids", []), "scope": rule["scope"]}
    table = dict(method="audited_text_region_projection", row_count=len(values), column_count=width,
                 cells=cells, header_rows=header_rows, header_status="agent_reviewed",
                 bbox=bbox, visual_dependency=False, footnote_block_ids=[], review=review,
                 omitted_visual_semantics=rule["scope"])
    return finalize(seed, concatenate(flattened), rule, table, bbox)


def recover(documents, canonical, policy, pdf_pages):
    """Apply source-specific layout approvals using reusable fail-closed operators."""
    source_blocks = {b["block_id"]: b for doc in documents.values() for b in doc["blocks"]}
    added = []
    for rule in policy["rules"]:
        seed = source_blocks[rule["seed_block_id"]]
        sid, page = seed["metadata"]["source_id"], seed["page"]
        text = canonical[sid, page]
        if sha256(seed["text"].encode()) != rule["seed_text_sha256"]:
            raise ValueError("Stale recovery review")
        mapper = lambda value: exact_map(value, text, sid, page)
        op = rule["operation"]
        if op == "restore_block":
            if seed["warnings"]:
                raise ValueError("Cannot restore a block with unresolved parser warnings")
            value = {key: deepcopy(seed[key]) for key in ("text", "text_map", "source_spans")}
            block = finalize(seed, value, rule)
        elif op == "cover_tokens":
            # Version/date are separate layout tokens; canonical order may differ.
            tokens = re.findall(r"v\d+(?:\.\d+)+|\d{4}\.\d{2}", seed["text"])
            if len(tokens) != 2 or compact("".join(tokens)) != compact(seed["text"]):
                raise ValueError("Unexpected cover metadata")
            block = finalize(seed, concatenate([mapper(t) for t in tokens]), rule)
        elif op == "omit_icon_clause":
            # Keep the heading and every remaining sentence, not just a gold quote.
            pattern = r"点击\s+可[^。]*。"
            matches = list(re.finditer(pattern, seed["text"]))
            if len(matches) != 1:
                raise ValueError("Expected one audited icon-dependent clause")
            hit = matches[0]
            values = [mapper(s.strip()) for s in (seed["text"][:hit.start()], seed["text"][hit.end():]) if s.strip()]
            block = finalize(seed, concatenate(values), rule)
        elif op == "project_cells":
            cells = {(c["row"], c["column"]): c for c in seed["table"]["cells"]}
            values = []
            for row in rule["rows"]:
                items = []
                for col in rule["columns"]:
                    cell = cells[row, col]
                    if cell["role"] != "cell":
                        raise ValueError("Merged cells require an explicit projection")
                    # Reuse validated cell maps: no alignment to another occurrence.
                    items.append({key: deepcopy(cell[key]) for key in ("text", "text_map", "source_spans")})
                values.append(items)
            block = build_table(seed, values, rule, rule["header_rows"], seed["bbox"])
        elif op == "crop_table":
            xs, ys = rule["column_edges"], rule["row_edges"]
            pdf_page = pdf_pages[sid, page]
            values = []
            for a, z in zip(ys, ys[1:]):
                texts = []
                for left, right in zip(xs, xs[1:]):
                    cropped = pdf_page.crop((left, a, right, z))
                    value = (cropped.extract_text(x_tolerance=2, y_tolerance=3) or "").strip()
                    texts.append(value)
                # Align the entire row once so repeated labels cannot steal
                # characters from the prose cell or a nearby diagram.
                row_value = mapper("\n".join(texts))
                row, offset = [], 0
                for value in texts:
                    row.append(slice_value(row_value, offset, offset + len(value), text, sid, page))
                    offset += len(value) + 1
                values.append(row)
            if any(not v["text"] for row in values for v in row):
                raise ValueError("Unexpected empty cell in audited text rectangle")
            block = build_table(seed, values, rule, rule["header_rows"], [xs[0], ys[0], xs[-1], ys[-1]])
        else:
            raise ValueError(f"Unknown recovery operation: {op}")
        validate_block(block, text)
        added.append(block)
    return added


def merge_additions(old, additions, documents):
    """Keep every old object intact and preserve document/reading order."""
    rank = {b["block_id"]: i for doc in documents.values() for i, b in enumerate(doc["blocks"])}
    old_ids = {b["block_id"] for b in old}
    if old_ids & {b["block_id"] for b in additions}:
        raise ValueError("Recovery must have new block identities")
    def key(b):
        return (b["metadata"]["source_id"],
                rank.get(b["metadata"].get("derived_from_block_id", b["block_id"]),
                         rank.get(b["metadata"].get("derived_from_block_ids", [""])[0], 10**9)))
    # Insert additions without sorting/reordering any parent objects.
    result = list(old)
    for addition in sorted(additions, key=key):
        index = next((i for i, b in enumerate(result) if key(b) > key(addition)), len(result))
        result.insert(index, addition)
    if [b for b in result if b["block_id"] in old_ids] != old:
        raise ValueError("Recovery changed parent corpus order")
    return result

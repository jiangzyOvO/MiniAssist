"""Geometry-based PDF table extraction, including open outer borders."""
from statistics import median
from collections import Counter


def close(a, b, tolerance=2):
    return abs(a-b) <= tolerance


def contains(box, char):
    x, y = (char["x0"]+char["x1"])/2, (char["top"]+char["bottom"])/2
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def grid_boundaries(cells, axis):
    values = sorted(v for c in cells for v in (c[axis], c[axis+2]))
    result = []
    for v in values:
        if not result or v-result[-1] > 1:
            result.append(v)
    return result


def grid_span(start, end, boundaries):
    """Use physical grid lines, not bounding boxes enlarged by merged cells."""
    return (min(range(len(boundaries)), key=lambda i: abs(boundaries[i]-end))
            - min(range(len(boundaries)), key=lambda i: abs(boundaries[i]-start)))


def extract_shaded_key_value_table(page):
    """DJI specification layout adapter: alternating shaded key/value rows.

    Only called within the bookmarked specification section. Repeated adjacent
    rectangles establish the divider, and shading boundaries establish rows.
    No product facts, QA answers, or page-number-specific text is inserted.
    """
    pairs = []
    for left in page.rects:
        for right in page.rects:
            if (close(left["x1"], right["x0"], .1)
                    and close(left["top"], right["top"], .1)
                    and close(left["bottom"], right["bottom"], .1)
                    and left["width"] > 40 and right["width"] > 100):
                pairs.append((left, right))
    if len(pairs) < 3:
        return None
    grid = Counter((round(a["x0"], 1), round(a["x1"], 1), round(b["x1"], 1))
                   for a, b in pairs).most_common(1)[0]
    if grid[1] < 3:
        return None
    x0, divider, x1 = grid[0]
    pairs = [(a, b) for a, b in pairs if close(a["x0"], x0, .2)
             and close(a["x1"], divider, .2) and close(b["x1"], x1, .2)]
    categories = [r for r in page.rects if close(r["x0"], x0, .2)
                  and close(r["x1"], x1, .2) and isinstance(r["non_stroking_color"], tuple)
                  and .80 < sum(r["non_stroking_color"])/len(r["non_stroking_color"]) < .98
                  and (sum(r["non_stroking_color"])/len(r["non_stroking_color"]) < .92
                       or not any(contains((divider, r["top"], x1, r["bottom"]), c)
                                  and c["text"].strip() for c in page.chars))]
    # Exclude running header, page footer, and the darker section-title bar.
    title_bars = [r for r in page.rects if close(r["x0"], x0, .2)
                  and close(r["x1"], x1, .2) and isinstance(r["non_stroking_color"], tuple)
                  and .65 < sum(r["non_stroking_color"])/len(r["non_stroking_color"]) <= .80]
    top = max([40] + [r["bottom"]+1 for r in title_bars])
    chars = [c for c in page.chars if x0 <= (c["x0"]+c["x1"])/2 <= x1
             and top < c["top"] and c["bottom"] < page.height*.95]
    if not chars:
        return None
    top, bottom = min(c["top"] for c in chars)-3, max(c["bottom"] for c in chars)+3
    boundaries = [top, bottom]
    for r in [a for a, _ in pairs] + categories:
        boundaries.extend([r["top"], r["bottom"]])
    ys = []
    for y in sorted(boundaries):
        if not ys or y-ys[-1] > 1:
            ys.append(y)
    cells, rows = [], []
    for a, b in zip(ys, ys[1:]):
        row_chars = [c for c in chars if a <= (c["top"]+c["bottom"])/2 < b]
        if not row_chars:
            continue
        ri = len(rows)
        category = any(close(a, r["top"], 1) and close(b, r["bottom"], 1) for r in categories)
        rows.append({"row": ri, "kind": "category" if category else "data"})
        boxes = [(x0, a, x1, b)] if category else [(x0, a, divider, b), (divider, a, x1, b)]
        for ci, box in enumerate(boxes):
            cp = page.filter(lambda o: o["object_type"] == "char" and contains(box, o))
            text = cp.extract_text(x_tolerance=2, y_tolerance=3) or ""
            cells.append({"row": ri, "column": ci, "bbox": list(box), "text": text,
                          "role": "cell", "row_span": 1, "col_span": 2 if category else 1})
        if category:
            cells.append({"row": ri, "column": 1, "bbox": None, "text": "",
                          "role": "covered_by_merged_cell", "covered_by": [ri, 0],
                          "row_span": 0, "col_span": 0})
    return {
        "bbox": [x0, min(c["bbox"][1] for c in cells if c["bbox"]),
                 x1, max(c["bbox"][3] for c in cells if c["bbox"])],
        "method": "shaded_key_value_layout", "row_count": len(rows), "column_count": 2,
        "cells": cells, "rows": rows, "header_rows": [],
        "header_status": "key_value_no_column_header", "column_roles": ["property", "value"],
        "footnote_block_ids": [], "footnote_status": "pending_numbered_reference_linking",
        "visual_dependency": any(not c["text"].strip() for c in cells if c["role"] == "cell"),
    }


def extend_open_table(page, seed):
    """Extend a header-only grid using real row rules and continuous dividers.

    No page number or product names are used. The missing outside borders are
    explicit reconstruction; internal dividers and row rules must be present.
    """
    if any(cell is None for row in seed.rows for cell in row.cells):
        # Add only missing outer borders. Keep actual internal rules so genuine
        # merged cells are not split or filled from unrelated neighbours.
        x0, top, x1, bottom = seed.bbox
        repaired = page.crop(seed.bbox).find_tables({
            "vertical_strategy": "lines", "explicit_vertical_lines": [x0, x1],
            "horizontal_strategy": "lines", "intersection_tolerance": 3,
        })
        if repaired:
            candidate = max(repaired, key=lambda t: len(t.cells))
            if len(candidate.cells) > len(seed.cells):
                return candidate, "open_border_reconstruction"
    if len(seed.rows) != 1:
        return seed, "pdf_grid"
    x0, top, x1, bottom = seed.bbox
    xs = sorted({round(c[0], 2) for c in seed.cells} | {round(c[2], 2) for c in seed.cells})
    internal = [x for x in xs if x0+3 < x < x1-3]
    if not internal:
        return seed, "pdf_grid"
    ys = [top, bottom]
    horizontal = [e for e in page.edges if e.get("orientation") == "h"
                  and e["bottom"] > bottom+2 and e["x0"] < x1 and e["x1"] > x0]
    candidates = sorted({round(e["top"], 1) for e in horizontal})
    for y in candidates:
        parts = [(e["x0"], e["x1"]) for e in horizontal if close(e["top"], y, 1)]
        if not parts or min(a for a, _ in parts) > x0+2 or max(b for _, b in parts) < x1-2:
            continue
        if any(any(e.get("orientation") == "v" and close(e["x0"], x)
                       and e["top"] <= ys[-1]+2 and e["bottom"] >= y-2
                       for e in page.edges) for x in internal):
            ys.append(y)
        else:
            break
    if len(ys) < 3:
        return seed, "pdf_grid"
    crop = page.crop((x0, top, x1, ys[-1]))
    tables = crop.find_tables({
        "vertical_strategy": "explicit", "explicit_vertical_lines": xs,
        "horizontal_strategy": "explicit", "explicit_horizontal_lines": ys,
        "intersection_tolerance": 3,
    })
    return (tables[0], "open_border_reconstruction") if tables else (seed, "pdf_grid")


def extract_tables(page):
    accepted, rejected = [], []
    for seed in page.find_tables():
        table, method = extend_open_table(page, seed)
        matrix = table.extract(x_tolerance=2, y_tolerance=3)
        filled = [v.strip() for row in matrix for v in row if v and v.strip()]
        # Reject diagram grids with little linguistic/numeric content.
        if len(table.rows) < 2 or len(table.columns) < 2 or len(filled) < 4 or median(map(len, filled)) < 2:
            rejected.append({"bbox": list(table.bbox), "reason": "sparse_or_header_only_grid"})
            continue
        if any(all(abs(a-b) < 2 for a, b in zip(table.bbox, t["bbox"])) for t in accepted):
            continue
        cells = []
        xs, ys = grid_boundaries(table.cells, 0), grid_boundaries(table.cells, 1)
        for ri, row in enumerate(table.rows):
            for ci, box in enumerate(row.cells):
                if box is None:
                    # A null slot is not automatically the value above it.
                    cells.append({"row": ri, "column": ci, "bbox": None, "text": "",
                                  "role": "unresolved_slot", "row_span": 0, "col_span": 0})
                    continue
                # A glyph belongs to one cell by centre, not every box it touches.
                cell_page = page.filter(lambda obj: obj["object_type"] == "char" and contains(box, obj))
                text = cell_page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                # Count geometric coverage instead of blindly forward-filling.
                row_span = grid_span(box[1], box[3], ys)
                col_span = grid_span(box[0], box[2], xs)
                cells.append({"row": ri, "column": ci, "bbox": list(box), "text": text,
                              "role": "cell", "row_span": row_span, "col_span": col_span})
        # Resolve null grid positions only when a physical spanning cell covers them.
        for c in cells:
            if c["role"] != "unresolved_slot":
                continue
            for parent in cells:
                if parent["bbox"] and parent["row"] <= c["row"] < parent["row"]+parent["row_span"] \
                        and parent["column"] <= c["column"] < parent["column"]+parent["col_span"]:
                    c.update(role="covered_by_merged_cell",
                             covered_by=[parent["row"], parent["column"]])
                    break
        accepted.append({
            "bbox": list(table.bbox), "method": method, "row_count": len(table.rows),
            "column_count": len(table.columns), "cells": cells,
            # Header inferred from leading row; validation report flags the heuristic.
            "header_rows": [0], "header_status": "leading_row_candidate",
            "footnote_block_ids": [], "footnote_status": "nearby_candidates_only",
            "visual_dependency": any(c["role"] == "cell" and not c["text"].strip() for c in cells),
        })
    return accepted, rejected

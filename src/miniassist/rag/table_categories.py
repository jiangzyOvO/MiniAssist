"""Explicit, same-table category context. Never remap copied chunk labels."""
import copy
import re


def compact(text):
    return ''.join(text.split())


def category_context(block, row, canonical):
    """Resolve each path level against preceding physical category cells.

    Scope stops at the next heading of the same or shallower depth. No cross-page
    inference, nearest-string search, QA labels, or metadata-only anchors.
    """
    table = block['table']
    metadata = {r['row']: r for r in table.get('rows', [])}
    path = metadata.get(row, {}).get('category_path', [])
    active = {}
    headings = sorted((r for r in metadata.values() if r.get('kind') == 'category'), key=lambda r: r['row'])
    for heading in headings:
        if heading['row'] >= row:
            break
        depth = len(heading.get('category_path', []))
        active = {d: h for d, h in active.items() if d < depth}
        active[depth] = heading
    anchors, unresolved = [], []
    for depth, label in enumerate(path, 1):
        heading = active.get(depth)
        if not heading or heading.get('category_path') != path[:depth]:
            unresolved.append(label)
            continue
        cells = [c for c in table['cells'] if c['row'] == heading['row'] and c['role'] == 'cell']
        if len(cells) != 1 or compact(cells[0]['text']) != compact(label):
            unresolved.append(label)
            continue
        cell = cells[0]
        sid, page = block['metadata']['source_id'], block['page']
        source = canonical[sid, page]
        covered = set()
        for m in cell['text_map']:
            a, z = m['text_start'], m['text_end']
            if not (0 <= a < z <= len(cell['text'])) or source[m['source_start']:m['source_end']] != cell['text'][a:z]:
                raise ValueError('Category source mapping mismatch')
            covered.update(range(a, z))
        if not {i for i, ch in enumerate(cell['text']) if not ch.isspace()} <= covered:
            raise ValueError('Incomplete category source mapping')
        spans = []
        for m in cell['text_map']:
            spans.append({'source_id': sid, 'source_sha256': block['metadata']['source_sha256'],
                          'page': page, 'block_id': block['block_id'],
                          'start_char': m['source_start'], 'end_char': m['source_end'],
                          'exact_text': source[m['source_start']:m['source_end']]})
        end = next((h['row'] for h in headings if h['row'] > heading['row']
                    and len(h.get('category_path', [])) <= depth), table['row_count'])
        anchors.append({'anchor_id': f"{block['block_id']}:category:{heading['row']}",
                        'text': cell['text'], 'cell_id': cell['cell_id'],
                        'origin': 'supplemental_category', 'source_spans': spans,
                        'scope': {'table_id': block['block_id'], 'category_row': heading['row'],
                                  'data_row_start': heading['row'] + 1, 'data_row_end_exclusive': end,
                                  'category_path': path[:depth], 'method': 'same_table_category_rows'}})
    return {'status': 'unresolved' if unresolved else ('supported' if path else 'not_required'),
            'anchors': anchors, 'unresolved_labels': unresolved}


def add_category_context(rows, blocks, canonical):
    result = copy.deepcopy(rows)
    for row in result:
        row['category_context'] = category_context(blocks[row['table_id']], row['row'], canonical)
    return result


def model_codes(text):
    """Only explicit alphanumeric model designations, not arbitrary product names."""
    return re.findall(r'型号\s*[：:]\s*([A-Za-z]+[A-Za-z0-9-]*\d[A-Za-z0-9-]*)', text)

"""Shared, QA-independent serialization for all three splitters.

Table rows repeat reviewed headers, physical merged cells and required notes.
The fixed/recursive baselines may subsequently cut these units; this is measured.
"""
from .text import literal, from_block, join


def prepare(blocks):
    by_id = {b['block_id']: b for b in blocks}
    units = []
    for b in blocks:
        common = {'source_id': b['metadata']['source_id'], 'source_version': b['metadata']['source_version'],
                  'source_sha256': b['metadata']['source_sha256'], 'block_id': b['block_id'],
                  'page': b['page'], 'section_path': b['section_path'], 'section_ids': b['section_ids'],
                  'release_date': b['metadata'].get('release_date'),
                  'dom_xpath': b['metadata'].get('dom_xpath'), 'links': b['metadata'].get('links', [])}
        if not b['table']:
            units.append(dict(common, unit_id=b['block_id'], kind=b['block_type'],
                              prefix=literal(''), body=from_block(b), suffix=literal('')))
            continue
        t = b['table']
        cells = {(c['row'], c['column']): c for c in t['cells']}
        def resolved(row, col):
            c = cells[(row, col)]
            if c['role'] == 'covered_by_merged_cell':
                c = cells[tuple(c['covered_by'])]
            if c['role'] != 'cell':
                raise ValueError('Unresolved cell in frozen corpus')
            return c
        header_parts = []
        for row in t['header_rows']:
            for col in range(t['column_count']):
                header_parts.append(join([literal(f'表头第{row+1}层/第{col+1}列：'), from_block(b, resolved(row, col))]))
        headers = join(header_parts, '\n')
        row_meta = {r['row']: r for r in t.get('rows', [])}
        for row in range(t['row_count']):
            if row in t['header_rows']:
                continue
            meta = row_meta.get(row, {})
            if meta.get('kind') == 'category':
                # Preserve actual category glyph provenance as its own unit.
                units.append(dict(common, unit_id=f"{b['block_id']}:category:{row}", kind='table_category',
                                  prefix=literal(''), body=from_block(b, resolved(row, 0)), suffix=literal('')))
                continue
            dependencies = (meta.get('footnote_block_ids', []) + t['review'].get('context_block_ids', [])
                            if t['method'] == 'shaded_key_value_layout'
                            else b['metadata'].get('required_context_block_ids', []))
            dependencies = list(dict.fromkeys(dependencies))
            notes = join([from_block(by_id[ident]) for ident in dependencies], '\n')
            # Category labels here are metadata-derived, not source evidence.
            category = ' / '.join(meta.get('category_path', []))
            prefix = join([headers, literal(('分类：'+category+'\n') if category else ''),
                           literal('第1列：'), from_block(b, resolved(row, 0)), literal('\n')], '\n')
            body = join([join([literal(f'第{col+1}列：'), from_block(b, resolved(row, col))])
                         for col in range(1, t['column_count'])], '\n')
            suffix = join([literal('\n关联说明：\n'), notes]) if notes['text'] else literal('')
            units.append(dict(common, unit_id=f"{b['block_id']}:row:{row}", kind='table_row',
                              row=row, category_path=meta.get('category_path', []),
                              required_context_block_ids=dependencies, prefix=prefix, body=body, suffix=suffix))
    for u in units:
        u['value'] = join([u['prefix'], u['body'], u['suffix']])
    return units

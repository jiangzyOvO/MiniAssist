"""Source-bound rows with separately declared category context supplements."""
import hashlib
import json
from functools import lru_cache
from .errors import CitationError
from .settings import ROOT


def document_fingerprint(doc):
    fields = ('chunk_id', 'source_id', 'source_sha256', 'text', 'text_map', 'unit_ids')
    data = {k: doc.get(k) for k in fields}
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build_rows(doc, blocks):
    """Compile only whole source cells already present in this chunk's mapping.

    Row membership comes from the frozen parsed table, never from label regexes.
    Missing cells/headers fail closed; categories copied as metadata are excluded.
    """
    def cell_part(block, cell):
        if cell['role'] == 'covered_by_merged_cell':
            cell = next(c for c in block['table']['cells']
                        if (c['row'], c['column']) == tuple(cell['covered_by']))
        if cell['role'] != 'cell' or not cell['text'].strip():
            return None
        maps, required, found, mapped_cell_positions = [], set(), set(), set()
        for cm in cell['text_map']:
            if cm['text_end']-cm['text_start'] != cm['source_end']-cm['source_start']:
                raise ValueError('Nonlinear source cell')
            for i in range(cm['text_start'], cm['text_end']):
                if not cell['text'][i].isspace():
                    mapped_cell_positions.add(i)
                    required.add(cm['source_start'] + i-cm['text_start'])
            for dm in doc['text_map']:
                if (dm['source_id'], dm['page'], dm['block_id']) != (block['metadata']['source_id'], block['page'], block['block_id']):
                    continue
                if dm['text_end']-dm['text_start'] != dm['source_end']-dm['source_start']:
                    raise ValueError('Nonlinear chunk mapping')
                lo, hi = max(cm['source_start'], dm['source_start']), min(cm['source_end'], dm['source_end'])
                if lo >= hi:
                    continue
                a, z = dm['text_start']+lo-dm['source_start'], dm['text_start']+hi-dm['source_start']
                ca, cz = cm['text_start']+lo-cm['source_start'], cm['text_start']+hi-cm['source_start']
                if doc['text'][a:z] != cell['text'][ca:cz]:
                    raise ValueError('Cell/chunk text mismatch')
                found.update(range(lo, hi))
                m = {**dm, 'text_start': a, 'text_end': z, 'source_start': lo, 'source_end': hi}
                if m not in maps:
                    maps.append(m)
        if (mapped_cell_positions != {i for i, char in enumerate(cell['text']) if not char.isspace()}
                or not required or not required <= found):
            return None
        return {'cell_id': cell['cell_id'], 'row': cell['row'], 'column': cell['column'],
                'text': cell['text'], 'text_map': maps}

    rows = []
    for uid in doc.get('unit_ids', []):
        if ':row:' not in uid:
            continue
        bid, row_number = uid.rsplit(':row:', 1)
        block = blocks.get(bid)
        if not block or not block.get('table'):
            continue
        if (block['metadata']['source_id'], block['metadata']['source_sha256']) != (doc['source_id'], doc['source_sha256']):
            raise ValueError('Table source mismatch')
        row = int(row_number)
        table = block['table']
        if row in table['header_rows'] or not 0 <= row < table['row_count']:
            continue
        members = [c for c in table['cells'] if c['row'] == row]
        headers = [c for c in table['cells'] if c['row'] in table['header_rows']]
        if len(members) != table['column_count'] or len(members) < 2:
            continue
        parts, head = [cell_part(block, c) for c in members], [cell_part(block, c) for c in headers]
        if any(p is None for p in parts + head):
            continue
        rows.append({'table_row_id': doc['chunk_id'] + ':' + uid, 'table_id': bid,
                     'row': row, 'headers': head, 'cells': parts})
    return rows


@lru_cache(maxsize=1)
def _registry():
    config = json.loads((ROOT/'config/citation_protocol.v2.json').read_text())
    path = ROOT/config['table_rows']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['table_rows_sha256']:
        raise CitationError('table_registry_hash_mismatch')
    return json.loads(raw)['documents']


def table_rows(doc):
    entry = _registry().get(doc['chunk_id'])
    if entry is None:
        return []
    if entry['document_fingerprint'] != document_fingerprint(doc):
        raise CitationError('table_document_mismatch')
    return entry['rows']


def model_table_rows(doc):
    return [{'table_row_id': row['table_row_id'], 'table_id': row['table_id'], 'row': row['row'],
             **({'category_context': row['category_context']} if 'category_context' in row else {}),
             'headers': [{k: p[k] for k in ('row', 'column', 'text')} for p in row['headers']],
             'cells': [{k: p[k] for k in ('row', 'column', 'text')} for p in row['cells']]}
            for row in table_rows(doc)]


def resolve_table_row(doc, row_id):
    row = next((r for r in table_rows(doc) if r['table_row_id'] == row_id), None)
    if row is None:
        raise CitationError('unknown_table_row_id')
    parts = [{**p, 'role': role} for role, cells in [('header', row['headers']), ('cell', row['cells'])] for p in cells]
    context = row.get('category_context', {})
    anchors = context.get('anchors', [])
    # Display separators are NOT a continuous quotation. Consumers must use parts.
    display = '；'.join(' / '.join(p['text'] for p in cells) for cells in (row['headers'], row['cells']) if cells)
    result = {'quote': display, 'model_quote': None, 'table_row_id': row_id,
            'match_mode': 'table_row_exact', 'quote_is_contiguous': False,
            'quote_parts': parts, 'text_map': [m for p in parts for m in p['text_map']]}
    if context:
        # Supplemental source coordinates MUST NOT masquerade as chunk offsets.
        result.update(citation_protocol='table_category_context_v2',
                      source_context=anchors, category_scope_status=context['status'],
                      context_expansion_chars=sum(len(a['text']) for a in anchors))
        result['quote'] = '；'.join([*[a['text'] for a in anchors], display])
        result['quote_parts'] = [*({'role': 'category', **a, 'text_map': []} for a in anchors), *parts]
    return result


def validate_category_models(claim_text, documents, references):
    """Narrow provenance guard, not a general claim entailment checker.

    Explicit model codes and copied category model codes require a source in this
    claim's own references. A legal row ID alone does not authorize a model name.
    """
    import re
    from .table_categories import model_codes
    candidates = set(model_codes(claim_text))
    for doc in documents:
        for code in model_codes(doc['text']):
            if re.search(r'(?<![A-Za-z0-9])' + re.escape(code) + r'(?![A-Za-z0-9])', claim_text):
                candidates.add(code)
    proof = '\n'.join(ref['quote'] for ref in references)
    if any(not re.search(r'(?<![A-Za-z0-9])' + re.escape(code) + r'(?![A-Za-z0-9])', proof) for code in candidates):
        raise CitationError('unsupported_category_model')

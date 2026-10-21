"""Fixed, recursive and structure-aware splitting with exact source mappings."""
import re
from .text import cut, join, literal, from_block
from ..parsing.models import stable_id

SEPARATORS = [r'\n\s*\n', r'\n', r'[。！？；](?:[”’」』])?', r'[，,、]']


def recursive_spans(text, size, start=0, level=0):
    """Recursively refine oversize paragraphs -> lines -> sentences -> clauses."""
    if len(text) <= size:
        return [(start, start+len(text))] if text else []
    if level == len(SEPARATORS):
        return [(start+i, start+min(i+size, len(text))) for i in range(0, len(text), size)]
    ends = [m.end() for m in re.finditer(SEPARATORS[level], text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    out, previous = [], 0
    for end in ends:
        out.extend(recursive_spans(text[previous:end], size, start+previous, level+1))
        previous = end
    return out


def windows(text, size, overlap, recursive=False):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    if not recursive:
        start = 0
        while start < len(text):
            end = min(start+size, len(text))
            yield start, end
            if end == len(text):
                break
            start = end-overlap
        return
    atoms = recursive_spans(text, size)
    i = 0
    while i < len(atoms):
        start, end = atoms[i]
        j = i+1
        while j < len(atoms) and atoms[j][1]-start <= size:
            end = atoms[j][1]
            j += 1
        yield start, end
        if j == len(atoms):
            break
        # Retain whole trailing atoms only, <= overlap, and always progress.
        k = j
        while k > i+1 and end-atoms[k-1][0] <= overlap:
            k -= 1
        if k < j and atoms[j][1]-atoms[k][0] > size:
            k = j
        i = k


def chunk(value, method, units, index, flags=None):
    meta = units[0]
    pages = sorted({m['page'] for m in value['text_map'] if m['page'] is not None})
    paths = list(dict.fromkeys(tuple(u['section_path']) for u in units))
    return dict(value, chunk_id=stable_id(method, index, value['text']), method=method,
                source_id=meta['source_id'], source_version=meta['source_version'],
                source_sha256=meta['source_sha256'], pages=pages,
                section_paths=[list(p) for p in paths],
                block_ids=sorted({m['block_id'] for m in value['text_map']}),
                unit_ids=list(dict.fromkeys(u['unit_id'] for u in units)),
                unit_types=sorted({u['kind'] for u in units}),
                release_dates=sorted({u['release_date'] for u in units if u['release_date']}),
                dom_xpaths=list(dict.fromkeys(u['dom_xpath'] for u in units if u['dom_xpath'])),
                links=[link for u in units for link in u['links']], flags=flags or [])


def flat_split(units, size, overlap, method):
    result = []
    for sid in dict.fromkeys(u['source_id'] for u in units):
        source_units = [u for u in units if u['source_id'] == sid]
        value = join([u['value'] for u in source_units], '\n\n')
        bounds, cursor = [], 0
        for u in source_units:
            bounds.append((cursor, cursor+len(u['value']['text']), u))
            cursor += len(u['value']['text'])+2
        for a, z in windows(value['text'], size, overlap, method == 'recursive'):
            touched = [u for left, right, u in bounds if left < z and right > a]
            if touched:
                result.append(chunk(cut(value, a, z), method, touched, len(result)))
    return result


def structure_split(units, blocks, size, overlap):
    result = []
    # Only actual retained headings can contribute source evidence.
    headings = {}
    for b in blocks:
        if b['block_type'] == 'heading' and b['section_ids']:
            headings[(b['metadata']['source_id'], b['section_ids'][-1])] = from_block(b)
    def ancestor_text(u):
        return join([headings[(u['source_id'], ident)] for ident in u['section_ids']
                     if (u['source_id'], ident) in headings], '\n')
    pending, key = [], None
    def flush():
        nonlocal pending
        if not pending:
            return
        ancestors = ancestor_text(pending[0])
        # A pathological heading chain must not prevent forward progress.
        prefix = join([ancestors, literal('\n')]) if ancestors['text'] else literal('')
        budget = size-len(prefix['text'])
        value = join([u['value'] for u in pending], '\n\n')
        bounds, cursor = [], 0
        for u in pending:
            bounds.append((cursor, cursor+len(u['value']['text']), u))
            cursor += len(u['value']['text'])+2
        if budget <= 0:
            result.append(chunk(join([prefix,value]), 'structure', pending, len(result), ['context_exceeds_budget']))
        else:
            for a,z in windows(value['text'],budget,min(overlap,budget-1),True):
                touched = [u for left,right,u in bounds if left<z and right>a]
                if touched:
                    result.append(chunk(join([prefix,cut(value,a,z)]),'structure',touched,len(result)))
        pending = []
    for u in units:
        next_key = (u['source_id'], tuple(u['section_ids']), u['release_date'])
        if next_key != key or u['kind'] in ('table_row', 'table_category', 'faq', 'specification'):
            flush()
        key = next_key
        if u['kind'] == 'table_row':
            prefix = join([ancestor_text(u), literal('\n'), u['prefix']])
            suffix = u['suffix']
            budget = size-len(prefix['text'])-len(suffix['text'])
            if budget <= 0:
                result.append(chunk(join([prefix,u['body'],suffix]),'structure',[u],len(result),['context_exceeds_budget']))
            else:
                spans = list(windows(u['body']['text'],budget,min(overlap,budget-1),True))
                for a,z in spans:
                    result.append(chunk(join([prefix,cut(u['body'],a,z),suffix]),'structure',[u],len(result),
                                        ['table_row_continuation'] if len(spans)>1 else []))
        else:
            pending.append(u)
            if u['kind'] in ('table_category','faq','specification'):
                flush()
    flush()
    return result


def split(units, blocks, method, size=500, overlap=80):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    if method in ('fixed','recursive'):
        return flat_split(units,size,overlap,method)
    if method == 'structure':
        return structure_split(units,blocks,size,overlap)
    raise ValueError(f'Unknown splitter: {method}')

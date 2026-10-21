"""QA-independent revision: release records stay together; attach orphan headings.

No retrieval scores, question IDs or gold annotations enter this module.
The original structure splitter remains available unchanged.
"""
from copy import deepcopy
from itertools import groupby

from .splitters import chunk, structure_split, windows
from .text import coordinates, cut, join, literal
from ..parsing.models import stable_id

UPDATE_HEADINGS = {'本次更新', '更新内容', '更新说明'}
PURE_TYPES = {'heading', 'table_category'}


def release_chunks(units, size, overlap):
    whole = join([u['value'] for u in units], '\n\n')
    if len(whole['text']) <= size:
        return [chunk(whole, 'structure_v2', units, 0, ['release_record_atomic'])]
    # The frozen parser marks the record date independently of section headings.
    # Everything before the first update heading is record identity/version context.
    at = next((i for i, u in enumerate(units) if u['kind'] == 'heading'
               and u['value']['text'].strip() in UPDATE_HEADINGS), None)
    if at is None:
        # Unknown layout: retain it honestly instead of guessing a version binding.
        return [chunk(cut(whole, a, z), 'structure_v2', units, i, ['release_layout_unrecognized'])
                for i, (a, z) in enumerate(windows(whole['text'], size, overlap, True))]
    header_units, body_units = units[:at], units[at:]
    header = join([u['value'] for u in header_units], '\n')
    sections, pending = [], []
    for u in body_units:
        if u['kind'] == 'heading' and pending:
            sections.append(pending)
            pending = []
        pending.append(u)
    if pending:
        sections.append(pending)
    result = []
    for section in sections:
        heading_units = section[:1] if section[0]['kind'] == 'heading' else []
        content_units = section[len(heading_units):]
        prefix = join([header, *[u['value'] for u in heading_units], literal('')], '\n')
        value = join([u['value'] for u in content_units], '\n\n')
        budget = size - len(prefix['text'])
        if budget <= 0:
            result.append(chunk(join([prefix, value]), 'structure_v2', header_units + section,
                                len(result), ['context_exceeds_budget']))
        elif not value['text']:
            result.append(chunk(prefix, 'structure_v2', header_units + section, len(result),
                                ['release_context_only']))
        else:
            for a, z in windows(value['text'], budget, min(overlap, budget - 1), True):
                rendered = join([prefix, cut(value, a, z)])
                retained = {m['block_id'] for m in rendered['text_map']}
                touched = [u for u in header_units + section if u['block_id'] in retained]
                result.append(chunk(rendered, 'structure_v2', touched, len(result),
                                    ['release_version_context_repeated']))
    return result


def is_pure(c):
    return bool(c['unit_types']) and set(c['unit_types']) <= PURE_TYPES


def related(left, right):
    if left['source_id'] != right['source_id'] or left['release_dates'] != right['release_dates']:
        return False
    return any(a == b[:len(a)] for a in left['section_paths'] for b in right['section_paths'])


def merge(left, right):
    result = deepcopy(right)
    result.update(join([left, right], '\n\n'))
    for key in ('pages', 'block_ids', 'unit_ids', 'unit_types', 'release_dates', 'dom_xpaths'):
        result[key] = list(dict.fromkeys(left[key] + right[key]))
    result['section_paths'] = [list(p) for p in dict.fromkeys(tuple(p) for p in
                                                            left['section_paths'] + right['section_paths'])]
    result['links'] = left['links'] + right['links']
    result['flags'] = list(dict.fromkeys(left['flags'] + right['flags'] + ['heading_attached']))
    return result


def remove_or_attach_headings(chunks, size):
    nonheading_coverage = set().union(*(coordinates(c) for c in chunks if not is_pure(c)))
    # Remove only redundant heading copies; all source coordinates remain in text.
    kept = [c for c in chunks if not (is_pure(c) and coordinates(c) <= nonheading_coverage)]
    result = []
    i = 0
    while i < len(kept):
        c = kept[i]
        if (is_pure(c) and i + 1 < len(kept) and related(c, kept[i + 1])
                and len(c['text']) + len(kept[i + 1]['text']) + 2 <= size):
            kept[i + 1] = merge(c, kept[i + 1])
        else:
            result.append(c)
        i += 1
    return result


def structure_split_v2(units, blocks, size=500, overlap=80):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    results = []
    # A dated record is a boundary even when its parser sections are separate.
    key = lambda u: (u['source_id'], u['release_date'])
    for (source, release_date), group in groupby(units, key=key):
        group = list(group)
        if release_date:
            results.extend(release_chunks(group, size, overlap))
        else:
            results.extend(structure_split(group, blocks, size, overlap))
    results = remove_or_attach_headings(results, size)
    for index, c in enumerate(results):
        c['method'] = 'structure_v2'
        c['chunk_id'] = stable_id('structure_v2', index, c['text'])
    return results

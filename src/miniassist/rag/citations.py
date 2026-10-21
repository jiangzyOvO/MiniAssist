"""Whitespace-only matching with original offsets; no fuzzy text correction."""
from .errors import CitationError


def resolve_reference(doc, citation):
    """Validate both ordinary text and whole-table-row references."""
    from .state import Citation
    from .table_evidence import resolve_table_row
    cite = Citation.model_validate(citation)
    if cite.evidence_id != doc['chunk_id']:
        raise CitationError('unknown_evidence_id')
    if cite.table_row_id is not None:
        return resolve_table_row(doc, cite.table_row_id)
    return resolve_citation(doc, cite.quote, cite.span_id)


def citable_spans(doc):
    """Maximal mapped runs separated only by whitespace; synthetic labels split runs."""
    text = doc["text"]
    covered = set()
    for m in doc["text_map"]:
        if (m["text_end"] - m["text_start"] != m["source_end"] - m["source_start"]
                or not 0 <= m["text_start"] < m["text_end"] <= len(text)):
            raise CitationError("nonlinear_mapping")
        covered.update(range(m["text_start"], m["text_end"]))
    spans, start = [], None
    for i, char in enumerate(text):
        if i in covered or char.isspace():
            if start is None and i in covered and not char.isspace():
                start = i
        elif start is not None:
            end = i
            while end > start and text[end - 1].isspace():
                end -= 1
            spans.append({"start": start, "end": end, "text": text[start:end]})
            start = None
    if start is not None:
        end = len(text.rstrip())
        spans.append({"start": start, "end": end, "text": text[start:end]})
    return spans


def _occurrences(text, query):
    pos = 0
    while (pos := text.find(query, pos)) != -1:
        yield pos
        pos += 1


def citation_spans(doc):
    """Offer whole mapped runs plus line windows with stable frozen-chunk IDs.

    Whole runs preserve PDF line-wrapped quotations. Line windows distinguish
    repeated table values within one run. Synthetic labels are never included.
    The full document text remains in the payload for semantic row/column context.
    """
    result = {}
    for span in citable_spans(doc):
        windows = [(span['start'], span['end'])]
        offset = span['start']
        for line in span['text'].splitlines(keepends=True):
            lo = offset + len(line) - len(line.lstrip())
            hi = offset + len(line.rstrip())
            if hi > lo:
                windows.append((lo, hi))
            offset += len(line)
        for start, end in windows:
            ident = f"{doc['chunk_id']}:{start}:{end}"
            result.setdefault(ident, {'span_id': ident, 'start': start, 'end': end,
                                      'text': doc['text'][start:end]})
    return list(result.values())


def resolve_quote(doc, quote, span_id=None):
    compact = "".join(c for c in quote if not c.isspace())
    if len(compact) < 4:
        raise CitationError("quote_too_short")
    offered = citation_spans(doc)
    if span_id is not None:
        spans = [s for s in offered if s['span_id'] == span_id]
        if not spans:
            raise CitationError("unknown_span_id")
    else:
        # Legacy callers keep the same global uniqueness requirement.
        spans = citable_spans(doc)
    candidates = []
    for span in spans:
        offsets = [i for i in range(span["start"], span["end"])
                   if not doc["text"][i].isspace()]
        normalized = "".join(doc["text"][i] for i in offsets)
        for pos in _occurrences(normalized, compact):
            candidates.append((offsets[pos], offsets[pos + len(compact) - 1] + 1))
    if not candidates:
        if span_id is not None:
            raise CitationError("quote_not_in_span")
        # Explain synthetic-label errors separately from altered/spliced quotes.
        normalized_all = "".join(c for c in doc["text"] if not c.isspace())
        raise CitationError("unmapped_synthetic_text" if compact in normalized_all
                            else "quote_not_found")
    # Prefer verbatim occurrences when available, but never choose an arbitrary source.
    exact = [(lo, hi) for lo, hi in candidates if doc["text"][lo:hi] == quote.strip()]
    candidates = exact or candidates
    resolved = {}
    for start, end in candidates:
        mappings = []
        for m in doc["text_map"]:
            lo, hi = max(start, m["text_start"]), min(end, m["text_end"])
            if lo < hi:
                mappings.append({**m, "text_start": lo, "text_end": hi,
                                 "source_start": m["source_start"] + lo - m["text_start"],
                                 "source_end": m["source_start"] + hi - m["text_start"]})
        # Repeated section headings may refer to the same original source coordinates.
        identity = tuple(sorted({(m["source_id"], str(m["page"]), m["source_start"], m["source_end"])
                                 for m in mappings}))
        resolved.setdefault(identity, (start, end, mappings))
    if len(resolved) != 1:
        raise CitationError("ambiguous_quote")
    start, end, mappings = next(iter(resolved.values()))
    if span_id is None:
        # Record the smallest offered window that contains the uniquely resolved
        # quote; never use this to choose among ambiguous original coordinates.
        span_id = min((s for s in offered if s['start'] <= start and end <= s['end']),
                      key=lambda s: (s['end']-s['start'], s['start']))['span_id']
    return {"quote": doc["text"][start:end], "model_quote": quote,
            "span_id": span_id,
            "match_mode": "exact" if doc["text"][start:end] == quote else "whitespace_only",
            "quote_start": start, "quote_end": end, "text_map": mappings}


def model_citation_spans(doc):
    """Whole mapped passages plus repeated-line locators, without redundant lines.

    Ordinary quotes need no model-supplied coordinates. Only repeated text benefits
    from narrower offered windows. Internal provenance windows remain unchanged.
    """
    whole = {(s['start'], s['end']) for s in citable_spans(doc)}
    spans = citation_spans(doc)
    counts = {}
    for s in spans:
        if (s['start'], s['end']) not in whole:
            key = ''.join(s['text'].split())
            counts[key] = counts.get(key, 0) + 1
    return [{'span_id': s['span_id'], 'text': s['text']} for s in spans
            if (s['start'], s['end']) in whole or counts.get(''.join(s['text'].split()), 0) > 1]


def resolve_citation(doc, quote, span_id=None):
    """Strict matching with narrowly audited recovery of same-passage locators.

    Never picks one of multiple original occurrences or crosses chunk boundaries.
    A fabricated range can recover only if it contains the unique quote inside a
    single mapped run. A line window can expand only into a quote containing that
    full line, across a line break. Arbitrary unrelated-window IDs still fail.
    """
    try:
        return resolve_quote(doc, quote, span_id)
    except CitationError as original:
        code = original.diagnostic['code']
        if code not in {'unknown_span_id', 'quote_not_in_span'} or not span_id:
            raise
        prefix = doc['chunk_id'] + ':'
        if not span_id.startswith(prefix):
            raise
        try:
            parts = span_id[len(prefix):].split(':')
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                raise ValueError()
            lo, hi = map(int, parts)
            if not 0 <= lo < hi <= len(doc['text']):
                raise ValueError()
            unique = resolve_quote(doc, quote)
        except (ValueError, CitationError):
            raise original from None
        start, end = unique['quote_start'], unique['quote_end']
        if code == 'unknown_span_id':
            allowed = lo <= start and end <= hi and any(
                s['start'] <= lo and hi <= s['end'] for s in citable_spans(doc))
        else:
            allowed = start <= lo and hi <= end and '\n' in doc['text'][start:end]
        if not allowed:
            raise original
        unique['locator_repair'] = {'reason': code, 'original_span_id': span_id,
            'resolved_span_id': unique['span_id'], 'method': 'unique_original_same_passage'}
        return unique

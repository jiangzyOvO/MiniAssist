"""Text operations that clip exact provenance rather than inherit parent ranges."""

def literal(text):
    return {'text': text, 'text_map': []}


def from_block(b, cell=None):
    obj = cell or b
    return {'text': obj['text'], 'text_map': [dict(m, source_id=b['metadata']['source_id'],
            page=b['page'], block_id=b['block_id']) for m in obj['text_map']]}


def join(parts, separator=''):
    result = literal('')
    for i, part in enumerate(parts):
        if i:
            result['text'] += separator
        offset = len(result['text'])
        result['text_map'].extend(dict(m, text_start=m['text_start']+offset,
                                      text_end=m['text_end']+offset) for m in part['text_map'])
        result['text'] += part['text']
    return result


def cut(value, start, end):
    if not 0 <= start <= end <= len(value['text']):
        raise ValueError('Invalid text slice')
    mappings = []
    for m in value['text_map']:
        a, z = max(start, m['text_start']), min(end, m['text_end'])
        if a < z:
            mappings.append(dict(m, text_start=a-start, text_end=z-start,
                                 source_start=m['source_start']+a-m['text_start'],
                                 source_end=m['source_start']+z-m['text_start']))
    return {'text': value['text'][start:end], 'text_map': mappings}


def coordinates(value):
    return {(m['source_id'], m['page'], i) for m in value['text_map']
            for i in range(m['source_start'], m['source_end'])}

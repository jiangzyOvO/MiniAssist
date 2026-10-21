"""Offline evidence/structure diagnostics. No retrieval or generated answers.

Gold is used only in this module, after chunks have been generated.
"""
from itertools import product
from math import ceil
from statistics import mean
from .text import coordinates
from ..parsing.models import sha256


def minimum_cover(required, chunk_sets, state_limit=100000):
    """Exact set cover with dominance pruning and bounded branch-and-bound.

    A state-limit exit is explicitly NOT reported as an exact minimum.
    """
    if not required:
        return {'status':'exact','count':0,'chunk_indices':[]}
    positions = {c:i for i,c in enumerate(sorted(required, key=str))}
    masks = {}
    for i, coverage in enumerate(chunk_sets):
        bits = sum(1<<positions[c] for c in required & coverage)
        if bits:
            masks.setdefault(bits,i)
    full = (1<<len(required))-1
    union = 0
    for mask in masks:
        union |= mask
    if union != full:
        return {'status':'uncovered','count':None,'chunk_indices':[]}
    candidates=[]
    for mask in sorted(masks,key=int.bit_count,reverse=True):
        if not any(mask & other == mask for other,_ in candidates):
            candidates.append((mask,masks[mask]))
    # Greedy is only an upper bound, never labelled an exact minimum by itself.
    left, upper = full, []
    while left:
        mask,index=max(candidates,key=lambda p:(p[0]&left).bit_count())
        left &= ~mask
        upper.append(index)
    best, visited, states = upper, {}, 0
    def visit(left, chosen):
        nonlocal best,states
        states += 1
        if states > state_limit:
            raise RuntimeError('search_limit')
        if not left:
            if len(chosen)<len(best):
                best=chosen
            return
        if len(chosen)>=visited.get(left,10**9):
            return
        visited[left]=len(chosen)
        gains=[(mask&left,index,mask) for mask,index in candidates if mask&left]
        lower=ceil(left.bit_count()/max(g.bit_count() for g,_,_ in gains))
        if len(chosen)+lower>=len(best):
            return
        # Every cover must choose a candidate containing this uncovered character.
        bits=[]
        rest=left
        while rest:
            bit=rest & -rest
            choices=[(g,index,mask) for g,index,mask in gains if g&bit]
            bits.append(choices)
            if len(choices)==1:
                break
            rest-=bit
        for _,index,mask in sorted(min(bits,key=len),key=lambda x:x[0].bit_count(),reverse=True):
            visit(left & ~mask,chosen+[index])
    try:
        visit(full,[])
    except RuntimeError:
        return {'status':'search_limit','count':None,'upper_bound':len(best),'chunk_indices':best}
    return {'status':'exact','count':len(best),'chunk_indices':best}


def anchor_set(anchor, canonical):
    key=(anchor['source_id'],anchor['pdf_page'])
    text=canonical[key]
    if sha256(text.encode())!=anchor['page_text_sha256'] or text[anchor['start_char']:anchor['end_char']]!=anchor['exact_text']:
        raise ValueError('Gold source anchor mismatch')
    return {(key[0],key[1],i) for i in range(anchor['start_char'],anchor['end_char']) if not text[i].isspace()}


def percentile(values, p):
    ordered=sorted(values)
    return ordered[max(0,ceil(len(ordered)*p)-1)] if ordered else 0


def evaluate(chunks, units, blocks, gold, canonical, size):
    chunk_sets=[coordinates(c) for c in chunks]
    all_coordinates=set().union(*chunk_sets)
    source_coordinates=set()
    for b in blocks:
        source_coordinates.update((b['metadata']['source_id'],b['page'],i) for m in b['text_map']
                                  for i in range(m['source_start'],m['source_end']))
    errors=[]
    for c in chunks:
        for m in c['text_map']:
            src=canonical[(m['source_id'],m['page'])]
            if not (0<=m['text_start']<m['text_end']<=len(c['text']) and
                    0<=m['source_start']<m['source_end']<=len(src)):
                errors.append('invalid_map_range:'+c['chunk_id'])
            elif c['text'][m['text_start']:m['text_end']] != src[m['source_start']:m['source_end']]:
                errors.append('invalid_map_text:'+c['chunk_id'])
            if m['source_id']!=c['source_id']:
                errors.append('cross_source_chunk:'+c['chunk_id'])
        if len(c['text'])>size and 'context_exceeds_budget' not in c['flags']:
            errors.append('unflagged_oversize:'+c['chunk_id'])
    if source_coordinates!=all_coordinates:
        errors.append('frozen_corpus_coordinate_coverage_changed')
    groups, questions, spans=[] ,[],[]
    for record in gold['records']:
        required_options=[]
        for g in record['groups']:
            options=[]
            for alt in g['alternatives']:
                anchors=alt['all_of_spans']+alt['context_anchors']
                sets=[anchor_set(s,canonical) for s in anchors]
                for s,required in zip(anchors,sets):
                    spans.append({'evidence_id':g['evidence_id'],'source_id':s['source_id'],'page':s['pdf_page'],
                                  'complete_in_corpus':required<=all_coordinates,
                                  'complete_in_one_chunk':any(required<=c for c in chunk_sets)})
                options.append(set().union(*sets))
            results=[minimum_cover(req,chunk_sets) for req in options]
            exact=[r for r in results if r['status']=='exact']
            best=min(exact,key=lambda r:r['count']) if exact else results[0]
            groups.append({'evidence_id':g['evidence_id'],'question_id':record['question_id'],
                           'role':g['role'],'title':g['title'],'minimum_cover':best,
                           'single_chunk_complete':any(any(req<=c for c in chunk_sets) for req in options)})
            if g['role']=='required':
                required_options.append(options)
        combinations=1
        for opts in required_options:
            combinations*=len(opts)
        if combinations>64:
            raise ValueError('Too many gold alternatives for exact oracle; report explicitly instead of approximation')
        results=[minimum_cover(set().union(*choice),chunk_sets) for choice in product(*required_options)]
        exact=[r for r in results if r['status']=='exact']
        best=min(exact,key=lambda r:r['count']) if exact else results[0]
        if exact and any(r['status']=='search_limit' for r in results) and best['count']>1:
            best=dict(best,status='search_limit',upper_bound=best['count'],count=None)
        questions.append({'question_id':record['question_id'],'question':record['question'],
                          'minimum_cover':best,'example_chunk_ids':[chunks[i]['chunk_id'] for i in best['chunk_indices']]})
    required_groups=[g for g in groups if g['role']=='required']
    row_checks=[]
    for u in units:
        if u['kind']!='table_row':
            continue
        prefix,body,suffix=map(coordinates,(u['prefix'],u['body'],u['suffix']))
        relevant=[i for i,c in enumerate(chunks) if u['unit_id'] in c['unit_ids'] and body & chunk_sets[i]]
        row_checks.append({'unit_id':u['unit_id'], 'source_id':u['source_id'],'page':u['page'],
                           'row_complete_in_one_chunk':any(prefix|body|suffix<=c for c in chunk_sets),
                           'fragments':len(relevant),
                           'fragments_with_header_and_key':sum(prefix<=chunk_sets[i] for i in relevant),
                           'fragments_with_required_notes':sum(suffix<=chunk_sets[i] for i in relevant)})
    steps=[coordinates(u['value']) for u in units if u['kind']=='list_item']
    lengths=[len(c['text']) for c in chunks]
    instances=sum(m['text_end']-m['text_start'] for c in chunks for m in c['text_map'])
    common_instances=sum(m['text_end']-m['text_start'] for u in units for m in u['value']['text_map'])
    minimums=[q['minimum_cover']['count'] for q in questions if q['minimum_cover']['status']=='exact']
    return {'errors':errors,'chunk_count':len(chunks),'length_chars':{'min':min(lengths),'median':percentile(lengths,.5),
             'p95':percentile(lengths,.95),'max':max(lengths),'mean':round(mean(lengths),2)},
            'oversize_chunks':sum(n>size for n in lengths),
            'unique_source_chars':len(source_coordinates),'covered_source_chars':len(all_coordinates&source_coordinates),
            'mapped_char_instances':instances,'common_input_mapped_instances':common_instances,
            'common_serialization_expansion':round(common_instances/len(source_coordinates),4),
            'source_repetition_ratio':round(1-len(source_coordinates)/instances,4),
            'extra_mapped_chars_vs_common_input':instances-common_instances,
            'gold_spans_total':len(spans),'gold_spans_retained':sum(s['complete_in_corpus'] for s in spans),
            'gold_spans_single_chunk':sum(s['complete_in_one_chunk'] for s in spans),
            'required_groups_total':len(required_groups),
            'required_groups_single_chunk':sum(g['single_chunk_complete'] for g in required_groups),
            'oracle_mean_min_chunks':round(mean(minimums),3) if questions and len(minimums)==len(questions) else None,
            'oracle_max_min_chunks':max(minimums) if questions and len(minimums)==len(questions) else None,
            'steps_total':len(steps),'steps_single_chunk':sum(any(s<=c for c in chunk_sets) for s in steps),
            'table_rows_total':len(row_checks),
            'table_rows_single_chunk':sum(r['row_complete_in_one_chunk'] for r in row_checks),
            'table_fragments':sum(r['fragments'] for r in row_checks),
            'table_fragments_with_headers':sum(r['fragments_with_header_and_key'] for r in row_checks),
            'table_fragments_with_notes':sum(r['fragments_with_required_notes'] for r in row_checks),
            'groups':groups,'questions':questions,'spans':spans,'table_checks':row_checks}

from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from miniassist.chunking.text import literal, join, cut, coordinates
from miniassist.chunking.splitters import windows, recursive_spans, split
from miniassist.chunking.units import prepare
from miniassist.chunking.evaluation import minimum_cover, anchor_set
from miniassist.parsing.models import sha256


def value(text,start=0,page=1):
    return {'text':text,'text_map':[{'text_start':0,'text_end':len(text),'source_start':start,
            'source_end':start+len(text),'source_id':'S1','page':page,'block_id':'b'}]}


def block(ident,text,start=0,page=1,section='s',kind='paragraph'):
    return {'block_id':ident,'block_type':kind,'page':page,'text':text,
            'text_map':[{'text_start':0,'text_end':len(text),'source_start':start,'source_end':start+len(text)}],
            'section_path':[section], 'section_ids':[section], 'table':None,
            'metadata':{'source_id':'S1','source_version':'test','source_sha256':'test'}}


class TextTests(unittest.TestCase):
    def test_slice_clips_provenance(self):
        clipped=cut(value('甲乙丙丁',10),1,3)
        self.assertEqual(clipped['text'],'乙丙')
        self.assertEqual(coordinates(clipped),{('S1',1,11),('S1',1,12)})
        self.assertEqual(clipped['text_map'][0]['text_start'],0)

    def test_disjoint_join_keeps_page_identity_and_literals_unmapped(self):
        merged=join([value('甲',10,1),literal('分类说明'),value('乙',20,2)],'|')
        self.assertEqual(coordinates(merged),{('S1',1,10),('S1',2,20)})
        self.assertEqual(merged['text_map'][-1]['text_start'],7)

    def test_repetition_does_not_create_new_source_coordinates(self):
        repeated=join([value('事实',8),value('事实',8)])
        self.assertEqual(len(coordinates(repeated)),2)


class SplitTests(unittest.TestCase):
    def test_fixed_windows_are_length_limited_with_exact_overlap(self):
        self.assertEqual(list(windows('abcdefghij',5,2)),[(0,5),(3,8),(6,10)])

    def test_recursive_windows_cover_input_without_gaps_and_progress(self):
        for text in ['一。二。三。四。五。','abcdefg'*20,'段一\n\n段二。长段三，继续。\n第四行']:
            for size in [5,9,20]:
                spans=list(windows(text,size,2,True))
                self.assertEqual(set().union(*(set(range(a,z)) for a,z in spans)),set(range(len(text))))
                self.assertTrue(all(z-a<=size for a,z in spans))
                self.assertTrue(all(spans[i+1][0]>spans[i][0] for i in range(len(spans)-1)))

    def test_recursive_preserves_short_sentence_boundary(self):
        self.assertEqual(list(windows('甲乙。丙丁。戊己。',7,3,True)),[(0,6),(3,9)])

    def test_invalid_parameters_fail(self):
        for size,overlap in [(0,0),(10,10),(10,-1)]:
            with self.assertRaises(ValueError):
                list(windows('text',size,overlap))

    def test_structured_chunks_do_not_cross_sections(self):
        blocks=[block('a','第一段。',section='a'),block('b','第二段。',10,section='b')]
        chunks=split(prepare(blocks),blocks,'structure',50,5)
        self.assertEqual(len(chunks),2)
        self.assertTrue(all(len(c['section_paths'])==1 for c in chunks))

    def test_real_heading_can_be_repeated_with_exact_provenance(self):
        blocks=[block('h','充电',0,kind='heading'),block('b','需要关闭电源。',10,page=2)]
        chunks=split(prepare(blocks),blocks,'structure',12,2)
        self.assertTrue(all({('S1',1,0),('S1',1,1)}<=coordinates(c) for c in chunks))
        self.assertEqual(set().union(*(coordinates(c) for c in chunks)),
                         {('S1',1,i) for i in range(2)}|{('S1',2,i) for i in range(10,17)})

    def test_long_table_row_repeats_header_key_and_notes(self):
        unit={'source_id':'S1','source_version':'test','source_sha256':'test','block_id':'b',
              'unit_id':'row','page':1,'section_path':[],'section_ids':[], 'release_date':None,
              'dom_xpath':None,'links':[],'kind':'table_row',
              'prefix':value('表头行键',0),'body':value('甲乙。丙丁。戊己。庚辛。',10),'suffix':value('脚注',30)}
        unit['value']=join([unit['prefix'],unit['body'],unit['suffix']])
        chunks=split([unit],[],'structure',14,2)
        required=coordinates(unit['prefix'])|coordinates(unit['suffix'])
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(required<=coordinates(c) for c in chunks))
        self.assertTrue(all(len(c['text'])<=14 for c in chunks))
        self.assertEqual(set().union(*(coordinates(c) for c in chunks)),coordinates(unit['value']))

    def test_too_large_required_context_is_explicitly_flagged(self):
        unit={'source_id':'S1','source_version':'test','source_sha256':'test','block_id':'b',
              'unit_id':'row','page':1,'section_path':[],'section_ids':[], 'release_date':None,
              'dom_xpath':None,'links':[],'kind':'table_row',
              'prefix':value('完整表头和行键',0),'body':value('事实',10),'suffix':value('必要脚注',30)}
        unit['value']=join([unit['prefix'],unit['body'],unit['suffix']])
        chunks=split([unit],[],'structure',10,2)
        self.assertEqual(len(chunks),1)
        self.assertIn('context_exceeds_budget',chunks[0]['flags'])


class OracleTests(unittest.TestCase):
    def test_exact_cover_beats_greedy(self):
        result=minimum_cover(set(range(1,7)),[{1,2,3,4},{1,2,5},{3,4,6}])
        self.assertEqual(result['status'],'exact')
        self.assertEqual(result['count'],2)
        self.assertEqual(set(result['chunk_indices']),{1,2})

    def test_required_fragments_must_all_be_present(self):
        self.assertEqual(minimum_cover({1,2,3},[{1,2},{3}])['count'],2)
        self.assertEqual(minimum_cover({1,2,3},[{1,2}])['status'],'uncovered')

    def test_duplicate_chunks_cannot_cover_missing_evidence(self):
        self.assertEqual(minimum_cover({1,2},[{1},{1}])['status'],'uncovered')

    def test_search_limit_does_not_claim_exact_minimum(self):
        result=minimum_cover({1,2,3},[{1,2},{2,3}],state_limit=0)
        self.assertEqual(result['status'],'search_limit')
        self.assertIsNone(result['count'])

    def test_anchor_validation_rejects_changed_text(self):
        a={'source_id':'S1','pdf_page':1,'start_char':0,'end_char':2,'exact_text':'错误',
           'page_text_sha256':sha256('原文'.encode())}
        with self.assertRaises(ValueError):
            anchor_set(a,{('S1',1):'原文'})


if __name__=='__main__':
    unittest.main()
